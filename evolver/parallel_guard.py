"""
并行线程防护模块 (Parallel Guard)
解决 Stage 4/5/6/7 并行执行时某个线程卡死拖垮整个 daemon 的问题。

核心功能:
1. run_with_timeout  — 单函数超时执行,超时返回错误而非无限阻塞
2. run_parallel      — 多函数并行执行,每个独立超时,互不影响
3. cleanup_zombies   — 僵尸线程巡检,每轮开始时调用

设计原则 (低性能硬件友好):
- 不用 ctypes/multiprocessing,纯 ThreadPoolExecutor + result(timeout)
- 超时后线程不强制 kill (Python 限制),但结果被丢弃,不影响主循环
- 零依赖,纯标准库
"""
import concurrent.futures
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple


# 默认超时 (秒) — 单个 Stage 最长执行时间
DEFAULT_STAGE_TIMEOUT = 30

# 僵尸线程追踪
_active_executors: List[concurrent.futures.ThreadPoolExecutor] = []
_lock = threading.Lock()


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = DEFAULT_STAGE_TIMEOUT,
    tag: str = "",
) -> Dict[str, Any]:
    """
    单函数超时执行。
    超时后返回 {"_timeout": True, "tag": tag, "timeout_sec": timeout},
    而非无限阻塞主循环。
    """
    if kwargs is None:
        kwargs = {}
    tag = tag or getattr(func, "__name__", "unknown")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(func, *args, **kwargs)
            try:
                result = fut.result(timeout=timeout)
                return result if isinstance(result, dict) else {"_result": result}
            except concurrent.futures.TimeoutError:
                print(f"  ⏰ [ParallelGuard] '{tag}' 超时 ({timeout}s),跳过")
                return {"_timeout": True, "tag": tag, "timeout_sec": timeout}
    except Exception as e:
        print(f"  💥 [ParallelGuard] '{tag}' 异常: {type(e).__name__}: {e}")
        return {"_error": f"{type(e).__name__}: {e}", "tag": tag}


def run_parallel(
    tasks: List[Tuple[str, Callable]],
    timeout: float = DEFAULT_STAGE_TIMEOUT,
    max_workers: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """
    多函数并行执行,每个独立超时。
    tasks: [(name, func), ...] — func 应是无参的(用 lambda 包裹)
    返回: {name: result_dict, ...}
    超时的 task 返回 {"_timeout": True}
    异常的 task 返回 {"_error": "..."}
    """
    if not tasks:
        return {}

    # 单任务直接顺序执行 (省去线程池开销)
    if len(tasks) == 1:
        name, func = tasks[0]
        try:
            result = func()
            return {name: result if isinstance(result, dict) else {"_result": result}}
        except Exception as e:
            return {name: {"_error": f"{type(e).__name__}: {e}"}}

    results: Dict[str, Dict[str, Any]] = {}

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}
            for name, func in tasks:
                fut = pool.submit(func)
                future_map[fut] = name

            for fut in concurrent.futures.as_completed(future_map, timeout=timeout + 5):
                name = future_map[fut]
                try:
                    result = fut.result(timeout=timeout)
                    results[name] = result if isinstance(result, dict) else {"_result": result}
                except concurrent.futures.TimeoutError:
                    print(f"  ⏰ [ParallelGuard] '{name}' 超时 ({timeout}s),跳过")
                    results[name] = {"_timeout": True, "tag": name}
                except Exception as e:
                    print(f"  💥 [ParallelGuard] '{name}' 异常: {type(e).__name__}: {e}")
                    results[name] = {"_error": f"{type(e).__name__}: {e}"}

    except concurrent.futures.TimeoutError:
        # as_completed 整体超时 — 未完成的 task 标记超时
        for name, _ in tasks:
            if name not in results:
                print(f"  ⏰ [ParallelGuard] '{name}' 整体超时,跳过")
                results[name] = {"_timeout": True, "tag": name}

    return results


def cleanup_zombies() -> int:
    """
    僵尸线程巡检回收。
    每轮主循环开始时调用,清理上一轮可能残留的线程池引用。
    返回清理的数量。
    """
    cleaned = 0
    with _lock:
        still_alive = []
        for executor in _active_executors:
            try:
                # 尝试 shutdown(wait=False),如果不报错说明还活着
                executor.shutdown(wait=False)
                # 检查是否还有线程在跑
                if executor._threads:
                    still_alive.append(executor)
                    cleaned += 1
            except Exception:
                pass
        _active_executors.clear()
        _active_executors.extend(still_alive)
    return cleaned


def get_thread_count() -> int:
    """返回当前活跃线程数 (含主线程)"""
    return threading.active_count()
