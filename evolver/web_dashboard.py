"""
Web Dashboard - 实时可视化监控面板

使用方式:
    # 在 main.py 中启动
    from evolver.web_dashboard import start_dashboard
    start_dashboard(daemon, port=8080)

访问: http://localhost:8080
"""

import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️ FastAPI 未安装，Web Dashboard 不可用")
    print("   安装: pip install fastapi uvicorn websockets")


class DashboardServer:
    """Web Dashboard 服务器"""

    def __init__(self, daemon=None, port: int = 8080):
        self.daemon = daemon
        self.port = port
        self._server_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._app = None
        self._active_websockets = []

        if FASTAPI_AVAILABLE:
            self._init_app()
        else:
            self._app = None

    def _init_app(self):
        """初始化 FastAPI 应用"""
        app = FastAPI(title="AI 自迭代 Dashboard", version="1.0")

        # ── 页面路由 ──
        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._get_html()

        # ── API: 系统状态 ──
        @app.get("/api/status")
        async def api_status():
            return self._get_status()

        # ── API: 迭代历史 ──
        @app.get("/api/history")
        async def api_history(limit: int = 30):
            return self._get_history(limit)

        # ── API: 基准数据 ──
        @app.get("/api/benchmark")
        async def api_benchmark():
            return self._get_benchmark()

        # ── API: 合约状态 ──
        @app.get("/api/contracts")
        async def api_contracts():
            return self._get_contracts()

        # ── API: 长期记忆库 ──
        @app.get("/api/memory")
        async def api_memory():
            return self._get_memory()

        # ── API: 控制 ──
        @app.post("/api/control/pause")
        async def api_pause():
            return self._control_pause()

        @app.post("/api/control/resume")
        async def api_resume():
            return self._control_resume()

        @app.post("/api/control/trigger")
        async def api_trigger():
            return self._control_trigger()

        # ── WebSocket 实时推送 ──
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._active_websockets.append(websocket)
            try:
                while not self._stop_event.is_set():
                    # 每秒推送状态更新
                    try:
                        status = self._get_status()
                        await websocket.send_json(status)
                    except Exception:
                        break
                    await websocket.receive_text()  # 等待 ping/pong
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self._active_websockets:
                    self._active_websockets.remove(websocket)

        self._app = app

    def _get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "running": False,
            "cycle_count": 0,
            "uptime": 0,
            "model": "N/A",
            "gpu_temp": None,
            "last_cycle_status": "N/A",
            "consecutive_no_change": 0,
            "memory_entries": 0,
        }

        if self.daemon:
            s = self.daemon.get_status()
            status["running"] = s.get("running", False)
            status["cycle_count"] = s.get("cycle_count", 0)
            status["uptime"] = s.get("uptime_sec", 0)
            status["consecutive_no_change"] = s.get("consecutive_no_change", 0)
            
            # 获取模型信息
            if self.daemon.evolver and hasattr(self.daemon.evolver, 'doubao'):
                client = self.daemon.evolver.doubao
                status["model"] = getattr(client, 'model_id', 'N/A')
            
            # 获取 GPU 温度
            if hasattr(self.daemon, '_last_gpu_temp'):
                status["gpu_temp"] = self.daemon._last_gpu_temp
            
            # 最近一轮状态
            history = getattr(self.daemon, '_history', [])
            if history:
                last = history[-1]
                status["last_cycle_status"] = "成功" if last.get("success", True) else "失败"
                status["last_cycle_errors"] = last.get("errors", [])[:3]
        
        # 记忆条目数
        try:
            from success_patterns import get_pattern_library
            lib = get_pattern_library()
            status["memory_entries"] = len(lib.patterns)
        except Exception:
            pass

        return status

    def _get_history(self, limit: int = 30) -> List[Dict]:
        """获取迭代历史"""
        if not self.daemon:
            return []
        
        history = getattr(self.daemon, '_history', [])
        result = []
        for item in history[-limit:]:
            stages = item.get("stages", {})
            result.append({
                "cycle": item.get("cycle", 0),
                "timestamp": item.get("timestamp", ""),
                "success": item.get("success", True),
                "objective": stages.get("objective", ""),
                "applied": stages.get("self_iteration", {}).get("applied_count", 0),
                "verdict": stages.get("benchmark", {}).get("verdict", "N/A"),
                "risk": stages.get("meta_drive", {}).get("risk_level", "N/A"),
            })
        return result

    def _get_benchmark(self) -> Dict[str, Any]:
        """获取基准数据"""
        from config import Config
        path = Config.LOG_DIR / "benchmark_history.json"
        if not path.exists():
            return {"iterations": [], "scores": [], "verdicts": []}
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            iterations = []
            scores = []
            verdicts = []
            
            for item in data[-50:]:
                if isinstance(item, dict):
                    iterations.append(item.get("iteration_id", 0))
                    scores.append(item.get("score_delta", 0))
                    verdicts.append(item.get("verdict", "持平"))
            
            return {
                "iterations": iterations,
                "scores": scores,
                "verdicts": verdicts,
                "total": len(data),
            }
        except Exception:
            return {"iterations": [], "scores": [], "verdicts": []}

    def _get_contracts(self) -> Dict[str, Any]:
        """获取合约状态"""
        from config import Config
        path = Config.LOG_DIR / "contract_latest.json"
        if not path.exists():
            return {"passed": 0, "failed": 0, "pass_ratio": 0}
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "passed": data.get("passed", 0),
                "failed": data.get("failed", 0),
                "skipped": data.get("skipped", 0),
                "pass_ratio": data.get("pass_ratio", 0),
                "timestamp": data.get("timestamp", ""),
            }
        except Exception:
            return {"passed": 0, "failed": 0, "pass_ratio": 0}

    def _get_memory(self) -> Dict[str, Any]:
        """获取长期记忆库"""
        try:
            from success_patterns import get_pattern_library
            lib = get_pattern_library()
            stats = lib.get_statistics()
            return {
                "total": stats.get("total", 0),
                "avg_improvement": stats.get("avg_improvement", 0),
                "top_dimensions": stats.get("top_dimensions", []),
                "recent": [
                    {
                        "file": p.file_path,
                        "improvement": p.quality_improvement,
                        "verdict": p.verdict,
                    }
                    for p in lib.get_recent_patterns(5)
                ],
            }
        except Exception:
            return {"total": 0}

    def _control_pause(self) -> Dict[str, Any]:
        """暂停 daemon"""
        if self.daemon:
            try:
                # 设置一个内部暂停标志
                setattr(self.daemon, '_paused_by_dashboard', True)
                return {"success": True, "message": "Daemon 已暂停"}
            except Exception as e:
                return {"success": False, "message": str(e)}
        return {"success": False, "message": "Daemon 未运行"}

    def _control_resume(self) -> Dict[str, Any]:
        """恢复 daemon"""
        if self.daemon:
            try:
                setattr(self.daemon, '_paused_by_dashboard', False)
                return {"success": True, "message": "Daemon 已恢复"}
            except Exception as e:
                return {"success": False, "message": str(e)}
        return {"success": False, "message": "Daemon 未运行"}

    def _control_trigger(self) -> Dict[str, Any]:
        """手动触发迭代"""
        if self.daemon:
            try:
                # 发送一个信号让 daemon 立即执行一轮
                setattr(self.daemon, '_force_cycle', True)
                return {"success": True, "message": "已触发手动迭代"}
            except Exception as e:
                return {"success": False, "message": str(e)}
        return {"success": False, "message": "Daemon 未运行"}

    def _get_html(self) -> str:
        """返回 Dashboard HTML 页面"""
        return HTML_TEMPLATE

    def start(self) -> bool:
        """启动 Dashboard 服务器"""
        if not FASTAPI_AVAILABLE:
            print("❌ FastAPI 未安装，无法启动 Dashboard")
            return False
        
        if self._server_thread and self._server_thread.is_alive():
            print("⚠️ Dashboard 已在运行")
            return True

        self._stop_event.clear()
        
        def _run():
            try:
                uvicorn.run(
                    self._app,
                    host="127.0.0.1",
                    port=self.port,
                    log_level="warning",
                )
            except Exception as e:
                print(f"❌ Dashboard 启动失败: {e}")

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        print(f"🌐 Web Dashboard 已启动: http://localhost:{self.port}")
        return True

    def stop(self):
        """停止 Dashboard 服务器"""
        self._stop_event.set()
        if self._server_thread:
            self._server_thread.join(timeout=2)


# ── HTML 模板 ──
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI 自迭代 Dashboard</title>
    <script src="https://cdn.bootcdn.net/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            font-size: 24px;
            margin-bottom: 20px;
            color: #58a6ff;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
            transition: border-color 0.2s;
        }
        .card:hover { border-color: #58a6ff; }
        .card-title {
            font-size: 12px;
            text-transform: uppercase;
            color: #8b949e;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 24px;
            font-weight: 600;
        }
        .card-value.running { color: #3fb950; }
        .card-value.stopped { color: #f85149; }
        .card-sub {
            font-size: 12px;
            color: #8b949e;
            margin-top: 4px;
        }
        .charts {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }
        .chart-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
        }
        .chart-card h3 {
            font-size: 14px;
            margin-bottom: 12px;
            color: #8b949e;
        }
        .table-wrap {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th {
            text-align: left;
            padding: 8px 12px;
            color: #8b949e;
            border-bottom: 1px solid #30363d;
        }
        td {
            padding: 8px 12px;
            border-bottom: 1px solid #21262d;
        }
        .status-ok { color: #3fb950; }
        .status-fail { color: #f85149; }
        .status-warn { color: #d29922; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        .badge.success { background: #1a7f37; color: #fff; }
        .badge.fail { background: #da3633; color: #fff; }
        .badge.neutral { background: #1f6feb; color: #fff; }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 8px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.8; }
        .btn-primary { background: #238636; color: #fff; }
        .btn-danger { background: #da3633; color: #fff; }
        .btn-secondary { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
        .btn-warning { background: #d29922; color: #fff; }
        @media (max-width: 768px) {
            .charts { grid-template-columns: 1fr; }
        }
        .refresh-note {
            font-size: 12px;
            color: #8b949e;
            margin-top: 12px;
            text-align: center;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🔄 AI 自迭代 Dashboard</h1>

    <div class="controls">
        <button class="btn btn-primary" onclick="triggerIteration()">▶️ 手动触发迭代</button>
        <button class="btn btn-warning" onclick="togglePause()">⏸️ 暂停/恢复</button>
        <button class="btn btn-secondary" onclick="refreshData()">🔄 刷新</button>
    </div>

    <div class="grid" id="stats-grid">
        <div class="card"><div class="card-title">运行状态</div><div class="card-value" id="status">-</div></div>
        <div class="card"><div class="card-title">迭代次数</div><div class="card-value" id="cycles">-</div></div>
        <div class="card"><div class="card-title">运行时长</div><div class="card-value" id="uptime">-</div></div>
        <div class="card"><div class="card-title">当前模型</div><div class="card-value" style="font-size:18px;" id="model">-</div></div>
        <div class="card"><div class="card-title">GPU 温度</div><div class="card-value" id="gpu">-</div></div>
        <div class="card"><div class="card-title">记忆条目</div><div class="card-value" id="memory">-</div></div>
        <div class="card"><div class="card-title">连续零进展</div><div class="card-value" id="no-change">-</div></div>
        <div class="card"><div class="card-title">最近一轮</div><div class="card-value" style="font-size:18px;" id="last-status">-</div></div>
    </div>

    <div class="charts">
        <div class="chart-card"><h3>📈 质量评分趋势</h3><canvas id="trendChart"></canvas></div>
        <div class="chart-card"><h3>📊 合约状态</h3><canvas id="contractChart"></canvas></div>
    </div>

    <div class="table-wrap">
        <h3 style="margin-bottom:12px;color:#8b949e;font-size:14px;">📋 最近迭代记录</h3>
        <table>
            <thead><tr><th>轮次</th><th>时间</th><th>目标</th><th>修改数</th><th>判定</th><th>风险</th></tr></thead>
            <tbody id="history-body"></tbody>
        </table>
    </div>
    <div class="refresh-note">🔄 自动刷新: 每 3 秒</div>
</div>

<script>
let ws = null;
let chart = null;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    ws.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            updateUI(data);
        } catch(e) {}
    };
    ws.onclose = function() {
        setTimeout(connectWebSocket, 3000);
    };
}

function updateUI(data) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = data.running ? '🟢 运行中' : '⚪ 已停止';
    statusEl.className = 'card-value ' + (data.running ? 'running' : 'stopped');

    document.getElementById('cycles').textContent = data.cycle_count || 0;
    document.getElementById('uptime').textContent = formatUptime(data.uptime || 0);
    document.getElementById('model').textContent = data.model || 'N/A';
    document.getElementById('gpu').textContent = data.gpu_temp !== null ? data.gpu_temp + '°C' : 'N/A';
    document.getElementById('memory').textContent = data.memory_entries || 0;
    document.getElementById('no-change').textContent = data.consecutive_no_change || 0;

    const lastStatus = document.getElementById('last-status');
    if (data.last_cycle_status === '成功') {
        lastStatus.textContent = '✅ 成功';
        lastStatus.className = 'card-value status-ok';
    } else if (data.last_cycle_status === '失败') {
        lastStatus.textContent = '❌ 失败';
        lastStatus.className = 'card-value status-fail';
    } else {
        lastStatus.textContent = '⏳ 等待中';
        lastStatus.className = 'card-value';
    }

    fetchHistory();
    fetchBenchmark();
    fetchContracts();
}

function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
}

async function fetchHistory() {
    try {
        const resp = await fetch('/api/history?limit=20');
        const data = await resp.json();
        const tbody = document.getElementById('history-body');
        tbody.innerHTML = data.map(item => `
            <tr>
                <td>#${item.cycle}</td>
                <td>${item.timestamp.slice(11,19)}</td>
                <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${item.objective || '-'}</td>
                <td>${item.applied}</td>
                <td><span class="badge ${item.verdict === '进步' ? 'success' : item.verdict === '退步' ? 'fail' : 'neutral'}">${item.verdict || '-'}</span></td>
                <td>${item.risk || '-'}</td>
            </tr>
        `).join('');
    } catch(e) {}
}

async function fetchBenchmark() {
    try {
        const resp = await fetch('/api/benchmark');
        const data = await resp.json();
        if (!data.iterations || data.iterations.length < 2) {
            document.getElementById('trendChart').parentElement.innerHTML = '<h3>📈 质量评分趋势</h3><p style="color:#8b949e;padding:20px;">数据不足，需要至少 2 轮基准测试</p>';
            return;
        }
        renderChart(data);
    } catch(e) {}
}

function renderChart(data) {
    const ctx = document.getElementById('trendChart').getContext('2d');
    if (chart) { chart.destroy(); }
    
    const colors = data.verdicts.map(v => {
        if (v === '进步') return '#3fb950';
        if (v === '退步') return '#f85149';
        return '#d29922';
    });

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.iterations,
            datasets: [{
                label: '质量评分变化',
                data: data.scores,
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                fill: true,
                tension: 0.3,
                pointBackgroundColor: colors,
                pointRadius: 5,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#8b949e' } }
            },
            scales: {
                x: { ticks: { color: '#8b949e' } },
                y: { ticks: { color: '#8b949e' } }
            }
        }
    });
}

async function fetchContracts() {
    try {
        const resp = await fetch('/api/contracts');
        const data = await resp.json();
        const ctx = document.getElementById('contractChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['通过', '失败', '跳过'],
                datasets: [{
                    data: [data.passed || 0, data.failed || 0, data.skipped || 0],
                    backgroundColor: ['#3fb950', '#f85149', '#d29922'],
                    borderColor: '#161b22',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#8b949e' }
                    }
                }
            }
        });
    } catch(e) {}
}

async function triggerIteration() {
    try {
        const resp = await fetch('/api/control/trigger', { method: 'POST' });
        const result = await resp.json();
        alert(result.message || '已触发');
    } catch(e) { alert('触发失败: ' + e.message); }
}

async function togglePause() {
    try {
        const resp = await fetch('/api/control/pause', { method: 'POST' });
        const result = await resp.json();
        if (result.success) {
            alert('已暂停');
        } else {
            // 尝试恢复
            const resp2 = await fetch('/api/control/resume', { method: 'POST' });
            const result2 = await resp2.json();
            alert(result2.message || '已恢复');
        }
    } catch(e) { alert('操作失败: ' + e.message); }
}

async function refreshData() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        updateUI(data);
    } catch(e) {}
}

// 自动刷新（备用）
setInterval(refreshData, 3000);

// 启动 WebSocket
connectWebSocket();
</script>
</body>
</html>
"""


# ── 启动函数 ──
_dashboard_server: Optional[DashboardServer] = None


def start_dashboard(daemon=None, port: int = 8080) -> bool:
    """启动 Web Dashboard（全局单例）"""
    global _dashboard_server
    if _dashboard_server is None:
        _dashboard_server = DashboardServer(daemon, port)
    return _dashboard_server.start()


def stop_dashboard():
    """停止 Web Dashboard"""
    global _dashboard_server
    if _dashboard_server:
        _dashboard_server.stop()
        _dashboard_server = None


def get_dashboard_status() -> Dict[str, Any]:
    """获取 Dashboard 状态"""
    if _dashboard_server:
        return {
            "running": _dashboard_server._server_thread and _dashboard_server._server_thread.is_alive(),
            "port": _dashboard_server.port,
        }
    return {"running": False}