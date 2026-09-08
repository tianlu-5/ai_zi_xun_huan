"""
长期记忆系统 - 基于 SQLite + 向量检索

功能：
1. 存储每次迭代的经验（目标、修改、结果、评分）
2. 按语义检索相关历史经验
3. 自动聚类和去重
4. 支持导出/导入经验库
"""

import json
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

from config import Config


@dataclass
class MemoryEntry:
    """一条记忆条目"""
    entry_id: str
    timestamp: str
    iteration: int
    objective: str
    file_paths: str          # 逗号分隔
    modifications_summary: str  # 修改摘要
    verdict: str              # 进步/持平/退步
    quality_delta: float      # 质量变化
    contract_pass_ratio: float
    embedding: Optional[List[float]] = None  # 向量
    tags: List[str] = field(default_factory=list)
    success_rate: float = 1.0
    times_used: int = 0


class MemoryStore:
    """长期记忆存储引擎（SQLite + 向量检索）"""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Config.LOG_DIR / "memory"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / "memory.db"
        self._init_db()
        self._embedding_model = None  # 延迟加载

    def _init_db(self):
        """初始化 SQLite 表"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                entry_id TEXT PRIMARY KEY,
                timestamp TEXT,
                iteration INTEGER,
                objective TEXT,
                file_paths TEXT,
                modifications_summary TEXT,
                verdict TEXT,
                quality_delta REAL,
                contract_pass_ratio REAL,
                tags TEXT,
                success_rate REAL,
                times_used INTEGER,
                embedding BLOB
            )
        """)

        # 创建索引加速查询
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_verdict ON memories(verdict)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_iteration ON memories(iteration)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")

        conn.commit()
        conn.close()

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本的向量（使用本地 Ollama embedding）"""
        import urllib.request
        try:
            body = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/embeddings",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            return json.loads(resp.read()).get("embedding", [])
        except Exception:
            return []

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import math
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def add(self, entry: MemoryEntry) -> str:
        """
        添加一条记忆
        """
        # 生成 embedding（如果有内容）
        text_to_embed = f"{entry.objective} {entry.modifications_summary} {entry.verdict}"
        embedding = self._get_embedding(text_to_embed)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO memories (
                entry_id, timestamp, iteration, objective, file_paths,
                modifications_summary, verdict, quality_delta,
                contract_pass_ratio, tags, success_rate, times_used, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.entry_id,
            entry.timestamp,
            entry.iteration,
            entry.objective,
            entry.file_paths,
            entry.modifications_summary,
            entry.verdict,
            entry.quality_delta,
            entry.contract_pass_ratio,
            json.dumps(entry.tags),
            entry.success_rate,
            entry.times_used,
            json.dumps(embedding) if embedding else None,
        ))

        conn.commit()
        conn.close()
        return entry.entry_id

    def search_by_semantic(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        按语义搜索相关记忆
        """
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM memories ORDER BY timestamp DESC LIMIT 100")
        rows = cursor.fetchall()
        conn.close()

        # 计算相似度排序
        results = []
        for row in rows:
            embed_json = row[12]  # embedding 列
            if embed_json:
                try:
                    embed = json.loads(embed_json)
                    sim = self._cosine_sim(query_embedding, embed)
                    if sim > 0.3:
                        results.append((sim, self._row_to_entry(row)))
                except Exception:
                    continue

        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:limit]]

    def search_by_file(self, file_path: str, limit: int = 5) -> List[MemoryEntry]:
        """按文件路径搜索相关记忆"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memories
            WHERE file_paths LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{file_path}%", limit))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_entry(row) for row in rows]

    def search_by_verdict(self, verdict: str, limit: int = 10) -> List[MemoryEntry]:
        """按判定结果搜索"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memories
            WHERE verdict = ?
            ORDER BY quality_delta DESC
            LIMIT ?
        """, (verdict, limit))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_entry(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆库统计信息"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT verdict, COUNT(*) FROM memories GROUP BY verdict")
        verdict_counts = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "total_entries": total,
            "verdict_distribution": verdict_counts,
            "storage_path": str(self.db_path),
        }

    def _row_to_entry(self, row) -> MemoryEntry:
        """将 SQL 行转换为 MemoryEntry"""
        return MemoryEntry(
            entry_id=row[0],
            timestamp=row[1],
            iteration=row[2],
            objective=row[3],
            file_paths=row[4],
            modifications_summary=row[5],
            verdict=row[6],
            quality_delta=row[7],
            contract_pass_ratio=row[8],
            tags=json.loads(row[9]) if row[9] else [],
            success_rate=row[10],
            times_used=row[11],
        )

    def format_for_prompt(self, query: str, limit: int = 3) -> str:
        """
        格式化记忆，注入到 Prompt 中
        """
        entries = self.search_by_semantic(query, limit=limit)
        if not entries:
            return "（暂无相关历史经验）"

        lines = ["=== 📚 相关历史经验（参考学习） ===\n"]
        for i, e in enumerate(entries, 1):
            lines.append(f"## 经验 {i}（迭代 #{e.iteration}）")
            lines.append(f"- **目标**：{e.objective[:120]}")
            lines.append(f"- **文件**：{e.file_paths}")
            lines.append(f"- **结果**：{e.verdict}（质量变化 {e.quality_delta:+.2%}）")
            lines.append(f"- **摘要**：{e.modifications_summary[:100]}")
            if e.tags:
                lines.append(f"- **标签**：{', '.join(e.tags)}")
            lines.append("")

        return "\n".join(lines)


# 全局单例
_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store