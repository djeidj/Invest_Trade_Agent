"""tracing 落库层:trace_session(会话表) + trace_span(调用明细表)。

概念对齐工业界:一次问答 = 一个 trace(会话),其中每次模型调用/工具调用
= 一个 span。LangSmith/Langfuse/OpenTelemetry 用的就是这套词汇,
我们用几十行 SQLite 实现最小同构版——概念懂了,以后换任何观测平台都不陌生。

教学点【成本快照】:cost_yuan 在入账时按"当时的牌价"算好存下来,
而不是只存 token 数。牌价会变,账单不能变——和发票一个道理。
"""
import sqlite3
import time
from typing import Any

from config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_session (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,            -- 会话开始时间
    question    TEXT NOT NULL,
    answer      TEXT,
    model       TEXT NOT NULL,
    status      TEXT DEFAULT 'running',   -- running / ok / error: ...
    duration_ms INTEGER,                  -- 整个会话耗时(含排队/网络)
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_yuan   REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trace_span (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,         -- 属于哪个会话
    seq         INTEGER NOT NULL,         -- 会话内序号,按发生顺序
    kind        TEXT NOT NULL,            -- 'llm' | 'tool'
    name        TEXT NOT NULL,            -- 'step 1' / 工具名
    detail      TEXT,                     -- 参数/结果摘要
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_yuan   REAL DEFAULT 0,
    latency_ms  INTEGER DEFAULT 0
);
"""


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)  # 幂等建表
    return conn


def start_session(question: str, model: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO trace_session (ts, question, model) VALUES (?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), question, model),
        )
        return int(cur.lastrowid)


def add_span(
    session_id: int,
    seq: int,
    kind: str,
    name: str,
    detail: str = "",
    pt: int = 0,
    ct: int = 0,
    cost: float = 0.0,
    latency_ms: int = 0,
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO trace_span"
            " (session_id, seq, kind, name, detail, prompt_tokens,"
            "  completion_tokens, cost_yuan, latency_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, seq, kind, name, detail, pt, ct, cost, latency_ms),
        )


def end_session(
    session_id: int,
    answer: str,
    pt: int,
    ct: int,
    cost: float,
    duration_ms: int,
    status: str = "ok",
) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE trace_session SET answer=?, prompt_tokens=?, completion_tokens=?,"
            " cost_yuan=?, duration_ms=?, status=? WHERE id=?",
            (answer, pt, ct, cost, duration_ms, status, session_id),
        )


def recent_sessions(limit: int = 10) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trace_session ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def session_spans(session_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trace_span WHERE session_id = ? ORDER BY seq", (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]
