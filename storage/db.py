"""SQLite 存储层。

教学点:为什么用 SQLite 而不是 MySQL/PostgreSQL?
单机项目、零运维、性能完全够用——选型的第一原则是"用最简单的够用方案",
而不是"用最炫的方案"。面试被问到架构选型时,能说出"为什么不用X"比堆技术栈更加分。
"""
import sqlite3
from typing import Any

from config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bar (
    symbol TEXT NOT NULL,          -- 股票代码
    date   TEXT NOT NULL,          -- 交易日 YYYY-MM-DD
    open REAL, high REAL, low REAL, close REAL,
    volume REAL,                   -- 成交量
    amount REAL,                   -- 成交额
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS news (
    url          TEXT PRIMARY KEY, -- 用链接做主键,天然去重
    symbol       TEXT NOT NULL,
    title        TEXT,
    content      TEXT,
    source       TEXT,
    publish_time TEXT
);
"""


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # sqlite 只建文件不建父目录
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 让查询结果能按列名取值
    conn.executescript(_SCHEMA)  # 幂等:表已存在就跳过
    return conn


def upsert_bars(rows: list[dict[str, Any]]) -> int:
    """写入/更新日K线。INSERT OR REPLACE:重复拉取不会产生脏数据。"""
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_bar"
            " (symbol, date, open, high, low, close, volume, amount)"
            " VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :amount)",
            rows,
        )
    return len(rows)


def upsert_news(rows: list[dict[str, Any]]) -> int:
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO news"
            " (url, symbol, title, content, source, publish_time)"
            " VALUES (:url, :symbol, :title, :content, :source, :publish_time)",
            rows,
        )
    return len(rows)
