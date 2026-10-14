"""向量存储与检索:SQLite 存向量 + numpy 暴力余弦。

教学点1【祛魅:向量库没有魔法】:
检索 = 查询向量与库内每个向量算余弦相似度,排序取前 k。
百级文档暴力算一遍 <1ms;语料到十万级、需要 ANN 索引时再上 FAISS/Chroma。
教学点2【防幻觉设计】:min_score 相似度阈值——低于阈值视为"语料里没有",
返回空,让 agent 诚实说"没找到",而不是拿不相干的内容硬答。
"""
import sqlite3

import numpy as np

from config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_vec (
    url          TEXT PRIMARY KEY,  -- 对应 news.url,重复建索引按 URL 覆盖
    symbol       TEXT NOT NULL,
    text         TEXT NOT NULL,     -- 被向量化的文本(标题+内容)
    title        TEXT,
    source       TEXT,
    publish_time TEXT,
    vec          BLOB NOT NULL      -- float32 数组的字节流(比 JSON 省一半空间)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def upsert(items: list[dict]) -> None:
    rows = [
        {**it, "vec": np.asarray(it["vec"], dtype=np.float32).tobytes()} for it in items
    ]
    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO news_vec"
            " (url, symbol, text, title, source, publish_time, vec)"
            " VALUES (:url, :symbol, :text, :title, :source, :publish_time, :vec)",
            rows,
        )


def search(
    query_vec: list[float],
    top_k: int = 4,
    min_score: float = 0.3,
    symbol: str = "",
) -> list[dict]:
    """余弦相似度检索,返回带元数据的片段(供引用溯源)。"""
    with _connect() as conn:
        sql = "SELECT url, symbol, text, title, source, publish_time, vec FROM news_vec"
        params: tuple = ()
        if symbol:
            sql += " WHERE symbol = ?"
            params = (symbol,)
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []

    mat = np.stack([np.frombuffer(r[6], dtype=np.float32) for r in rows])
    q = np.asarray(query_vec, dtype=np.float32)
    # 余弦相似度 = 点积 / (模长乘积);+1e-9 防零向量除零
    sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
    order = np.argsort(sims)[::-1]  # 从高到低

    hits = []
    for i in order:
        if sims[i] < min_score or len(hits) >= top_k:
            break
        r = rows[i]
        hits.append(
            {
                "url": r[0],
                "symbol": r[1],
                "text": r[2],
                "title": r[3],
                "source": r[4],
                "publish_time": r[5],
                "score": float(sims[i]),
            }
        )
    return hits
