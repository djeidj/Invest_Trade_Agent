"""把 news 表语料向量化,建检索索引(news_vec 表)。幂等:重复跑按 URL 覆盖。

教学点:入库语料的质量决定检索上限——空内容跳过;
text 用"标题+内容"拼接,标题往往比正文更能代表主题。

运行: uv run python -m scripts.build_index
"""
from rag import embedder, store
from storage import db


def main() -> None:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT url, symbol, title, content, source, publish_time FROM news"
            " WHERE content IS NOT NULL AND content != ''"
        ).fetchall()

    items = [
        {
            "url": r["url"],
            "symbol": r["symbol"],
            "text": f"{r['title']}。{r['content']}",
            "title": r["title"],
            "source": r["source"],
            "publish_time": r["publish_time"],
        }
        for r in rows
    ]
    print(f"待向量化 {len(items)} 条...")
    vecs = embedder.embed([it["text"] for it in items])
    for it, v in zip(items, vecs):
        it["vec"] = v
    store.upsert(items)
    print(f"索引完成: news_vec 共 {len(items)} 条向量")


if __name__ == "__main__":
    main()
