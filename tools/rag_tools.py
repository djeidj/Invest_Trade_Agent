"""RAG 检索工具:从向量库取【非结构化文本】。

collection 是"语料类别"(每个是独立索引),用 config.settings.RAGCollection 白名单约束:
- 现在只有 NEWS("news"),由 stock_tools.ingest_stock_news 写入。
- 加新类别:只在 RAGCollection 里加成员 + 提供对应写入器;别放开给模型自由填字符串,
  否则模型会乱编 collection 名 → 命中空库或污染索引。pydantic 会把它渲染成 enum 白名单。
"""
from langchain.tools import tool

from config.settings import RAGCollection
from storage import vecdb


@tool
def query_rag(
    query: str,
    symbol: str = "",
    collection: RAGCollection = RAGCollection.NEWS,
    top_k: int = 4,
) -> str:
    """检索向量库里的非结构化文本(默认个股新闻),语义匹配。

    回答新闻/舆情类问题前先用本工具。

    Args:
        query(str): 想找的内容描述,例如 "i茅台 APP 上架新品"
        symbol(str): 可选,6位股票代码限定某只股票
        collection(RAGCollection): 语料类别枚举,目前只有 "news"
        top_k(int): 返回块数,默认 4
    """
    hits = vecdb.search(query, symbol=symbol, collection=collection, k=top_k)
    if not hits:
        return (
            f"RAG collection '{collection}' 中未检索到相关内容(未命中)。"
            "若该类别新闻尚未入库,请先调用 ingest_stock_news 抓取再检索;"
            "确认无数据则如实告知用户,不要编造。"
        )

    lines = [
        f"[{i + 1}] ({h['symbol']}, 相关度 {h['score']:.2f})\n{h['text']}"
        for i, h in enumerate(hits)
    ]
    return "检索到以下内容(引用时请标注[编号]):\n" + "\n\n".join(lines)
