"""非结构化文本的向量检索地基:Chroma,【按语料类别分 collection】。

设计:collection 名 = 语料类别(news / research / announcement ...)。
- 类别是粗、稳、互不相干的划分 → 各自独立索引,不同语料不互相污染相关度分数。
- 高基数、需组合筛选的维度(symbol)留在 metadata 里用 where 过滤,不建 collection
  (按 symbol 建库会失去跨股票检索、还撞 collection 名合法性)。
新增一类文本 = 换个 collection 名调用,零改本模块代码 → 通用。

注意分工:结构化数字(实时价/K线/财报数值)不放这里,走精确查/直连返回。
本模块被 tools/news_tools(写入) + tools/rag_tools(检索) 使用,构成个股新闻的读穿式 RAG。

关键坑(都实测踩过):
- OpenAIEmbeddings 的维度字段是 `dimensions`,不是 `embedding_dimensions`。
- GLM embedding-3:单请求批量上限 64(`chunk_size=64`),且要 `check_embedding_ctx_length=False`。
- Chroma 默认 L2 距离,余弦要显式 `collection_metadata={"hnsw:space":"cosine"}`。
- 要"相关度分数(0~1)"用 `similarity_search_with_relevance_scores`(它内部 1-距离),
  别用 `Chroma.query()`(返回 dict、给的是距离,越小越像)。
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

from config.settings import (
    CHROMA_PATH,
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    GLM_API_KEY,
    GLM_BASE_URL,
)

_embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    base_url=GLM_BASE_URL,
    api_key=GLM_API_KEY,
    dimensions=EMBED_DIMENSIONS,
    check_embedding_ctx_length=False,  # 关掉 tiktoken 检查(GLM 词表不认)
    chunk_size=64,                       # GLM embedding 单请求上限 64 条
)

# 中文分隔符优先,避免把句子切得支离破碎
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", "。", "!", "?", ";", ",", " ", ""],
)


def _vs(collection: str) -> Chroma:
    return Chroma(
        collection_name=collection,
        embedding_function=_embeddings,
        persist_directory=str(CHROMA_PATH),
        collection_metadata={"hnsw:space": "cosine"},
    )


def put(text: str, *, symbol: str, signature: str, collection: str) -> int:
    """把一段文本切块、向量化后写入指定 collection。返回写入块数。

    symbol 存 metadata 供检索过滤;类别用 collection 名区分(不再占一个 type 字段)。
    块 id = "{signature}#{序号}" 确定性的:add_documents 按 id upsert,重复写覆盖不堆积。
    """
    chunks = _splitter.split_text(text)
    if not chunks:
        return 0
    docs = [
        Document(
            page_content=c,
            metadata={"symbol": symbol, "signature": signature, "chunk": i},
        )
        for i, c in enumerate(chunks)
    ]
    ids = [f"{signature}#{i}" for i in range(len(chunks))]
    _vs(collection).add_documents(docs, ids=ids)
    return len(docs)


def search(
    query: str,
    *,
    symbol: str = "",
    collection: str,
    k: int = 4,
    min_score: float = 0.2,
) -> list[dict]:
    """在指定 collection 内做语义检索,可选按 symbol 过滤。

    similarity_search_with_relevance_scores 返回相关度(0~1,越大越相关);
    低于 min_score 视为"没有",过滤掉。symbol 空 = 跨该类别全部语料检索。
    返回 [{text, symbol, signature, score}]。
    """
    # _vs(collection) 会在 collection 名非法(空/超长/非法字符)时抛错,一并放进 try:
    # 上层把"建库失败"也当作未命中处理,不会让 query_rag 因为模型乱传 collection 而崩。
    kwargs = {"filter": {"symbol": symbol}} if symbol else {}
    try:
        vs = _vs(collection)
        results = vs.similarity_search_with_relevance_scores(query, k=k, **kwargs)
    except Exception:
        return []  # 空库/非法 collection/异常一律当作未命中,交给上层提示

    hits = []
    for doc, score in results:
        if score < min_score:
            continue
        m = doc.metadata
        hits.append(
            {
                "text": doc.page_content,
                "symbol": m.get("symbol", ""),
                "signature": m.get("signature", ""),
                "score": float(score),
            }
        )
    return hits


def list_collections() -> list[dict]:
    """列出持久化目录里所有 collection 及块数,供调试页展示。

    用原生 PersistentClient 只读列举,不走 _vs()(_vs 会 get_or_create,顺手建出垃圾 collection)。
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return [{"name": c.name, "count": c.count()} for c in client.list_collections()]
