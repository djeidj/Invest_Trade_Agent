"""vecdb 集成测:真 embedding,但落到 isolated_vecdb 的临时目录,不污染真实库。"""
from tests.conftest import needs_key


@needs_key
def test_put_then_search_symbol_scoped(isolated_vecdb):
    n = isolated_vecdb.put(
        "贵州茅台旗下华茅酒上架i茅台,新品定价488元每瓶。",
        symbol="600519",
        signature="url-a",
        collection="news",
    )
    assert n >= 1
    hits = isolated_vecdb.search("华茅 上架", symbol="600519", collection="news")
    assert hits, "应命中刚写入的语料"
    assert hits[0]["symbol"] == "600519"


@needs_key
def test_search_other_symbol_is_miss(isolated_vecdb):
    isolated_vecdb.put("宁德时代注册宁的易购商标。", symbol="300750", signature="u1", collection="news")
    # 别的股票不应命中
    assert isolated_vecdb.search("商标", symbol="999999", collection="news") == []


@needs_key
def test_unknown_collection_is_miss_not_crash(isolated_vecdb):
    # 不存在的 collection / 空库 → 返回空,不抛异常
    assert isolated_vecdb.search("随便", symbol="600519", collection="nonexistent") == []


@needs_key
def test_idempotent_put_same_signature(isolated_vecdb):
    # 相同 signature 重复写 → 块 id 相同 → upsert 覆盖,不堆积
    isolated_vecdb.put("同一篇新闻正文。", symbol="600519", signature="dup", collection="news")
    isolated_vecdb.put("同一篇新闻正文。", symbol="600519", signature="dup", collection="news")
    hits = isolated_vecdb.search("新闻正文", symbol="600519", collection="news")
    sigs = [h["signature"] for h in hits]
    assert sigs.count("dup") <= 1, "重复写不应产生重复块"
