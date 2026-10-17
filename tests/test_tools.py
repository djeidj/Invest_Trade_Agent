"""工具层测:全部 mock 掉 akshare/vecdb,只验证"取数→解析→返回"的编排逻辑。"""
import pandas as pd

import tools.stock_tools as st
from tests.conftest import needs_key


def test_get_financial_report_formats(monkeypatch):
    df = pd.DataFrame(
        {
            "选项": ["常用指标", "常用指标", "常用指标"],
            "指标": ["营业总收入", "归母净利润", "经营现金流量净额"],
            "20260630": [9.0e10, 3.3e10, 5.0e10],
        }
    )
    monkeypatch.setattr(st.ak, "stock_financial_abstract", lambda symbol: df)
    out = st.get_financial_report.invoke({"stock_code": "600519", "periods": 1})
    assert "600519" in out
    assert "900.00亿" in out


def test_get_financial_report_network_error_is_graceful(monkeypatch):
    def boom(symbol):
        raise RuntimeError("ak down")

    monkeypatch.setattr(st.ak, "stock_financial_abstract", boom)
    out = st.get_financial_report.invoke({"stock_code": "600519"})
    assert "获取财报失败" in out  # 不抛异常,转成文字


def test_ingest_stock_news_calls_vecdb_put(monkeypatch):
    news = pd.DataFrame(
        [
            {"新闻标题": "标题A", "新闻内容": "正文A", "新闻链接": "http://a"},
            {"新闻标题": "标题B", "新闻内容": "", "新闻链接": "http://b"},  # 空正文应跳过
        ]
    )
    calls = []
    monkeypatch.setattr(st.ak, "stock_news_em", lambda symbol: news)
    monkeypatch.setattr(
        st.vecdb, "put",
        lambda text, symbol, signature, collection: calls.append(signature) or 1,
    )
    out = st.ingest_stock_news.invoke({"stock_code": "600519"})
    assert calls == ["http://a"], "只写入有正文的新闻,空正文跳过"
    assert "写入 RAG 共 1 个文本块" in out


@needs_key
def test_query_rag_hit_and_miss(isolated_vecdb, monkeypatch):
    import tools.rag_tools as rt

    isolated_vecdb.put("比亚迪发布新电池。", symbol="002594", signature="s1", collection="news")
    hit = rt.query_rag.invoke({"query": "新电池", "symbol": "002594"})
    assert "检索到以下内容" in hit and "[1]" in hit
    miss = rt.query_rag.invoke({"query": "不存在的内容xyz", "symbol": "000000"})
    assert "未检索到" in miss
