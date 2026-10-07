"""新闻数据源(akshare 封装,数据来自东方财富个股新闻)。

和 quotes.py 同样的防腐层套路:中文列名在边界处翻译成标准 schema。
"""
import pandas as pd

import akshare as ak

from datasource._retry import with_retry

_NEWS_COLUMNS = {
    "新闻标题": "title",
    "新闻内容": "content",
    "发布时间": "publish_time",
    "文章来源": "source",
    "新闻链接": "url",
}


@with_retry
def fetch_stock_news(symbol: str, limit: int = 20) -> list[dict]:
    """拉取一只股票的最新个股新闻。"""
    df: pd.DataFrame = ak.stock_news_em(symbol=symbol)
    df = df.rename(columns=_NEWS_COLUMNS)
    rows = df[list(_NEWS_COLUMNS.values())].head(limit).to_dict("records")
    for r in rows:
        r["symbol"] = symbol
        # 内容太长的截断,防止后面塞进模型 prompt 时爆 token
        if r.get("content"):
            r["content"] = str(r["content"])[:500]
    return rows
