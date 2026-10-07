"""Day1 验收脚本:拉取自选股的行情+新闻并入库。

运行方式(在项目根目录):
    python -m scripts.fetch_daily

教学点:循环里每只股票独立 try/except。
一只股票的接口失败不应该让整批任务挂掉——
批量任务的单条失败要"记录并继续",这是数据管道的常识。
"""
import time

from config import WATCHLIST
from datasource import news, quotes
from storage import db


def main() -> None:
    for symbol, name in WATCHLIST.items():
        try:
            n_bars = db.upsert_bars(quotes.fetch_daily_hist(symbol))
            n_news = db.upsert_news(news.fetch_stock_news(symbol))
            print(f"[OK] {name}({symbol}): 日K {n_bars} 条, 新闻 {n_news} 条")
        except Exception as e:  # noqa: BLE001 单票失败不影响其他票
            print(f"[FAIL] {name}({symbol}): {e}")
        time.sleep(1)  # 限速:对免费接口保持礼貌,降低再次触发反爬的概率

    # 简单对账:库里现在有多少数据
    with db.get_conn() as conn:
        bars = conn.execute("SELECT COUNT(*) AS n FROM daily_bar").fetchone()["n"]
        news_count = conn.execute("SELECT COUNT(*) AS n FROM news").fetchone()["n"]
    print(f"\n入库完成:daily_bar 共 {bars} 条,news 共 {news_count} 条")


if __name__ == "__main__":
    main()
