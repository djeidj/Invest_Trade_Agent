"""行情数据源(多源容灾:东财主源 + 腾讯备源)。

教学点1【防腐层】:akshare 各源的原始列名、字段语义都不同,
必须在边界处翻译成统一 schema,上层代码永远面对干净数据。

教学点2【多源容灾(failover)】:任何单一免费源都不可靠(限流/改版/封IP),
主源失败自动切备源,调用方无感知。今天东财把我们限流了,
failover 正好上线——工程能力就是这么踩坑踩出来的。

教学点3【同名不同义】:腾讯接口的 amount 字段实际是成交量(手),
不是成交额!不同源的"同名列"语义可能不同,映射前必须逐字段核实,
否则数据会悄悄变脏——脏数据比报错可怕,因为它不报错。
"""
from datetime import date, timedelta

import akshare as ak
import pandas as pd

from datasource._retry import with_retry

# 东财源:中文列名 -> 标准列名(成交量/成交额都有)
_EM_COLUMNS = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}

# 腾讯源字段语义(2026-10-07 经 scripts/check_tx_amount 量纲检验+交叉验证):
#   volume=成交量(股), amount=成交额(元), turnover=换手率(小数,暂不使用)
# 注意 volume 单位与东财("手")不一致,在 _fetch_from_tx 中换算
_TX_COLUMNS = {
    "date": "date",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "amount",
}


# 所有源适配后必须交付的完整字段契约
STANDARD_FIELDS = ["date", "open", "high", "low", "close", "volume", "amount"]


def _to_rows(df: pd.DataFrame, columns: dict[str, str], symbol: str) -> list[dict]:
    """防腐层:清洗 -> 翻译列名 -> 补全契约 -> 统一字典列表。

    输出契约:每一行必须含全部标准字段,缺失补 None。
    调用方(sqlite 绑定、以后的 agent 工具)才能永远信任这个结构。
    """
    df = df.rename(columns=columns)
    # 重命名可能撞出重名:如源的 amount 改名为 volume,而源本身已有 volume 列
    # 所以去重必须放在 rename 之后(保留第一列)
    df = df.loc[:, ~df.columns.duplicated()]
    rows = (
        df.reindex(columns=STANDARD_FIELDS)  # 缺失字段补 NaN,而不是悄悄丢掉或报 KeyError
        .to_dict("records")
    )
    for r in rows:
        r["symbol"] = symbol
        r["date"] = str(r["date"])
        for k, v in r.items():
            if pd.isna(v):  # NaN -> None,sqlite 才能存成 NULL
                r[k] = None
    return rows


def _date_range(days: int) -> tuple[str, str]:
    start = (date.today() - timedelta(days=days)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")
    return start, end


@with_retry
def _fetch_from_east(symbol: str, days: int) -> list[dict]:
    """主源:东方财富。前复权(qfq)保证价格序列跨除权日可比。"""
    start, end = _date_range(days)
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq"
    )
    return _to_rows(df, _EM_COLUMNS, symbol)


@with_retry
def _fetch_from_tx(symbol: str, days: int) -> list[dict]:
    """备源:腾讯财经。同样支持 qfq,日期格式同为 YYYYMMDD。"""
    start, end = _date_range(days)
    df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
    df["volume"] = df["volume"] * 100  # 腾讯源成交量单位是股,统一换成"手"对齐东财(1手=100股)
    print(f"    [tx] 腾讯源原始列: {list(df.columns)}")  # 临时调试:观察真实列名,稳定后删
    return _to_rows(df, _TX_COLUMNS, symbol)


def fetch_daily_hist(symbol: str, days: int = 120) -> list[dict]:
    """对外入口:主源失败自动切备源(failover),调用方无感知。"""
    try:
        return _fetch_from_east(symbol, days)
    except Exception as e:  # noqa: BLE001 主源任何失败都触发切换
        print(f"    [failover] 东财源失败({type(e).__name__}),切换腾讯源")
        return _fetch_from_tx(symbol, days)
