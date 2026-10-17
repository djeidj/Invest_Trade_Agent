"""股票工具层。全是结构化数字,一律精确查/直连返回,都不进向量 RAG:

- get_stock_price        实时行情:直连、当场返回(价格会变,不缓存)。
- get_stock_history      历史K线:时间序列,直连返回(主源东财,失败切腾讯)。
- get_financial_report   财报关键指标:新浪 abstract(非东财),直连返回进上下文,不自建库。

原则:结构化数字一律直连/精确取、当场返回;会话记忆交给 checkpointer,不再手搓缓存库。
向量 RAG 只该留给真正的文本(新闻/研报/财报叙述),这类文本源目前还没有,故暂不接 RAG。
"""
from datetime import date, timedelta

from langchain.tools import tool
import akshare as ak

from config.settings import RAGCollection
from storage import vecdb

def _code64(stock_code: str) -> str:
    """归一化成 6 位纯代码,用作缓存主键与展示。"""
    return "".join(ch for ch in stock_code if ch.isdigit())[:6] or stock_code


def _tx_symbol(stock_code: str) -> str:
    """腾讯源要带小写交易所前缀(600519 -> sh600519 / 000001 -> sz000001)。"""
    code = _code64(stock_code)
    ex = "sh" if code[:1] == "6" else "sz" if code[:1] in ("0", "3") else "bj"
    return f"{ex}{code}"


def _records_to_text(df, label: str, max_rows: int = 4) -> str:
    """把 DataFrame 最近若干行渲染成"字段: 值"文本,每行内联表名/只保留非空字段。

    避免 df.to_string():财报宽表几百列直接 dump 会瞬间撑爆模型上下文。
    表名内联进每一行,是为了切块检索时每个块都自带来源标题,不会切出"只有标题没数据"的空块。
    """
    lines = []
    for _, row in df.head(max_rows).iterrows():
        pairs = [f"{k}={v}" for k, v in row.items() if v is not None and str(v) not in ("", "nan", "None")]
        if pairs:
            lines.append(f"[{label}] " + "; ".join(pairs))
    return "\n".join(lines)


@tool
def get_stock_price(stock_code: str) -> str:
    """获取A股股票的【实时】行情(最新价/涨跌幅/今开/最高/最低/成交量额)。实时数据直连、不进 RAG。

    主源东财实时盘口;失败自动切腾讯源最新行情(多源容灾——单一免费源常被限流,实测踩过)。
    当用户问"现在多少钱""今天涨跌"时用本工具。
    Args:
        stock_code(str): 6位股票代码,例如 "600519"
    """
    code = _code64(stock_code)

    # 主源:东财单股盘口+实时(比全市场快照省得多)
    try:
        df = ak.stock_bid_ask_em(symbol=code)
        d = dict(zip(df.iloc[:, 0], df.iloc[:, 1])) if df.shape[1] == 2 else df.iloc[0].to_dict()

        def g(*keys):  # 取第一个存在且非空的字段,缺省 "-"
            for k in keys:
                if k in d and str(d[k]) not in ("", "nan", "None"):
                    return d[k]
            return "-"

        return (
            f"股票: {code}(东财实时)\n"
            f"最新价: {g('最新', '现价', '最新价')} 元\n"
            f"涨跌幅: {g('涨幅', '涨跌幅')}%  涨跌额: {g('涨跌', '涨跌额')} 元\n"
            f"今开: {g('开盘', '今开')} 元  最高: {g('最高')} 元  最低: {g('最低')} 元\n"
            f"成交量: {g('总手', '成交量')} 手  成交额: {g('金额', '成交额')} 元"
        )
    except Exception:
        pass  # 东财不可用 -> 落到下面腾讯备源

    # 备源:腾讯历史最后一根(不同域名,东财限流时可用);用最近两根现算涨跌幅
    try:
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=40)).strftime("%Y%m%d")
        tx = ak.stock_zh_a_hist_tx(symbol=_tx_symbol(code), start_date=start, end_date=end, adjust="qfq")
        if tx.empty:
            return f"未找到 {code} 的行情数据(东财/腾讯均无返回)"
        last = tx.iloc[-1]
        prev = tx.iloc[-2] if len(tx) >= 2 else None
        chg_pct = chg_amt = "-"
        if prev is not None and prev["close"]:
            chg_amt = round(float(last["close"]) - float(prev["close"]), 2)
            chg_pct = round(chg_amt / float(prev["close"]) * 100, 2)
        vol_shou = round(float(last["volume"]) / 100)  # 腾讯 volume 单位=股,除以100 转手(与东财口径统一)
        return (
            f"股票: {code}(腾讯源·最新收盘,东财实时暂不可用)\n"
            f"最新价: {last['close']} 元\n"
            f"涨跌幅: {chg_pct}%  涨跌额: {chg_amt} 元\n"
            f"今开: {last['open']} 元  最高: {last['high']} 元  最低: {last['low']} 元\n"
            f"成交量: {vol_shou} 手  成交额: {last['amount']} 元"
        )
    except Exception as e:
        return f"获取实时价格失败(东财+腾讯双源均不可用): {e}"


@tool
def get_stock_history(stock_code: str, period: str = "daily", start_date: str = "", end_date: str = "") -> str:
    """获取A股【历史K线】数据(开高低收/成交量等),直接返回内容。

    历史K线是结构化时间序列,精确查询即可,不进 RAG、不做语义检索。
    每次直接联网取最新数据(主源东财,失败切腾讯)。
    Args:
        stock_code(str): 6位股票代码,例如 "600519"
        period(str): 周期 "daily"/"weekly"/"monthly",默认 "daily"
        start_date(str): 开始日期 "YYYYMMDD",默认近一年
        end_date(str): 结束日期 "YYYYMMDD",默认今天
    """
    start = start_date or "20240101"
    end = end_date or date.today().strftime("%Y%m%d")
    code = _code64(stock_code)
    try:
        try:
            df = ak.stock_zh_a_hist(symbol=code, period=period, start_date=start, end_date=end, adjust="qfq")
        except Exception:
            df = ak.stock_zh_a_hist_tx(symbol=_tx_symbol(code), start_date=start, end_date=end, adjust="qfq")
        if df.empty:
            return f"未获取到 {stock_code} 的历史数据(period={period})"
        recent = df.tail(10)
        text = _records_to_text(recent, f"{code} 历史K线 {period} {start}-{end}", max_rows=10)
        return text
    except Exception as e:
        return f"获取历史数据失败(东财+腾讯双源均不可用): {e}"


# 关注指标 -> 新浪 stock_financial_abstract 的指标行名
_SINA_METRICS = {
    "revenue": "营业总收入",
    "net_profit": "归母净利润",
    "op_cashflow": "经营现金流量净额",
}


def _to_float(v) -> float | None:
    if v is None or str(v) in ("", "nan", "None", "NaT", "—", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_abstract(df, max_periods: int = 8) -> list[dict]:
    """把新浪 abstract(指标×报告期宽表)转成 [{report_date, revenue, ...}, ...] 倒序。

    abstract 列是 '选项'/'指标'/各 YYYYMMDD 报告期;每个目标指标占一行,沿日期列取值。
    缺行/缺值一律 None,不崩。一次调用即可拿到全部历史期。
    """
    if df is None or df.empty or "指标" not in df.columns:
        return []
    date_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
    recs: dict[str, dict] = {}
    for field, name in _SINA_METRICS.items():
        match = df[df["指标"] == name]
        if match.empty:
            continue
        row = match.iloc[0]  # 同名多行取第一(通常"常用指标"分组)
        for c in date_cols:
            val = _to_float(row[c])
            if val is None:
                continue
            s = str(c)
            rd = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
            recs.setdefault(rd, {"report_date": rd})[field] = val
    rows = sorted(recs.values(), key=lambda x: x["report_date"], reverse=True)[:max_periods]
    for rec in rows:
        for field in _SINA_METRICS:
            rec.setdefault(field, None)
    return rows


def _fmt_yi(v) -> str:
    if v is None:
        return "-"
    if abs(v) >= 1e8:
        return f"{v / 1e8:.2f}亿"
    return f"{v:,.0f}"


def _format_financial(code: str, rows: list[dict], source_note: str) -> str:
    """结构化行 → 给模型看的文本(直接作为工具返回值进上下文)。"""
    if not rows:
        return f"未获取到 {code} 的财报数据"
    lines = [f"股票 {code} 财务关键指标(最近{len(rows)}期,{source_note}):"]
    for r in rows:
        lines.append(
            f"- {r['report_date']}: 营业总收入={_fmt_yi(r.get('revenue'))} "
            f"归母净利={_fmt_yi(r.get('net_profit'))} 经营现金流={_fmt_yi(r.get('op_cashflow'))}"
        )
    return "\n".join(lines)


@tool
def get_financial_report(stock_code: str, periods: int = 4) -> str:
    """获取A股财报关键指标(营业总收入/归母净利润/经营现金流),最近数期,直接返回。

    数据源:新浪 stock_financial_abstract(非东财)。结果直接进对话上下文——
    工具输出本就随 checkpointer 存在会话里,无需再自建数据库缓存。
    Args:
        stock_code(str): 6位股票代码,例如 "600519"
        periods(int): 返回最近几期,默认4
    """
    code = _code64(stock_code)
    try:
        df = ak.stock_financial_abstract(symbol=code)
        rows = _parse_abstract(df)
        if not rows:
            return f"未找到 {stock_code} 的财报数据"
        return _format_financial(code, rows[:periods], "新浪")
    except Exception as e:
        return f"获取财报失败: {e}"



@tool
def ingest_stock_news(stock_code: str) -> str:
    """抓取某只A股的近期新闻并存入向量库(RAG 语料)。只回确认信息,不回新闻本身。

    当 query_rag 检索不到该股票新闻时才调用;写入后请再用 query_rag 检索作答。
    Args:
        stock_code(str): 6位股票代码,例如 "600519"
    """
    code = "".join(c for c in stock_code if c.isdigit())[:6] or stock_code
    try:
        df = ak.stock_news_em(symbol=code)
        if df.empty:
            return f"未抓到 {code} 的新闻。"
        total = 0
        for _, r in df.iterrows():
            title = str(r.get("新闻标题", "")).strip()
            content = str(r.get("新闻内容", "")).strip()
            url = str(r.get("新闻链接", "")).strip() or f"{code}-{total}"
            text = f"{title}。{content}"
            if not content:
                continue
            # signature=新闻链接 → 确定性块 id,重复抓取同一篇新闻只覆盖不堆积
            total += vecdb.put(text, symbol=code, signature=url, collection=RAGCollection.NEWS)
        return (
            f"已抓取 {code} 的 {len(df)} 条新闻、写入 RAG 共 {total} 个文本块。"
            f'请调用 query_rag(query="想了解的内容", symbol="{code}") 检索后作答。'
        )
    except Exception as e:
        return f"抓取新闻失败(东财源不可用): {e}"