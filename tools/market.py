"""行情/新闻工具层:把 storage 的查询能力包装成 LLM 可调用的工具。

教学点1【工具 = schema + 执行器】:
- TOOL_DEFS(JSON)是给模型看的:工具叫什么、干什么用、要什么参数
- _TOOLS 里的 Python 函数是真正干活的
- 模型永远只输出"调用请求",执行权在我们手里

教学点2【工具读库,不直连外网】:
数据在 D1 已入库,工具从本地 SQLite 读——快、稳、可重入、不触发反爬。
外网抓取属于定时任务(fetch_daily),与 agent 运行时解耦。

教学点3【description 是工具的"使用说明书"】:
工具没被模型选中的第一原因就是 description 含糊。
写法要点:说清"什么时候该用我"、参数格式、给例子。

教学点4【工具报错也是反馈,不是崩溃】:
模型传错参数时,返回一段"怎么改"的说明文字,而不是抛异常。
模型看到错误说明有能力自我纠正——错误信息就是提示词。
"""
import json
from typing import Any

from config import WATCHLIST
from rag import embedder, store
from storage import db


def _resolve_symbol(symbol: str) -> str:
    """宽容输入:'sh600519'/'600519'/'贵州茅台' 统一解析成6位代码。"""
    s = str(symbol).strip().lower()
    for prefix in ("sh", "sz", "."):
        s = s.replace(prefix, "")
    if s in WATCHLIST:
        return s
    for code, name in WATCHLIST.items():  # 试 中文名 <-> 代码
        if name in str(symbol) or str(symbol) in name:
            return code
    return s  # 解析不了就原样返回,让查询结果说话


def get_daily_bars(symbol: str, days: int = 10) -> str:
    code = _resolve_symbol(symbol)
    rows = db.get_bars(code, days)
    if not rows:
        available = ", ".join(f"{c}({n})" for c, n in WATCHLIST.items())
        return f"数据库中没有 {symbol} 的日K数据。当前库中可用的股票: {available}"
    lines = [
        f"{r['date']} 开{r['open']} 高{r['high']} 低{r['low']} 收{r['close']} 量{r['volume']}股"
        for r in rows
    ]
    return f"{WATCHLIST.get(code, code)} 最近{len(rows)}个交易日:\n" + "\n".join(lines)


def get_stock_news(symbol: str, limit: int = 5) -> str:
    code = _resolve_symbol(symbol)
    rows = db.get_news(code, limit)
    if not rows:
        available = ", ".join(f"{c}({n})" for c, n in WATCHLIST.items())
        return f"数据库中没有 {symbol} 的新闻。当前库中可用的股票: {available}"
    items = [
        f"[{r['publish_time']}] {r['title']}({r['source']}): {r['content']}" for r in rows
    ]
    return f"{WATCHLIST.get(code, code)} 最近{len(rows)}条新闻:\n" + "\n".join(items)


def search_news(query: str, symbol: str = "", top_k: int = 4) -> str:
    """语义检索新闻语料(RAG)。返回带[编号]的片段,模型引用时必须标注来源。

    和 get_stock_news(按股票列最新,关键词型)互补:
    search_news 按"语义相似"找内容,能命中标题里没有关键词的报道。
    """
    hits = store.search(
        embedder.embed_one(query),
        top_k=top_k,
        symbol=_resolve_symbol(symbol) if symbol else "",
    )
    if not hits:
        # 防幻觉设计:检索不到就明说,让模型对用户诚实
        return "新闻语料中未检索到相关内容。请如实告知用户语料中没有相关信息,不要编造。"
    lines = [
        f"[{i + 1}] [{h['publish_time']}] {h['title']}({h['source']}, 相关度{h['score']:.2f})\n"
        f"    {h['text'][:200]}"
        for i, h in enumerate(hits)
    ]
    return "检索到以下新闻片段(回答引用时必须标注[编号]和来源):\n" + "\n".join(lines)


# --- 工具 schema:给模型看的"菜单" ---
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_daily_bars",
            "description": (
                "查询指定A股股票最近N个交易日的日K线(开高低收/成交量)。"
                "当用户询问某只股票的价格、涨跌、走势时调用。"
                "symbol 为6位股票代码(如 600519);用户说股票名时,用你的知识转成代码。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "6位股票代码,如 600519"},
                    "days": {
                        "type": "integer",
                        "description": "最近N个交易日,默认10,最大120",
                        "default": 10,
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "description": (
                "查询指定A股股票的近期新闻(标题/来源/时间/摘要)。"
                "当用户询问某只股票的消息面、舆情、最近发生什么事时调用。"
                "symbol 为6位股票代码。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "6位股票代码,如 600519"},
                    "limit": {
                        "type": "integer",
                        "description": "返回条数,默认5",
                        "default": 5,
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "语义检索新闻语料库(库内全部个股新闻的正文)。"
                "当用户询问某条报道的具体内容/原文、或 get_stock_news 的标题不足以回答时调用。"
                "query 用一句自然的中文描述想找的内容,可带 symbol 限定某只股票。"
                "返回带[编号]的片段,回答时必须标注[编号]和来源。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "想找的内容,如'i茅台APP维护'",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "可选,限定某只股票的6位代码",
                        "default": "",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回条数,默认4",
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# --- 执行器:给机器看的 ---
_TOOLS = {
    "get_daily_bars": get_daily_bars,
    "get_stock_news": get_stock_news,
    "search_news": search_news,
}


def execute_dict(name: str, args: dict) -> str:
    """执行一次工具调用(dict 入参版,给 LangGraph 节点等结构化调用方使用)。"""
    fn = _TOOLS.get(name)
    if fn is None:
        return f"未知工具 {name}。可用工具: {', '.join(_TOOLS)}"
    try:
        return str(fn(**args))
    except TypeError as e:
        return f"参数不匹配: {e}。参数定义见工具 schema,请修正后重试。"
    except Exception as e:  # noqa: BLE001 工具故障转成文字反馈给模型
        return f"工具执行失败: {e}"


def execute(name: str, arguments_json: str) -> str:
    """执行一次工具调用(字符串入参版:模型吐的 arguments 是原始 JSON 文本)。"""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return f"参数不是合法JSON: {arguments_json!r}"
    return execute_dict(name, args)
