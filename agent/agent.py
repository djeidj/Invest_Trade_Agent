import sqlite3
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import (
    CHECKPOINTER_DB_PATH,
    GLM_API_KEY,
    GLM_BASE_URL,
    GLM_MODEL,
    SUMMARIZE_KEEP_MESSAGES,
    SUMMARIZE_TRIGGER_TOKENS,
)
from config.sys_prompt import stock_sys_prompt
from tools import (
    get_financial_report,
    get_stock_history,
    get_stock_price,
    get_weather,
    ingest_stock_news,
    query_rag,
)

model = init_chat_model(
    model=GLM_MODEL,
    model_provider="openai",
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL,
)

# SqliteSaver:按 thread_id 存取会话历史,让多轮对话有记忆(高层 create_agent 直接吃 checkpointer)。
conn = sqlite3.connect(CHECKPOINTER_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

agent = create_agent(
    model=model,
    tools=[
        get_stock_price,       # 实时行情:直连返回
        get_stock_history,     # 历史K线:直连返回(东财→腾讯 failover)
        get_financial_report,  # 财报关键指标:新浪直连返回,进上下文
        query_rag,             # 新闻:先语义检索 RAG 语料
        ingest_stock_news,     # 未命中 -> 抓个股新闻写入 RAG
        get_weather,
    ],
    system_prompt=SystemMessage(content=stock_sys_prompt),
    checkpointer=checkpointer,
    # 上下文压缩:消息数超阈值时,LangChain 内置中间件把较早消息摘要成一条、
    # 保留最近 KEEP 条原文(且保证 AI/Tool 消息成对不被拆散),长会话不再无限膨胀。
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", SUMMARIZE_TRIGGER_TOKENS),
            keep=("messages", SUMMARIZE_KEEP_MESSAGES),
        )
    ],
)


def _messages_to_trace(messages: list) -> list[dict]:
    """把本轮 messages 转成可序列化的轨迹,给前端调试展示(工具调用一目了然)。"""
    trace: list[dict] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            trace.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            entry: dict = {"role": "assistant", "content": m.content}
            if getattr(m, "tool_calls", None):
                entry["tool_calls"] = [
                    {"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls
                ]
            trace.append(entry)
        elif isinstance(m, ToolMessage):
            trace.append({"role": "tool", "content": str(m.content)[:2000]})
    return trace


def run_agent_turn(thread_id: str, prompt: str) -> dict:
    """跑一轮,返回 {"thread_id", "answer", "trace"}。给 API/调试用。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    messages = result["messages"]
    answer = None
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            answer = message.content
            break
    return {"thread_id": thread_id, "answer": answer, "trace": _messages_to_trace(messages)}


def run_agent(thread_id: str, prompt: str, verbose: bool = True) -> str | None:
    out = run_agent_turn(thread_id, prompt)
    if verbose:
        for step in out["trace"]:
            print("=" * 50, "\n[" + step["role"] + "]")
            for tc in step.get("tool_calls", []):
                print(f"  [tool] {tc['name']}({tc['args']})")
            if step["content"]:
                print(str(step["content"])[:300])
    return out["answer"]


def stream_agent_turn(thread_id: str, prompt: str):
    """流式跑一轮,逐个 yield 事件字典(给 SSE 前端):
      token       模型正文增量(打字机)
      tool_call   模型请求调用某工具(完整参数,来自 updates)
      tool_result 工具返回(预览)
      final       收尾:answer + 完整 trace
      error       异常
    token 取自 messages 模式(逐 chunk),工具事件取自 updates 模式(完整消息),
    两者分工避免把最终正文重复发一遍。
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                msg, meta = chunk["data"]
                if meta.get("langgraph_node") == "model" and isinstance(msg, AIMessageChunk):
                    text = msg.content
                    if isinstance(text, str) and text:
                        yield {"type": "token", "text": text}
            elif chunk["type"] == "updates":
                update = chunk["data"]
                for _node, node_update in update.items():
                    for m in (node_update or {}).get("messages", []):
                        if isinstance(m, AIMessage):
                            for tc in m.tool_calls or []:
                                yield {"type": "tool_call", "name": tc["name"], "args": tc["args"]}
                        elif isinstance(m, ToolMessage):
                            yield {"type": "tool_result", "tool": m.name, "preview": str(m.content)[:200]}

        # 流结束后再取一次最终状态(移出循环):此时末条 AIMessage 才是带正文的最终回答
        msgs = _thread_state(thread_id)
        answer = next(
            (str(m.content) for m in reversed(msgs) if isinstance(m, AIMessage) and m.content), None
        )
        yield {"type": "final", "thread_id": thread_id, "answer": answer, "trace": _messages_to_trace(msgs)}
    except Exception as e:
        yield {"type": "error", "text": str(e)}


def _thread_state(thread_id: str) -> list:
    """取某 thread 的对话消息(checkpointer 里已存的状态)。"""
    snap = agent.get_state({"configurable": {"thread_id": thread_id}})
    return snap.values.get("messages", []) if snap and snap.values else []


def list_threads(limit: int = 50) -> list[dict]:
    """列出历史会话(给前端切换用)。读 checkpointer 的 checkpoints 表,非破坏性。"""
    rows = conn.execute(
        "SELECT thread_id, MAX(checkpoint_id) AS latest FROM checkpoints"
        " GROUP BY thread_id ORDER BY latest DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for thread_id, _latest in rows:
        msgs = _thread_state(thread_id)
        title = next(
            (str(m.content) for m in msgs if isinstance(m, HumanMessage)), ""
        )[:40]
        out.append({"thread_id": thread_id, "title": title, "count": len(msgs)})
    return out


def get_history(thread_id: str) -> dict:
    """返回某历史会话的完整轨迹(给前端回放)。"""
    return {"thread_id": thread_id, "trace": _messages_to_trace(_thread_state(thread_id))}


def delete_thread(thread_id: str) -> None:
    """删除某会话的 checkpoint(checkpointer 官方 API),从历史列表移除。不可逆。"""
    checkpointer.delete_thread(thread_id)


if __name__ == "__main__":
    thread = str(uuid.uuid4())
    answer = run_agent(thread, "贵州茅台最近10个交易日的收盘价走势如何?")
    print("\n\n===== 最终回答 =====")
    print(answer)
