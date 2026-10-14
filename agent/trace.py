"""tracing 中间件:包住 run_agent_stream,把事件流"抽头"落库。

教学点1【为什么包一层而不是改引擎】:
D6 的事件协议在这里兑现第二波红利——事件流本身就是天然的观测点。
本包装器用 yield 原样转发事件(传输层和前端零感知),同时旁路记账。
引擎/传输/观测三方解耦:以后想把记录换成发 Langfuse,只改这个文件。
反过来,把写库代码直接塞进 stream.py 也能跑,但引擎从此背上存储依赖。

教学点2【cost 快照】:每个 span 入库时按 config 牌价把成本算成钱存下。
牌价是会变的,账单不能变。

教学点3【会话总时长】:wrapper 的 t0 到 end_session 是完整会话耗时,
通常大于各 span 耗时之和——差值就是"非 LLM/非工具"的开销
(排队、网络、SQLite 写库),这也是观测的价值:看见看不见的时间。
"""
import time
from collections import deque

from agent.stream import run_agent_stream
from config import GLM_MODEL, GLM_PRICE_IN, GLM_PRICE_OUT
from storage import trace as trace_db


def cost_yuan(prompt_tokens: int, completion_tokens: int) -> float:
    """按 config 牌价(元/百万 tokens)算一次调用的成本。"""
    return prompt_tokens / 1e6 * GLM_PRICE_IN + completion_tokens / 1e6 * GLM_PRICE_OUT


def traced_stream(client, question: str, max_steps: int = 8):
    """事件流透传 + 旁路记账。签名与 run_agent_stream 一致,可直接替换。"""
    session_id = trace_db.start_session(question, GLM_MODEL)
    t0 = time.perf_counter()
    seq = 0
    prompt_total = completion_total = 0
    cost_total = 0.0
    answer = ""
    pending_tools: deque = deque()  # tool_call 事件先入队,等配对的 tool_result

    try:
        for ev in run_agent_stream(client, question, max_steps):
            if ev["type"] == "usage":  # 一次 LLM 调用结束 → 记一个 llm span
                seq += 1
                cost = cost_yuan(ev["prompt_tokens"], ev["completion_tokens"])
                prompt_total += ev["prompt_tokens"]
                completion_total += ev["completion_tokens"]
                cost_total += cost
                trace_db.add_span(
                    session_id, seq, "llm", f"step {ev['step']}", detail="模型调用",
                    pt=ev["prompt_tokens"], ct=ev["completion_tokens"],
                    cost=cost, latency_ms=ev["latency_ms"],
                )
            elif ev["type"] == "tool_call":
                pending_tools.append((ev["name"], ev["args"]))
            elif ev["type"] == "tool_result":  # 与 tool_call 按 FIFO 配对
                seq += 1
                name, args = pending_tools.popleft()
                trace_db.add_span(
                    session_id, seq, "tool", name,
                    detail=f"{args} -> {ev['preview']}",
                    latency_ms=ev.get("latency_ms", 0),
                )
            elif ev["type"] == "final":
                answer = ev["text"]
            yield ev  # 原样转发:传输层/前端完全不知道 trace 的存在

        trace_db.end_session(
            session_id, answer, prompt_total, completion_total,
            round(cost_total, 6), round((time.perf_counter() - t0) * 1000), status="ok",
        )
    except Exception as e:  # 出错的会话也要留档,错误本身就是最重要的观测对象
        trace_db.end_session(
            session_id, answer, prompt_total, completion_total,
            round(cost_total, 6), round((time.perf_counter() - t0) * 1000),
            status=f"error: {e}",
        )
        raise
