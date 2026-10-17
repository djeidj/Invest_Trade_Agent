"""FastAPI 调试后端 —— 只做"传输",不含业务智能。

暴露:
- GET  /                返回调试前端(static/index.html)
- POST /api/chat        跑一轮 agent,返回 {thread_id, answer, trace}
- GET  /api/rag         返回当前向量库 collection 概览(调试用)

engine(agent.agent.run_agent_turn)与传输解耦:换前端/换协议不改引擎。
thread_id 由前端保持并回传,配合 SqliteSaver checkpointer 实现多轮记忆;
点"新会话"前端换个 uuid 即可开新线程。
"""
import json
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent.agent import delete_thread, get_history, list_threads, run_agent_turn, stream_agent_turn
from storage import vecdb

app = FastAPI(title="TradeInvestAgent 调试台")
STATIC = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    prompt: str
    thread_id: str | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    return run_agent_turn(thread_id, req.prompt)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式:边跑边吐 token / 工具调用事件,前端即时展示。"""
    thread_id = req.thread_id or str(uuid.uuid4())

    def sse():
        for ev in stream_agent_turn(thread_id, req.prompt):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/api/threads")
def threads():
    """历史会话列表,供前端选择切换。"""
    return {"threads": list_threads()}


@app.get("/api/history/{thread_id}")
def history(thread_id: str):
    """回放某历史会话(读 checkpointer,不重跑)。"""
    return get_history(thread_id)


@app.delete("/api/threads/{thread_id}")
def remove_thread(thread_id: str):
    """删除某会话(checkpoint 全清,不可逆)。"""
    delete_thread(thread_id)
    return {"deleted": thread_id}


@app.get("/api/rag")
def rag():
    return {"collections": vecdb.list_collections()}
