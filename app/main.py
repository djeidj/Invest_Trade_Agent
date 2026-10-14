"""FastAPI 入口 —— 把 agent 引擎挂上 HTTP。本文件只做"传输",不做"智能"。

教学点1【分层】:
- GET  /          返回静态聊天页面
- POST /api/chat  把 run_agent_stream 的事件字典包成 SSE 帧推给浏览器
引擎(agent/stream.py)不知道 HTTP 的存在;本文件不知道 agent 循环的细节。
两边只通过"事件字典"这个协议解耦——换传输层时引擎零改动。

教学点2【为什么选 SSE 而不是 WebSocket】:
聊天是"用户问一句 → 服务器持续答"的单向流。SSE(Server-Sent Events):
- 就是普通 HTTP 响应,Content-Type 是 text/event-stream,不用升级协议
- 帧格式极简:"data: <内容>\\n\\n",内容习惯用 JSON
- 浏览器断线自动重连(EventSource 原生行为;我们用 fetch 自己解析也同理)
WebSocket 是双向通道,适合联机游戏/协同编辑这类频繁互推的场景,
单向聊天用它属于杀鸡用牛刀(还要自己处理心跳/重连)。

教学点3【为什么不需要 CORS】:
页面和接口由同一个 FastAPI 进程伺服 → 同源 → 浏览器同源策略天然放行。
将来前后端分开部署(比如页面挂 CDN),才需要加 CORSMiddleware。

教学点4【同步端点不堵车】:
端点用 def(不是 async def):OpenAI SDK 是同步的,FastAPI 会把 def 端点
丢进线程池跑;返回的同步生成器 Starlette 也会在工作线程里迭代,
不会卡住事件循环。这是"同步 SDK + FastAPI"的标准姿势。

运行: uv run uvicorn app.main:app --reload
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from agent.trace import traced_stream  # D7:换观测版引擎——事件透传+旁路落库
from config import GLM_API_KEY, GLM_BASE_URL

app = FastAPI(title="TradeInvestAgent")
# 进程启动时建一次客户端,所有请求复用连接(和 D2 教的客户端复用一致)
client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    """请求体 schema:类型不对/缺字段,FastAPI+pydantic 自动返回 422,不用手写校验。"""

    question: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):
    """SSE 端点:把事件生成器包成 'data: {...}\\n\\n' 帧。

    ensure_ascii=False:中文以原文传输而非 \\uXXXX 转义,包体小、可读。
    """
    def sse():
        for ev in traced_stream(client, req.question):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
