"""API/SSE 传输测:把 stream_agent_turn 换成假 generator,验证 SSE 帧格式/路由,不跑 LLM。"""
import json

import pytest
from fastapi.testclient import TestClient

import app.main as api

client = TestClient(api.app)


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "TradeInvestAgent" in r.text


def test_threads_and_rag_endpoints_ok():
    assert client.get("/api/threads").status_code == 200
    assert "threads" in client.get("/api/threads").json()
    r = client.get("/api/rag")
    assert r.status_code == 200 and "collections" in r.json()


def test_sse_stream_frames(monkeypatch):
    fake = [
        {"type": "tool_call", "name": "get_stock_price", "args": {"stock_code": "600519"}},
        {"type": "token", "text": "你好"},
        {"type": "final", "answer": "你好!", "trace": [], "thread_id": "t"},
    ]

    def fake_stream(thread_id, prompt):
        yield from fake

    monkeypatch.setattr(api, "stream_agent_turn", fake_stream)

    with client.stream("POST", "/api/chat/stream", json={"prompt": "hi", "thread_id": "t"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())

    # 每个事件应是一帧 `data: {json}\n\n`
    frames = [ln for ln in body.split("\n\n") if ln.startswith("data: ")]
    assert len(frames) == len(fake)
    first = json.loads(frames[0][len("data: "):])
    assert first["type"] == "tool_call" and first["name"] == "get_stock_price"
    last = json.loads(frames[-1][len("data: "):])
    assert last["type"] == "final" and last["answer"] == "你好!"
