"""测试隔离:绝不碰真实 data/。把 Chroma 持久化目录 monkeypatch 到 pytest 临时目录。

约定:所有涉及外部网络的单元测试都 mock 掉(akshare/LLM);
vecdb 的读写用 isolated_vecdb fixture 落到 tmp_path,测完即弃。
"""
import os

import pytest

try:  # 让需要 GLM_API_KEY 的用例在采集阶段就能看到 key
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

needs_key = pytest.mark.skipif(
    not os.getenv("GLM_API_KEY"), reason="需要 GLM_API_KEY(embedding 真调用)"
)


@pytest.fixture
def isolated_vecdb(tmp_path, monkeypatch):
    """把 vecdb 的 CHROMA_PATH 指到临时目录,返回该模块。测试不污染真实向量库。"""
    from storage import vecdb

    monkeypatch.setattr(vecdb, "CHROMA_PATH", tmp_path / "chroma")
    return vecdb
