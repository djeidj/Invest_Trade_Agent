"""全局配置。

教学点:配置与代码分离。换股票池、换模型、换 key 都不改业务代码,
这在面试里叫"12-Factor 应用的配置外置原则"。
"""
import os
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 从项目根目录的 .env 读取环境变量

BASE_DIR = Path(__file__).parent.parent
CHECKPOINTER_DB_PATH = BASE_DIR / "data" / "checkpointer.db"
# Chroma 向量库持久化目录(D5 用 LangChain Chroma 替换了 SQLite+numpy 暴力余弦)
CHROMA_PATH = BASE_DIR / "data" / "chroma"

# --- 智谱 GLM(OpenAI 兼容协议)---
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
# 智谱的 OpenAI 兼容端点。模型名以当天官网为准,Day2 接模型时再一起确认
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-5.3-flash")

# --- 向量化模型(RAG 用,和聊天模型共用同一个 OpenAI 兼容端点)---
EMBED_MODEL = os.getenv("EMBED_MODEL", "embedding-3")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "512"))

# --- 上下文压缩(LangChain SummarizationMiddleware 阈值)---
# 累计 prompt token 达 TRIGGER 就把较早的消息摘要成一条,只保留最近 KEEP 条原文。
# 用绝对 tokens 而非 fraction:GLM 经 OpenAI 兼容端点没有 profile,fraction 会报错。
# TRIGGER 设保守值(远低于 GLM 实际窗口,安全);想更贴真实窗口按官网调。
SUMMARIZE_TRIGGER_TOKENS = int(os.getenv("SUMMARIZE_TRIGGER_TOKENS", "800000"))
SUMMARIZE_KEEP_MESSAGES = int(os.getenv("SUMMARIZE_KEEP_MESSAGES", "6"))

# --- RAG 语料类别(= Chroma collection 名)。单一事实来源:所有工具都从这里引,别各写各的字面量 ---
# 新增一类:① 在这里加一个成员 ② 提供对应写入器(@tool 或脚本)写这个 collection。
# 用作工具入参类型时,pydantic 会把它渲染成 enum 白名单,模型只能选、编不出别的。
class RAGCollection(StrEnum):
    NEWS = "news"          # 个股新闻,由 stock_tools.ingest_stock_news 写入
    # RESEARCH = "research"      # 将来:研报
    # ANNOUNCEMENT = "announcement"  # 将来:公告

# --- 牌价(元/百万 tokens),D7 成本核算用 ---
# 查证:GLM-4.5-Air 牌价为输入0.8/输出2元每百万tokens(以官网定价页为准);
# 免费档模型把两个值改成 0 即可,核算机制不变。
# 注意:牌价只影响新账单——历史会话的 cost 已按当时牌价"快照"入库,不追溯改写。
GLM_PRICE_IN = float(os.getenv("GLM_PRICE_IN", "0.8"))
GLM_PRICE_OUT = float(os.getenv("GLM_PRICE_OUT", "2.0"))

