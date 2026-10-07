"""全局配置。

教学点:配置与代码分离。换股票池、换模型、换 key 都不改业务代码,
这在面试里叫"12-Factor 应用的配置外置原则"。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 从项目根目录的 .env 读取环境变量

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "tradeinvest.db"

# --- 智谱 GLM(OpenAI 兼容协议)---
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
# 智谱的 OpenAI 兼容端点。模型名以当天官网为准,Day2 接模型时再一起确认
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-5.3-flash")

# --- 自选股池:A股6位代码 -> 名称 ---
# 改这里就能换股票池,业务代码不用动
WATCHLIST: dict[str, str] = {
    "600519": "贵州茅台",
    "000001": "平安银行",
    "300750": "宁德时代",
    "601899": "紫金矿业",
    "002230": "科大讯飞",
}
