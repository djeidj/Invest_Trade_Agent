"""每日投研简报生成器 —— 注意:这是一个 workflow,不是 agent!

教学点【workflow vs agent 选型】(出自 Anthropic《Building Effective Agents》,面试高频):
- workflow:流程由代码写死,LLM 只在固定环节被调用 → 确定、便宜、可测试
- agent:LLM 自主决定每一步调什么工具 → 灵活,但贵、慢、可能失控

简报是每天重复的确定性任务,流程完全可预知 → 用 workflow:
  1. 数据收集:直接查库(确定性!根本不需要 function calling)
  2. LLM 环节一:逐股点评(数据已在 prompt 里,模型无需工具)
  3. LLM 环节二:全局总结
"用户会问什么不可预知"的交互追问,才值得用 agent。

注意:工具层(get_daily_bars 等)是公用的——workflow 和 agent 都消费它。

运行: uv run python -m scripts.gen_briefing
输出: reports/briefing_YYYY-MM-DD.md
"""
from datetime import date
from pathlib import Path

from openai import OpenAI

from config import GLM_API_KEY, GLM_BASE_URL, GLM_MODEL, WATCHLIST
from tools.market import get_daily_bars, get_stock_news

REPORTS_DIR = Path(__file__).parent.parent / "reports"

STOCK_PROMPT = (
    "你是严谨的投研分析师。基于以下该股票的行情与新闻数据,写一段不超过120字的点评:"
    "先说走势特征(必须引用具体价格数字),再用一句话概括消息面,结尾一句风险提示。"
    "数据里没有的信息不要编造。\n\n数据:\n{data}"
)

SUMMARY_PROMPT = (
    "你是投研团队主管。基于以下各股点评,写今日总结(200字以内):"
    "先概括整体特征(普涨/普跌/分化),再点名最值得关注的一只并说明理由,"
    "最后一句免责声明。\n\n各股点评:\n{comments}"
)


def ask(client: OpenAI, prompt: str) -> str:
    """无工具的纯文本 LLM 调用——workflow 里的标准环节。"""
    resp = client.chat.completions.create(
        model=GLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def main() -> None:
    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)

    sections: list[str] = []
    comments: list[str] = []
    for symbol, name in WATCHLIST.items():
        data = get_daily_bars(symbol, 5) + "\n\n" + get_stock_news(symbol, 3)
        comment = ask(client, STOCK_PROMPT.format(data=data))
        sections.append(f"### {name}({symbol})\n\n{comment}\n")
        comments.append(f"【{name}】{comment}")
        print(f"[OK] {name} 点评完成")

    summary = ask(client, SUMMARY_PROMPT.format(comments="\n\n".join(comments)))
    print("[OK] 全局总结完成")

    today = date.today().isoformat()
    report = (
        f"# 投研简报 · {today}\n\n## 今日总结\n\n{summary}\n\n## 个股点评\n\n"
        + "\n".join(sections)
    )
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"briefing_{today}.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n简报已生成: {out}")


if __name__ == "__main__":
    main()
