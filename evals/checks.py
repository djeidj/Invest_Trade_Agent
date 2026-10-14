"""检查器实现:确定性优先,LLM judge 兜底。

教学点1【检查器分级】:
- 确定性(contains / contains_any / citations / refusal / db_close_prices):
  便宜、稳定、可复现——能用确定性检查的地方绝不用模型
- LLM judge:语义级验收("是否真列出了新闻")只有模型能判,
  但 judge 自己也是 LLM,有稳定性差、手松、自我偏好等风险,
  所以:temperature=0(压随机性)+ 严格指令(拿不准判不合格)+ 只做兜底

教学点2【为什么 db_close_prices 不要求 100% 命中】:
模型有权概述("收盘价从 X 涨到 Y"省略中间日),100% 门槛会造成
"答案其实正确但被误杀"。60% 足以挡住编造——编一道题的数字容易,
编对 6 成真实数字难。阈值本身就是一种设计决策。
"""
import json
import re

from openai import OpenAI

from storage import db

# 拒答类回答的标志词(覆盖我们 SYSTEM_PROMPT 教的几种说法)
REFUSAL_MARKS = ["未找到", "没有", "未检索", "查不到", "不存在"]

# judge 提示词模板:只输出 JSON,方便程序解析(JSON 模式的最朴素替代)
JUDGE_TMPL = (
    "你是严格的评测官。判断【回答】是否满足【评测标准】。\n"
    "规则:只依据标准判断,不要脑补;拿不准就判不合格。\n"
    '只输出JSON,格式: {{"pass": true或false, "reason": "一句话理由"}}\n\n'
    "【问题】{question}\n【回答】{answer}\n【评测标准】{criteria}"
)


def check_contains(answer: str, spec: dict) -> tuple[bool, str]:
    ok = spec["value"] in answer
    return ok, f"需包含 {spec['value']!r}"


def check_contains_any(answer: str, spec: dict) -> tuple[bool, str]:
    hit = next((v for v in spec["values"] if v in answer), None)
    return hit is not None, f"需包含任一 {spec['values']}"


def check_citations(answer: str, spec: dict) -> tuple[bool, str]:
    # pattern 支持用例级自定义,默认匹配 [1] [2] 这类编号标记
    ok = bool(re.search(spec.get("pattern", r"\[\d+\]"), answer))
    return ok, "回答需含[编号]引用标记"


def check_refusal(answer: str, spec: dict) -> tuple[bool, str]:
    # marks 支持用例级自定义拒答标志词,默认用通用词表
    marks = spec.get("marks", REFUSAL_MARKS)
    ok = any(m in answer for m in marks)
    return ok, f"回答需含拒答标志词之一 {marks}"


def check_db_close_prices(answer: str, spec: dict) -> tuple[bool, str]:
    """ground truth 从库里现拉:最近 N 日收盘价,答案至少命中 60%。

    兼容两种格式:11.73(两位小数)和 1732(整数写法,去尾零)。
    """
    bars = db.get_bars(spec["symbol"], spec["days"])
    if not bars:
        return False, f"库里没有 {spec['symbol']} 的数据,用例前提失效(先跑 fetch_daily)"
    hits = 0
    for r in bars:
        c = r["close"]
        candidates = {f"{c:.2f}", f"{c:.2f}".rstrip("0").rstrip(".")}
        if any(x in answer for x in candidates):
            hits += 1
    need = -(-len(bars) * 6 // 10)  # ceil(len*0.6),向上取整
    return hits >= need, f"命中 {hits}/{len(bars)} 个收盘价(需>={need})"


def check_llm_judge(
    client: OpenAI, answer: str, spec: dict, question: str
) -> tuple[bool, str]:
    """语义级验收:judge 模型按标准打分,只输出 JSON。"""
    try:
        resp = client.chat.completions.create(
            model=spec.get("judge_model", None) or _default_model(),
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_TMPL.format(
                        question=question, answer=answer, criteria=spec["criteria"]
                    ),
                }
            ],
            temperature=0,  # 评委要稳,不要灵感
        )
        text = resp.choices[0].message.content or ""
        s, e = text.find("{"), text.rfind("}")  # 防御:从文本里抠出 JSON 再解析
        data = json.loads(text[s : e + 1]) if 0 <= s < e else {}
        return bool(data.get("pass")), f"judge: {str(data.get('reason', ''))[:80]}"
    except Exception as e:  # noqa: BLE001 judge 挂了=检查失败,不能静默放过
        return False, f"judge 执行失败: {e}"


def _default_model() -> str:
    from config import GLM_MODEL

    return GLM_MODEL


def run_check(client: OpenAI, spec: dict, answer: str, question: str) -> tuple[bool, str]:
    """检查器分发器:用例里写 type,这里路由到具体实现。"""
    t = spec["type"]
    if t == "contains":
        return check_contains(answer, spec)
    if t == "contains_any":
        return check_contains_any(answer, spec)
    if t == "citations":
        return check_citations(answer, spec)
    if t == "refusal":
        return check_refusal(answer, spec)
    if t == "db_close_prices":
        return check_db_close_prices(answer, spec)
    if t == "llm_judge":
        return check_llm_judge(client, answer, spec, question)
    return False, f"未知检查类型 {t}"
