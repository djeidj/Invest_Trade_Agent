"""eval 跑批器:逐用例跑 agent → 跑检查器 → 汇总成绩单 → 退出码。

教学点1【eval 是黑盒】:run_agent 只收到 question,像真用户一样。
不侵入引擎、不读中间状态——这样 eval 测的才是"系统对外行为"。

教学点2【exit code 是给机器看的】:
全过 exit 0,有挂 exit 1。将来接 CI(GitHub Actions)时,
`python -m evals.runner` 一条命令就能挡住"改坏老功能"的合并。
人看成绩单,机器看退出码——同一个工具服务两种读者。

运行: uv run python -m evals.runner       (全量,约 4~6 分钟)
      uv run python -m evals.runner 2     (只跑前 2 条,冒烟用)
"""
import sys
import time

from openai import OpenAI

from agent.loop import run_agent
from config import GLM_API_KEY, GLM_BASE_URL
from evals.cases import CASES
from evals.checks import run_check


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(CASES)
    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
    passed = 0
    t_start = time.time()

    for case in CASES[:limit]:
        t0 = time.time()
        try:
            answer = run_agent(client, case["question"], verbose=False)
        except Exception as e:  # noqa: BLE001 agent 崩溃也算用例失败,不能中断跑批
            answer = f"<agent异常: {e}>"

        results = [run_check(client, spec, answer, case["question"]) for spec in case["checks"]]
        ok = all(c_ok for c_ok, _ in results)
        passed += ok

        print(f"{'[PASS]' if ok else '[FAIL]'} {case['id']:<28} {time.time() - t0:5.1f}s")
        for c_ok, detail in results:
            print(f"        {'OK  ' if c_ok else 'FAIL'} {detail}")
        if not ok:
            print(f"        回答: {answer[:120]}")

    total = min(limit, len(CASES))
    print("=" * 60)
    print(f"通过 {passed}/{total}  用时 {time.time() - t_start:.0f}s")
    sys.exit(0 if passed == total else 1)  # CI 门禁:非 0 即失败


if __name__ == "__main__":
    main()
