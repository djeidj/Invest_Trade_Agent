"""网络诊断脚本:用对照实验定位 eastmoney 不通的原因。

调试方法论【差分诊断】:不要猜,做对照——
  同一个目标,只改变一个条件(走代理 vs 绕过代理),对比结果。
  另外设一个对照组(bigmodel.cn,已知是通的),排除"整个网络都挂了"的解释。

运行: uv run python -m scripts.diag_network
"""
import urllib.request

import requests

EM_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    "?secid=1.600519&fields1=f1&fields2=f51&klt=101&fqt=1"
    "&beg=20260901&end=20261007"
)


def try_request(label: str, url: str, bypass_proxy: bool = False) -> bool:
    try:
        if bypass_proxy:
            s = requests.Session()
            s.trust_env = False  # 无视环境变量代理 + Windows 系统代理(注册表)
            r = s.get(url, timeout=10)
        else:
            r = requests.get(url, timeout=10)
        print(f"[OK]   {label}: HTTP {r.status_code} ({r.elapsed.total_seconds():.2f}s)")
        return True
    except Exception as e:  # noqa: BLE001 诊断脚本,打印一切
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")
        return False


def main() -> None:
    proxies = urllib.request.getproxies()  # requests 用的就是这套:环境变量+注册表
    print("1) Python 实际会用的代理配置:")
    print("   ", proxies if proxies else "(无) —— 说明请求是直连的")

    print("\n2) 实验组:走默认配置(带代理)请求 eastmoney")
    via_proxy_ok = try_request("eastmoney(带代理)", EM_URL)

    print("\n3) 实验组:绕过一切代理请求 eastmoney")
    direct_ok = try_request("eastmoney(直连)", EM_URL, bypass_proxy=True)

    print("\n4) 对照组:走默认配置请求 bigmodel.cn(已知能通)")
    try_request("bigmodel.cn", "https://open.bigmodel.cn/api/paas/v4/")

    print("\n=== 结论判读 ===")
    if direct_ok and not via_proxy_ok:
        print("元凶是代理:代理把 eastmoney 转发到了被东财拒绝的出口。")
        print("修复:.env 里加一行 NO_PROXY=eastmoney.com 再跑(详见我的说明)。")
    elif via_proxy_ok:
        print("现在全通了——刚才是瞬时故障/限流,直接重跑 fetch_daily 即可。")
    else:
        print("直连也不通:IP 被限或更深层的网络问题。告诉我,我们换备用数据源。")


if __name__ == "__main__":
    main()
