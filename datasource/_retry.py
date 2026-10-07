"""重试装饰器:对付免费接口的瞬时故障。

今天的 RemoteDisconnected 就是活教材:网络抖动/限流/代理抽风是常态,不是异常。
数据管道的设计假设应该是"接口随时会抖一下",重试是标配。

教学点【什么该重试,什么不该】:
- 该重试:瞬时错误(断连、超时、限流)——再试一次很可能就成功
- 不该重试:永久错误(代码写错、参数不对)——重试 100 次结果都一样,纯属浪费
这里为了简单统一重试,生产级做法是区分错误类型,或直接用 tenacity 库。
指数退避(exponential backoff):每次失败后等待时间翻倍,避免被打挂的接口刚喘过气又被我们打一顿。
"""
import functools
import time
from collections.abc import Callable

MAX_ATTEMPTS = 3
BASE_DELAY = 1.0  # 秒


def with_retry(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 简化处理:所有异常都重试
                if attempt == MAX_ATTEMPTS:
                    raise  # 三次都失败,把异常抛给调用方
                delay = BASE_DELAY * 2 ** (attempt - 1)
                print(f"    [retry] {fn.__name__} 第{attempt}次失败({e}), {delay:.0f}s 后重试")
                time.sleep(delay)

    return wrapper
