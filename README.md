# TradeInvestAgent — A股投研 Agent

> 基于 LangChain 高层 `create_agent` 的智能体:自主调用行情/财报/新闻工具回答 A 股问题,
> 带多轮记忆(SqliteSaver checkpointer)、多数据源容灾、读穿式新闻 RAG,并附一个 FastAPI 调试台。

## 架构

```
应用层   app/          FastAPI 调试台:聊天 + 工具轨迹 + RAG 状态(SSE 流式 /api/chat/stream,会话增删查)
Agent层  agent/        create_agent(model, tools, system_prompt, checkpointer, SummarizationMiddleware)
工具层   tools/        @tool 封装:行情/财报直连、新闻读穿式 RAG、通用工具
存储层   storage/      vecdb.py = Chroma 向量库(按语料类别分 collection)
配置层   config/       settings(GLM/embedding/路径/RAGCollection 枚举/压缩阈值) + sys_prompt(工作流提示)
数据源   akshare       东财/腾讯/新浪 多源,按数据类型选、带 failover
```

## 目录

```
agent/agent.py        create_agent 装配 + run_agent_turn / stream_agent_turn(返回 answer + trace) + 会话 list/delete + CLI
tools/stock_tools.py  get_stock_price / get_stock_history / get_financial_report / ingest_stock_news
tools/rag_tools.py    query_rag(collection 参数用 RAGCollection 枚举约束)
tools/normal_tools.py get_weather(通用工具占位)
storage/vecdb.py      put / search / list_collections(Chroma, 单 collection per 类别)
config/settings.py    GLM/embedding 配置 + RAGCollection(StrEnum) + 上下文压缩阈值
config/sys_prompt.py  投研工作流系统提示
app/main.py           FastAPI 路由(只做传输,不含智能)
app/static/index.html 调试前端
tests/                pytest:纯逻辑 / 工具(akshare mock)/ vecdb(临时库隔离)/ API(SSE 传输)
```

## 工具与数据源

| 工具 | 数据 | 来源 / 策略 | 是否进 RAG |
|---|---|---|---|
| `get_stock_price` | 实时行情 | 东财 `stock_bid_ask_em` → 失败切腾讯 | 否,直连返回 |
| `get_stock_history` | 历史K线 | 东财 `stock_zh_a_hist` → 失败切腾讯 | 否,直连返回 |
| `get_financial_report` | 营收/净利/经营现金流 | 新浪 `stock_financial_abstract`(非东财) | 否,直连返回 |
| `ingest_stock_news` | 个股新闻(生产者) | 东财 `stock_news_em` → 写入 Chroma | 是 |
| `query_rag` | 新闻检索(消费者) | Chroma 语义检索 | — |

**数据源韧性**:akshare 免费源常被限流,已避免硬依赖东财——行情/历史做东财→腾讯 failover,
财报直接用新浪;个股新闻暂只东财源,已隔离在 `ingest_stock_news` 一处便于换源。

## 读穿式新闻 RAG

模型工作流(由系统提示约束):`query_rag` 先检索 → 未命中则 `ingest_stock_news` 抓取写入 → 再 `query_rag` 取回作答。
- 向量库**按语料类别分 collection**(`news` …),类别名由 `config.RAGCollection` 枚举约束,模型不能乱编。
- `symbol` 存 metadata 做过滤(高基数维度不建独立库,保留跨股票检索)。
- 结构化数字(价/史/财报)不进向量库——语义检索对数字无意义,一律精确查/直连。

## 快速开始

```bash
# 1. 装依赖(uv 按 uv.lock 精确复现,需 Python >= 3.13)
uv sync

# 2. 配置密钥(编辑 .env,已 gitignore)
#    GLM_API_KEY=... (GLM_BASE_URL/GLM_MODEL/EMBED_MODEL 可选)

# 3a. 命令行跑一轮 agent
uv run python -m agent.agent

# 3b. 起调试台,浏览器打开 http://127.0.0.1:8000
uv run uvicorn app.main:app --reload

# 4. 跑测试(akshare/LLM 全 mock,向量测落临时目录,不污染 data/)
uv run python -m pytest -q
```

## 关键设计取舍(可作为面试讨论点)

- **高层 `create_agent` 而非手写 LangGraph 节点/边**:ReAct 循环 + checkpointer 是框架标配,
  高层栈更贴生产;省下的是样板,买到的不是更强智能而是标准化。
- **工具结果直接进上下文,不自建缓存库**:有 checkpointer 做会话记忆,再叠一层取数缓存是重复造轮子。
- **RAG 只用于非结构化文本**:向量检索的价值在"模糊问题→相关段落",数字/表格走精确查。
- **collection 名/数据源由代码定,不让模型现编**:枚举白名单避免模型乱填污染索引。
- **上下文压缩用官方 `SummarizationMiddleware`,不自己写**:消息 token 超阈值自动摘要旧消息、保留近期且
  不拆散 AI/Tool 消息对;能用框架件就不重造。
- **测试全 mock 外部依赖 + 隔离临时目录**:akshare/LLM 打桩、向量读写落 `tmp_path`,保证确定性且不污染真实数据。

## 现状与待办

已完成:create_agent 编排 + 多轮记忆、`SummarizationMiddleware` 上下文压缩、多源容灾数据工具、读穿式新闻 RAG、
FastAPI 调试台(SSE 流式 + 历史会话增删查)、pytest 单测/传输测。

尚未做(诚实标注):
- 端到端测试与 CI 流水线(当前是单元/传输层)
- 可观测性(tracing / token 成本核算)
- IM 集成(飞书/企业微信)、定时任务、部署上线
- 更多文本语料(研报/公告)接入 RAG
