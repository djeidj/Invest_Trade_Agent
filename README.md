# TradeInvestAgent — 自动化投研 Agent

> 每天自动收集自选股(A股)的行情、新闻,生成带引用来源的研究简报;
> 支持追问:Agent 自主调用工具查数据、读财报后回答。

## 架构

```
数据层(akshare + SQLite)
   → 工具层(标准 schema 的 tool 封装 / MCP server)
   → Agent 层(手写 agent loop → 迁移 LangGraph 多 agent 编排)
   → 应用层(FastAPI 流式接口 + 前端 demo)
   → 质量层(tracing 可观测 + eval 回归集)
```

## 开发路线

- [x] D1 数据管道:akshare 行情/新闻 → SQLite(防腐层设计)
- [ ] D2 工具层封装 + LLM 客户端(智谱 GLM,OpenAI 兼容协议)
- [ ] D3 手写 agent loop(function calling / 重试 / 失败处理)
- [ ] D4 多 agent 简报生成(迁移 LangGraph,附手写版对比)
- [ ] D5 财报 RAG 追问,回答带引用溯源
- [ ] D6 FastAPI + 流式输出 + 前端 demo
- [ ] D7 tracing 可观测:每步 trace / 延迟 / token 成本
- [ ] D8 eval 回归集:固定问题集量化对比不同模型/提示词
- [ ] D9-10 MCP server + 定时任务 + 部署 + 演示视频

## 快速开始

```bash
# 1. 安装依赖(uv 自动创建虚拟环境并按 uv.lock 精确复现)
uv sync

# 2. 配置 key
cp .env.example .env   # 填入 GLM_API_KEY

# 3. 拉取自选股数据入库
uv run python -m scripts.fetch_daily
```
