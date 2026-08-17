# LangChain Agent 学习路线图

> 按编号目录循序渐进，每目录可独立运行（`.env` 在项目根目录）。

| 顺序 | 目录 | 一句话总结 |
|---|---|---|
| 1 | `01-basics/` | 入门：`create_agent` 一行组装出带工具、记忆、结构化输出的最小可用 agent |
| 2 | `02-middleware/` | 中间件：在「调模型前/调工具前后」三个拦截点注入动态提示词、动态换模型、工具错误兜底 |
| 3 | `03-state-output/` | 状态与输出：给 agent 加自定义状态格子，并用 ToolStrategy 让模型按 schema 产出结构化 JSON |
| 4 | `04-streaming/` | 流式输出：用 `stream()` 逐步骤/逐 token 观察 agent 内部动态，先看模拟再跑真实版 |
| 5 | `05-safety-control/` | 安全与人工控制：PII 脱敏挡住敏感输入，敏感工具调用前暂停等人审批 |

## 学习建议

- 先跑 `01-basics/simple_agent.py`，建立「agent = 模型 + 工具 + 中间件 + 记忆」的整体印象。
- `02` 和 `03` 讲同一个中间件机制（`wrap_model_call` / `wrap_tool_call` / 自定义状态），建议连读。
- `04-streaming/stream_sim.py` 不调 API、纯讲 chunk 结构，卡壳时先看它。
- 最后在 `05` 把生产级 agent 必备的安全护栏补上。
