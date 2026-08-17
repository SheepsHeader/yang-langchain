# 中间件方法的执行时机（Middleware Timing）

> 环境：langchain `1.3.15` + `ai_agent` conda 环境，源码级确认
> 源码位置：`langchain/agents/middleware/types.py`（钩子定义）、`langchain/agents/factory.py`（图的编排）
> 配套代码：`../study/custom_state.py`（`PrefMiddleware` 实战）
> 相邻参考：`../study/custom-state-memory.md`

---

## 💬 提问：中间件的方法什么时候执行？在 model 回答前后吗？

**不是简单的"回答前后"。** agent 是**循环**（模型→工具→模型…），所以这些方法分两类触发时机：

| 时机 | 钩子 | 执行次数 |
|---|---|---|
| 整场一次 | `before_agent` / `after_agent` | 一次 `invoke()` 只跑一遍 |
| 每轮一次 | `before_model` / `wrap_model_call` / `after_model` / `wrap_tool_call` | agent 每循环一圈各一次，会跑多次 |

---

## 📐 完整执行顺序（一次带工具的 invoke）

```
invoke()
  before_agent ───────────── ① 整场开场，仅一次
   ┌─ agent 主循环 ───────────────────────────┐
   │ before_model ─────── ② 这轮调模型前        │ ← 返回 dict = 状态更新
   │ wrap_model_call ──── ③ 包住模型调用        │ ← 改请求/响应/重试/短路
   │    ↓ 真正调模型                            │
   │ after_model ──────── ④ 这轮模型回答后      │ ← 返回 dict = 状态更新
   │ wrap_tool_call ───── ⑤ 包住每个工具调用    │ ← 改参数/结果/重试
   │    ↓ 真正调工具                            │
   └────── 回到 ② 再来一轮 ─────────────────────┘
  after_agent ───────────── ⑥ 整场收尾，仅一次
```

---

## 🔑 关键点

1. **agent 循环导致多次触发**：模型可能连续调好几轮工具，所以 `before_model` / `after_model` / `wrap_model_call` 每一轮循环各触发一次；`before_agent` / `after_agent` 只在首尾各一次。

2. **源码佐证（factory.py 建图逻辑）**：
   - `before_agent` 是图的**入口节点**（`entry_node`）
   - `before_model` 是**循环入口**（`loop_entry_node`，工具调完回跳到这里）
   - `after_model` 是**循环出口**（`loop_exit_node`，模型调完后在此路由：有工具 → 去工具节点，无 → 去退出节点）
   - `after_agent` 是**出口节点**（`exit_node`，仅一次）

3. **`wrap_model_call` 与 `before_model` 的区别（最实用的一对）**：
   - `before_model` 返回 dict = **状态更新**，改不了模型请求
   - `wrap_model_call` 能**改模型请求本身**（`request.override(...)`）、改响应、重试、短路
   - 时机只差一层：`before_model` 在该轮真正调模型之前；`wrap_model_call` 把"调模型"这一下整个包住

4. **实践对应**：`../study/custom_state.py` 的 `PrefMiddleware` 用 `wrap_model_call` 注入用户偏好，正是因为 `before_model` 改不了系统消息。

---

## 相关文件

- `../study/custom_state.py` — `PrefMiddleware.wrap_model_call` 实战：调模型前注入用户偏好
- `../study/custom-state-memory.md` — 自定义状态：加格子 vs 让模型看见格子的区别
- `../study/wrap_model.py` — `@wrap_model_call` 装饰器写法（动态换模型）
