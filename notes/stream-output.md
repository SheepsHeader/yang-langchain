# Stream 流式输出 —— agent.stream 两种粒度彻底搞懂

> 来源：https://langchain-doc.cn/v1/python/langchain/agents.html
> 环境：langchain `1.3.15` + `ai_agent` conda 环境 + DeepSeek，已实测验证
> 配套代码：`../study/stream_output.py`（真实跑通）、`../study/stream_sim.py`（模拟 chunk 演化，不调 API）
> 相邻参考：`../study/simple_agent.py`（`invoke` 一次性调用）、`../study/wrap_tool.py`（工具错误处理中间件）

---

## 📌 一句话点破

`agent.stream()` 和 `agent.invoke()` 的区别不在结果，而在**过程能不能看见**：
`invoke` 憋到全部跑完才返回一次；`stream` 把过程切成一帧一帧吐出来，实时看到"模型决定调工具 → 工具返回结果 → 模型生成总结"的完整链路。

**stream 有两种粒度**，选哪个由 `stream_mode` 决定：

| stream_mode | 吐什么 | 每帧的形状 | 适合看 |
|---|---|---|---|
| `"values"` | 每步一个**完整状态快照** | `{"messages": [到目前为止的全部消息]}` | 流程：调了谁、结果、最终回答 |
| `"messages"` | 每个 **token 的增量文本** | `(message_chunk, metadata)` | 打字机效果的逐字生成过程 |

---

## 🧩 values 模式：按"步骤"整块吐

`chunk` 是一个大字典，`chunk["messages"]` 是**到目前为止的全部对话**，每来一帧就追加一条。所以 `chunk["messages"][-1]` 永远是"最新那条消息"。动态演化一共 4 帧（`stream_sim.py` 里逐帧模拟了这个过程）：

```
帧1  [Human(搜索 AI 新闻并总结发现)]
帧2  [Human(...), AI(tool_calls=search_ai_news)]          ← 模型决定调工具
帧3  [Human(...), AI(tool_calls), Tool(新闻内容)]          ← 工具执行完，结果回填
帧4  [Human(...), AI(tool_calls), Tool(...), AI(总结...)]  ← 模型基于结果生成回答
```

消费循环（见 `stream_output.py` 上半段）：

```python
for chunk in agent.stream({"messages": [...]}, stream_mode="values"):
    latest_message = chunk["messages"][-1]
    tool_calls = getattr(latest_message, "tool_calls", None) or []  # ② 见下方坑 2
    if tool_calls:                                          # ① 先判断工具调用
        print(f"正在调用工具：{[tc['name'] for tc in tool_calls]}")
    elif latest_message.content:
        print(f"智能体：{latest_message.content}")
```

---

## ⚡ messages 模式：按 token 逐字吐

这是大家直觉里"流式输出"的样子（ChatGPT 打字效果）。每个 `msg_chunk` 是模型吐出的**一小段增量文本**，`metadata` 里的 `langgraph_node` 标明这段增量来自哪个节点（模型 / 某个工具）。

```python
last_node = None
for msg_chunk, metadata in agent.stream({"messages": [...]}, stream_mode="messages"):
    node = metadata.get("langgraph_node", "?")
    if node != last_node:               # 节点一换就换行打标签，否则全部挤成一行
        print(f"\n── {node} ──", flush=True)
        last_node = node
    if msg_chunk.content:               # 只打印文本；工具调用的 token 碎片 content 为空
        print(msg_chunk.content, end="", flush=True)
print()
```

运行效果：

```
── model ──   我来帮你搜索一些 AI 相关的新闻...
── tools ──   1. DeepSeek 发布新一代推理模型；2. ...
── model ──   根据搜索结果总结：模型能力提升、Agent 生态发展、成本大降。
```

**代价**：没有完整的 `messages` 列表，只有零散增量文本，所以 values 模式里那套"取最新一条完整消息"的写法在这里用不了。

---

## 🐛 踩过的坑（按踩坑顺序）

1. **判断顺序：先 `tool_calls` 后 `content`。** 模型可能在同一条消息里既写思考文字、又同时发起工具调用——若先判断 `content`，工具名会被 `if` 分支吞掉，`elif` 永远走不到。
2. **`getattr` 兜底。** values 模式第一帧是原始 `HumanMessage`，它**没有** `tool_calls` 属性，直接访问会 `AttributeError`。用 `getattr(msg, "tool_calls", None) or []` 让所有消息类型都安全。
3. **输入必须是 `{"messages": [...]}`。** 和 `invoke` 一样只认 `messages` 通道，写错键名会收到空消息列表，DeepSeek 直接 400。
4. **工具可能被并行调用多次。** 实测 DeepSeek 有时一次触发 3 次 `search_ai_news`（同参数），tool_calls 里会出现 3 条、工具结果重复 3 遍——这是模型行为，不是代码 bug。
5. **Windows 控制台默认 GBK。** 模型可能回 emoji，开头加 `sys.stdout.reconfigure(encoding="utf-8")`，否则打印乱码。

---

## 📋 结论速查表

| 概念 | 一句话 |
|---|---|
| stream 和 invoke | invoke 等全部跑完返回一次；stream 把过程一帧帧吐出来 |
| `stream_mode="values"` | 每步一个完整状态快照，`chunk["messages"][-1]` 取最新一条 |
| `stream_mode="messages"` | 每 token 一小段增量 `(msg_chunk, metadata)`，`metadata["langgraph_node"]` 标注来源 |
| 两模式选谁 | 看流程选 values；看逐字生成选 messages |
| 坑 1 | 判断顺序：先 `tool_calls` 后 `content` |
| 坑 2 | `HumanMessage` 没有 `tool_calls`，用 `getattr` 兜底 |
| 坑 3 | 输入键名必须是 `messages` |
| messages 模式的代价 | 没有完整消息列表，只能拼增量文本 |

## 相关文件

- `../study/stream_output.py` — 真实流式示例：values 模式流程观察 + messages 模式逐字输出
- `../study/stream_sim.py` — 不调 API 的模拟器：展示 values 模式 chunk 字典 4 帧演化，及同样消费逻辑的输出
- `../study/simple_agent.py` — `invoke` 一次性调用、context / checkpointer / response_format
