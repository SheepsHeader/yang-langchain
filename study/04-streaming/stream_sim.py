"""
模拟 agent.stream(stream_mode="values") 产出的 chunk 字典，看它每一帧的动态变化。
用普通 Python 对象代替真实消息，不调 API，方便理解结构。

真实 chunk 结构：chunk = {"messages": [到目前为止的全部消息]}
每来一帧，messages 就追加一条，chunk["messages"][-1] 永远是"最新那条消息"。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")


class Msg:
    """模拟一条消息。真实类型：HumanMessage（用户）/ AIMessage（模型）/ ToolMessage（工具结果）"""

    def __init__(self, role, content="", tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []  # 只有 AIMessage 会有，别的消息 getattr 兜底成 []

    def __repr__(self):
        calls = " ".join(f"调{t['name']}" for t in self.tool_calls)
        body = self.content or calls or "(空)"
        return f"{self.role}({body})"


# 四帧模拟。真实运行时这些由 langgraph 逐帧产生，这里直接模拟结果。
# 观察重点：每一帧 messages 都比上一帧多一条，[-1] 指向的"最新消息"在变。
frames = [
    # 帧1：用户消息刚进来，messages 只有一条
    {"messages": [Msg("Human", "搜索 AI 新闻并总结发现")]},
    # 帧2：模型决定调工具。这条 AIMessage 只有 tool_calls、没有文字
    {"messages": [
        Msg("Human", "搜索 AI 新闻并总结发现"),
        Msg("AI", tool_calls=[{"name": "search_ai_news"}]),
    ]},
    # 帧3：工具执行完，结果作为 ToolMessage 追加进来
    {"messages": [
        Msg("Human", "搜索 AI 新闻并总结发现"),
        Msg("AI", tool_calls=[{"name": "search_ai_news"}]),
        Msg("Tool", "1. DeepSeek 发布新一代推理模型；2. OpenAI 开源智能体框架；3. 国内模型降价。"),
    ]},
    # 帧4：模型拿到工具结果，生成最终回答
    {"messages": [
        Msg("Human", "搜索 AI 新闻并总结发现"),
        Msg("AI", tool_calls=[{"name": "search_ai_news"}]),
        Msg("Tool", "1. DeepSeek 发布新一代推理模型；2. OpenAI 开源智能体框架；3. 国内模型降价。"),
        Msg("AI", "根据搜索结果总结：大模型能力提升、Agent 生态发展、成本大降。"),
    ]},
]

print("======== 逐帧看 chunk 字典的动态变化 ========")
for i, chunk in enumerate(frames, 1):
    print(f"────── 帧 {i} ──────")
    print(f"  chunk['messages'] 长度：{len(chunk['messages'])}")
    for msg in chunk["messages"]:
        print(f"    {msg}")
    print(f"  → 最新一条 chunk['messages'][-1]：{chunk['messages'][-1]}")
    print()

print("======== 用 stream_output.py 的消费逻辑跑一遍 ========")
for chunk in frames:
    latest_message = chunk["messages"][-1]
    tool_calls = getattr(latest_message, "tool_calls", None) or []
    if tool_calls:
        print(f"正在调用工具：{[tc['name'] for tc in tool_calls]}")
    elif latest_message.content:
        print(f"智能体：{latest_message.content}")
