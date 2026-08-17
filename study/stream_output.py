"""
流式输出（agent.stream）示例 —— 逐块观察智能体的每一步状态，而不是等全部结束后一次性拿结果。
https://langchain-doc.cn/v1/python/langchain/agents.html

【本文件验证什么】
    create_agent 出来的 agent 自带 .stream()，stream_mode="values" 时每次产出该时间点的完整图状态。
    在流式循环里能实时看到：模型决定调用工具 → 工具返回结果 → 模型基于结果生成最终回答。

【关键坑】
    stream 的输入同样必须是 {"messages": [...]}（同 invoke），并且流式状态下消息会逐条追加，
    所以循环里取 chunk["messages"][-1]（最新一条）即可 —— 它可能是普通回答，也可能带 tool_calls。
"""
import sys

import dotenv

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，模型可能回 emoji，改成 UTF-8
from langchain.agents import create_agent
from langchain.tools import tool

dotenv.load_dotenv()                          # 加载 .env 里的 API key


# ============================== 模拟工具 ==============================
@tool
def search_ai_news(query: str) -> str:
    """搜索 AI 相关新闻，传入查询关键词，返回匹配的新闻摘要。"""
    # 模拟实现：真实场景这里会去调新闻 API / 搜索引擎，本示例直接返回写死的字符串，
    # 仅用于验证"工具结果 → 模型总结"这条链路走通。
    return (
        "1. DeepSeek 发布新一代推理模型，长上下文能力大幅提升；"
        "2. OpenAI 开源智能体框架，可编排多个子智能体协作；"
        "3. 国内大模型 API 价格持续下调，推理成本一年降了九成。"
    )
# ==============================================================================


# 组装智能体：model 用 DeepSeek，只挂这一个新闻搜索工具
agent = create_agent(
    model="deepseek-chat",
    tools=[search_ai_news],
)

# 流式运行：stream_mode="values" 产出完整状态字典，messages 里会逐条追加新消息
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "搜索 AI 新闻并总结发现"}]
}, stream_mode="values"):
    # 每个块包含该时间点的完整状态
    latest_message = chunk["messages"][-1]
    # 先判断 tool_calls：模型可能在同一条消息里既写思考文本又带 tool_calls，
    # 若先判断 content，工具调用名会被 if 分支吞掉，永远走不到 elif。
    # 用 getattr 兜底：values 模式第一块是原始 HumanMessage，它没有 tool_calls 属性。
    tool_calls = getattr(latest_message, "tool_calls", None) or []
    if tool_calls:
        print(f"正在调用工具：{[tc['name'] for tc in tool_calls]}")
    elif latest_message.content:
        print(f"智能体：{latest_message.content}")

# =====================================================================
# 第二种：stream_mode="messages" —— 逐 token 流式（打字机效果）
#   values 模式按"步骤"整块吐状态；messages 模式把生成过程按 token 吐，
#   每次 yield (message_chunk, metadata)。message_chunk.content 是一小段增量文本，
#   拼接起来就是完整回答 —— 代价是没有完整 messages 列表，只能看到文本流。
# =====================================================================
print("\n========== messages 模式：逐字流式输出 ==========")
last_node = None
for msg_chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "搜索 AI 新闻并总结发现"}]},
    stream_mode="messages",
):
    # metadata 里的 langgraph_node：这段增量来自哪个节点（模型 / 某个工具）。
    # 节点一变化就换行打印标签，否则连续 token 会挤成一行分不清谁在说。
    node = metadata.get("langgraph_node", "?")
    if node != last_node:
        print(f"\n── {node} ──", flush=True)
        last_node = node
    # msg_chunk：一小段增量文本；只打印真正的文本 —— 工具调用的 token 碎片 content 为空。
    if msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)  # end="" 不换行，flush 立即输出
print()  # 收尾换行
