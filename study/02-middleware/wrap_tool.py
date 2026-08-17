"""
工具执行错误处理（wrap_tool_call）示例 —— 验证工具出错时 agent 不崩，而是把错误回传给模型
https://langchain-doc.cn/v1/python/langchain/agents.html#工具调用错误处理

核心思想：工具抛异常默认会中断整个 agent（对话直接报错退出）。用 @wrap_tool_call 写一个中间件，
在【每次调用工具前后】拦截调用：工具成功就原样放行，工具抛异常就把异常转成一条 ToolMessage
（内容是"工具错误，请重试"）塞回对话 —— 模型看到这条消息后自行决定重试或换一种问法，agent 得以继续。

【本文件主要验证什么】
    工具执行抛异常（search 直接 raise NotImplementedError）时，agent 不会崩，
    而是把错误作为 ToolMessage 回传给模型，模型重试后最终给出兜底回答。

【怎么验证，以及一个关键坑】
    ① 跑起来看输出：完整链路里会出现「AI 调工具 → ToolMessage 错误 → AI 重试/放弃」，
       中间那条 ToolMessage 就是本中间件的功劳。
    ② 关键坑：invoke 的输入必须是 {"messages": [...]}（消息数组），不能写 {"message": "..."}。
       create_agent 的输入 schema 只认 messages 通道，传错键名会让 state["messages"] 为空，
       模型第一次调用就收到空消息列表 → DeepSeek 直接 400 "Empty input messages"。
"""
import sys
import dotenv

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，模型可能回 emoji，改成 UTF-8
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool

dotenv.load_dotenv()                          # 加载 .env 里的 API key


# ============================== 工具错误处理核心 ==============================
@wrap_tool_call
def handle_tool_errors(request, handler):
    """工具执行异常时，向模型回传一条自定义错误消息。"""
    # @wrap_tool_call：把函数升级成中间件，LangChain 每次【调用工具前】都会先执行它。
    # request：本次工具调用信息。request.tool_call["id"] 是这次调用的唯一 id，
    #          回传的 ToolMessage 必须带上它，模型才能把错误消息"对上号"。
    # handler：真正执行工具的函数（每调用一次就真执行一次工具）。
    try:
        return handler(request)
    except Exception as e:
        # 工具成功 → 原样放行；工具抛异常 → 拦截下来，转成错误消息回传给模型。
        # 注意：若这里不拦截（直接 raise），异常会冒泡出去，整个 agent 对话就此中断。
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
# ==============================================================================


@tool
def search(query: str) -> str:
    """搜索方法，传入查询参数返回结果"""
    # 工具函数：docstring 会被当作工具说明喂给模型，让它知道何时调用。
    # 此处刻意抛异常，用来触发上面的错误处理中间件 —— 真实工具出错时就是走到这条路径。
    raise NotImplementedError


# 组装智能体：
#   middleware=[handle_tool_errors]  把错误处理中间件挂进中间件链（工具出错时不中断）
agent = create_agent(
    model="deepseek-chat",
    tools=[search],
    middleware=[handle_tool_errors]
)

# 完整链路：invoke({"messages": [HumanMessage(...)]})
#   → 模型收到问题 → 决定调用 search 工具
#   → search 抛 NotImplementedError → 中间件拦截 → 回传 ToolMessage（内容=工具错误）
#   → 模型看到错误消息 → 换一种问法重试 search → 再次失败 → 模型放弃并给兜底回答
# 预期输出：messages 里能看到
#   [AIMessage(带 tool_calls) → ToolMessage(工具错误) → AIMessage(兜底回答)]
result = agent.invoke(
    {"messages": [HumanMessage(content="查询一下今天的天气怎么样")]}
)

print(result)
