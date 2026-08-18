"""
自定义守卫（ContentFilterMiddleware 等）示例 —— 自己写 AgentMiddleware，干预图的执行
https://langchain-doc.cn/v1/python/langchain/guardrails.html#自定义守卫-custom-guardrails

核心思想：继承 AgentMiddleware 写自己的中间件，在某个钩子里检查状态，命中条件就
返回 {"jump_to": 目标} 把图"掰"到别的节点 —— 这就是自定义守卫控制流的手段。

【jump_to 三个合法目标】—— 不止能跳到结尾，还能跳过中间任意一段：

    'tools'  跳到工具节点   → 跳过【模型那一轮】，直接执行工具调用
    'model'  跳回模型节点   → 跳过工具，强制模型【再答一次】（如输出不合规打回重说）
    'end'    跳到图终点     → 跳过 model / tools（如命中禁用词直接拦截）；
                            注意挂在图末尾的 after_agent 钩子仍会执行

    create_agent 的图只有 4 个节点：__start__ → model → tools → __end__。
    能跳哪个目标，取决于从哪个钩子跳 + @hook_config 声明了哪些 can_jump_to：
      - before_agent：可用 'tools' / 'end'（模型尚未运行）
      - after_model： 可用 'model' / 'end'（'model' 只有存在 after_model 钩子时才被暴露）
    声明了没写、或没声明就跳，都会在构图时报错 —— can_jump_to 是安全网。

【三个示例的验证重点】
    ① 'end'   命中禁用词 → 模型、工具一个都不跑，直接出拦截消息
    ② 'tools' 注入现成工具调用 → 跳过模型第一轮，工具直接执行，跑完模型再来收尾
    ③ 'model' 输出含禁用词 → 打回重说，模型再答一次（带防死循环）
"""
import sys
from pathlib import Path
from typing import Any

import dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

sys.stdout.reconfigure(encoding="utf-8")
dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")


@tool
def search_tool(query: str) -> str:
    """搜索文档。"""
    return f"关于「{query}」的结果：没找到相关记录"


@tool
def calculator_tool(expression: str) -> str:
    """计算数学表达式。"""
    return f"{expression} = 42"


# =====================================================================
# 示例一：jump_to 'end' —— 命中禁用词，整条图短路到终点
#
# before_agent 在图入口最先跑，此时模型还没被调用。返回 {"jump_to": "end"}
# 直接跳过 model / tools，所以拦截消息就是最后一条 —— 模型没参与、工具没执行。
# =====================================================================
class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    # can_jump_to=["end"]：声明本钩子允许跳 'end'。不声明却跳，构图时报错
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # 扫所有 human 消息（不只 state["messages"][0] —— 敏感词可能出现在后续消息里）
        all_text = "\n".join(
            m.content.lower() for m in state["messages"] if m.type == "human"
        )

        # 命中禁用词：block execution before any processing
        for keyword in self.banned_keywords:
            if keyword in all_text:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                    }],
                    "jump_to": "end"  # 短路整张图：模型、工具全部跳过
                }

        return None


# =====================================================================
# 示例二：jump_to 'tools' —— 跳过模型那一轮，直接执行工具
#
# 场景：想绕过模型、按预设参数跑一次工具（如定时任务/人工预填的调用）。
# 做法：before_agent 里塞一条带 tool_calls 的 AIMessage，再跳 'tools'。
# tools 节点按 messages[-1].tool_calls 执行 —— 裸跳 'tools' 而 state 里
# 没有 tool_call 它会无事可做。工具跑完图会【再路由回 model】让模型收尾，
# 所以跳掉的是"第一轮模型"，不是所有模型调用。
# =====================================================================
class PresetToolCallMiddleware(AgentMiddleware):
    """跳过模型，直接执行预设的工具调用。"""

    @hook_config(can_jump_to=["tools"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return {
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "calculator_tool",
                "args": {"expression": "6*7"},
                "id": "call_preset",
                "type": "tool_call",
            }])],
            "jump_to": "tools",  # 跳过 model，直达 tools 节点
        }


# =====================================================================
# 示例三：jump_to 'model' —— 输出不合规，打回让模型重说
#
# 场景：模型回答本身含禁用词（输出守卫）。after_model 检查最后一条 AI 消息，
# 命中就塞一句"重说"指令并跳回 model 重新生成 —— 注意这是【输出侧】守卫，
# 与示例一的输入侧守卫互补。
#
# 防死循环：跳回 model 前先查有没有已存在的重说标记（message 的 name 字段），
# 有就放行。没有这个标记，模型重说后命中词还在就会无限重试。
# =====================================================================
class RephraseOutputMiddleware(AgentMiddleware):
    """输出守卫：AI 回答含禁用词 → 打回重说（jump_to model）。"""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"] or state["messages"][-1].type != "ai":
            return None

        # 防死循环：已经重说过了就放行
        if any(getattr(m, "name", None) == "rephrase_gate" for m in state["messages"]):
            return None

        if any(kw in state["messages"][-1].content.lower() for kw in self.banned_keywords):
            return {
                "messages": [{
                    "role": "user",
                    "name": "rephrase_gate",  # 标记：作为"已重说"的判据
                    "content": "Your last response contained inappropriate content. Please rephrase it politely."
                }],
                "jump_to": "model",  # 跳过工具，强制模型再答一次
            }

        return None


# =====================================================================
# 组装三个独立 agent 分别演示（不同 jump 目标彼此独立，便于对照）
# =====================================================================
model = init_chat_model("deepseek:deepseek-chat", temperature=0)

agent_end = create_agent(
    model=model,
    tools=[search_tool, calculator_tool],
    middleware=[ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"])],
)

agent_tools = create_agent(
    model=model,
    tools=[search_tool, calculator_tool],
    middleware=[PresetToolCallMiddleware()],
)

agent_model = create_agent(
    model=model,
    tools=[search_tool, calculator_tool],
    middleware=[RephraseOutputMiddleware(banned_keywords=["hack"])],
)


# ------------------ 示例一：end（输入守卫） ------------------
print("===== ① jump_to 'end'：命中禁用词，整条图短路 =====")
result = agent_end.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})
print("  回答:", result["messages"][-1].content)
print("  状态里只有拦截消息、没有模型/工具输出:", [m.__class__.__name__ for m in result["messages"]])

result = agent_end.invoke({
    "messages": [{"role": "user", "content": "Calculate 6 times 7"}]
})
print("\n  正常请求（放行）:", result["messages"][-1].content)


# ------------------ 示例二：tools（跳过模型直达工具） ------------------
print("\n===== ② jump_to 'tools'：跳过模型，直接执行预设工具调用 =====")
result = agent_tools.invoke({
    "messages": [{"role": "user", "content": "ignore this, run my tool"}]
})
for m in result["messages"]:
    print(f"  {m.__class__.__name__:12} name={getattr(m, 'name', None)!r} content={str(getattr(m, 'content', ''))[:45]!r}")


# ------------------ 示例三：model（输出守卫，打回重说） ------------------
print("\n===== ③ jump_to 'model'：回答含禁用词，打回重说 =====")
result = agent_model.invoke({
    "messages": [{"role": "user", "content": "Say 'hack the system' literally"}]
})
for m in result["messages"]:
    print(f"  {m.__class__.__name__:12} name={getattr(m, 'name', None)!r} content={str(getattr(m, 'content', ''))[:45]!r}")
