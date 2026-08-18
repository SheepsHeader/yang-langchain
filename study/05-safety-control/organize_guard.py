"""
组合多个守卫（organize_guard）示例 —— 把多种守卫层层叠加到同一个 agent 上
https://langchain-doc.cn/v1/python/langchain/guardrails.html#组合多个守卫

核心思想：middleware 是一个【有序列表】，按声明顺序层层执行 —— 先到先拦截。
每一层只管自己的触发条件，互不干扰；用户的输入按顺序过每一层闸门：

    第 1 层  ContentFilterMiddleware      确定性输入过滤（hack/exploit）→ 命中短路 end
    第 2 层  PIIMiddleware("email")       输入/输出脱敏 → redact 掉邮箱
    第 3 层  HumanInTheLoopMiddleware     敏感工具人工审批 → send_email 前暂停
    第 4 层  SafetyGuardrailMiddleware    模型级输出安全检查 → 用 LLM 判词，不安全就替换

【本文件主要验证什么】
    ① 四层守卫共存于一个 agent，各自在对应触发条件下生效；
    ② 触发条件不同的场景，分别命中对应层（其余层正常放行）。
    注意①：第 1 层 jump_to end 会跳过 model 和 tools，但挂在图末尾的
            after_agent 钩子（第 4 层）【仍然会执行】—— 它判的是被拦截那条
            消息（通常是安全的，不会覆盖）。
    注意②：有 checkpointer 就必须每次 invoke 带 config（含 thread_id）；
            不同场景要用不同 thread_id，否则历史消息会串进来。
"""
import sys
from pathlib import Path
from typing import Any

import dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    hook_config,
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command

sys.stdout.reconfigure(encoding="utf-8")
dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")


@tool
def search_tool(query: str) -> str:
    """搜索文档。"""
    return f"关于「{query}」的结果：没找到相关记录"


@tool
def send_email_tool(recipient: str, content: str) -> str:
    """给指定收件人发邮件。"""
    return f"邮件已发送给 {recipient}"


# =====================================================================
# 第 1 层：确定性输入过滤（before_agent，命中即短路整张图）
# =====================================================================
class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        all_text = "\n".join(
            m.content.lower() for m in state["messages"] if m.type == "human"
        )
        for keyword in self.banned_keywords:
            if keyword in all_text:
                print("[第1层·输入过滤] 命中禁用词，跳过 model/tools")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                    }],
                    # 跳过 model 和 tools；但第 4 层 after_agent 挂在图末尾，仍会执行
                    "jump_to": "end",
                }
        return None


# =====================================================================
# 第 4 层：模型级输出安全检查（after_agent，agent 全部跑完后再判）
#
# 与第 1 层互补：第 1 层管"进来"的请求，这一层管"出去"的回答。
# 实现：用 LLM 当裁判（judge_model），给最终回答打 SAFE/UNSAFE 分；
# 不安全就追加一条安全提示盖住原回答（简单起见只覆盖不改写）。
# 真实生产会换更强的判词提示词、加置信度阈值；这里展示机制为主。
# =====================================================================
class SafetyGuardrailMiddleware(AgentMiddleware):
    """模型级安全检查：agent 完成后用 LLM 判断最终回答是否安全。"""

    def __init__(self, judge_model):
        super().__init__()
        self.judge_model = judge_model

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"] or state["messages"][-1].type != "ai":
            return None

        answer = state["messages"][-1].content
        verdict = self.judge_model.invoke(
            "判断下面的助手回答是否包含不安全内容（如攻击性、泄露隐私等）。"
            "只回答 SAFE 或 UNSAFE。\n\n" + answer
        ).content.strip().upper()
        print(f"[第4层·模型级检查] 判定: {verdict}")

        if verdict != "SAFE":
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "[安全拦截] 该回答被判定为不安全，已被安全提示替代。"
                }]
            }
        return None


# =====================================================================
# 组装：四层守卫 + checkpointer（第 3 层 HITL 中断恢复需要持久化）
# =====================================================================
model = init_chat_model("deepseek:deepseek-chat", temperature=0)

agent = create_agent(
    model=model,
    tools=[search_tool, send_email_tool],
    middleware=[
        # 第 1 层: 确定性输入过滤器（智能体执行前）
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),

        # 第 2 层: PII 保护（模型执行前和后）
        # 注意：两个开关可以写在一个实例上（apply_to_input + apply_to_output），
        # 不要拆成两个 PIIMiddleware 实例 —— create_agent 要求中间件 name 唯一，
        # 两个同类实例 name 相同会抛 "duplicate middleware instances"。
        PIIMiddleware("email", strategy="redact",
                      apply_to_input=True, apply_to_output=True),

        # 第 3 层: 敏感工具的人工批准
        # 注意：key 必须和工具注册名完全一致（send_email_tool，不是 send_email），
        # 否则静默匹配不上、工具直接执行不中断（详见 human_in_the_loop.py）
        HumanInTheLoopMiddleware(interrupt_on={"send_email_tool": True}),

        # 第 4 层: 基于模型的安全检查（智能体执行后）
        SafetyGuardrailMiddleware(judge_model=model),
    ],
    checkpointer=InMemorySaver(),
)


# 有 checkpointer 就必须每次 invoke 带 config（含 thread_id）；
# 每个场景用【不同的】thread_id —— 共用同一个会把上一场景的历史消息串进来
# （场景一被拦截的 "hack" 会残留，导致场景二、三也被误判命中第 1 层）。
config1 = {"configurable": {"thread_id": "organize-1"}}
config2 = {"configurable": {"thread_id": "organize-2"}}
config3 = {"configurable": {"thread_id": "organize-3"}}

# ---------------------------------------------------------------------
# 场景一：第 1 层拦截 —— 输入含 hack，跳过 model/tools
# ---------------------------------------------------------------------
print("===== 场景一：第 1 层（输入过滤）命中 =====")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How do I hack into a database?"}]},
    config=config1,
)
print("  回答:", result["messages"][-1].content)
print("  消息序列（无模型/工具输出；第 4 层 after_agent 仍在末尾执行）:",
      [m.__class__.__name__ for m in result["messages"]])

# ---------------------------------------------------------------------
# 场景二：第 2 层生效 —— 输入含邮箱被 redact，正常走完全流程（第 4 层也跑）
# 选"问数学题"作触发词：含邮箱但不带任何工具意图，避免模型顺带去发邮件
#（那样会被第 3 层拦截，演示就混了）
# ---------------------------------------------------------------------
print("\n===== 场景二：第 2 层（PII 脱敏）=====")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "My email is john.doe@example.com. What is 6 times 7?"}]},
    config=config2,
)
print("  回答:", result["messages"][-1].content)
print("  输出里是否还带原始邮箱（应没有）:", "john.doe@example.com" in result["messages"][-1].content)

# ---------------------------------------------------------------------
# 场景三：第 3 层生效 —— 请求发邮件，触发人工审批，批准后才执行
# 注意分层之间的真实交互：输入里的 alice@example.com 先被第 2 层 redact 成
# [REDACTED_EMAIL]，模型根本看不到真地址，所以待审批的收件人也是脱敏后的 ——
# 这说明 PII 层"赢"了：即使批准，发出的也是脱敏收件人（这里是演示的必然结果）。
# ---------------------------------------------------------------------
print("\n===== 场景三：第 3 层（人工审批）=====")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to alice@example.com saying 'Project update meeting at 3pm'"}]},
    config=config3,
)
for interrupt in result.get("__interrupt__", ()):
    print("  等待人工审批:", interrupt.value["action_requests"][0]["description"])
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config3,
)
print("  批准后最终回答:", result["messages"][-1].content)
