"""
人工审批（HumanInTheLoopMiddleware）示例 —— 敏感工具调用前暂停，等人批准
https://langchain-doc.cn/v1/python/langchain/agents.html#人工介入

核心思想：给每个工具配置是否打断（interrupt_on）。模型想调用敏感工具
（发邮件、删库）时，agent 先暂停、把"请求"抛给外面的人；人用
Command(resume=...) 带着决定恢复，批准才真正执行工具，安全操作自动放行。

【本文件主要验证什么】
    ① 模型请求 send_email_tool 时 agent 停在中断点，工具没有真正执行；
    ② 人工 approve 后，send_email_tool 才被放行执行，对话继续。

【几个关键概念速览】（细节在代码对应处展开）
    - interrupt_on：每个工具一个开关。key 必须和【工具注册名】完全一致。
    - interrupt()：langgraph 的"暂停"原语。图停在它那一行，控制权交回调用方。
    - Command(resume=...)：恢复时喂给挂起中的 interrupt() 的返回值，即人工的决定。
    - checkpointer + thread_id：图"暂停到哪了"是持久化存的，没有它们就无法恢复。
    - 观测"工具有没有真执行"：数消息条数不可靠，要看有没有对应工具名的 ToolMessage。
"""
import sys
from pathlib import Path

import dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.stdout.reconfigure(encoding="utf-8")
dotenv.load_dotenv(Path(__file__).parent.parent / ".env")


@tool
def search_tool(query: str) -> str:
    """搜索文档。"""
    return f"关于「{query}」的结果：没找到相关记录"


@tool
def send_email_tool(recipient: str, content: str) -> str:
    """给指定收件人发邮件。"""
    return f"邮件已发送给 {recipient}"


@tool
def delete_database_tool(db_name: str) -> str:
    """删除指定数据库。"""
    return f"数据库 {db_name} 已删除"


model = init_chat_model("deepseek:deepseek-chat", temperature=0)

agent = create_agent(
    model=model,
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 要求批准敏感操作
                # 注意：key 必须和【工具注册名】完全一致（@tool 函数默认注册名 = 函数名，
                # 所以这里是 send_email_tool，不是 send_email）。
                #
                # 匹配逻辑：中间件拿 tool_call["name"] 去 interrupt_on 里查，
                #     if (config := self.interrupt_on.get(tool_call["name"])) is not None:
                #         ... 才进入中断流程
                # 所以 key 配错 = get() 返回 None = 静默跳过中断 = 工具直接执行，不报错。
                # 更坑的是 __init__ 不校验 key 是否对应真实工具：写 True 配错名字，
                # 无警告无异常，静默失效。官方文档示例就有这个坑（tool 叫 send_email_tool
                # 却配 "send_email"），照抄必踩。@tool(name="send_email") 显式改名后
                # 配 "send_email" 才会匹配 —— 本质是"等于工具注册名"。
                "send_email_tool": True,
                "delete_database_tool": True,
                # 自动批准安全操作（False = 不中断，模型调用 search_tool 直接执行）
                "search_tool": False,
            }
        ),
    ],
    # 在中断期间持久化状态
    checkpointer=InMemorySaver(),
)

# 人工审核需要一个线程 ID 来进行持久化
#（恢复时靠它找到"暂停到哪了"，相当于对话的身份证）
config = {"configurable": {"thread_id": "some_id"}}

# =====================================================================
# 第一次 invoke：启动图，从头正常跑
#
# 为什么这里不用 Command？因为图还没有任何挂起的中断 —— {"messages": [...]}
# 是普通输入，从图的入口进入。中断还没发生，自然没有"恢复"一说。
# =====================================================================
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to alice@example.com saying 'Project update meeting at 3pm'"}]},
    config=config,
)

print("=== 第一次 invoke 结束，agent 停在中断点 ===")

# =====================================================================
# 读中断信息：result["__interrupt__"]
#
# 中断发生时，langgraph 把控制权交回调用方，并在返回结果里带一个特殊 key
# __interrupt__（中断对象元组；没中断就是空元组）。interrupt.value 就是
# 中间件传给 interrupt() 的那个 dict，结构：
#   {"action_requests": [{name, args, description}, ...],   # 待审批的工具调用清单
#    "review_configs": [...]}                               # 每个动作的审批配置
# 所以打印 description 的取法：value["action_requests"][0]["description"]。
# value 是 dict 不是对象 —— 写成 interrupt.value.description 会 AttributeError。
# =====================================================================
for interrupt in result.get("__interrupt__", ()):
    print(f"等待人工审批: {interrupt.value['action_requests'][0]['description']}")

# =====================================================================
# 观测"邮件有没有真的发出去" —— 别数消息条数！
#
# 工具只有真正执行了，状态里才会多出一条 ToolMessage（name=工具名）。
# 数条数不可靠：若模型先调了 search_tool（自动放行）再要调 send_email_tool，
# 条数会是 4，但 send_email_tool 照样没执行。所以正确做法是按
# "ToolMessage 且 name == send_email_tool" 过滤 —— 有 = 执行了，没有 = 没执行。
# =====================================================================
print("邮件有没有真的发出去？看状态里有没有 send_email_tool 的 ToolMessage（工具执行了才会有）：")
emails_sent = [m for m in result["messages"]
               if m.__class__.__name__ == "ToolMessage" and getattr(m, "name", None) == "send_email_tool"]
print("  send_email_tool 是否已执行:", len(emails_sent) > 0)
print("  当前消息:", [m.__class__.__name__ for m in result["messages"]])

# =====================================================================
# 恢复：为什么这里要 Command(resume=...)？
#
# 中断的本质：图还停在中间件里那个 interrupt(hitl_request) 调用的"那一行"，
# 等它的返回值。此刻你再传 {"messages": [...]} 没用 —— 那会走图的入口，
# 而不是回答这个挂起的调用。Command(resume=...) 就是 langgraph 的专用通道：
# "把这个值喂给正在挂起的 interrupt()"。喂进去的值恰好就是 interrupt()
# 的返回值，中间件拿到后做：
#     decisions = interrupt(hitl_request)["decisions"]   # 挂起时这里交出控制权
# 恢复传的 {"decisions": [{"type": "approve"}]} 就是它的返回值，
# ["decisions"] 取出人工决定，approve 放行 send_email_tool。
#
# 能恢复的前提：checkpointer（InMemorySaver）+ 同一个 thread_id。
# 图"暂停到哪了"是持久化的，缺了任何一个都无法续跑。
# Command 就两种用途：resume= 喂值给挂起的中断（本项目）；goto=/update=
# 手动控制图流向或改状态。
# =====================================================================
print("\n=== 人工批准，恢复对话 ===")
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
)

# 恢复后模型正常收尾。审批通过的最硬证据：状态里出现了 send_email_tool
# 的 ToolMessage（用同样的按名字过滤定位，别用 result["messages"][-2] 猜位置）。
print("恢复后模型的回答:", result["messages"][-1].content)
tool_msgs = [m for m in result["messages"]
             if m.__class__.__name__ == "ToolMessage" and getattr(m, "name", None) == "send_email_tool"]
print("审批通过的证据（send_email_tool 执行产生的 ToolMessage）:", tool_msgs[0].content)

# =====================================================================
# 附：ToolMessage 长什么样（邮件发出后，状态里那条）
#
# ToolMessage(
#     content='邮件已发送给 alice@example.com',   # 工具函数 return 的字符串
#     name='send_email_tool',                     # 工具名 —— 判断"哪个工具执行了"看它
#     id='0edda3fa-...',                          # 消息自身 ID
#     tool_call_id='call_00_...',                 # ← 指向模型 AI 消息里同 id 的 tool_call，
# )                                               #   形成"模型请求 → 工具结果"的配对
#
# 判断"发出去没"：name == send_email_tool 的 ToolMessage 存在，且 tool_call_id
# 能对上 AI 消息里那个 send_email_tool 的 tool_call。没执行就压根没这条消息。
# =====================================================================
