"""自定义状态（Custom State）—— 给智能体的"小本子"加格子。

文档"记忆"小节讲的其实是这个：除了自动维护的对话历史（messages），
你可以给智能体的状态加自定义字段，让它在对话期间记住"非对话事实"。

本例子：加一个 user_preferences 栏，存用户偏好，
通过中间件 wrap_model_call 在每次调模型前把偏好写进系统提示。

注意（1.0 的坑）：
- 自定义状态必须继承 AgentState，且必须是 TypedDict（dict 带类型注解），
  不能再是 Pydantic 模型或 dataclass。
- 中间件的 before_model 返回 dict 是"状态更新"，不是改 system_prompt；
  要让模型看到状态，用下面的 wrap_model_call 直接改模型请求。
- 文件后半部是方式二：只传 state_schema= 不配中间件 → 字段能跟踪但模型看不到，
  和方式一的差别在那边注释里说清。
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent  # AgentState: 本子的版式（自带 messages 栏）
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

load_dotenv(Path(__file__).parent.parent / ".env")  # 从项目根目录 .env 加载 API key


# =====================================================================
# ① 定义版式：继承 AgentState → 自带 messages（对话记录）栏，
#    再加一栏 user_preferences 存"需要一直记着、但不在对话里"的事实。
# =====================================================================
class CustomState(AgentState):
    user_preferences: dict


# =====================================================================
# ② 中间件：声明本子版式 + 每次模型开口前把偏好念给模型听。
#
#    wrap_model_call 钩子：包裹模型的每次调用。
#    在这里把 request 里的状态读出来，override 一个系统消息写进去，
#    模型就能"记着"这份偏好。request 是只读的，用 override 产生新请求。
# =====================================================================
class PrefMiddleware(AgentMiddleware):
    state_schema = CustomState

    def wrap_model_call(self, request, handler):
        prefs = request.state.get("user_preferences")
        if prefs:
            request = request.override(system_message=SystemMessage(
                content=f"用户偏好（务必遵守）：{prefs}。直接按偏好回答，不要再反问用户。"
            ))
        return handler(request)


model = init_chat_model("deepseek:deepseek-chat", temperature=0)
agent = create_agent(
    model=model,
    middleware=[PrefMiddleware()],  # 状态版式随中间件挂载到 agent
)

# invoke 时把偏好塞进本子（消息之外的自定义栏）
result = agent.invoke({
    "messages": [{"role": "user", "content": "推荐一个菜"}],
    "user_preferences": {"口味": "清淡", "忌口": "香菜"},
})

print(result["messages"][-1].content)
# 期望：模型按偏好推荐清蒸鲈鱼之类，而不是可乐鸡翅。


print("###############自定义状态2：与上面的功能不一样#######################")

# =====================================================================
# 方式二：只声明版式，不配中间件（state_schema= 快捷参数）
#
# 【功能】等价于中间件里的 `state_schema = CustomState2` 那半句——只是给本子加格子：
#   字段会被【跟踪】（收进状态、invoke 输出里有、配 checkpointer 可跨会话存档），
#   但模型节点只把 messages(+system_prompt) 喂给模型，user_preferences 躺在
#   request.state 里【读不到】→ 推荐菜不会按"热爱辣味、忌口香葱"来。
#
# 【场景】格子不只有"给模型看"一个用途，以下情况它够用：
#   ① 中间件通讯簿：一个中间件写状态，另一个中间件读，数据在非模型环节流动；
#   ② 旁路数据：计数、审计、flag 等不该进模型提示的字段，外层代码从结果里读；
#   ③ 跨会话持久：配合 checkpointer，自定义字段跟 messages 一起存档。
#   【用错场景】想让模型按偏好回答、却没中间件去注入 → 存了也白存，要用方式一。
# =====================================================================
class CustomState2(AgentState):
    user_preferences: dict

agent2 = create_agent(
    model,
    state_schema=CustomState2
)
# 能跟踪但模型看不见 → 结果不会体现"热爱辣味、忌口香葱"
result2 = agent2.invoke({
    "messages": [{"role": "user", "content": "推荐一个菜"}],
    "user_preferences": {"口味": "热爱辣味", "忌口": "香葱"},
})

print(result2["messages"][-1].content)