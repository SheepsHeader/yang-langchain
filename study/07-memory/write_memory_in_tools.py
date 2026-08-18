from dataclasses import dataclass

import dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

dotenv.load_dotenv()


# InMemoryStore 将数据保存到内存字典中。在生产环境中使用基于数据库的存储。
store = InMemoryStore() # [!code highlight]

@dataclass
class Context:
    user_id: str

# TypedDict 定义了供 LLM 使用的用户信息结构
class UserInfo(TypedDict):
    name: str

# 允许智能体更新用户信息的工具（适用于聊天应用）
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    # 访问 store - 与提供给 `create_agent` 的 store 相同
    store = runtime.store # [!code highlight]
    user_id = runtime.context.user_id # [!code highlight]
    # 在 store 中存储数据 (namespace, key, data)
    store.put(("users",), user_id, user_info) # [!code highlight]
    return "Successfully saved user info."

# context 是每次 invoke 时动态传入的"运行期元数据"，和 store 职责不同：
#   - store：长期记忆，跨对话/线程存活，存数据本身
#   - context：一次调用的上下文身份，用完即弃，存"这次是谁在调、该用哪份数据"
# context_schema 声明工具能拿到的上下文结构；invoke 时传实际值；
# 工具内通过 runtime.context 取用。它只进工具，不会拼进发给 LLM 的对话消息。
# 典型用途：user_id、tenant_id、鉴权信息、request_id 等"这次调用属于谁"的元数据。
# 想跨调用保留的东西放 store；想让模型在对话里看到的信息直接写进 messages。
agent = create_agent(
    model="deepseek-chat",
    tools=[save_user_info],
    store=store, # [!code highlight]
    context_schema=Context # 声明运行时上下文的结构：工具能拿到 user_id
)

# 运行智能体
response = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    # 在 context 中传入 user_id 以识别正在更新谁的信息
    context=Context(user_id="user_123") # [!code highlight] 本次调用由 user_123 发起，工具据此把名字存到他的抽屉
)
print(response)

# 您可以直接访问 store 来获取该值
result = store.get(("users",), "user_123").value
print(result)