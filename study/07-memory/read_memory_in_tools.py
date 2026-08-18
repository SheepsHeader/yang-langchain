from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_deepseek import ChatDeepSeek
from langgraph.store.memory import InMemoryStore
import dotenv

dotenv.load_dotenv()

@dataclass
class Context:
    user_id: str

# InMemoryStore 将数据保存到内存字典中。在生产环境中使用基于数据库的存储。
store = InMemoryStore() # [!code highlight]

# 使用 put 方法向 store 写入示例数据
store.put( # [!code highlight]
    ("users",),  # 用于将相关数据分组的命名空间（用于用户数据的 users 命名空间）
    "user_123",  # 命名空间内的 Key（用户 ID 作为 Key）
    {
        "name": "John Smith",
        "language": "English",
    }  # 为给定用户存储的数据
)

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info."""
    # 访问 store - 与提供给 `create_agent` 的 store 相同
    store = runtime.store # [!code highlight]
    user_id = runtime.context.user_id
    # 从 store 检索数据 - 返回带有 value 和 metadata 的 StoreValue 对象
    user_info = store.get(("users",), user_id) # [!code highlight]
    return str(user_info.value) if user_info else "Unknown user"

model = ChatDeepSeek(model="deepseek-v4-flash")

# context 是每次 invoke 时动态传入的"运行期元数据"，和 store 职责不同：
#   - store：长期记忆，跨对话/线程存活，存数据本身
#   - context：一次调用的上下文身份，用完即弃，存"这次是谁在调、该用哪份数据"
# context_schema 声明工具能拿到的上下文结构；invoke 时传实际值；
# 工具内通过 runtime.context 取用。它只进工具，不会拼进发给 LLM 的对话消息。
# 典型用途：user_id、tenant_id、鉴权信息、request_id 等"这次调用属于谁"的元数据。
# 想跨调用保留的东西放 store；想让模型在对话里看到的信息直接写进 messages。
agent = create_agent(
    model=model,
    tools=[get_user_info],
    # 将 store 传递给智能体 - 使智能体能够在运行工具时访问 store
    store=store, # [!code highlight]
    context_schema=Context # 声明运行时上下文的结构：工具能拿到 user_id
)

# 运行智能体并打印结果
result = agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_123") # [!code highlight] 本次调用由 user_123 发起，工具据此找/存他的数据
)
print(result)
