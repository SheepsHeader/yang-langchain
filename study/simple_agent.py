from dataclasses import dataclass  # @dataclass：根据带类型注解的字段自动生成 __init__/__repr__/__eq__
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent  # 高阶 API：一行创建"循环调用工具直到停止"的 agent 图
from langchain.chat_models import init_chat_model  # 按 "provider:model" 字符串初始化聊天模型
from langchain.tools import tool, ToolRuntime  # @tool: 函数→BaseTool；ToolRuntime: 工具内自动注入的运行时对象
from langgraph.checkpoint.memory import InMemorySaver  # 记忆 checkpointer：跨轮次持久化对话状态（内存版）


load_dotenv(Path(__file__).parent.parent / ".env")  # 从项目根目录 .env 加载 API key 等环境变量

# 定义系统提示
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。

你可以使用两个工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""

# =====================================================================
# @dataclass —— 装饰器（改变类的行为）
#   作用：根据类体中带类型注解的字段，自动生成 __init__ / __repr__ / __eq__ 等方法。
#   在此的意义：create_agent 需要从数据类推断 schema（context_schema 参数），
#              普通类它拿不到字段结构。类体里的字段注解同时作为 schema 定义。
# =====================================================================
@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str  # 字段注解：必填 str。运行时每次 invoke 临时传入，不写入对话记忆

# =====================================================================
# @tool —— 装饰器（改变函数的行为）
#   作用：把普通 Python 函数包装成 BaseTool，使它能被 LLM 调用：
#       1. 从函数签名（参数名 + 类型注解）自动推断 JSON schema，模型按此填参数
#       2. 从 docstring 生成工具描述，模型据此判断"该用哪个工具"
#       3. 调用时自动校验参数、序列化结果
#   注意：@tool 要求函数必须有类型注解，否则 schema 推断会失败。
# =====================================================================
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阴雨连绵！"

# =====================================================================
# @tool + 泛型注解 ToolRuntime[Context] —— 本文件最特殊的一处
#
#   runtime: ToolRuntime[Context] 有两层含义：
#   ① 触发运行时注入：工具参数只要名为 runtime、类型注解为 ToolRuntime，
#      LangGraph 执行工具时就会自动注入运行时对象，无需传参（模型也不会填它）。
#   ② [Context] 是泛型参数：标注 runtime.context 是 Context 类型实例，
#      所以能直接 .user_id 而不报类型错误；与 create_agent 的 context_schema=Context 呼应。
#
#   ToolRuntime 暴露的属性：state(图状态)、context(本次运行的上下文)、
#   tool_call_id、config、store(持久化存储)、stream_writer、tools(所有可用工具)。
# =====================================================================
@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id  # runtime 由框架注入，context 来自 invoke 时传入的 Context(user_id="1")
    return "Florida" if user_id == "1" else "SF"

# 配置模型
model = init_chat_model(
    "deepseek:deepseek-chat",  # "provider:model" 格式；temperature=0 降低随机性
    temperature=0
)

# =====================================================================
# @dataclass —— 作为 response_format 输出 schema
#   模型最终回答必须符合此结构，解析结果在 response['structured_response']。
#   - punny_response: str           —— 必填字段，模型必须给出
#   - weather_conditions: str | None —— 联合类型注解（PEP 604，等价 Optional[str]），
#                                       可选字段，模型可填可空（None）
# =====================================================================
@dataclass
class ResponseFormat:
    """代理的响应模式。"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None

# 设置记忆
checkpointer = InMemorySaver()  # 跨 invoke 保存对话状态；配合 config 里的 thread_id 区分会话

# 创建代理
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,         # ← 声明 Context 数据类为运行时上下文 schema
    response_format=ResponseFormat, # ← 声明 ResponseFormat 为结构化输出 schema
    checkpointer=checkpointer
)

# 运行代理
# `thread_id` 是给定对话的唯一标识符（相同 thread_id = 同一段记忆/会话）。
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")  # context 是本次运行的临时数据，非对话记忆
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="佛罗里达今天依然是'阳光灿烂'的一天！阳光正在播放'rey-dio'热门歌曲！我得说，这是进行'solar-bration'的完美天气！如果你希望下雨，恐怕这个想法已经'被冲走'了——预报仍然'清晰地'灿烂！",
#     weather_conditions="佛罗里达总是阳光明媚！"
# )

# 注意，我们可以使用相同的 `thread_id` 继续对话。
response = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢！"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="你真是'雷'厉风行地欢迎！帮助你保持'当前'天气总是'轻而易举'。我只是'云'游四方，等待随时'淋浴'你更多预报。祝你在佛罗里达的阳光下度过'sun-sational'的一天！",
#     weather_conditions=None
# )
