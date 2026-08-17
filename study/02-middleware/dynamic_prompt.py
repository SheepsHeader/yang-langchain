"""
动态系统提示（Dynamic System Prompt）示例
https://langchain-doc.cn/v1/python/langchain/agents.html#动态系统提示

核心思想：不用 system_prompt 写死开场白，而是在【每次调用模型前】现算一句，
根据这次请求注入的上下文生成不同的系统提示 —— 这就是「动态提示词注入」。
"""
from typing import TypedDict                  # 声明上下文形状用
from langchain.agents import create_agent     # 创建智能体
from langchain.agents.middleware import dynamic_prompt, ModelRequest  # 动态提示词核心
from langchain.tools import tool              # 装饰器：函数变工具
import dotenv

dotenv.load_dotenv()                          # 加载 .env 里的 API key


# 运行时上下文的"形状"：context 参数里只允许 user_role 一个字段
class Context(TypedDict):
    user_role: str


# ============================== 动态提示词注入核心 ==============================
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示。"""
    # @dynamic_prompt：把函数升级成中间件，LangChain 每次【调用模型前】都会先执行它，
    # 并把返回值当作本轮的 System Prompt 注入给模型 —— "注入"就指这一步。
    #
    # request：本次请求信息。request.runtime.context 就是 invoke 时传的 context 字典
    #（对话内容走 messages，context 是走旁路塞进来的、供提示词读取的额外信息）。

    # 读取注入进来的 user_role；没传默认 "user"（安全取值，不会报错）
    user_role = request.runtime.context.get("user_role", "user")

    base_prompt = "你是一个有帮助的助手。"

    # 动态的精髓：context 一变，拼出来的提示词就变 → 模型口吻跟着变
    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术响应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单解释概念，避免使用行话。"
    return base_prompt
# ================================================================================


@tool
def web_search(query: str) -> str:
    """这是联网搜索关键字查询相关信息的方法"""
    # 工具函数：docstring 会被当作工具说明喂给模型，让它知道何时调用。
    # 此处为演示写死返回值，真正的联网需另行实现。
    return """
    机器学习（Machine Learning）是一种让计算机"自己学会做事"的技术，而不是靠人一条条写死规则。

    它的核心思路很简单：给你一堆例子，让机器从例子里自己找出规律，然后用这个规律去处理没见过的
    新数据。

    打个比方：你不用教孩子"有毛、会叫、有四条腿的就是狗"，只要带他看过几只狗和猫，他自然就分得
    清了。机器学习就是这个道理——你喂给程序大量"狗和猫的照片"，程序自己总结出区别，下次见到新照
    片它就能判断。

    它和传统编程最大的区别在于：传统编程是"人写规则，电脑执行"；机器学习是"人给数据，电脑自己
    总结规则"。

    如今机器学习的应用无处不在：手机里的语音识别、刷脸支付、购物推荐、天气预报、自动驾驶，背后
    都是它。深度学习是机器学习的一个分支，近几年靠着大数据和强大的计算芯片，把机器的"自学能力"
    推到了以前不敢想的高度，也成就了今天的 ChatGPT 和各类智能体（Agent）。
    """;


# 组装智能体 —— 两个参数是动态注入的关键：
#   middleware=[user_role_prompt]  把动态提示词函数挂进中间件链（不写死 system_prompt）
#   context_schema=Context         声明上下文形状，校验/提示 invoke 时传的 context
agent = create_agent(
    model="deepseek-chat",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# 注入不同的 user_role → 动态提示词走不同分支 → 同一问题得到不同风格的回答
# 完整链路：invoke(context={"user_role": ...})
#   → context 装进 request.runtime.context
#   → 调用模型前执行 user_role_prompt → 现算系统提示
#   → 提示词注入给模型 → 按对应口吻回答
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "解释机器学习"}]},
    context={"user_role": "expert"}       # 注入点
)

result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "解释机器学习"}]},
    context={"user_role": "beginner"}     # 注入点
)

print("如果你是专家的话，我会这么回答：")
print(result1["messages"][-1].content)

print("如果你是一个初学者，我会告诉你:")
print(result2["messages"][-1].content)
