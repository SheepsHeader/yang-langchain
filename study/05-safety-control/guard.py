"""
PII 中间件（PIIMiddleware）示例 —— 在发送给模型前对用户输入做脱敏
https://langchain-doc.cn/v1/python/langchain/agents.html#PII

三种策略：
    redact —— 把 PII 替换成占位符标签（如 <email>）
    mask   —— 部分遮盖（如 4532-****-****-9010）
    block  —— 检测到就抛异常，直接阻止本次请求

【本文件主要验证什么】
    三种策略各自生效：普通输入里的 email 被 redact、信用卡被 mask，
    含 API key 的输入触发 block 抛错（用 try/except 接住，看异常类型）。
"""
import sys
from pathlib import Path

import dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool

sys.stdout.reconfigure(encoding="utf-8")
dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")


@tool
def customer_service_tool(issue: str) -> str:
    """处理客户工单。"""
    return f"已记录工单：{issue}"


@tool
def email_tool(recipient: str, content: str) -> str:
    """给指定收件人发邮件。"""
    return f"已发送邮件给 {recipient}"


model = init_chat_model("deepseek:deepseek-chat", temperature=0)

agent = create_agent(
    model=model,
    tools=[customer_service_tool, email_tool],
    middleware=[
        # 在发送给模型之前，将用户输入中的电子邮件编辑掉
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # 遮盖用户输入中的信用卡
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # 阻止 API 密钥 - 如果检测到则抛出错误
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

# 场景一：正常输入，email 被 redact、信用卡被 mask，模型正常应答
result = agent.invoke({
    "messages": [{"role": "user", "content": "My email is john.doe@example.com and card is 4242-4242-4242-4242"}]
})
print("场景一（redact + mask）:")
print("模型实际收到的 user message:", result["messages"][0].content)
print("模型回答:", result["messages"][-1].content)

# 场景二：输入含 API key，block 策略抛错拦截
try:
    agent.invoke({
        "messages": [{"role": "user", "content": "My api key is sk-1234567890abcdefghijklmnopqrstuv"}]
    })
    print("场景二：未触发拦截（不符合预期）")
except Exception as e:
    print("\n场景二（block）:")
    print(f"抛出的异常类型: {type(e).__name__}: {e}")
