"""
结构化输出（Structured Output）示例 —— 对比两种让模型按固定 schema 输出的策略
https://langchain-doc.cn/v1/python/langchain/agents.html#结构化输出

核心思想：模型默认输出自由文本，但很多场景要求它必须返回符合某个数据结构的 JSON
（比如"提取联系信息，必须给 name/email/phone 三个字段"）。create_agent 的
response_format 参数支持两种策略实现这个目标：
    - ToolStrategy：把输出 schema 伪装成一个"工具"，让模型通过调用工具来产出结构化结果
    - ProviderStrategy：用模型 provider 原生的 response_format 强制模型直接输出 JSON
      （OpenAI 侧的 json_schema 模式）

【本文件主要验证什么】
    同一份 Pydantic schema，ToolStrategy 和 ProviderStrategy 是否都能产出结构化结果，
    结果统一放在 result["structured_response"] 里，两个结果对比着看。

【怎么验证，以及一个关键坑】
    ① 跑起来看输出：先打印 ToolStrategy 结果（能通），再打印 ProviderStrategy 结果。
    ② 关键坑：ProviderStrategy 走 provider 原生的 json_schema response_format，
       DeepSeek 目前只支持 text 和 json_object，不支持 json_schema，
       所以 agent2.invoke 会直接 400 "This response_format type is unavailable now"。
       ToolStrategy 靠工具调用约束输出、不依赖 response_format，DeepSeek 下能正常跑。
       这就是 DeepSeek 上结构化输出要用 ToolStrategy 的原因。
"""
import dotenv
from pydantic import BaseModel
from langchain.agents import create_agent  # 高阶 API：一行创建"循环调用工具直到停止"的 agent 图
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy  # 两种结构化输出策略（本文件对比核心）


dotenv.load_dotenv()  # 从项目根目录 .env 加载 API key 等环境变量

# ============================== 输出 schema ==============================
# Pydantic 模型同时充当两种策略的 schema 来源：字段名 + 类型就是模型必须遵循的结构，
# 两种策略都从这里推断输出形状，解析结果再还原成 ContactInfo 实例。
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str
# =========================================================================

# ============================== 策略一：ToolStrategy（能跑通） ==============================
# 把 ContactInfo 包装成一个"结构化输出工具"，模型通过调用工具返回 JSON，再解析成 Pydantic。
# 走工具调用链路，不依赖 provider 的 response_format —— 任何支持工具调用的模型都能用。
agent1 = create_agent(
    model="deepseek-chat",
    response_format=ToolStrategy(ContactInfo)  # ← 指定策略：schema → 工具
)

# 运行 agent1：结构化结果取 result["structured_response"]
result1 = agent1.invoke({
    "messages": [{"role": "user", "content": "从以下内容提取联系信息：John Doe, john@example.com, (555) 123-4567"}]
})

print("使用ToolStrategy输出的结构化结果如下：")
print(result1["structured_response"])
# ================================================================================

# ============================== 策略二：ProviderStrategy（DeepSeek 下会报错） ==============================
# 走 provider 原生结构化输出：请求时带 response_format=json_schema 强制模型直接输出 JSON。
# OpenAI 支持，但 DeepSeek 目前不支持 json_schema 类型 → 调用时会 400（见头部"关键坑"）。
agent2 = create_agent(
    model="deepseek-chat",
    response_format=ProviderStrategy(ContactInfo)  # ← 指定策略：schema → provider 原生 json_schema
)

result2 = agent2.invoke({  # 关键坑：DeepSeek 下会直接 400，脚本在此中断
    "messages": [{"role": "user", "content": "从以下内容提取联系信息：John Doe, john@example.com, (555) 123-4567"}]
})

print("使用ProviderStrategy输出的结构化结果如下：")
print(result2["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
# ===========================================================================================
