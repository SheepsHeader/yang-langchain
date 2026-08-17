"""
模型动态选择（wrap_model_call）示例 —— 验证动态选择模型是否生效
https://langchain-doc.cn/v1/python/langchain/agents.html#模型动态选择

核心思想：不用 create_agent 写死一个模型，而是【每次调用模型前】先看一眼当前对话状态，
根据复杂度（这里看消息条数）决定本次请求用哪个模型 —— 对话短用便宜快的，长对话换更强的。

【本文件主要验证什么】
    动态模型选择到底有没有生效 —— 即中间件是否真的按消息条数切换了模型。

【怎么验证，以及一个关键坑】
    ① 最可靠：在 @wrap_model_call 中间件里 print 选中的 model.model_name，
       每次调用都会打一行日志，直接看到「这个请求走了哪个模型」。
    ② 辅助：看 response["messages"][-1].response_metadata["model_name"]。
       但注意 —— 若两个模型名在服务端是同一别名的不同叫法，这个字段会一样，看不出区别。
       实测 deepseek-chat 当前被服务端别名解析成 deepseek-v4-flash，
       所以之前拿 "deepseek-chat" vs "deepseek-v4-flash" 当模型对时验证不了。
       现在改用 v4-flash vs v4-pro，两者真实不同，response 里也能对上号。
"""
import sys
import dotenv

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，模型可能回 emoji，改成 UTF-8
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

dotenv.load_dotenv()

# 两个候选模型：basic 便宜、advanced 更强。注意别用服务端互为别名的两个名字当对比
#（如 deepseek-chat 当前就等价于 deepseek-v4-flash），否则看不出切换。
basic_model = ChatDeepSeek(model="deepseek-v4-flash")
advanced_model = ChatDeepSeek(model="deepseek-v4-pro")

# ============================== 动态选择核心 ==============================
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂性选择模型。"""
    # @wrap_model_call：把函数升级成中间件，LangChain 每次【调用模型前】都会先执行它，
    # 然后调用 handler 继续后面的调用链 —— "换模型"就在这一步拦截完成。
    #
    # request：本次请求信息。request.state["messages"] 是当前对话的消息列表，
    # 中间件就靠它拿到"这次对话有多长"，据此决定用哪个模型。

    # 验证点①：消息条数 > 5 走 advanced，否则走 basic
    message_count = len(request.state["messages"])

    model = advanced_model if message_count > 5 else basic_model
    print(f"[middleware] message_count={message_count} -> 选择 model={model.model_name}")
    # 验证点②：直接打印选中的模型名 —— 这是判断动态选择是否生效的最直接依据。

    # request.override(model=model)：返回一个替换了 model 的【新】request（原对象不变，
    # 不可变风格）。注意不要写成 request.model = model —— 那是废弃写法，会告警。
    return handler(request.override(model=model))
# ============================================================================

agent = create_agent(
    model=basic_model,  # 默认模型（中间件不介入时兜底用）
    middleware=[dynamic_model_selection]
)

# 短对话：只有 1 条 + 追加的 1 条提问 = 2 条，期望走 basic_model
messages1 = [
      {"role": "user", "content": f"只有1条消息"}
  ]

# 长对话：8 条 + 追加的 1 条提问 = 9 条，期望走 advanced_model
messages2 = [
      {"role": "user", "content": f"第 {i} 条消息"}
      for i in range(8)
  ]

# 追加一条"自报家门"的提问，方便对照：中间件日志说走了哪个模型，
# response 里的 model_name 是否和它一致（验证点②）。
messages1.append({"role": "user", "content": f"你是什么版本的模型？你的名称是？比如deepseek-chat、deepseek-v4之类的"})
messages2.append({"role": "user", "content": f"你是什么版本的模型？你的名称是？比如deepseek-chat、deepseek-v4之类的"})

response1 = agent.invoke(
    {"messages": messages1}
)

response2 = agent.invoke(
    {"messages": messages2}
)

# 预期输出对照（v4-flash vs v4-pro 服务端可区分）：
#   [middleware] message_count=2 -> 选择 model=deepseek-v4-flash   → response1 model_name 也是 v4-flash
#   [middleware] message_count=9 -> 选择 model=deepseek-v4-pro     → response2 model_name 也是 v4-pro
print(response1)
print(response2)
