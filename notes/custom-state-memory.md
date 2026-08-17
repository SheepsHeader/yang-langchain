# 自定义状态（Custom State）—— 文档「记忆」小节彻底搞懂

> 来源：https://langchain-doc.cn/v1/python/langchain/agents.html#记忆
> 环境：langchain `1.3.15` + `ai_agent` conda 环境 + DeepSeek，已实测验证
> 配套代码：`../study/custom_state.py`（由 `self_memory.py` 改名而来）
> 相邻参考：`../study/simple_agent.py`（context / checkpointer / response_format）、`../study/wrap_model.py`（动态模型选择）

---

## 📌 背景：文档这段为什么看得一头雾水

文档「记忆」小节的**代码是不完整的**：

```python
class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState

    def before_model(self, state, runtime) -> dict | None:
        ...   # ← 占位符！文档没写实现
```

它只演示了「怎么**声明**记忆格子」，没演示「怎么**读**格子里的内容」，所以看着迷糊是正常的。下面四问四答把缺的部分补全。

---

## 💬 你的提问 1：这些自定义实现记忆的知识点，能用老奶奶都能听懂的话解释一遍吗？

### 一句话点破

这个「记忆」小节讲的不是"记住上一轮聊了什么"（那是 `checkpointer` 的事），而是：**给智能体的"小本子"加格子。**

### 老奶奶版：服务员小王

你雇了个服务员小王，他手里有个小本子，工作流程是：**每说一句话就记进本子，然后照着本子上的内容说话。**

- **默认情况**：本子上只有一栏"对话记录"（`messages`），你说了啥、他答了啥，全记这一栏。模型能"记得"你五分钟前说的话，不是因为脑子好，而是**每次说话都把整本对话重新读一遍**。
- **问题**：有些事不适合记在对话栏。比如你进门第一句说"我不吃香菜"，过了十轮才说"来碗汤"——让小王翻十轮对话找"不吃香菜"，费劲还可能翻丢。
- **自定义状态**：给本子**加一栏**，比如"顾客偏好"栏，专门放"需要一直记着、但不属于某句话"的事实。
- **AgentState**：本子的**版式模板**。规定本子一定有一栏 `messages`，你可以自己加别的栏。
- **中间件（middleware）**：给小王配的小助手，能在**见顾客前**（调模型前）把偏好念给模型听，也能在**送客后**把新发现写回本子。

### 四样"跟记忆沾边"的东西，必须分清

| 东西 | 是什么 | 类比 | 会不会自己消失 |
|---|---|---|---|
| `messages` 对话历史 | 每轮聊天的记录 | 本子上的"对话栏" | 一直攒，越来越多 |
| `checkpointer` + `thread_id` | 把**整本本子**存档，下次同一 thread_id 再翻开 | 档案柜 | 不存档就没了 |
| `context`（`Context(user_id=...)`） | 每次调用**临时**递进去的便条 | 便条 | **用完就扔，不是记忆！** |
| **自定义状态** | 本子上**加一栏**，记"非对话事实" | 加栏 | 跟对话一起存（配合 checkpointer） |

**最容易混的是 `context` 和自定义状态**：长得像（都在 `invoke` 时传），但 `context` 不写进本子、不传给模型、跨轮次读不到；自定义状态是真的写进本子、贯穿整段对话。

### 两个坑（1.0 起）

1. 自定义状态**必须继承 `AgentState`，且是 `TypedDict`**，不能用 Pydantic / dataclass。
   （`simple_agent.py` 里 `Context` 用 `@dataclass` 是对的——那是 context 不是 state，不受此约束。）
2. 中间件 `before_model` 返回的 dict 是**状态更新**，不是改 system_prompt。实测 `return {"system_prompt": ...}` 无效（模型收到的还是 `None`）。

### 能跑的例子（已验证）

核心：中间件 `wrap_model_call` 钩子，每次调模型前把状态里的偏好写进系统提示。

```python
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

class CustomState(AgentState):        # ① 加栏：对话记录之外多一个"用户偏好"格
    user_preferences: dict

class PrefMiddleware(AgentMiddleware):  # ② 小助手：模型开口前把偏好念给模型听
    state_schema = CustomState

    def wrap_model_call(self, request, handler):
        prefs = request.state.get("user_preferences")
        if prefs:
            request = request.override(system_message=SystemMessage(
                content=f"用户偏好（务必遵守）：{prefs}。直接按偏好回答。"
            ))
        return handler(request)

agent = create_agent(model=init_chat_model("deepseek:deepseek-chat", temperature=0),
                     middleware=[PrefMiddleware()])

r = agent.invoke({
    "messages": [{"role": "user", "content": "推荐一个菜"}],
    "user_preferences": {"口味": "清淡", "忌口": "香菜"},   # invoke 时塞进本子
})
print(r["messages"][-1].content)
# → 根据您清淡口味且忌口香菜的偏好，推荐清蒸鲈鱼（不加香菜）
```

**去掉中间件**时模型会无视偏好（实测推荐可乐鸡翅还安利香菜）——这就是自定义状态的意义：让"非对话事实"**显式、可靠**地进入模型视野。

---

## 💬 你的提问 2：为什么案例中有用到 @wrap_model 注解，你的却没有？

文档「记忆」和「动态模型选择」**两处**都出现了这个钩子，写法不同是场景决定的：

| | 装饰器 `@wrap_model_call` | 类方法 `wrap_model_call` |
|---|---|---|
| 文档位置 | 「动态模型选择」小节 | 「记忆/自定义状态」小节 |
| 配套文件 | `wrap_model.py` | `custom_state.py` |
| 要干什么 | 只拦截模型调用、换模型 | 声明 `state_schema` + 注入状态 |
| 关键点 | 不需要自定义状态 | **必须把 `state_schema` 挂上去** |

**核心原因**：自定义状态的版式 `state_schema` 要跟钩子绑在同一个中间件对象上。装饰器写法（`@wrap_model_call`）只管"调模型前拦一下"，而「记忆」这节的主角是自定义状态，所以用类写法把 `state_schema` 和钩子绑一起。

> ⚠️ **后续修正**：源码里 `wrap_model_call` 其实支持 `state_schema` 关键字参数，
> 所以 `@wrap_model_call(state_schema=CustomState)` 也成立。
> 类写法是官方推荐样式（一个类可同时放多个钩子 + 工具 + 状态），不是唯一写法。

---

## 💬 你的提问 3：没有这个自定义节点，agent 其实也有一个 AgentState，自定义的只是在这个基础上增加新字段，是吗？

**对，理解完全正确。** 只是用词上"节点"应改为"状态版式 / State Schema"（节点是图里的概念，如 model 节点、工具节点）。

内置版式 `AgentState` 出厂自带三个字段（已查源码）：

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]            # 对话记录（必有）
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, ...]]      # 内部控制：下一步去哪个节点，临时/私有
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]    # 结构化输出结果（不进输入）
```

`class CustomState(AgentState): user_preferences: dict` 就是在**这三个字段之外加新字段**，加出来的字段和内置字段地位完全一样，记在同一个本子上。

类比：内置版式是出厂"标准小本子"（对话栏 + 两个内部栏目），自定义状态 = 加你自己的栏目。

---

## 💬 你的提问 4：这两个注解是会为方法生成一个同名的类吗？

**会，但关键是：装饰器返回的不是类，是那个类的实例。**

`@wrap_model_call` 的源码实现：

```python
def decorator(func):
    middleware_name = func.__name__              # 类名 = 函数名
    return type(                                 # ① 用 type() 动态造一个子类
        middleware_name,                         #    类名（如 "dynamic_model_selection"）
        (AgentMiddleware,),                      #    继承 AgentMiddleware
        {
            "state_schema": state_schema or AgentState,   # 类的属性
            "tools": tools or [],
            "wrap_model_call": wrapped,          #    你的函数被塞进类里当方法
        },
    )()                                          # ② 立刻实例化，返回的是实例
```

干了三件事：

1. 用 `type()` 动态造一个**同名子类**：函数 `dynamic_model_selection` → 类 `dynamic_model_selection`（继承 `AgentMiddleware`）。
2. 你的函数体被包进 `wrapped`，成为该类的 `wrap_model_call` 方法。
3. **实例化**这个类，把实例塞回原变量。

于是 `middleware=[dynamic_model_selection]` 里传的，是一个 **`AgentMiddleware` 子类的实例**——函数变量在装饰后就"变"成对象了。

### 类写法 = 同一件事的显式版

```python
class PrefMiddleware(AgentMiddleware):   # ← 显式写类
    state_schema = CustomState
    def wrap_model_call(self, request, handler): ...

middleware=[PrefMiddleware()]            # ← 显式实例化
```

装饰器把三步压缩成一行注解，类写法拆开写。**运行时形态完全一样：一个 `AgentMiddleware` 子类 + 它的实例。**

---

## 📋 结论速查表

| 概念 | 一句话 |
|---|---|
| 文档「记忆」小节实际讲什么 | 自定义状态：给 AgentState 加字段，不是 checkpointer 那种跨会话记忆 |
| AgentState | 内置状态版式：`messages`（必有）+ `jump_to` + `structured_response`，自定义在其上加字段 |
| 自定义状态要求 | 继承 AgentState、必须 TypedDict、配合中间件或 `state_schema` 参数 |
| 声明方式 | 类中间件（推荐，绑状态+钩子） 或 `state_schema=CustomState` 快捷方式 或 `@wrap_model_call(state_schema=...)` |
| 怎么让模型"看到"状态 | `wrap_model_call` 里 `request.override(system_message=...)`；`before_model` 返回 dict 只是状态更新 |
| 两个钩子写法 | 装饰器 `@wrap_model_call` 返回 **实例**；类方法 `wrap_model_call` 是显式写法，二者运行时等价 |
| 与 context 的区别 | context 每次 invoke 临时传入、不写进本子、不是记忆；自定义状态贯穿对话 |

## 相关文件

- `../study/custom_state.py` — 自定义状态跑通示例（原 `self_memory.py`，文档粘贴版是坏的，已替换）
- `../study/wrap_model.py` — `@wrap_model_call` 装饰器写法（动态换模型）；**装饰器原理已写进该文件注释**（type() 动态建类 → 返回实例）
- `../study/simple_agent.py` — context / checkpointer / response_format
