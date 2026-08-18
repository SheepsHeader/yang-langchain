# MySQL 持久化 Agent — 设计笔记

> 配套代码：`main.py`（终端循环对话，`!q` 退出）
> 学习方式：关键代码 + 小巧思（为什么这么写）
> 日期：2026-08-18

---

## 一、核心设计：两层持久化（都落 MySQL，重启不丢）

一个"重启后还记得"的助手，其实要解决**两件不同的事**：

| 层 | 存的是什么 | 作用 | 实现 |
|---|---|---|---|
| **checkpointer** | 每一轮的**对话历史**（messages） | 重启后模型还记得"我们刚聊了什么" | `PyMySQLSaver` |
| **store** | **长期记忆**（用户事实） | 跨对话/跨重启存活，像人物档案 | `PyMySQLStore` |

心智模型：**checkpointer = 对话备忘录，store = 人物档案**。备忘录记这次聊到哪；档案记这个人是谁。

关键代码——LangChain v1 的 `create_agent` 一次性把两层都接上：

```python
agent = create_agent(
    model=model,
    tools=[save_memory, recall_memories],
    system_prompt=SYSTEM_PROMPT,
    store=store,          # 长期记忆（人物档案）
    checkpointer=saver,   # 对话历史（备忘录）
    context_schema=Context,
)
```

---

## 二、关键代码逐段拆解

### 1. 凭据放 .env，建库用 `setup()` 前先保证库存在

```python
MYSQL_PASSWORD = os.environ["MYSQL_PASSWORD"]  # 从 .env 读，绝不硬编码
DB_URI = f"mysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

def ensure_database() -> None:
    conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT,
                           user=MYSQL_USER, password=MYSQL_PASSWORD)
    conn.cursor().execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB} CHARACTER SET utf8mb4")
    conn.commit(); conn.close()
```

- `PyMySQLStore.setup()` / `PyMySQLSaver.setup()` 只建**表**（幂等，已存在则跳过）
- 但连接串 `mysql://.../agent_memory` 里的**库**必须先存在，否则连不上
- 所以先用 root 裸连（不带库名），`CREATE DATABASE IF NOT EXISTS` 兜底

### 2. 连接生命周期：`with` 包住整个 REPL

```python
with (
    PyMySQLStore.from_conn_string(DB_URI) as store,
    PyMySQLSaver.from_conn_string(DB_URI) as saver,
):
    store.setup()
    saver.setup()
    ...  # 整个 while True 对话循环在这里
```

- `from_conn_string` 返回的是**上下文管理器**：连接由 `with` 托管
- 建表、对话、存记忆全都要连接活着 → 把整个循环放进 `with` 块
- 退出 `with`（含异常退出）自动关连接，不用手写 `close()`

### 3. 单用户续接的关键：固定 `thread_id`

```python
config = {"configurable": {"thread_id": USER_ID}}   # USER_ID = "my-user"
```

- checkpointer 按 `thread_id` 区分"这是哪段对话"
- 单用户就把 `thread_id` **写死成同一个** → 每次启动都是同一段对话的续写
- 想支持多用户，就把 `thread_id` 换成各用户的 id

### 4. 工具通过 `runtime` 拿 store 和上下文

```python
@dataclass
class Context:
    user_id: str

@tool
def save_memory(content: str, runtime: ToolRuntime[Context]) -> str:
    runtime.store.put(("memories", runtime.context.user_id), str(uuid.uuid4()), {"content": content})
    return "已记住。"
```

- `context_schema=Context` 声明工具能拿到的"这次是谁在调"
- 工具内 `runtime.store` = 同一个 store；`runtime.context` = 本次的用户身份
- `("memories", user_id)` 命名空间：第 0 层"记忆库"，第 1 层"哪个用户"

---

## 三、小巧思（设计时的取舍）

### ① 换持久化 = 换一行（接口的力量）
所有 demo（`put/get/search`、`runtime.store`、`store=` 参数）都走 **`BaseStore` 接口**。
`InMemoryStore()` → `PyMySQLStore.from_conn_string(...)` 只是换实现，调用方零改动。
**这就是为什么先把接口学明白，比背某个存储的 API 值钱。**

### ② `thread_id` 复用 `USER_ID`
单用户场景下，用户身份和对话线程是同一个东西 → 一个常量两用，不用多维护一个配置。

### ③ Windows 控制台是 GBK，emoji 会炸
模型回复里出现 `☕` → `UnicodeEncodeError: 'gbk' codec can't encode`。
```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```
`errors="replace"` 兜底：万一还有编码不了的字符，用 `?` 顶替而不是崩溃。

### ④ 建库用 utf8mb4
MySQL 8 默认 utf8mb4，但显式声明更稳——中文、emoji 都能存，避免默认字符集截断的坑。

### ⑤ 凭据放 .env，绝不硬编码（血的教训）
数据库密码和 API key 一样是**秘密**：提交到 git → push 到远程 → 永久泄漏。
- `.env` 已在 `.gitignore` 里（连 DeepSeek key 也是这么放的）
- 代码里 `os.environ["MYSQL_PASSWORD"]` 读取，用 `KeyError` 快速失败提醒你没配
- 提交前自查：`grep -rniE "MYSQL_PASSWORD|password" study/` 别把真密码带进版本库

### ⑥ 包 3.0.0 的三个坑（实测踩过）
1. `langgraph.store.mysql.__init__` 会**强拉全部三个驱动**（pymysql/aiomysql/asyncmy），缺一个 import 就炸 → 全装
2. `PyMySQLSaver` **不在顶层导出**，得从 `langgraph.checkpoint.mysql.pymysql` 导入
3. `PyMySQLStore` **没有 `index` 配置**（不像 `PostgresStore` 支持 pgvector）→ 没有语义 `search(query=...)`，`recall` 只能整批取出。想要语义搜索得换 PostgresStore

---

## 四、一句话总结

> **checkpointer 管"备忘录"（对话历史），store 管"人物档案"（长期记忆）；
> 固定 thread_id 让重启续接对话，with 托管连接生命周期；
> 换存储只换一行，因为一切走 BaseStore 接口。**
