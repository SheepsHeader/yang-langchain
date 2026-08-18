"""单用户 MySQL 持久化 Agent 小助手（终端循环对话）。

玩法：聊几轮 → 退出（!q）→ 重启再聊 → 观察：
  1. 对话历史续接（checkpointer：重启后模型还记得之前聊过什么）
  2. 长期记忆存活（store：启动时打印已有记忆条数，问答时 agent 能 recall）

依赖：pip install langgraph-checkpoint-mysql[pymysql] aiomysql asyncmy（已装进 ai_agent env）
模型：DeepSeek（项目根 .env 里的 DEEPSEEK_API_KEY）
"""
import os
import sys
import uuid
from dataclasses import dataclass

import dotenv
import pymysql
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.store.mysql import PyMySQLStore

dotenv.load_dotenv()

# Windows 控制台默认 GBK 编码，模型回复里的 emoji/特殊字符会炸，统一改 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8")

# --- MySQL 连接（凭据从 .env 读取，不进版本库） ---
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ["MYSQL_PASSWORD"]
MYSQL_DB = os.environ.get("MYSQL_DB", "agent_memory")
DB_URI = f"mysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
USER_ID = "my-user"  # 单用户，写死；对话历史线程名也用同一个


def ensure_database() -> None:
    """建库（幂等）。PyMySQLStore.setup() 只能建表，库得先存在。"""
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, connect_timeout=5
    )
    conn.cursor().execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB} CHARACTER SET utf8mb4")
    conn.commit()
    conn.close()


# --- 运行期上下文：工具能知道"这次是哪个用户" ---
@dataclass
class Context:
    user_id: str


# --- 长期记忆工具（读写 store，数据落 MySQL 的 store 表） ---
@tool
def save_memory(content: str, runtime: ToolRuntime[Context]) -> str:
    """把用户透露的个人信息、偏好等值得长期记住的事实，存进长期记忆。"""
    uid = runtime.context.user_id
    runtime.store.put(("memories", uid), str(uuid.uuid4()), {"content": content})
    return "已记住。"


@tool
def recall_memories(runtime: ToolRuntime[Context]) -> str:
    """取出这个用户存过的所有长期记忆，供回答问题时参考。"""
    uid = runtime.context.user_id
    items = runtime.store.search(("memories", uid))
    if not items:
        return "（暂无长期记忆）"
    return "\n".join(f"- {i.value['content']}" for i in items)


SYSTEM_PROMPT = """你是用户的贴心助手，拥有持久化的长期记忆：
- 用户透露个人信息/偏好时，用 save_memory 存下来；
- 回答涉及用户过往信息（名字、喜好、说过的事等）时，先调 recall_memories。
对话历史也会被完整保留，所以用户之前聊过什么你都能直接记得。"""


def main() -> None:
    ensure_database()

    with (
        PyMySQLStore.from_conn_string(DB_URI) as store,
        PyMySQLSaver.from_conn_string(DB_URI) as saver,
    ):
        # 自行建表（幂等，已存在则跳过）
        store.setup()
        saver.setup()

        model = ChatDeepSeek(model="deepseek-chat")
        agent = create_agent(
            model=model,
            tools=[save_memory, recall_memories],
            system_prompt=SYSTEM_PROMPT,
            store=store,          # 长期记忆
            checkpointer=saver,   # 对话历史落盘
            context_schema=Context,
        )

        # 单用户固定 thread_id：重启后用同一个线程续接对话
        config = {"configurable": {"thread_id": USER_ID}}

        existing = store.search(("memories", USER_ID))
        print(f"== 长期记忆已有 {len(existing)} 条 ==")
        print("== 输入 !q 退出；聊几轮后重启，看历史/记忆是否还在 ==")

        while True:
            try:
                user_input = input("\n你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input == "!q":
                break
            if not user_input:
                continue

            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                context=Context(user_id=USER_ID),
            )
            reply = result["messages"][-1].content
            print("助手 >", reply)


if __name__ == "__main__":
    main()
