"""单用户 MySQL 持久化 Agent 小助手（研究目录 07-memory 的 mysql 示例包）。

持久化原理（LangGraph 长期记忆的两层，都落 MySQL，进程重启不丢）：
- checkpointer（PyMySQLSaver）→ 对话历史跨进程续接：重启后同一 thread_id 继续聊，模型记得之前聊过什么
- store（PyMySQLStore）→ 长期记忆（用户事实）：agent 通过工具读写，跨对话/重启存活

运行：python study/07-memory/mysql/main.py（在项目根目录执行，保证 dotenv 找到 .env）
退出：对话中输入 !q
"""
