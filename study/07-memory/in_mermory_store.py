from langgraph.store.memory import InMemoryStore


def embed(texts: list[str]) -> list[list[float]]:
    # 把文字变成一串数字"指纹"，供语义搜索算相似度。
    # 这里是占位实现，随便造数字；真实项目替换为嵌入模型，
    # 如 langchain 的 init_embeddings("openai:text-embedding-3-small")。
    return [[1.0, 2.0] * len(texts)]


# InMemoryStore：LangGraph 的"长期记忆"存储，跨对话/线程共享，而非仅上一轮。
# 数据存在内存字典里，程序退出即丢；生产环境用基于数据库的存储（如 PostgresStore / Redis）。
# index 配置是开启语义搜索的关键 —— 不配置时 search 的 query 参数无效。
store = InMemoryStore(index={"embed": embed, "dims": 2})  # dims: 每条文本转成几个数字

user_id = "my-user"
application_context = "chitchat"
# namespace = 抽屉便签，分层分类记忆：第 0 层是谁的（用户），第 1 层是哪个场景（闲聊）。
namespace = (user_id, application_context)

# 把一条记忆放进 namespace 抽屉，名为 "a-memory"，值为任意 JSON 结构。
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "topic": "language",  # 自设计标签字段，供 search 的 filter 精确匹配用
    },
)
# 按名字精确取回某条记忆（get 靠 namespace + key，不做语义匹配）。
item = store.get(namespace, "a-memory")
# 语义搜索：先按 filter 精确筛选 value 内 topic 字段 = "language" 的记忆，
# 再用 query 的语义（向量相似度）对剩下的结果排序。
items = store.search(
    namespace, filter={"topic": "language"}, query="language preferences"
)
print(items)
