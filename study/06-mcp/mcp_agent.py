# 访问多个 MCP 服务器（stdio + streamable_http）
import asyncio
import os
import sys

import dotenv

sys.stdout.reconfigure(encoding="utf-8")
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

model = ChatDeepSeek(model="deepseek-v4-flash")

# stdio_mcp.py 是本目录下的本地工具服务器（加减乘）
math_server = os.path.join(os.path.dirname(__file__), "stdio_mcp.py")


async def main():
    # 注意：langchain-mcp-adapters 0.1.0+ 已移除 async with 用法，
    # 直接 new 出来调 get_tools() 即可（每个工具调用会自己建会话）
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",  # 本地子进程通信
                "command": sys.executable,
                "args": [math_server],
            },
            "weather": {
                "transport": "streamable_http",  # 基于 HTTP 的远程服务器
                # 先跑 python http_mcp.py 在 8000 端口把天气服务器拉起来
                "url": "http://localhost:8000/mcp",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_agent(model, tools)
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
    )
    print(math_response)
    print(weather_response)


if __name__ == "__main__":
    asyncio.run(main())
