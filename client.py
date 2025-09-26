import asyncio
from typing import Any, Dict, List, Optional
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import  stdio_client
from contextlib import AsyncExitStack

class EmailMCPClient:
    def __init__(self):

        self.session : Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None



    async def connect(self,server_path: str="email_server.py"):
        server_params = StdioServerParameters(
            command="python",
            args=[server_path]
        )
        #connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio,self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write))
        #initialize the connection
        await self.session.initialize()


    async def list_tools(self):
        if not self.session:
            raise RuntimeError("Client not connected. Call connect() first.")
        tools = await self.session.list_tools()
        return tools.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        if not self.session:
            raise RuntimeError("Client not connected. Call connect() first.")
        result = await self.session.call_tool(tool_name, arguments)
        return result.content[0].text if result.content else None



async def main():
    client = EmailMCPClient()
    try:
        print("connecting")
        await client.connect("email_server.py")
        print("connected")
        tools = await client.list_tools()
        print("Available tools:", [tool.name for tool in tools])
        result = await client.call_tool("search_emails", {"query": "test", "max_results": 1})
        print("Tool call result:", result)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            await client.exit_stack.aclose()
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
