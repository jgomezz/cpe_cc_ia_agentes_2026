"""
s3_lab2_3_agent_with_mcp  —  Agente que consulta un MCP Server para obtener información de estudiantes y normativas académicas
─────────────────────────────────────────────────────────
"""

import asyncio

from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


async def main():

    # Configuracion de la comunicacion al MCP Server por stdio (stdin/stdout)
    server_params = StdioServerParameters(
        command="python",
        args=["sesion_03/s3_lab2_1_mcp_server.py"],
    )

    # Conectar al MCP Server usando los parametros definidos
    async with stdio_client(server_params) as (read, write):
        # Crea una sesion del cliente MCP
        async with ClientSession(read, write) as session:
            
            await session.initialize()

            # Cargar y listar las tools disponibles en el MCP Server
            tools = await load_mcp_tools(session)
            for t in tools:
                print(f"   • {t.name}")




if __name__ == "__main__":
    asyncio.run(main())