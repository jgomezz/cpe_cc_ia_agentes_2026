"""
s3_lab2_3_agent_with_mcp  —  Agente que consulta un MCP Server para obtener información de estudiantes y normativas académicas
─────────────────────────────────────────────────────────
"""

import asyncio

from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from model import get_llm
from langchain.agents import create_agent

SYSTEM_PROMPT = """
        Eres un asistente académico que ayuda a estudiantes con información sobre sus carreras, normativas y procedimientos. 
        Usa las tools disponibles para consultar datos de estudiantes y buscar normativas relevantes. 
        Responde de manera clara y precisa, guiando al estudiante en sus consultas académicas.
        """          


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

            # Crear el agente con las tools MCP
            llm = get_llm("ollama")

            agent = create_agent(
                            llm,
                            tools=tools,
                            system_prompt=SYSTEM_PROMPT)
                                           

            result = await agent.ainvoke(
                {"messages": [("human", "Necesito los datos del estudiante T20231.")]},
                {"recursion_limit": 15},
            )

            print(f"🤖 {result['messages'][-1].content}")


if __name__ == "__main__":
    
    # Ejecutar la función principal asincrónica
    asyncio.run(main())