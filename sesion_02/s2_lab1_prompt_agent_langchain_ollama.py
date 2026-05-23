from dotenv import load_dotenv
import os

load_dotenv()

model = os.getenv("OLLAMA_MODEL","llama3.1:8b")
base_url = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")

# Conexion con Ollama
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model=model,
    base_url=base_url,
    temperature=0.0,
)

# Creacion de un agente basico con Langchain Ollama

from langchain.agents import create_agent
from langchain_core.tools import tool

# Creacion de funciones para el agente

@tool
def obtener_informacion_Tecsup(query:str) -> str:
    """
    Contiene información sobre TECSUP.
    """
    # Aquí podrías implementar una consulta real a una base de datos o API
    return "TECSUP es una institución educativa peruana especializada en carrearas técnica y cursos especializados fundadda en 1984"


basic_agent = create_agent(
    model = llm,
    tools = [ 
                obtener_informacion_Tecsup,
            ],
    )


result = basic_agent.invoke(
    {"messages": [("human", "Dame información de TECSUP")]}
)
print(f"Respuesta: {result['messages'][-1].content}")


# Mostrar el historial de mensajes y llamadas a herramientas

for msg in result["messages"]:
    print(f"\n[{msg.type.upper()}]")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  → Tool call: {tc['name']}({tc['args']})")
    if hasattr(msg, 'content') and msg.content:
        print(f"  {msg.content[:200]}")
