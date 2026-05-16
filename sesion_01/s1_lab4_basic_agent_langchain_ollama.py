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
def calcular(expresion:str) -> str:
    """
    Calcula el resultado de una expresión matemática.
    """
    try:
        resultado = eval(expresion)
        return str(resultado)
    except Exception as e:
        return f"Error al calcular: {str(e)}"


basic_agent = create_agent(
    model = llm,
    tools = [calcular],
    )

result = basic_agent.invoke(
    {"messages": [("human", "¿Cuánto es 4827 * 3961?")]}
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