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

@tool
def informacion_Peru(consulta:str) -> str:
    """
    Contiene información sobre el Perú. Usalo para preguntas sobre PBI y  población.
    """
    # Aquí podrías implementar una consulta real a una base de datos o API
    informacion = {
        "pbi": "El PBI del Perú en 2025 fue de 340 mil millones de dólares.",
        "población": "La población del Perú en 2025 fue de aproximadamente 34 millones de habitantes."
    }
    consulta = consulta.lower()
    for clave, valor in informacion.items():
        if clave in consulta or any(palabra in consulta for palabra in clave.split()):
#        if clave in consulta:
            return valor
    return "No tengo información sobre esa consulta específica."


basic_agent = create_agent(
    model = llm,
    tools = [ 
                calcular ,
                obtener_informacion_Tecsup,
                informacion_Peru
            ],
    )

'''
result = basic_agent.invoke(
    {"messages": [("human", "¿Cuánto es 4827 * 3961?")]}
)
print(f"Respuesta: {result['messages'][-1].content}")

result = basic_agent.invoke(
    {"messages": [("human", "Dame información de TECSUP")]}
)
print(f"Respuesta: {result['messages'][-1].content}")
'''

result = basic_agent.invoke(
    {"messages": [("human", "Dame el valor del PBI del Perú")]}
)
print(f"Respuesta: {result['messages'][-1].content}")

result = basic_agent.invoke(
    {"messages": [("human", "Dame el valor de la poblacion del Perú")]}
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


# Ejercicio : Crear un funcion que me obtenga la fecha y hora del dia, 