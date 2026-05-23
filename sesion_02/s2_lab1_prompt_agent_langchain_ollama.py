from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from model import get_llm

load_dotenv()

# Conexion con Ollama
from langchain_ollama import ChatOllama

"""
model = os.getenv("OLLAMA_MODEL","llama3.1:8b")
base_url = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")

llm = ChatOllama(
    model=model,
    base_url=base_url,
    temperature=0.0,
)
"""

"""

model = os.getenv("OPENAI_MODEL","gpt-4o-mini")

llm = ChatOpenAI(
    model=model,
    temperature=0.0,
)
"""

llm = get_llm(name="openai")


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
    return "TECSUP (Tecnológico Superior) es una institución educativa peruana especializada en carrearas técnica y cursos especializados fundadda en 1984"

@tool
def informacion_carrera_software_Tecsup(query:str) -> str:
    """
    Contiene información sobre la carrera de Diseño y Desarrollo de Software en TECSUP.

    """
    # Aquí podrías implementar una consulta real a una base de datos o API
    info = """
        La carrera de Diseño y Desarrollo de Software en TECSUP tiene una duración de 3 años, 
        dividida en 6 semestres. Los lenguajes de programación que se enseñan incluyen Python, Java, JavaScript, C#, entre otros. 

    """


    return info

# SYSTEM PROMPT

SYSTEM_PROMPT = """
    Eres TecsupGPT un asistente útil usado para ayudar a las consultas relacionadas con TECSUP. 
    Responde con la mayor precisión posible.

    ## HERRAMIENTAS DISPONIBLES:
    
    1. obtener_informacion_Tecsup(query:str)
      Contiene información sobre TECSUP. Usalo para preguntas sobre historia, carreras, cursos, etc.

    2. informacion_carrera_software_Tecsup(query:str)
      Contiene información sobre la carrera de Diseño y Desarrollo de Software en TECSUP.
        Usalo para preguntas sobre el plan de estudios, duración, lenguajes de programación, etc.

    ## FORMATO DE RESPUESTA:
    Responder siempre en castellano

    ## RESTRICCIONES:
    - No inventes información. Si no sabes la respuesta, di que no lo sabes.

    """

basic_agent = create_agent(
    model = llm,
    tools = [ 
                obtener_informacion_Tecsup,
                informacion_carrera_software_Tecsup,
            ],
    system_prompt=SYSTEM_PROMPT,
    )


result = basic_agent.invoke(
    {"messages": [("human", "Dame información de la carrera de software de TECSUP")]}
)
print(f"Respuesta: {result['messages'][-1].content}")


# Mostrar el historial de mensajes y llamadas a herramientas

'''
for msg in result["messages"]:
    print(f"\n[{msg.type.upper()}]")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  → Tool call: {tc['name']}({tc['args']})")
    if hasattr(msg, 'content') and msg.content:
        print(f"  {msg.content[:200]}")
'''