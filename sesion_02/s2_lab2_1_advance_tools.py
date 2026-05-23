from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import requests

from model import get_llm

@tool
def get_exchange_rate(query:str) -> str:
    """
        Busca información actualizada del tipo de cambio del dolar en el BCRP y
        lo devuelve en soles peruanos. Utiliza la API del BCRP para obtener el tipo de cambio más reciente.
    """
    fecha_base = "2026-05-21"
    url = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD04640PD/json/" + fecha_base 
    data = requests.get(url).json()
    return data["periods"][-1]["values"][0]

@tool
def python_repl(code: str) -> str:
    """Ejecuta código Python y retorna el resultado.
    Usar para cálculos matemáticos, conversiones, procesamiento de datos.
    El código debe asignar el resultado a una variable llamada 'resultado'."""
    try:
        local_vars = {}
        
        exec(code, {"__builtins__": __builtins__}, local_vars)

        if "resultado" in local_vars:
            return str(local_vars["resultado"])
        
        return "Código ejecutado. Define una variable 'resultado' para ver el output."
    
    except Exception as e:
        return f"Error al ejecutar código: {e}"



tools = [get_exchange_rate,python_repl]

llm = get_llm(name="ollama")

llm_with_tools = llm.bind_tools(tools)


# Definir el SYSTEM PROMPT para el agente

SYSTEM_PROMPT = """
    Eres un asistente experto en economia y finanzas. Responde en español y sé conciso. 
    
    ## HERRAMIENTAS DISPONIBLES:
    1.- Usa get_exchange_rate para informacion sobre el tipo de cambio del dolar en soles peruanos. 
    2. Usa python_repl para cálculos. Asigna el resultado a la variable 'resultado'.
    3. Puedes usar MÚLTIPLES herramientas en secuencia para resolver tareas complejas.
    4. Explica tu razonamiento paso a paso.

"""


# ----------------------------------------
# 1.- Definir los Nodos del grafo
# ----------------------------------------

# Reasoning Node
def reasoning(state: MessagesState) -> dict:
    """
        Nodo de razonamiento que decide qué herramienta usar en base a la consulta del usuario
    """
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    reponse = llm_with_tools.invoke(messages)

    return {"messages": [reponse]}

# Tool Node
tool_executer = ToolNode(tools)

# ----------------------------------------
# 2.- Definir la logica de decisión del grafo
# ----------------------------------------
def decision_logic(state: MessagesState) -> str:
    """
        Lógica de decisión que determina qué herramienta usar en base a la consulta del usuario
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# ----------------------------------------
# 3.- Construir el grafo
# ----------------------------------------

graph = StateGraph(MessagesState)

# Agregar nodos
graph.add_node("reasoning", reasoning)
graph.add_node("node_tools", tool_executer)

# Definir el inicio del grafo
graph.set_entry_point("reasoning")

# Agregar los Edges entre los nodos
graph.add_conditional_edges(
    "reasoning",  # Nodo de razonamiento
    decision_logic,  # Lógica de decisión
    {
        "tools": "node_tools",  # Si la lógica decide usar herramientas, ir al nodo "tools"
        "end" : END        # Si la lógica decide no usar herramientas, terminar el grafo
    }
)

graph.add_edge("node_tools", "reasoning")  # Después de ejecutar la herramienta, volver al nodo de razonamiento

# Compilar el grafo
agent = graph.compile()


if __name__ == "__main__":
    
    # query = "¿Cuál es el tipo de cambio del dolar hoy?"
    
    # query = "¿Cuál es la temperatura de hoy?"
    
    query = "Busca el precio actual del dolar y calcula cuánto valen 3.5 dolares en soles"
    
    '''
    response = get_exchange_rate(query)
    print(response)
    '''

    result =  agent.invoke({"messages": [("human", query)]})
    print(f"Respuesta: {result['messages'][-1].content}")
