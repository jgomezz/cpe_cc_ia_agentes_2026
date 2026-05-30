from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import requests

from model import get_llm

from datetime import date, timedelta

@tool
def get_exchange_rate() -> str:
    """        
    Busca información actualizada del tipo de cambio del dolar en el BCRP y
    lo devuelve en soles peruanos. Utiliza la API del BCRP para obtener el tipo de cambio más reciente.
    """
    try:

        # Pedimos los últimos 7 días para asegurar al menos un valor
        # (BCRP no publica fines de semana ni feriados)
        
        hoy = date.today()
        fecha_base = hoy - timedelta(days=7)
        
        fmt_fecha_base = f"{fecha_base.isoformat()}/{hoy.isoformat()}"
        print(f"Fecha base -> {fmt_fecha_base}")

        url = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD04640PD/json/" + fmt_fecha_base 
        
        print(f"URL = {url}")

        respuesta = requests.get(url, timeout=10)
        respuesta.raise_for_status()
        data = respuesta.json()

        if not data.get("periods"):
            return "Error: el BCRP no devolvió datos en el rango consultado."
 
        ultimo = data["periods"][-1]
        valor = ultimo["values"][0]
        fecha_dato = ultimo.get("name", "fecha desconocida")
 
        return f"Tipo de cambio (BCRP, {fecha_dato}): S/ {valor} por USD"
    
    except requests.Timeout:
        return "Error: el BCRP no respondió en 10 segundos. Reintenta más tarde."
    except requests.RequestException as e:
        return f"Error de red al consultar BCRP: {e}"
    except (KeyError, IndexError, ValueError) as e:
        return f"Error procesando respuesta del BCRP: {e}"

@tool
def python_repl(code: str) -> str:
    """Ejecuta código Python y retorna el resultado.
    Usar para cálculos matemáticos, conversiones, procesamiento de datos.
    El código debe asignar el resultado a una variable llamada 'resultado'.
 
    IMPORTANTE: este es un sandbox aislado. NO puede llamar a otras tools.
    Si necesitas un valor de otra tool, primero invoca esa tool por separado
    y luego pasa el número literal a python_repl.
    """
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
    1. Usa get_exchange_rate para informacion sobre el tipo de cambio del dólar en soles peruanos. No tiene argumentos. 
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

    # Inyectar el SystemMessage solo si no está ya presente
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

def build_graph():

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

    return graph


from langgraph.checkpoint.memory import MemorySaver

# Máximo de iteraciones del grafo por invocación.
# 15 es conservador; sube a 25 (default LangGraph) si tu agente es más complejo.
RECURSION_LIMIT = 15

def get_agent():

    graph = build_graph()

    # Compilar el grafo
    agent = ( graph.compile(checkpointer=MemorySaver())
                    .with_config({"recursion_limit": RECURSION_LIMIT})
            )

    return agent


from langgraph.errors import GraphRecursionError

if __name__ == "__main__":

    # Agente
    agent = get_agent()

    print(f"🛡  recursion_limit fijado en {RECURSION_LIMIT} para todas las invocaciones")
    print("💾 Memoria: RAM (volátil)")
 
    config_jaime = {"configurable": {"thread_id": "session-jaime"}}
 
    conversacion = [
        "Hola soy Jaime, busca el precio actual del dolar y calcula cuánto valen 142 dolares en soles.",
      #  "¿Y cuánto sería el IGV (18%) sobre ese monto en soles?",
      #  "¿Cuántos dólares dije al inicio?",   # ← prueba de memoria
        "¿Cómo me llamo?, ¿Y de cuanto era el precio actual del dolar?",   # ← prueba de memoria
    ]
 
    print("\n" + "═" * 70)
    print("  CONVERSACIÓN DE JAIME  (thread_id = session-jaime)")
    print("═" * 70)
 
    for msg in conversacion:
        print(f"\n👤 {msg}")
        try:
            result = agent.invoke({"messages": [("human", msg)]}, config_jaime)
            # Mostrar tools usadas en este turno (didáctico)
            #'''
            for m in result["messages"]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        args_preview = str(tc["args"])[:60]
                        print(f"  🔧 {tc['name']}({args_preview})")
            #'''
            print(f"🤖 {result['messages'][-1].content}")
        except GraphRecursionError:
            print(f"⚠️  El agente excedió {RECURSION_LIMIT} iteraciones. Posible loop infinito.")


    config_maria = {"configurable": {"thread_id": "session-maria"}}
 
    conversacion = [
        "Hola soy Maria, busca el precio actual del dolar y calcula cuánto valen 1000 dolares en soles.",
       # "¿Y cuánto sería el IGV (18%) sobre ese monto en soles?",
       # "¿Cuántos es el IGV en soles?",   # ← prueba de memoria
         "¿Cómo me llamo?",   # ← prueba de memoria
    ]

    print("\n" + "═" * 70)
    print("  CONVERSACIÓN DE MARIA  (thread_id = session-maria)")
    print("═" * 70)
 
    for msg in conversacion:
        print(f"\n👤 {msg}")
        try:
            result = agent.invoke({"messages": [("human", msg)]}, config_maria)
            # Mostrar tools usadas en este turno (didáctico)
            '''
            for m in result["messages"]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        args_preview = str(tc["args"])[:60]
                        print(f"  🔧 {tc['name']}({args_preview})")
            #'''
            print(f"🤖 {result['messages'][-1].content}")
        except GraphRecursionError:
            print(f"⚠️  El agente excedió {RECURSION_LIMIT} iteraciones. Posible loop infinito.")
