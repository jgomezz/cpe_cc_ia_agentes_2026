"""
s3_lab3_1_planner_agent — Agente Planificador (Plan → Execute → Reflect) — Institución Educativa
──────────────────────────────────────────────────────────────────
Para tareas complejas (3+ pasos), un solo agente ReAct se confunde.
La solución: dividir en TRES nodos especializados.

  ┌─────────┐    ┌──────────┐    ┌─────────┐
  │  PLAN   │ →  │ EXECUTE  │ →  │ REFLECT │
  └─────────┘    └──────────┘    └─────────┘
                       ↑                │
                       │   "necesita    │
                       └───  más"  ─────┘

from langchain_core.tools import tool
"""

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from model import get_llm
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage


# ═══════════════════════════════════════════════════════════
# TOOLS (datos académicos de Institución Educativa)
# ═══════════════════════════════════════════════════════════

@tool
def consultar_estudiante(codigo: str) -> str:
    """Consulta datos académicos de un estudiante de Institución Educativa."""
    estudiantes = {
        "T20231": "María Quispe | Desarrollo de Software | Ciclo 3 | Promedio 16.0 | Asistencia 95%",
        "T20232": "Carlos Mendoza | Big Data | Ciclo 4 | Promedio 14.6 | Asistencia 70% (ALERTA)",
        "T20233": "Lucía Vargas | Redes | Ciclo 2 | Promedio 11.0 | Asistencia 50% (RIESGO)",
    }
    return estudiantes.get(codigo.upper(), f"Estudiante '{codigo}' no encontrado.")


@tool
def consultar_normativa(tema: str) -> str:
    """Consulta normativas académicas de Institución Educativa."""
    normativas = {
        "beca":            "Beca Institución Educativa: promedio mínimo 16, sin desaprobados, asistencia ≥85%.",
        "riesgo":          "Riesgo académico: promedio <13 o asistencia <80%. Requiere tutoría obligatoria.",
        "reincorporacion": "Reincorporación: trámite con Coordinación, máx. 2 ciclos consecutivos.",
        "titulacion":      "Requisitos: 4 créditos mínimos, modalidades: tesis o suficiencia profesional.",
    }
    return normativas.get(tema.lower(), f"Sin normativa sobre '{tema}'.")


@tool
def calculadora(expresion: str) -> str:
    """Calcula una expresión matemática. Ejemplo: '16 - 13' o '0.95 * 20'."""
    try:
        return f"Resultado: {eval(expresion)}"
    except Exception as e:
        return f"Error: {e}"


tools = [consultar_estudiante, consultar_normativa, calculadora]


# ═══════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════

llm = get_llm("ollama")
llm_with_tools = llm.bind_tools(tools)


# ═══════════════════════════════════════════════════════════
# ESTADO con campos "plan" y "reflection"
# ═══════════════════════════════════════════════════════════
class PlannerState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    reflection: str


# ═══════════════════════════════════════════════════════════
# NODO 1: PLAN  (descompone la tarea)
# ═══════════════════════════════════════════════════════════
PLAN_PROMPT = """
Eres un planificador experto en tareas académicas. 
descompones la tarea del usuario en 3 pasos.
Sigue un formato exacto:
Paso 1: [paso]
Paso 2: [paso]
Paso 3: [paso]
Responde SOLO con el plan, sin explicaciones adicionales.
"""

def planner_node(state: PlannerState)-> dict:
    """El nodo planificador recibe el mensaje del usuario y genera un plan de acción."""
    
    user_message = state["messages"][-1].content

    reponse = llm.invoke([
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=user_message),

    ])

    plan = reponse.content

    return {
        "plan": plan,
        "messages": HumanMessage(content=f"Ejecuta este plan : \n{plan} \n\n Tarea solicitada es : {user_message}")
    }



# ═══════════════════════════════════════════════════════════
# NODO 2: EXECUTE  (ejecuta el plan)
# ═══════════════════════════════════════════════════════════
def executor_node(state)-> dict:
    pass

# ═══════════════════════════════════════════════════════════
# NODO 3: REFLECT  (reflexiona sobre el plan)
# ═══════════════════════════════════════════════════════════
def reflection_node(state)-> dict:
    pass


def should_use_tools(state: PlannerState) -> str:
    """Si el LLM pidió tools, ir a tools. Si no, ir a reflexión."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "reflect"

def route_after_reflection(state: PlannerState) -> str:
    return "end" if state.get("reflection") == "approved" else "executor"

# ═══════════════════════════════════════════════════════════
# GRAFO: PLAN → EXECUTE ↔ TOOLS → REFLECT
# ═══════════════════════════════════════════════════════════
graph = StateGraph(PlannerState)

graph.add_node("planner",  planner_node)
graph.add_node("executor", executor_node)
graph.add_node("tools",    ToolNode(tools))
graph.add_node("reflect",  reflection_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_use_tools, {
    "tools":   "tools",
    "reflect": "reflect",
})
graph.add_edge("tools", "executor")
graph.add_conditional_edges("reflect", route_after_reflection, {
    "executor": "executor",
    "end":      END,
})

# El agente planificador con límite de recursión para evitar loops infinitos
agent = graph.compile().with_config({"recursion_limit": 25})

# ═══════════════════════════════════════════════════════════
# VISUALIZAR GRAFO
# ═══════════════════════════════════════════════════════════
print(agent.get_graph().draw_mermaid())