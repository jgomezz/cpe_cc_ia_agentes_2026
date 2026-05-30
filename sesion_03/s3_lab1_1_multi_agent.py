"""
Lab 1.1 — Sistema Multi-Agente con Patrón Supervisor (Institución Educativa)
──────────────────────────────────────────────────────────────
Un supervisor clasifica al usuario y delega a un agente especializado:

    Usuario → Supervisor → [Estudiantes | Docentes | Coordinación] → Respuesta

Pre-requisitos:
──────────────────────────────────────────────────────────────
   .env → OLLAMA_MODEL=ministral-3:14b

"""

from dotenv import load_dotenv
load_dotenv()

import unicodedata
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from model import get_llm


# ═══════════════════════════════════════════════════════════
# HELPER: normalizar texto (quitar tildes + lowercase)
# ═══════════════════════════════════════════════════════════

def normalizar(texto: str) -> str:
    """Convierte 'Cálculo' → 'calculo', 'Programación' → 'programacion'."""
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    return sin_tildes.lower().strip()


# ═══════════════════════════════════════════════════════════
# TOOLS POR ESPECIALIDAD (Institución Educativa)
# ═══════════════════════════════════════════════════════════

# ── Tools del agente de ESTUDIANTES ──
@tool
def consultar_notas(codigo_estudiante: str) -> str:
    """Consulta las notas del estudiante en el ciclo actual.
    Códigos disponibles: T20231, T20232, T20233.
    """
    notas = {
        "T20231": "Matemática: 16 | Programación: 18 | Inglés: 14 | Promedio: 16",
        "T20232": "Bases de Datos: 12 | Redes: 15 | Algoritmos: 17 | Promedio: 14.6",
        "T20233": "Cálculo: 09 | Física: 11 | Programación: 13 | Promedio: 11 (en riesgo)",
    }
    codigo = codigo_estudiante.strip().upper()
    if codigo not in notas:
        return (f"Estudiante '{codigo_estudiante}' NO EXISTE en el sistema. "
                f"Códigos válidos: {', '.join(notas.keys())}.")
    return notas[codigo]


@tool
def consultar_asistencia(codigo_estudiante: str, curso: str) -> str:
    """Consulta el porcentaje de asistencia del estudiante en un curso específico.
    Códigos disponibles: T20231, T20232, T20233.
    Cursos disponibles: matemática, bases de datos, cálculo, programación, etc.
    """
    asistencias = {
        ("T20231", "matematica"):     "Asistencia: 95% (19/20 clases). Estado: OK",
        ("T20232", "bases de datos"): "Asistencia: 70% (14/20 clases). Estado: ALERTA (mínimo 80%)",
        ("T20233", "calculo"):        "Asistencia: 50% (10/20 clases). Estado: DESAPROBADO por inasistencia",
    }
    codigo = codigo_estudiante.strip().upper()
    curso_norm = normalizar(curso)
    key = (codigo, curso_norm)

    if key not in asistencias:
        registros = [f"{c} en '{cu}'" for (c, cu) in asistencias.keys()]
        return (f"NO HAY registro de asistencia para {codigo} en '{curso}'. "
                f"Registros disponibles: {' · '.join(registros)}.")
    return asistencias[key]


# ── Tools del agente de DOCENTES ──
@tool
def consultar_horario_docente(codigo_docente: str) -> str:
    """Consulta el horario semanal de un docente.
    Códigos disponibles: D1001, D1002, D1003.
    """
    horarios = {
        "D1001": "Lunes 8-10: Programación I (Aula 201) | Mié 14-16: Algoritmos (Lab 305)",
        "D1002": "Martes 10-12: Bases de Datos (Lab 102) | Jueves 16-18: Big Data (Aula 401)",
        "D1003": "Lunes-Viernes 18-20: Inglés Técnico (Aula 105)",
    }
    codigo = codigo_docente.strip().upper()
    if codigo not in horarios:
        return (f"Docente '{codigo_docente}' NO EXISTE en el sistema. "
                f"Códigos válidos: {', '.join(horarios.keys())}.")
    return horarios[codigo]


@tool
def registrar_calificacion(codigo_estudiante: str, curso: str, nota: float) -> str:
    """Registra una calificación para un estudiante en un curso.
    La nota debe estar entre 0 y 20 (escala peruana). Aprobado: ≥13.
    """
    if not (0 <= nota <= 20):
        return f"ERROR: la nota {nota} es inválida. Debe estar entre 0 y 20."
    estado = "APROBADO" if nota >= 13 else "DESAPROBADO"
    return f"✅ Registrado: {codigo_estudiante} | {curso} | Nota: {nota} | {estado}"


# ── Tools del agente de COORDINACIÓN ──
@tool
def buscar_normativa(tema: str) -> str:
    """Busca normativas y reglamentos académicos de Institución Educativa.
    Temas disponibles: matrícula, convalidación, reincorporación, titulación, beca.
    """
    normativas = {
        "matricula":       "Reglamento: la matrícula se realiza dentro del cronograma oficial. Extemporánea: recargo del 10%.",
        "convalidacion":   "Solicitud con notas mín. 14 del curso original. Plazo: hasta 2da semana del ciclo.",
        "reincorporacion": "Estudiantes con licencia: trámite en Coordinación. Máx. 2 ciclos consecutivos.",
        "titulacion":      "Modalidades: tesis, suficiencia profesional, examen de competencias. 4 créditos mín.",
        "beca":            "Beca Institución Educativa: promedio mínimo 16, sin desaprobados. Renovación semestral.",
    }
    tema_norm = normalizar(tema)
    results = [v for k, v in normativas.items() if k in tema_norm]

    if not results:
        return (f"NO HAY normativa sobre '{tema}'. "
                f"Temas disponibles: {', '.join(normativas.keys())}.")
    return "\n".join(results)


@tool
def consultar_carrera(carrera: str) -> str:
    """Consulta información sobre las carreras técnicas que ofrece Institución Educativa.
    Carreras disponibles: software, redes, datos, industrial.
    """
    carreras = {
        "software":   "Desarrollo de Software · 6 ciclos · 3 años · 120 créditos · Aula+Laboratorio",
        "redes":      "Redes y Comunicaciones · 6 ciclos · 3 años · 118 créditos · Lab. Cisco",
        "datos":      "Big Data y Ciencia de Datos · 6 ciclos · 3 años · 120 créditos · Proyectos reales",
        "industrial": "Automatización Industrial · 6 ciclos · 3 años · 124 créditos · Lab. PLC y robótica",
    }
    carrera_norm = normalizar(carrera)
    results = [v for k, v in carreras.items() if k in carrera_norm]

    if not results:
        return (f"NO HAY carrera registrada como '{carrera}'. "
                f"Carreras disponibles en el sistema: {', '.join(carreras.keys())}.")
    return "\n".join(results)


# ═══════════════════════════════════════════════════════════
# ESTADO COMPARTIDO DEL GRAFO
# ═══════════════════════════════════════════════════════════

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str


# ═══════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════

llm = get_llm("ollama")


# ═══════════════════════════════════════════════════════════
# NODO 1: SUPERVISOR
# ═══════════════════════════════════════════════════════════

SUPERVISOR_PROMPT = """Eres el supervisor del asistente virtual de Institución Educativa.
Tu ÚNICA tarea: clasificar la solicitud del usuario.

AGENTES DISPONIBLES:
- estudiantes:  Consultas sobre notas, asistencia, situación académica del alumno.
- docentes:     Horarios de docentes, registro de calificaciones.
- coordinacion: Normativas, reglamentos, información de carreras, trámites.

Responde ÚNICAMENTE con UNA palabra: estudiantes, docentes o coordinacion.
Si no estás seguro, responde: coordinacion
"""


def supervisor_node(state: MultiAgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        *state["messages"],
    ])

    text = response.content.strip().lower()

    if "estudiantes" in text:
        decision = "estudiantes"
    elif "docentes" in text:
        decision = "docentes"
    elif "coordinacion" in text or "coordinación" in text:
        decision = "coordinacion"
    else:
        decision = "coordinacion"

    print(f"  📋 Supervisor → {decision}")
    return {"next_agent": decision}


# ═══════════════════════════════════════════════════════════
# AGENTES ESPECIALIZADOS — System prompts anti-alucinación
# ═══════════════════════════════════════════════════════════

# Regla compartida por todos los agentes: NO inventar.
ANTI_HALUCINACION = (
    "REGLAS ESTRICTAS:\n"
    "1. USA SIEMPRE las herramientas para obtener datos. No respondas de memoria.\n"
    "2. RESPONDE EXACTAMENTE lo que devuelve la herramienta. NO agregues:\n"
    "   - URLs inventadas (ej: Institución Educativa.edu.pe/...)\n"
    "   - Artículos, secciones o reglamentos no mencionados\n"
    "   - Carreras, cursos o servicios no listados\n"
    "   - Detalles adicionales 'de relleno'\n"
    "3. Si la herramienta devuelve 'NO EXISTE' o 'NO HAY', díselo al usuario tal cual.\n"
    "4. Si falta información, di 'no tengo esa información' en vez de inventar.\n"
)

estudiantes_agent = create_agent(
    llm,
    tools=[consultar_notas, consultar_asistencia],
    system_prompt=(
        "Eres el agente de atención a estudiantes de Institución Educativa.\n"
        "Respondes sobre notas, asistencia y situación académica.\n\n"
        + ANTI_HALUCINACION
    ),
)

docentes_agent = create_agent(
    llm,
    tools=[consultar_horario_docente, registrar_calificacion],
    system_prompt=(
        "Eres el agente de apoyo docente de Institución Educativa.\n"
        "Ayudas con horarios y registro de calificaciones.\n\n"
        + ANTI_HALUCINACION
    ),
)

coordinacion_agent = create_agent(
    llm,
    tools=[buscar_normativa, consultar_carrera],
    system_prompt=(
        "Eres el agente de Coordinación Académica de Institución Educativa.\n"
        "Respondes sobre normativas, reglamentos, carreras y trámites.\n\n"
        + ANTI_HALUCINACION
    ),
)


def estudiantes_node(state: MultiAgentState) -> dict:
    result = estudiantes_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1].content
    return {"messages": [AIMessage(content=f"[Estudiantes] {last}")]}


def docentes_node(state: MultiAgentState) -> dict:
    result = docentes_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1].content
    return {"messages": [AIMessage(content=f"[Docentes] {last}")]}


def coordinacion_node(state: MultiAgentState) -> dict:
    result = coordinacion_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1].content
    return {"messages": [AIMessage(content=f"[Coordinación] {last}")]}


# ═══════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════

def route_to_agent(state: MultiAgentState) -> str:
    return state.get("next_agent", "coordinacion")


# ═══════════════════════════════════════════════════════════
# CONSTRUIR GRAFO
# ═══════════════════════════════════════════════════════════

graph = StateGraph(MultiAgentState)

graph.add_node("supervisor",   supervisor_node)
graph.add_node("estudiantes",  estudiantes_node)
graph.add_node("docentes",     docentes_node)
graph.add_node("coordinacion", coordinacion_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_to_agent, {
    "estudiantes":  "estudiantes",
    "docentes":     "docentes",
    "coordinacion": "coordinacion",
})

graph.add_edge("estudiantes",  END)
graph.add_edge("docentes",     END)
graph.add_edge("coordinacion", END)

agent = (
    graph.compile(checkpointer=MemorySaver())
         .with_config({"recursion_limit": 15})
)


# ═══════════════════════════════════════════════════════════
# VISUALIZAR GRAFO
# ═══════════════════════════════════════════════════════════

print("=" * 50)
print("  GRAFO DEL AGENTE (Mermaid)")
print("=" * 50)
print(agent.get_graph().draw_mermaid())
print()
print("📋 Copia el código Mermaid y pégalo en https://mermaid.live")


# ═══════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-multi-agente"}}

    preguntas = [
        "¿Cuáles son las notas del estudiante T20232?",              # → estudiantes
        "¿Cuál es el horario del docente D1001?",                    # → docentes
#        "¿Qué requisitos hay para la beca Institución Educativa?",                  # → coordinacion
#        "Quiero registrar nota 17 a T20231 en Programación I.",      # → docentes
#        "¿Qué carreras ofrece Institución Educativa en el área de software?",       # → coordinacion
#        "¿Cuál es la asistencia de T20233 en Cálculo?",              # → estudiantes
    ]

    print("═" * 60)
    print("  ASISTENTE VIRTUAL Institución Educativa (Supervisor + 3 especialistas)")
    print("  Modelo: ministral-3:14b · Modo anti-alucinación")
    print("═" * 60)

    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n── Pregunta {i}/{len(preguntas)} ──")
        print(f"👤 {pregunta}")
        result = agent.invoke({"messages": [("human", pregunta)]}, config)
        print(f"🤖 {result['messages'][-1].content}")

    print("\n" + "═" * 60)