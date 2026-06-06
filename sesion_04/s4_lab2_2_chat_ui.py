import os
import time
import logging
from typing import Annotated, TypedDict

import streamlit as st

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver




# ════════════════════════════════════════════════════════════
# STREAMLIT UI
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ARIA — Institución Educativa",
    layout="centered",
)


# ════════════════════════════════════════════════════════════
# AGENTE Institución Educativa (cacheado con @st.cache_resource)
# ════════════════════════════════════════════════════════════

@st.cache_resource
def create_agent():
    """Crea el agente UNA SOLA VEZ por sesión.
    Sin @st.cache_resource, Streamlit recrearía el agente en cada interacción."""

    @tool
    def consultar_carrera(nombre: str) -> str:
        """Consulta información de una carrera de Institución Educativa.
        Carreras: software, redes, datos, industrial."""
        carreras = {
            "software":   "🎓 Desarrollo de Software · 6 ciclos · S/ 1850/mes",
            "redes":      "🌐 Redes y Comunicaciones · 6 ciclos · S/ 1750/mes",
            "datos":      "📊 Big Data y Ciencia de Datos · 6 ciclos · S/ 1900/mes",
            "industrial": "⚙️ Automatización Industrial · 6 ciclos · S/ 1850/mes",
        }
        nombre = nombre.lower().strip()
        if nombre not in carreras:
            return f"❌ Carrera '{nombre}' no encontrada. Disponibles: {', '.join(carreras.keys())}"
        return carreras[nombre]

    @tool
    def consultar_notas(codigo: str) -> str:
        """Consulta notas del estudiante. Códigos: T20231, T20232, T20233."""
        notas = {
            "T20231": "📚 María Quispe: Matemática 16, Programación 18, Inglés 14 | Promedio: 16",
            "T20232": "📚 Carlos Mendoza: Bases de Datos 12, Redes 15, Algoritmos 17 | Promedio: 14.6",
            "T20233": "⚠️ Lucía Vargas: Cálculo 09, Física 11, Programación 13 | Promedio: 11 (en riesgo)",
        }
        return notas.get(codigo.upper(), f"❌ Estudiante '{codigo}' no encontrado")

    @tool
    def registrar_consulta(motivo: str, prioridad: str) -> str:
        """Registra una consulta para Coordinación.
        prioridad: baja | media | alta | urgente"""
        return f"✅ Consulta registrada\n   Motivo: {motivo}\n   Prioridad: {prioridad}\n   ID: TK-4001"

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    tools_list = [consultar_carrera, consultar_notas, registrar_consulta]

    from model import get_llm_optimized
    from model import get_llm
    # ⭐ Optimización 1: num_predict=512 (respuestas más cortas → más rápidas)
    #llm = get_llm_optimized("vllm")
    llm = get_llm("ollama")


    llm_with_tools = llm.bind_tools(tools_list)

    SYSTEM_PROMPT = """Eres ARIA, asistente académico de Institución Educativa.
Usa SIEMPRE las herramientas. No inventes información.
Responde en español, sé concisa y amigable."""

    def reasoning(state):
        msgs = state["messages"]
        sys = [m for m in msgs if isinstance(m, SystemMessage)]
        others = [m for m in msgs if not isinstance(m, SystemMessage)]
        if not sys:
            sys = [SystemMessage(content=SYSTEM_PROMPT)]
        if len(others) > 20:
            others = others[-20:]
        return {"messages": [llm_with_tools.invoke(sys + others)]}

    def should_continue(state):
        last = state["messages"][-1]
        return "tools" if hasattr(last, "tool_calls") and last.tool_calls else "end"

    graph = StateGraph(AgentState)
    graph.add_node("reasoning", reasoning)
    graph.add_node("tools", ToolNode(tools_list))
    graph.set_entry_point("reasoning")
    graph.add_conditional_edges("reasoning", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "reasoning")
    return graph.compile(checkpointer=MemorySaver())


agent = create_agent()

st.title("🎓 ARIA")
st.caption("Asistente Académico de Institución Educativa · Ministral-3:14b (100% local)")
# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Configuración")

    thread_id = st.text_input(
        "Thread ID (para mantener memoria)",
        value="user-session-001",
    )

    st.divider()

    st.header("💡 Preguntas de ejemplo")
    examples = [
        "¿Qué carreras de software ofrece Institución Educativa?",
        "¿Cuáles son las notas de T20231?",
        "Información de la carrera de redes",
        "Registra consulta urgente: T20233 en riesgo académico",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.pending_message = ex

    st.divider()

    st.header("📋 Datos de prueba")
    st.markdown("""
                **Estudiantes:**
                - T20231 → María (excelente)
                - T20232 → Carlos (regular)
                - T20233 → Lucía (en riesgo)

                **Carreras:**
                - software, redes, datos, industrial
                """)

    st.divider()

    st.caption("🔒 100% local · datos privados · $0 de costo")

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Estado: historial de mensajes ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "info" in msg:
            st.caption(msg["info"])


# ── Input: chat o botón de ejemplo ──
pending = st.session_state.pop("pending_message", None)
prompt = pending or st.chat_input("¿En qué puedo ayudarte?")

if prompt:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Procesar con el agente (con spinner)
    with st.chat_message("assistant"):
        with st.spinner("Pensando... (modelo local, puede tardar 5-15 segundos)"):
            start = time.time()
            try:
                result = agent.invoke(
                    {"messages": [("human", prompt)]},
                    {
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": 15,
                    },
                )
                response = result["messages"][-1].content
                elapsed = time.time() - start

                # Detectar tools usadas
                tools_used = []
                for msg in result["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tools_used.extend([tc["name"] for tc in msg.tool_calls])
                tools_used = list(set(tools_used))

                # Mostrar respuesta
                st.markdown(response)

                # Info debajo (tools, tiempo)
                info_parts = []
                if tools_used:
                    info_parts.append(f"🔧 Tools: {', '.join(tools_used)}")
                info_parts.append(f"⏱️ {elapsed:.1f}s")
                info = " | ".join(info_parts)
                st.caption(info)

                # Logging
                logging.info(
                    f"thread={thread_id} | query='{prompt[:50]}' | "
                    f"tools={tools_used} | latency={elapsed:.2f}s"
                )

                # Guardar en historial
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "info": info,
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
                logging.error(f"thread={thread_id} | ERROR: {e}")