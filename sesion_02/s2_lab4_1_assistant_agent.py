from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import os

from langchain_core.tools import tool
from model import get_llm

from langchain_core.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


BASE_URL = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
#MODEL = os.getenv("OLLAMA_MODEL","llama3.1:8b")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Planes_Telefonos_Doc")
#QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ── Cargar vectorstore existente (sin re-indexar) ───────────

# Opción C: Ollama bge-m3
embeddings_model = OllamaEmbeddings(model="bge-m3",base_url=BASE_URL)

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings_model,
    url=QDRANT_URL,
    #api_key=QDRANT_API_KEY,
    collection_name=QDRANT_COLLECTION,
)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10},   # busca 10, elige 3 diversos
)

# ── Tools ───────────────────────────────────────────────────

@tool
def search_docs(query: str) -> str:
    """
    Busca en la documentación de CloudTech.
    Úsala para: políticas, planes, precios, procedimientos técnicos.
    """
    docs = mmr_retriever.invoke(query)

    if not docs:
        return "Sin resultados relevantes."
    
    partes = []
    for d in docs:
        nombre_archivo = d.metadata['source'].split('/')[-1]
        partes.append(f"[{nombre_archivo}]: {d.page_content}")

    return "\n\n---\n\n".join(partes)

@tool
def server_status(server_id: str) -> str:
    """
    Verificar el estado de un servidor de la empresa
    """
    servidores = {
        "srv-Lurin" : "activo | CPU : 23% | Plan : Pro",
        "srv-Nazca" : "saturado | CPU : 95% | Plan : Básico",
        "srv-Lurin" : "caido | CPU : 0% | Plan : Enterprise"

    }
    return servidores.get(server_id, f"{server_id} no encontrado")


# ── Agente ──────────────────────────────────────────────────

tools = [search_docs,server_status]

llm = get_llm(name="openai")

llm_with_tools = llm.bind_tools(tools)


# Definir el SYSTEM PROMPT para el agente

SYSTEM_PROMPT = """
    Eres un asistente experto en economia y finanzas. Responde en español y sé conciso. 
    
    ## HERRAMIENTAS DISPONIBLES:
    1. search_docs para preguntas sobre documentación, políticas, planes. 
    2. Userver_status para ver problemas en el servidor.
    3. NUNCA inventes. Cita fuentes cuando uses documentación.
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

    query = "¿Cómo hago backup?"

    result =  agent.invoke({"messages": [("human", query)]})
    print(f"Respuesta: {result['messages'][-1].content}")
