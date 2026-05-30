from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import os

from langchain_core.tools import tool


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

#@tool
def search_docs(query: str) -> str:

    docs = mmr_retriever.invoke(query)

    if not docs:
        return "Sin resultados relevantes."
    
    partes = []
    for d in docs:
        nombre_archivo = d.metadata['source'].split('/')[-1]
        partes.append(f"[{nombre_archivo}]: {d.page_content}")

    return "\n\n---\n\n".join(partes)


if __name__ == "__main__":

    query = "¿Cómo hago backup?"

    print(search_docs(query))