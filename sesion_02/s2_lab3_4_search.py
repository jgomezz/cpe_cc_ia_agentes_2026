from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import os

BASE_URL = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
#MODEL = os.getenv("OLLAMA_MODEL","llama3.1:8b")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Planes_Telefonos_Doc")
#QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# ── Cargar vectorstore existente (sin re-indexar) ───────────
# Opción A: OpenAI
# embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Opción B: Ollama
# embeddings_model = OllamaEmbeddings(model="nomic-embed-text",base_url=BASE_URL)

# Opción C: Ollama bge-m3
embeddings_model = OllamaEmbeddings(model="bge-m3",base_url=BASE_URL)

# Opción D: Ollama qwen3-embedding:4b
# embeddings_model = OllamaEmbeddings(model="qwen3-embedding:4b",base_url=BASE_URL)


vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings_model,
    url=QDRANT_URL,
    #api_key=QDRANT_API_KEY,
    collection_name=QDRANT_COLLECTION,
)

# Verificar
client = QdrantClient(
                      url=QDRANT_URL
                      #, api_key=QDRANT_API_KEY
                      )
info = client.get_collection(QDRANT_COLLECTION)
print(f"✅ Conectado a Qdrant: {info.points_count} vectores en '{QDRANT_COLLECTION}'\n")


# ── Dos retrievers para comparar ────────────────────────────
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10},   # busca 10, elige 3 diversos
)


# ── Comparar lado a lado ────────────────────────────────────
queries = [
    "¿Cómo hago backup?",
    "¿Qué incluye cada plan?",
    "Mi servidor está lento",
]

for q in queries:
    print(f"\n{'═' * 60}")
    print(f"🔍 {q}")
    print('═' * 60)

    print("\n📊 SIMILARITY  (los 3 más parecidos):")
    for i, doc in enumerate(similarity_retriever.invoke(q), 1):
        fuente = doc.metadata["source"].split("/")[-1]
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  {i}. [{fuente}] {preview}...")

    print("\n🎯 MMR  (relevantes Y diversos):")
    for i, doc in enumerate(mmr_retriever.invoke(q), 1):
        fuente = doc.metadata["source"].split("/")[-1]
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  {i}. [{fuente}] {preview}...")