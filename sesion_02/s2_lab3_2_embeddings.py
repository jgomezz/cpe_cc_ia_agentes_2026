
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_document(archivo):
    loader = TextLoader(archivo)
    documentos = loader.load()
    return documentos



import numpy as np

# Funciones utilitarias

# ── Función de similitud coseno ─────────────────────────────
def similitud(v1, v2):
    """Mide qué tan similares son dos vectores (1.0 = idénticos)."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def buscar(embeddings_model, query: str, top_k: int = 3):
    """Encuentra los chunks más parecidos a una pregunta."""
    q_vec = embeddings_model.embed_query(query)
    scored = [(similitud(q_vec, v), i) for i, v in enumerate(vectores)]
    scored.sort(reverse=True)
    return scored[:top_k]



if __name__ == "__main__":


    # 1. CARGAR DATOS

    dir = Path("docs")

    documentos = []
    for archivo in dir.glob("*.txt"):
        print(f"Archivo: {archivo.name}")
        docs = load_document(archivo)
        documentos.extend(docs)

    print(f"Total de documentos cargados: {len(documentos)}")

    #print(documentos[0].page_content[:500])  # Imprime los primeros 500 caracteres del primer documento

    # 2. FRAGMENTAR EN CHUNKS

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documentos)

    print(f"Total de chunks generados: {len(chunks)}")  

    # 3. CREAR EMBEDDINGS

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vectores = embedding_model.embed_documents([c.page_content for c in chunks] )

    print(f" {len(vectores)} vectores · {len(vectores[0])} dimensiones cada uno\n")

    # 4. EVALUAR EMBEDDING

    query = "¿Cuanto cuesta el plan Pro?"

    for  score, idx  in  buscar(embedding_model, query) :
    
        fuente = chunks[idx].metadata["source"].split("/")[-1]
        preview = chunks[idx].page_content[:100].replace("\n", " ")
        print(f"  {score:.3f} chunks[{idx}]  [{fuente}]  {preview}...")