
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
