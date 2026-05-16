from dotenv import load_dotenv
import os

load_dotenv()

model = os.getenv("OLLAMA_MODEL","llama3.1:8b")
base_url = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")

# Conexion con Ollama

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model=model,
    base_url=base_url,
    temperature=0.0,
)

prompt = "¿Cuál es la capital de Francia?"

reponse = llm.invoke(prompt)

print("Respuesta:", reponse.content)
