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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Uso de Chain con Langchain Ollama

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente experto en {tema}. Responde en español y sé conciso."),
    ("human", "{pregunta}")
])

chain = prompt | llm | StrOutputParser()

respuesta = chain.invoke({
    "tema": "geografía",
    "pregunta": "¿Cuál es la capital de Francia?"
})

print("Respuesta:", respuesta)

respuesta = chain.invoke({
    "tema": "Inteligencia Artificial",
    "pregunta": "¿Qué es un agente de IA?"
})

print("Respuesta:", respuesta)