from dotenv import load_dotenv
import os

load_dotenv()

model = os.getenv("OPENAI_MODEL","gpt-4o-mini")

# Conexion con OpenAI

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=model,
    temperature=0.0,
)

prompt = "¿Cuál es la capital de Francia?"

reponse = llm.invoke(prompt)

print("Respuesta:", reponse.content)
