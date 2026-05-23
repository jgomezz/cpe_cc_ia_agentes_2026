from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm(name):

    llm = None

    if name == "ollama":
       model = os.getenv("OLLAMA_MODEL","llama3.1:8b")
       base_url = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")

       llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.0,
      )
    elif name == "openai":
        model = os.getenv("OPENAI_MODEL","gpt-4o-mini")

        llm = ChatOpenAI(
            model=model,
            temperature=0.0,
        )

    return llm