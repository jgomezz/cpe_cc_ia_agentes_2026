"""
model.py — Helper para obtener el LLM configurado.
Permite alternar entre Ollama y OpenAI sin tocar los labs.
"""
import os
from dotenv import load_dotenv
load_dotenv()


def get_llm(name: str = "ollama", temperature: float = 0):
    """Retorna el LLM configurado según el nombre."""
    if name == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    elif name == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )
    raise ValueError(f"LLM desconocido: {name}")
