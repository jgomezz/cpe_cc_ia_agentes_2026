from dotenv import load_dotenv
import os

load_dotenv()

name = os.getenv("NAME","Unknown")
path = os.getenv("PATH","Unknown")
print("NAME:", name)
print("PATH:", path)


from openai import OpenAI

model = os.getenv("LLAMA_MODEL","llama3.1:8b")
base_url = os.getenv("LLAMA_BASE_URL","http://localhost:11434")


llm = OpenAI(
    base_url=base_url + "/v1"
)
