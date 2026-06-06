from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
from model import get_llm
from langchain.agents import create_agent

'''

curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "Dame un programa en Python para calcular el area de un triangulo"}'

curl -X POST http://localhost:8000/chat/stream \
        -H "Content-Type: application/json" \
        -d '{"message": "Dame un programa en Python para calcular el area de un triangulo"}'

'''


# ════════════════════════════════════════════════════════════
# AGENT
# ════════════════════════════════════════════════════════════
    

llm = get_llm("ollama")
agent = create_agent(llm)


# ════════════════════════════════════════════════════════════
# FAST API
# ════════════════════════════════════════════════════════════
    
app = FastAPI(title="Basic API")

# ════════════════════════════════════════════════════════════
# 1. API REST
# ════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    tools_used: list[str] = []

@app.post("/chat" , response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:   
    """Endpoint para recibir mensajes y responder usando el agente."""

    result = agent.invoke(
        {"messages": [("human", request.message)]},
        {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 15}
    )

    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tools_used.extend(tc["name"] for tc in msg.tool_calls)


    return ChatResponse(
        response=result["messages"][-1].content,
        thread_id=request.thread_id,
        tools_used=list(set(tools_used)),
    )


# ════════════════════════════════════════════════════════════
# 2. STREAMING
# ════════════════════════════════════════════════════════════

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk

class ChatStreamRequest(BaseModel):
    message: str
    thread_id: str = "default"


@app.post("/chat/stream")
async def chat_stream(body: ChatStreamRequest):
    def generar():

        # config es el diccionario de configuracion del agente
        config = {
            "configurable": {"thread_id": body.thread_id},
            "recursion_limit": 15,
        }

        # stream_mode="messages" es el modo de streaming que devuelve el agente
        result =agent.stream(
            {"messages": [("human", body.message)]},
            config,
            stream_mode="messages",
        )

        # result es un generator que devuelve tuplas de (chunk, _)
        for chunk, _ in result:
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

    return StreamingResponse(generar(), media_type="text/plain")




# ════════════════════════════════════════════════════════════
# 3. EJECUTAR
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("  Docs:   http://localhost:8000/docs")

    uvicorn.run(app, host="localhost", port=8000)




