from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn


from model import get_llm
from langchain.agents import create_agent


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
# 3. EJECUTAR
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("  Docs:   http://localhost:8000/docs")

    uvicorn.run(app, host="localhost", port=8000)




