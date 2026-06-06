from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
    
app = FastAPI(title="Basic API")

# ════════════════════════════════════════════════════════════
# 1. API REST
# ════════════════════════════════════════════════════════════

@app.get("/chat")
async def chat(): 
    return {"mensaje": "Hola Mundo"}

# ════════════════════════════════════════════════════════════
# 3. EJECUTAR
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("  Docs:   http://localhost:8000/docs")

    uvicorn.run(app, host="localhost", port=8000)




