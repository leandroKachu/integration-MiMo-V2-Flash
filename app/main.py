from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from app.llm import ask_llm

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print("🚀 Sun li.AI API iniciada")
    yield
    # SHUTDOWN
    print("🛑 API finalizada")


app = FastAPI(
    title="Sun li.AI Assistant",
    version="2.0.0",
    description="Assistente IA com contexto de conversa",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODELS
# ============================================
class Message(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    messages: List[Message]
    uuid: Optional[str] = None
    name: Optional[str] = None

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Sun li.AI Assistant",
        "version": "2.0.0"
    }

@app.post("/ask")
def ask(payload: AskRequest):
    
    # Extrai última pergunta do usuário
    question = get_user_question(payload.messages)
    if not question:
        return {"response": "Nenhuma pergunta encontrada."}
    
    # Nome padrão caso não fornecido
    name = payload.name or "usuário"
    
    # Converte Pydantic models para dict
    context_messages = [
        {"role": msg.role, "content": msg.content} 
        for msg in payload.messages
    ]
    
    # Processa COM contexto
    answer = ask_llm(question, name, context_messages)
    
    return answer


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_user_question(messages: List[Message]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content
    return None