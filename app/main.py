from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from app.llm import ask_llm



@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    # load_vector_store(
    #     model.get_sentence_embedding_dimension()
    # )
    # print("🚀 API iniciada")

    yield

    # SHUTDOWN
    print("🛑 API finalizada")
    
    
app = FastAPI(title="Local RAG + Gemini", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # em prod, restrinja depois
    allow_credentials=True,
    allow_methods=["*"],        # inclui OPTIONS automaticamente
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str
class AskRequest(BaseModel):
    messages: List[Message]
    uuid: Optional[str] = None
    name: Optional[str] = None
    
@app.post("/ask")
def ask(payload: AskRequest):
    print(payload)
    question = get_user_question(payload.messages)
    if not question:
        return {"response": "Nenhuma pergunta encontrada."}

    answer = ask_llm(question, payload.name)
    return answer

def get_user_question(messages: list[Message]) -> str | None:
    print(messages)
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content
    return None
