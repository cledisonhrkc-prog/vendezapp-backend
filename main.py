import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import google.generativeai as genai

app = FastAPI(title="VendeZapp.AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """Você é a VEXI, inteligência operacional do comércio.
Você controla estoque, vendas, produção, marketing e financeiro.
Tom: próximo, direto, levemente informal. Máximo 4 linhas por mensagem.
Nunca diga que é IA. Se perguntarem: "Sou a Vexi, cuido de tudo aqui! 🍦"
Use emojis com moderação (1-2 por mensagem).
Sempre responda em português brasileiro."""

class ChatRequest(BaseModel):
    message: str
    customer_name: Optional[str] = None
    tenant_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    action: Optional[str] = None

@app.get("/")
async def root():
    return {"status": "online", "service": "VendeZapp.AI Backend", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "gemini": bool(GEMINI_API_KEY)}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        context = SYSTEM_PROMPT
        if request.customer_name:
            context += f"\nCliente: {request.customer_name}"
        
        response = model.generate_content(
            f"{context}\n\nCliente disse: {request.message}\n\nResponda como VEXI:"
        )
        
        return ChatResponse(reply=response.text, action=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read-invoice")
async def read_invoice(message: str):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""Leia esta descrição de nota fiscal e extraia os itens em JSON:
{message}

Retorne JSON com: fornecedor, itens (descricao, quantidade, unidade, valor_unitario, valor_total), total_nota"""
        
        response = model.generate_content(prompt)
        return {"extracted": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
