import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI(title="VendeZapp.AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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
    return {"status": "healthy", "groq": bool(GROQ_API_KEY)}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    try:
        async with httpx.AsyncClient() as client:
            context = SYSTEM_PROMPT
            if request.customer_name:
                context += f"\nCliente: {request.customer_name}"
            
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": context},
                        {"role": "user", "content": request.message}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            
            return ChatResponse(reply=reply, action=None)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Groq API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read-invoice")
async def read_invoice(message: str):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    try:
        async with httpx.AsyncClient() as client:
            prompt = f"""Leia esta descrição de nota fiscal e extraia os itens em JSON:
{message}

Retorne JSON com: fornecedor, itens (descricao, quantidade, unidade, valor_unitario, valor_total), total_nota"""
            
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "Você é um extrator de dados de notas fiscais. Retorne apenas JSON válido."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            data = response.json()
            return {"extracted": data["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
