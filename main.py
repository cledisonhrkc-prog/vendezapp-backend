import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

@app.get("/health")
async def health():
    return {"status": "ok", "groq": bool(GROQ_API_KEY)}

@app.post("/chat")
async def chat(data: dict):
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "qwen/qwen3.8-27b",
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": "Você é a VEXI, assistente operacional de comércio. Responda em português brasileiro, curto e direto, máximo 4 linhas. Use 1 emoji."},
                    {"role": "user", "content": data.get("message", "oi")}
                ],
            },
        )
        if r.status_code != 200:
            return {"error": r.text[:500]}
        return {"reply": r.json()["choices"][0]["message"]["content"]}
