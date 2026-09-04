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
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gemma2-9b-it", "messages": [{"role": "user", "content": data.get("message", "oi")}]},
        )
        if r.status_code != 200:
            return {"error": r.text[:500]}
        return {"reply": r.json()["choices"][0]["message"]["content"]}
