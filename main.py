import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

@app.get("/health")
async def health():
    return {"status": "ok", "key_len": len(GROQ_API_KEY), "key_start": GROQ_API_KEY[:8]}

@app.post("/chat")
async def chat(data: dict):
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": data.get("message", "oi")}]},
        )
        return {"status_code": r.status_code, "body": r.text[:500]}
