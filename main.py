import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

@app.get("/health")
async def health():
    return {"status": "ok", "groq": bool(GROQ_API_KEY)}

@app.post("/chat")
async def chat(data: dict):
    async with httpx.AsyncClient() as c:
        r = await c.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":data["message"]}]})
        return {"reply": r.json()["choices"][0]["message"]["content"]}
