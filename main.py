import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

load_dotenv()

app = FastAPI(title="AI Chat Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversations: dict[str, list[dict]] = {}

SYSTEM_PROMPT = (
    "You are a kind, warm, and patient AI assistant designed to help elderly users. "
    "Always be polite, respectful, and gentle in your tone. "
    "Use simple, clear language — avoid jargon or technical terms. "
    "Keep your responses concise and conversational since they will be spoken aloud. "
    "If the user seems confused, kindly offer to explain further. "
    "Address them with warmth, as you would a respected elder."
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.get("/")
async def serve_demo():
    return FileResponse("static/index.html")


@app.get("/mic-test.html")
async def serve_mic_test():
    return FileResponse("mic-test.html")


@app.get("/simple-mic-test.html")
async def serve_simple_mic_test():
    return FileResponse("simple-mic-test.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    history = conversations.setdefault(req.session_id, [])
    history.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        raw_text = response.choices[0].message.content

        import re
        assistant_text = re.sub(r'<think>[\s\S]*?</think>', '', raw_text).strip()
        if not assistant_text:
            assistant_text = raw_text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")

    history.append({"role": "assistant", "content": assistant_text})

    
    if len(history) > 20:
        conversations[req.session_id] = history[-20:]

    return ChatResponse(reply=assistant_text, session_id=req.session_id)


@app.delete("/chat/{session_id}")
async def clear_history(session_id: str):
    conversations.pop(session_id, None)
    return {"status": "cleared"}
