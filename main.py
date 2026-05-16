"""
SHL Assessment Recommender - Main FastAPI Application
"""
import os
import json
import time
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import SHLAgent
agent: Optional[SHLAgent] = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup."""
    global agent
    print("Initializing SHL Assessment Agent...")
    agent = SHLAgent()
    await agent.initialize()
    print("Agent ready!")
    yield
    print("Shutting down...")
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Message(BaseModel):
    role: str  
    content: str
class ChatRequest(BaseModel):
    messages: list[Message]
class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str
class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a multi-turn conversation and return agent reply with optional recommendations.
    The API is stateless - full conversation history must be provided each call.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400, detail=f"Invalid role: {msg.role}. Must be 'user' or 'assistant'"
            )
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        result = await agent.chat(messages)
        return ChatResponse(**result)
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
