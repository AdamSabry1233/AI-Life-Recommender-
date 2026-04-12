"""
Phase 6 — FastAPI backend.

Exposes the recommendation engine over HTTP with session management.
One ChatEngine instance shared across all requests (loaded at startup).
One ChatSession per session_id (in-memory, lost on restart).

Endpoints:
    POST /recommend        single-shot intent → recommendations
    POST /chat             multi-turn message handler
    GET  /history          past FAISS sessions
    GET  /health           liveness check
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine import ChatEngine, ChatSession

# ── session store (in-memory) ─────────────────────────────────────────────────

_sessions: dict[str, ChatSession] = {}

def get_session(session_id: str) -> ChatSession:
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession()
    return _sessions[session_id]

# ── app lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading recommendation engine …")
    app.state.engine = ChatEngine(
        checkpoint_path=os.getenv("CHECKPOINT_PATH", "mf_best.pt"),
        data_dir=os.getenv("DATA_DIR", "data"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
    )
    print("Engine ready.")
    yield
    # save any open sessions on shutdown
    for session in _sessions.values():
        app.state.engine.save(session)

app = FastAPI(title="AI Life Recommender", version="0.6.0", lifespan=lifespan)

# ── request / response models ─────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    intent:     str
    session_id: str = ""          # auto-generated if blank

class ChatRequest(BaseModel):
    message:    str
    session_id: str

class ChatResponse(BaseModel):
    response:   str
    type:       str               # recommendations | answer | refinement | history
    session_id: str

class HistoryEntry(BaseModel):
    intent:    str
    timestamp: int
    movie:     dict
    food:      dict
    book:      dict
    habit:     dict
    summary:   str

# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=ChatResponse)
def recommend(req: RecommendRequest):
    """Single-shot: create a fresh session, run the full graph, return results."""
    engine     = app.state.engine
    session_id = req.session_id or str(uuid.uuid4())
    session    = ChatSession()
    _sessions[session_id] = session

    response, resp_type = engine.process(session, req.intent)
    return ChatResponse(response=response, type=resp_type, session_id=session_id)


@app.post("/session/start", response_model=ChatResponse)
def start_session(session_id: str = ""):
    """Start a new session — returns the first onboarding question."""
    engine     = app.state.engine
    session_id = session_id or str(uuid.uuid4())
    session    = ChatSession()
    _sessions[session_id] = session

    first_question = engine.start_onboarding()
    return ChatResponse(response=first_question, type="onboarding", session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Multi-turn: resume an existing session or start a new one."""
    engine  = app.state.engine
    session = get_session(req.session_id)

    response, resp_type = engine.process(session, req.message)
    return ChatResponse(response=response, type=resp_type, session_id=req.session_id)


@app.post("/chat/save")
def save_session(req: ChatRequest):
    """Explicitly save the current session to FAISS (called on UI disconnect)."""
    engine  = app.state.engine
    session = get_session(req.session_id)
    engine.save(session)
    return {"status": "saved"}


@app.get("/history", response_model=list[HistoryEntry])
def history():
    """Return all stored past interactions from the FAISS metadata."""
    engine = app.state.engine
    return engine.store.metadata
