"""FastAPI application entrypoint exposing RAG Agent and Memory endpoints."""
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent import rag_agent
from app.memory import memory_manager
from app.config import settings
from app.ui import HTML_DASHBOARD

app = FastAPI(
    title="Advanced RAG Agent with Context Compression & Semantic Memory",
    description="Production-grade RAG API featuring hybrid search, Cohere ReRank compression, and Mem0 long-term memory.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or prompt", json_schema_extra={"example": "Write a quick merge sort routine."})
    user_id: str = Field(..., description="Unique user identifier", json_schema_extra={"example": "usr_99"})
    session_id: str = Field(..., description="Chat session identifier", json_schema_extra={"example": "sess_1234"})


class Telemetry(BaseModel):
    retrieval_latency_ms: float = Field(..., description="Hybrid retrieval latency in milliseconds")
    rerank_latency_ms: float = Field(..., description="Cohere ReRank compression latency in milliseconds")
    generation_latency_ms: float = Field(..., description="Response generation latency in milliseconds")
    total_latency_ms: float = Field(..., description="Total end-to-end execution latency in milliseconds")


class ContextDocument(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    response: str
    telemetry: Telemetry
    context_documents: Optional[List[ContextDocument]] = None


class UserMemoriesResponse(BaseModel):
    user_id: str
    memories: List[str]
    count: int


@app.get("/", response_class=HTMLResponse)
def root_dashboard():
    """Serves the interactive visual testing dashboard for real-time validation."""
    return HTMLResponse(content=HTML_DASHBOARD)


@app.get("/api/status")
def api_status():
    """Returns JSON API status and docs pointer."""
    return {
        "status": "online",
        "service": "Advanced RAG Agent with Context Compression & Semantic Memory",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store": "qdrant",
        "compressor": "cohere-rerank",
        "memory_store": "mem0-semantic"
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    POST /api/chat: Accepts query, user_id, and session_id.
    Retrieves documents via hybrid parent retriever, compresses context using Cohere ReRank,
    adapts to stored user memories, and returns response with execution latency telemetry.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = rag_agent.process_chat(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {str(e)}")


@app.get("/api/users/{user_id}/memories", response_model=UserMemoriesResponse)
def get_user_memories_endpoint(user_id: str):
    """
    GET /api/users/{user_id}/memories: Retrieves current semantic facts stored for a user.
    """
    memories = memory_manager.get_user_memories(user_id)
    return {
        "user_id": user_id,
        "memories": memories,
        "count": len(memories)
    }


@app.post("/api/run-tests")
def run_tests_endpoint():
    """Runs the automated pytest suite and returns verification results."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v"],
        capture_output=True,
        text=True
    )
    return {
        "exit_code": result.returncode,
        "success": result.returncode == 0,
        "output": result.stdout or result.stderr
    }
