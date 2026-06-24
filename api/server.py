"""
api/server.py — FastAPI Backend
================================
Provides REST API endpoints for the Codebase Assistant.
Can be started via Uvicorn:
    uvicorn api.server:app --reload --port 8000
"""

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.search import HybridRetriever
from agents.workflow import run_agent
from graph.builder import GraphBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    mode: str = "search"  # "search" | "agent"
    top_k: int = 5


class SearchResultItem(BaseModel):
    file_path: str
    symbol: str
    type: str
    start_line: int
    end_line: int
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    results: List[SearchResultItem] = []


# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Codebase Assistant API",
    description="RAG Search and Agent Endpoints for the Codebase Assistant.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
retriever: Optional[HybridRetriever] = None
graph_db: Optional[GraphBuilder] = None


@app.on_event("startup")
def startup_event():
    global retriever, graph_db
    retriever = HybridRetriever()
    graph_db = GraphBuilder()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query_codebase(req: QueryRequest):
    """
    Main endpoint for answering codebase questions.
    If mode == "search", it returns standard RAG results.
    If mode == "agent", it invokes the LangGraph agent for multi-step reasoning.
    """
    if req.mode == "agent":
        try:
            answer = run_agent(req.query)
            return QueryResponse(answer=answer, results=[])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    else:
        # Standard Hybrid Search
        results = retriever.search(req.query, top_k=req.top_k)
        
        items = []
        for r in results:
            items.append(SearchResultItem(
                file_path=r.file_path,
                symbol=r.symbol,
                type=r.chunk_type,
                start_line=r.start_line,
                end_line=r.end_line,
                content=r.content,
                score=r.score
            ))
            
        return QueryResponse(
            answer="Here are the top semantic matches from the codebase.",
            results=items
        )


@app.get("/graph/callers")
def get_callers(symbol: str):
    """Returns functions that call the requested symbol."""
    callers = graph_db.get_callers(symbol)
    return {"symbol": symbol, "callers": callers}


@app.get("/graph/callees")
def get_callees(symbol: str):
    """Returns functions called by the requested symbol."""
    callees = graph_db.get_callees(symbol)
    return {"symbol": symbol, "callees": callees}
