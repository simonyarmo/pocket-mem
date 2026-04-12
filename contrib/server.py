"""
server.py — FastAPI HTTP sidecar for pocket-mem.

Exposes pocket-mem as a language-agnostic HTTP API so any process
(Python, Node, Go, etc.) can use memory without importing the package.

Start:
    uvicorn memory_agent.server:app --host 0.0.0.0 --port 8000

Environment variables (ingestion model — always Ollama/local):
    OLLAMA_BASE_URL   LLM endpoint  (default: http://localhost:11434/v1)
    OLLAMA_MODEL      model name    (default: qwen2.5:7b)
    MEMORY_PATH       storage root  (default: ./memory)

Environment variables (answer model — optional, any OpenAI-compatible API):
    ANSWER_BASE_URL   API base URL for recall(mode='answer')
                      e.g. https://api.anthropic.com/v1
                      Falls back to OLLAMA_BASE_URL if not set.
    ANSWER_MODEL      Model name for answers
                      e.g. claude-haiku-4-5-20251001
                      Falls back to OLLAMA_MODEL if not set.
    ANSWER_API_KEY    API key for the answer endpoint (e.g. your Anthropic key).
                      Falls back to "ollama" if not set.

Endpoints:
    POST /observe   — ingest a conversation turn (non-blocking)
    POST /recall    — query memory, returns context string or answer
    GET  /stats     — node/edge counts for a project
    GET  /health    — liveness probe
"""
from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from memory_agent import MemoryAgent, LLMConfig

app = FastAPI(title="pocket-mem", version="0.1.0")

# ---------------------------------------------------------------------------
# Agent cache — one MemoryAgent per (project, path) pair.
# The embedding model is heavy; caching avoids reloading it per request.
# ---------------------------------------------------------------------------
_agents: dict[str, MemoryAgent] = {}


def _llm_config() -> LLMConfig:
    ingest_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ingest_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    answer_base = os.environ.get("ANSWER_BASE_URL") or None
    answer_model = os.environ.get("ANSWER_MODEL") or None
    answer_key = os.environ.get("ANSWER_API_KEY") or None
    return LLMConfig(
        base_url=ingest_base,
        model=ingest_model,
        answer_base_url=answer_base,
        answer_model=answer_model,
        answer_api_key=answer_key,
    )


def _default_path() -> str:
    return os.environ.get("MEMORY_PATH", "./memory")


def _get_agent(project: str, path: str | None) -> MemoryAgent:
    resolved = path or _default_path()
    key = f"{project}:{resolved}"
    if key not in _agents:
        _agents[key] = MemoryAgent(project, path=resolved, llm=_llm_config())
    return _agents[key]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ObserveRequest(BaseModel):
    project: str
    user_input: str
    agent_response: str
    path: str | None = None


class ObserveResponse(BaseModel):
    status: str = "queued"
    project: str


class RecallRequest(BaseModel):
    project: str
    query: str
    mode: Literal["context", "answer"] = "context"
    path: str | None = None


class RecallResponse(BaseModel):
    project: str
    query: str
    mode: str
    result: str


class StatsResponse(BaseModel):
    project: str
    node_count: int
    entity_count: int
    chunk_count: int
    topic_count: int
    tone_count: int
    edge_count: int


class FlushRequest(BaseModel):
    project: str
    path: str | None = None


class FlushResponse(BaseModel):
    status: str = "ok"
    project: str


class HealthResponse(BaseModel):
    status: str = "ok"


class ConfigResponse(BaseModel):
    ingest_model: str
    answer_model: str
    ingest_base_url: str
    answer_base_url: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse()


@app.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    """Return active model configuration."""
    cfg = _llm_config()
    return ConfigResponse(
        ingest_model=cfg.model,
        answer_model=cfg.answer_model or cfg.model,
        ingest_base_url=cfg.base_url,
        answer_base_url=cfg.answer_base_url or cfg.base_url,
    )


@app.post("/flush", response_model=FlushResponse)
def flush(req: FlushRequest) -> FlushResponse:
    """Block until all background ingestion for this project is complete."""
    agent = _get_agent(req.project, req.path)
    agent.flush()
    return FlushResponse(project=req.project)


@app.post("/observe", response_model=ObserveResponse)
def observe(req: ObserveRequest) -> ObserveResponse:
    """
    Ingest a conversation turn. Returns immediately — ingestion runs
    in the background thread pool inside MemoryAgent.
    """
    agent = _get_agent(req.project, req.path)
    agent.observe(user_input=req.user_input, agent_response=req.agent_response)
    return ObserveResponse(project=req.project)


@app.post("/recall", response_model=RecallResponse)
def recall(req: RecallRequest) -> RecallResponse:
    """
    Query memory for a project. Returns a context string (mode='context')
    or a synthesized answer (mode='answer').
    """
    if req.mode not in ("context", "answer"):
        raise HTTPException(
            status_code=422,
            detail=f"mode must be 'context' or 'answer', got {req.mode!r}",
        )
    agent = _get_agent(req.project, req.path)
    result = agent.recall(req.query, mode=req.mode)
    if not isinstance(result, str):
        result = str(result)
    return RecallResponse(
        project=req.project,
        query=req.query,
        mode=req.mode,
        result=result,
    )


@app.get("/stats", response_model=StatsResponse)
def stats(project: str, path: str | None = None) -> StatsResponse:
    """Return node/edge counts for a project."""
    agent = _get_agent(project, path)
    s = agent.stats()
    return StatsResponse(project=project, **s)
