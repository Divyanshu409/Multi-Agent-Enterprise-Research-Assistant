from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The research question to answer over the corpus.")


class TraceStep(BaseModel):
    node: str
    duration_s: float
    output_summary: str
    llm: str | None = None
    embedder: str | None = None


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    citations: list[str]
    subtasks: list[str]
    revision_count: int
    retrieval_expansion_count: int
    nodes_run: list[str]
    trace: list[TraceStep]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    index_size: int
