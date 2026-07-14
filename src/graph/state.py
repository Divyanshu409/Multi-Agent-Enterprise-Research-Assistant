from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class RetrievedChunk(TypedDict):
    chunk_id: str
    doc_id: str
    company: str
    section: str
    text: str
    score: float
    subtask: str  


class CriticVerdict(TypedDict):
    verdict: str          
    feedback: str
    route_to: str          


class GraphState(TypedDict):
    # --- input --------------------------------------------------------------
    query: str

    # --- planner output -------------------------------------------------------
    subtasks: list[str]

    # --- retriever output -----------------------------------------------------
    retrieved_chunks: list[RetrievedChunk]

    # --- writer output ---------------------------------------------------------
    draft_answer: str

    # --- critic output ---------------------------------------------------------
    critic_verdict: CriticVerdict | None

    # --- loop control -----------------------------------------------------------
    revision_count: int         
    retrieval_expansion_count: int  

    # --- finalizer output --------------------------------------------------------
    final_answer: str
    citations: list[str]         

    # --- observability ------------------------------------------------------------
    trace: Annotated[list[dict], operator.add]
