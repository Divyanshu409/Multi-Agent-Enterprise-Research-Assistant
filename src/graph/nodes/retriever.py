from __future__ import annotations

import logging
import time
from typing import Callable

import faiss
import numpy as np

from src.config import settings
from src.graph.state import GraphState, RetrievedChunk
from src.providers.base import EmbeddingProvider

log = logging.getLogger(__name__)


def make_retriever_node(
    embedder: EmbeddingProvider,
    index: faiss.Index,
    chunk_metadata: list[dict],
    top_k: int | None = None,
) -> Callable[[GraphState], dict]:
    k = top_k or settings.top_k_per_subtask

    def retriever(state: GraphState) -> dict:
        t0 = time.time()
        subtasks = state.get("subtasks") or [state["query"]]

        
        best_by_id: dict[str, RetrievedChunk] = {}
        for subtask in subtasks:
            query_vec = np.array(embedder.embed([subtask]), dtype=np.float32)
            scores, ids = index.search(query_vec, k)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(chunk_metadata):
                    continue
                meta = chunk_metadata[idx]
                chunk_id = meta["chunk_id"]
                candidate: RetrievedChunk = {
                    "chunk_id": chunk_id,
                    "doc_id": meta["doc_id"],
                    "company": meta["company"],
                    "section": meta["section"],
                    "text": meta["text"],
                    "score": float(score),
                    "subtask": subtask,
                }
                existing = best_by_id.get(chunk_id)
                if existing is None or candidate["score"] > existing["score"]:
                    best_by_id[chunk_id] = candidate

        retrieved = sorted(best_by_id.values(), key=lambda c: c["score"], reverse=True)
        
        is_expansion = bool(state.get("retrieved_chunks"))
        expansion_count = state.get("retrieval_expansion_count", 0) + (1 if is_expansion else 0)

        return {
            "retrieved_chunks": retrieved,
            "retrieval_expansion_count": expansion_count,
            "trace": [{
                "node": "retriever",
                "duration_s": round(time.time() - t0, 3),
                "output_summary": f"{len(retrieved)} unique chunk(s) across {len(subtasks)} subtask(s)",
                "embedder": embedder.name,
            }],
        }

    return retriever
