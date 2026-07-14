from __future__ import annotations

import json
import logging
import time

import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import HealthResponse, QueryRequest, QueryResponse, TraceStep
from src.config import settings
from src.graph.graph import build_graph
from src.providers import get_embedding_provider, get_llm_provider
from src.providers.embeddings import LocalTfidfEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(
    title="Multi-Agent Enterprise Research Assistant",
    description="Plans, retrieves, drafts, and self-verifies cited answers over a SEC-filing corpus.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {}


@app.on_event("startup")
def load_dependencies() -> None:
    log.info("Loading LLM provider=%s, embedding provider=%s", settings.llm_provider, settings.embedding_provider)

    _model_by_provider = {
        "anthropic": settings.anthropic_model,
        "openai": settings.openai_model,
        "gemini": settings.gemini_model,
        "mock": "mock",
    }
    _key_by_provider = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "gemini": settings.gemini_api_key,
        "mock": None,
    }
    _state["llm"] = get_llm_provider(
        settings.llm_provider,
        model=_model_by_provider[settings.llm_provider],
        api_key=_key_by_provider[settings.llm_provider],
    )

    if not settings.faiss_index_path.exists():
        raise RuntimeError(
            f"No FAISS index found at {settings.faiss_index_path}. "
            "Run `python -m src.ingest.build_index` first."
        )

    if settings.embedding_provider == "local-tfidf":
        _state["embedder"] = LocalTfidfEmbedding.load(str(settings.data_processed_dir / "tfidf_vectorizer.pkl"))
    else:
        _state["embedder"] = get_embedding_provider(
            settings.embedding_provider,
            model_name=settings.sentence_transformer_model
            if settings.embedding_provider == "sentence-transformers"
            else settings.openai_embedding_model,
        )

    _state["index"] = faiss.read_index(str(settings.faiss_index_path))
    _state["chunk_metadata"] = [json.loads(line) for line in open(settings.chunks_path, encoding="utf-8")]
    _state["graph"] = build_graph(
        llm=_state["llm"], embedder=_state["embedder"],
        faiss_index=_state["index"], chunk_metadata=_state["chunk_metadata"],
    )

    try:
        import os
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        _state["mlflow_enabled"] = True
    except Exception as e:  # pragma: no cover
        log.warning("MLflow unavailable, run tracking disabled: %s", e)
        _state["mlflow_enabled"] = False

    log.info("Startup complete. Index has %d chunks.", _state["index"].ntotal)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if "index" not in _state:
        raise HTTPException(status_code=503, detail="Service still starting up.")
    return HealthResponse(
        status="ok",
        llm_provider=_state["llm"].name,
        embedding_provider=_state["embedder"].name,
        index_size=_state["index"].ntotal,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if "graph" not in _state:
        raise HTTPException(status_code=503, detail="Service still starting up.")

    t0 = time.time()
    try:
        result = _state["graph"].invoke({
            "query": request.query,
            "subtasks": [], "retrieved_chunks": [], "draft_answer": "",
            "critic_verdict": None, "revision_count": 0, "retrieval_expansion_count": 0,
            "final_answer": "", "citations": [], "trace": [],
        })
    except Exception as e:
        log.exception("Graph execution failed")
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {e}") from e

    latency = time.time() - t0
    nodes_run = [step["node"] for step in result["trace"]]

    if _state.get("mlflow_enabled"):
        try:
            import mlflow
            with mlflow.start_run(run_name=f"query-{int(t0)}"):
                mlflow.log_param("query", request.query[:250])
                mlflow.log_param("llm_provider", _state["llm"].name)
                mlflow.log_param("embedding_provider", _state["embedder"].name)
                mlflow.log_metric("latency_s", latency)
                mlflow.log_metric("revision_count", result["revision_count"])
                mlflow.log_metric("retrieval_expansion_count", result["retrieval_expansion_count"])
                mlflow.log_metric("citation_count", len(result["citations"]))
                mlflow.log_metric("nodes_run_count", len(nodes_run))
                mlflow.log_text(result["final_answer"], "final_answer.txt")
        except Exception:  
            log.exception("MLflow logging failed (non-fatal, continuing)")

    return QueryResponse(
        query=request.query,
        final_answer=result["final_answer"],
        citations=result["citations"],
        subtasks=result["subtasks"],
        revision_count=result["revision_count"],
        retrieval_expansion_count=result["retrieval_expansion_count"],
        nodes_run=nodes_run,
        trace=[TraceStep(**step) for step in result["trace"]],
    )
