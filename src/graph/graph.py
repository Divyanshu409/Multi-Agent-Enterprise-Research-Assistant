from __future__ import annotations

import logging

import faiss
from langgraph.graph import END, StateGraph

from src.config import settings
from src.graph.nodes.critic import make_critic_node
from src.graph.nodes.finalizer import finalizer
from src.graph.nodes.planner import make_planner_node
from src.graph.nodes.retriever import make_retriever_node
from src.graph.nodes.writer import make_writer_node
from src.graph.state import GraphState
from src.providers import get_embedding_provider, get_llm_provider
from src.providers.base import EmbeddingProvider, LLMProvider

log = logging.getLogger(__name__)


def route_after_critic(state: GraphState) -> str:
    verdict = state.get("critic_verdict") or {}
    if verdict.get("verdict") == "PASS":
        return "finalizer"

    desired = verdict.get("route_to", "writer")
    revision_count = state.get("revision_count", 0)
    expansion_count = state.get("retrieval_expansion_count", 0)

    if desired == "retriever" and expansion_count < settings.max_retrieval_expansions:
        return "retriever"
    if revision_count < settings.max_writer_revisions:
        return "writer"
    # Both retry budgets exhausted -- force-finalize rather than loop forever.
    return "finalizer"


def build_graph(
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    faiss_index: faiss.Index | None = None,
    chunk_metadata: list[dict] | None = None,
):
    
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
    llm = llm or get_llm_provider(
        settings.llm_provider,
        model=_model_by_provider[settings.llm_provider],
        api_key=_key_by_provider[settings.llm_provider],
    )

    if embedder is None or faiss_index is None or chunk_metadata is None:
        from src.ingest.build_index import load_raw_documents  
        import json

        if settings.embedding_provider == "local-tfidf":
            from src.providers.embeddings import LocalTfidfEmbedding

            embedder = LocalTfidfEmbedding.load(str(settings.data_processed_dir / "tfidf_vectorizer.pkl"))
        else:
            embedder = get_embedding_provider(
                settings.embedding_provider,
                model_name=settings.sentence_transformer_model
                if settings.embedding_provider == "sentence-transformers"
                else settings.openai_embedding_model,
            )
        faiss_index = faiss.read_index(str(settings.faiss_index_path))
        chunk_metadata = [json.loads(line) for line in open(settings.chunks_path, encoding="utf-8")]

    graph = StateGraph(GraphState)
    graph.add_node("planner", make_planner_node(llm))
    graph.add_node("retriever", make_retriever_node(embedder, faiss_index, chunk_metadata))
    graph.add_node("writer", make_writer_node(llm))
    graph.add_node("critic", make_critic_node(llm))
    graph.add_node("finalizer", finalizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", "retriever": "retriever", "finalizer": "finalizer"},
    )
    graph.add_edge("finalizer", END)

    return graph.compile()


def run_query(query: str, **build_kwargs) -> GraphState:
    """Convenience entrypoint used by the API layer and eval harness."""
    app = build_graph(**build_kwargs)
    initial_state: GraphState = {
        "query": query,
        "subtasks": [],
        "retrieved_chunks": [],
        "draft_answer": "",
        "critic_verdict": None,
        "revision_count": 0,
        "retrieval_expansion_count": 0,
        "final_answer": "",
        "citations": [],
        "trace": [],
    }
    return app.invoke(initial_state)
