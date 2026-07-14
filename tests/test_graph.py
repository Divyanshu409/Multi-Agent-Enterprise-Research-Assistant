from __future__ import annotations

import faiss
import numpy as np
import pytest

from src.graph.graph import build_graph, route_after_critic
from src.providers.embeddings import LocalTfidfEmbedding
from src.providers.mock_provider import MockProvider


@pytest.fixture
def tiny_index():
    docs = [
        {"chunk_id": "TEST_001", "doc_id": "TEST", "company": "TestCo",
         "section": "ITEM 1A. RISK FACTORS", "text": "TestCo faces significant currency risk from its European operations."},
        {"chunk_id": "TEST_002", "doc_id": "TEST", "company": "TestCo",
         "section": "ITEM 7. MD&A", "text": "TestCo revenue grew 12% year over year to $4.2 billion."},
        {"chunk_id": "TEST_003", "doc_id": "TEST", "company": "TestCo",
         "section": "ITEM 1A. RISK FACTORS", "text": "TestCo is exposed to supply chain concentration risk in Asia."},
    ]
    texts = [d["text"] for d in docs]
    embedder = LocalTfidfEmbedding()
    embedder.fit(texts)
    vectors = np.array(embedder.embed(texts), dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return embedder, index, docs


def test_graph_compiles(tiny_index):
    embedder, index, docs = tiny_index
    app = build_graph(llm=MockProvider(), embedder=embedder, faiss_index=index, chunk_metadata=docs)
    assert app is not None


def test_full_run_reaches_finalizer(tiny_index):
    embedder, index, docs = tiny_index
    app = build_graph(llm=MockProvider(), embedder=embedder, faiss_index=index, chunk_metadata=docs)
    state = app.invoke({
        "query": "What currency risk does TestCo face?",
        "subtasks": [], "retrieved_chunks": [], "draft_answer": "",
        "critic_verdict": None, "revision_count": 0, "retrieval_expansion_count": 0,
        "final_answer": "", "citations": [], "trace": [],
    })
    assert state["final_answer"] != ""
    node_sequence = [step["node"] for step in state["trace"]]
    assert node_sequence[0] == "planner"
    assert node_sequence[-1] == "finalizer"
    assert "retriever" in node_sequence
    assert "writer" in node_sequence
    assert "critic" in node_sequence


def test_revision_loop_is_bounded(tiny_index):
    embedder, index, docs = tiny_index
    app = build_graph(llm=MockProvider(), embedder=embedder, faiss_index=index, chunk_metadata=docs)
    state = app.invoke({
        "query": "Summarize all risks and financial results for TestCo",
        "subtasks": [], "retrieved_chunks": [], "draft_answer": "",
        "critic_verdict": None, "revision_count": 0, "retrieval_expansion_count": 0,
        "final_answer": "", "citations": [], "trace": [],
    })
    from src.config import settings
    assert state["revision_count"] <= settings.max_writer_revisions
    writer_calls = sum(1 for s in state["trace"] if s["node"] == "writer")
    assert writer_calls == state["revision_count"]
    assert state["trace"][-1]["node"] == "finalizer"


def test_route_after_critic_pass_goes_to_finalizer():
    state = {"critic_verdict": {"verdict": "PASS", "route_to": "finalizer"},
             "revision_count": 0, "retrieval_expansion_count": 0}
    assert route_after_critic(state) == "finalizer"


def test_route_after_critic_respects_revision_cap():
    from src.config import settings
    state = {"critic_verdict": {"verdict": "FAIL", "route_to": "writer"},
             "revision_count": settings.max_writer_revisions, "retrieval_expansion_count": 0}
    assert route_after_critic(state) == "finalizer"


def test_route_after_critic_can_request_more_retrieval():
    state = {"critic_verdict": {"verdict": "FAIL", "route_to": "retriever"},
             "revision_count": 0, "retrieval_expansion_count": 0}
    assert route_after_critic(state) == "retriever"


def test_route_after_critic_respects_expansion_cap():
    from src.config import settings
    state = {"critic_verdict": {"verdict": "FAIL", "route_to": "retriever"},
             "revision_count": 0, "retrieval_expansion_count": settings.max_retrieval_expansions}
    assert route_after_critic(state) in ("writer", "finalizer")
