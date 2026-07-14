from __future__ import annotations

import json

from src.graph.nodes.critic import _citation_coverage
from src.graph.nodes.finalizer import finalizer
from src.graph.nodes.planner import _parse_subtasks
from src.providers.mock_provider import MockProvider


def test_parse_subtasks_valid_json():
    raw = json.dumps({"subtasks": ["a", "b"]})
    assert _parse_subtasks(raw, fallback_query="fallback") == ["a", "b"]


def test_parse_subtasks_falls_back_on_bad_json():
    assert _parse_subtasks("not json", fallback_query="fallback query") == ["fallback query"]


def test_parse_subtasks_caps_at_four():
    raw = json.dumps({"subtasks": ["a", "b", "c", "d", "e", "f"]})
    assert len(_parse_subtasks(raw, fallback_query="x")) == 4


def test_citation_coverage_all_cited():
    draft = "Revenue grew 10% [chunk_A_001]. Costs also rose [chunk_A_002]."
    assert _citation_coverage(draft) == 1.0


def test_citation_coverage_partial():
    draft = "Revenue grew 10%. [chunk_A_001] This sentence has no citation."
    assert _citation_coverage(draft) == 0.5


def test_citation_coverage_empty_draft():
    assert _citation_coverage("") == 0.0


def test_finalizer_extracts_unique_citations():
    state = {
        "draft_answer": "Fact one [chunk_A_001]. Fact two [chunk_A_002]. Fact one again [chunk_A_001].",
        "critic_verdict": {"verdict": "PASS", "feedback": ""},
    }
    result = finalizer(state)
    assert result["citations"] == ["A_001", "A_002"]
    assert "force-finalized" not in result["final_answer"]


def test_finalizer_adds_caveat_when_forced():
    state = {
        "draft_answer": "Some claim [chunk_A_001].",
        "critic_verdict": {"verdict": "FAIL", "feedback": "still missing something"},
    }
    result = finalizer(state)
    assert "force-finalized" in result["final_answer"]
    assert "still missing something" in result["final_answer"]


def test_mock_provider_planner_role():
    llm = MockProvider()
    system = "ROLE: planner\nsome instructions"
    out = llm.complete(system=system, user="QUERY: compare Apple and Microsoft AI risk")
    data = json.loads(out)
    assert "subtasks" in data
    assert 1 <= len(data["subtasks"]) <= 3


def test_mock_provider_unknown_role():
    llm = MockProvider()
    out = llm.complete(system="ROLE: nonsense", user="anything")
    assert "no rule" in out
