from __future__ import annotations

import json
import re

from src.providers.base import LLMProvider


class MockProvider(LLMProvider):

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        role_match = re.search(r"ROLE:\s*(\w+)", system)
        role = role_match.group(1) if role_match else "unknown"

        if role == "planner":
            return self._mock_plan(user)
        if role == "writer":
            return self._mock_draft(user)
        if role == "critic":
            return self._mock_critique(user)
        return "[mock] no rule for this role; returning empty completion"

    def _mock_plan(self, user: str) -> str:
        query_match = re.search(r"QUERY:\s*(.+)", user)
        query = query_match.group(1).strip() if query_match else user.strip()
        parts = re.split(r"\bvs\.?\b|\band\b|,", query, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()][:3] or [query]
        subtasks = [f"Find evidence about: {p}" for p in parts]
        return json.dumps({"subtasks": subtasks})

    def _mock_draft(self, user: str) -> str:

        chunk_ids = re.findall(r"\[(chunk_\w+)\]", user)
        cited = chunk_ids[:2] or ["chunk_unknown"]
        citations = " ".join(f"[{c}]" for c in cited)
        return (
            f"Based on the retrieved evidence, here is a draft answer {citations}. "
            f"This mock draft cites {len(cited)} of {len(chunk_ids)} retrieved chunks "
            f"to exercise the critic's coverage check."
        )

    def _mock_critique(self, user: str) -> str:

        has_citation = bool(re.search(r"\[chunk_\w+\]", user))
        verdict = "PASS" if has_citation else "FAIL"
        feedback = (
            "Draft cites retrieved evidence." if has_citation
            else "Draft contains unsupported claims with no citation markers."
        )
        next_step = "finalizer" if has_citation else "writer"
        return json.dumps({"verdict": verdict, "feedback": feedback, "next_step": next_step})

    @property
    def name(self) -> str:
        return "mock:deterministic"
