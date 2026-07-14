from __future__ import annotations

import re
import time

from src.graph.state import GraphState

_CITATION_RE = re.compile(r"\[?chunk_([\w\-]+)\]?")


def finalizer(state: GraphState) -> dict:
    t0 = time.time()
    draft = state.get("draft_answer", "")
    verdict = state.get("critic_verdict") or {}

    citations = sorted(set(_CITATION_RE.findall(draft)))

    forced = verdict.get("verdict") != "PASS"
    caveat = ""
    if forced:
        caveat = (
            "\n\n[Note: this answer was force-finalized after reaching the maximum "
            "revision limit without the critic fully passing it. Remaining critic "
            f"feedback: {verdict.get('feedback', 'n/a')}]"
        )

    final_answer = draft + caveat

    return {
        "final_answer": final_answer,
        "citations": citations,
        "trace": [{
            "node": "finalizer",
            "duration_s": round(time.time() - t0, 3),
            "output_summary": f"{len(citations)} unique citation(s), force_finalized={forced}",
        }],
    }
