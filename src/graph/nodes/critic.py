from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable

from src.config import settings
from src.graph.state import GraphState
from src.providers.base import LLMProvider

log = logging.getLogger(__name__)

_PROMPT = (settings.prompts_dir / "critic.txt").read_text(encoding="utf-8")
_CITATION_RE = re.compile(r"\[?chunk_[\w\-]+\]?")


def _citation_coverage(draft: str) -> float:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _CITATION_RE.search(s))
    return cited / len(sentences)


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[chunk_{c['chunk_id']}] ({c['company']} -- {c['section']})\n{c['text']}")
    return "\n\n".join(blocks)


def make_critic_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    def critic(state: GraphState) -> dict:
        t0 = time.time()
        draft = state.get("draft_answer", "")
        context = _format_context(state.get("retrieved_chunks", []))
        coverage = _citation_coverage(draft)

        user_prompt = (
            f"QUESTION: {state['query']}\n\n"
            f"EVIDENCE CHUNKS:\n{context}\n\n"
            f"DRAFT ANSWER TO GRADE:\n{draft}"
        )
        raw = llm.complete(system=_PROMPT, user=user_prompt, max_tokens=800)

        try:
            data = json.loads(raw)
            verdict = data.get("verdict", "FAIL").upper()
            feedback = data.get("feedback", "")
            next_step = data.get("next_step", "writer")
        except json.JSONDecodeError:
            log.warning("critic: could not parse LLM verdict JSON, defaulting to FAIL")
            verdict = "FAIL"
            feedback = "Critic response was not valid JSON; treating as failed verification."
            next_step = "writer"

        if verdict == "PASS" and coverage < settings.min_citation_coverage:
            verdict = "FAIL"
            next_step = "writer"
            feedback = (
                f"{feedback} [Automated check] Only {coverage:.0%} of sentences carry a "
                f"citation marker; minimum required is {settings.min_citation_coverage:.0%}."
            ).strip()

        result = {"verdict": verdict, "feedback": feedback, "route_to": next_step}
        return {
            "critic_verdict": result,
            "trace": [{
                "node": "critic",
                "duration_s": round(time.time() - t0, 3),
                "output_summary": f"verdict={verdict}, citation_coverage={coverage:.0%}",
                "llm": llm.name,
            }],
        }

    return critic

