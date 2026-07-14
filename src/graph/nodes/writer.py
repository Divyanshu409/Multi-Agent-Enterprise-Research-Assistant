from __future__ import annotations

import time
from typing import Callable

from src.config import settings
from src.graph.state import GraphState
from src.providers.base import LLMProvider

_PROMPT = (settings.prompts_dir / "writer.txt").read_text(encoding="utf-8")


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[chunk_{c['chunk_id']}] ({c['company']} -- {c['section']})\n{c['text']}")
    return "\n\n".join(blocks)


def make_writer_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    def writer(state: GraphState) -> dict:
        t0 = time.time()
        context = _format_context(state.get("retrieved_chunks", []))
        critic_verdict = state.get("critic_verdict")
        feedback_block = ""
        if critic_verdict and critic_verdict.get("feedback"):
            feedback_block = (
                f"\n\nPREVIOUS CRITIC FEEDBACK (address every point in this revision):\n"
                f"{critic_verdict['feedback']}"
            )

        user_prompt = (
            f"QUESTION: {state['query']}\n\n"
            f"CONTEXT (retrieved evidence chunks):\n{context}"
            f"{feedback_block}"
        )
        draft = llm.complete(system=_PROMPT, user=user_prompt, max_tokens=3000)

        revision_count = state.get("revision_count", 0)
        return {
            "draft_answer": draft,
            "revision_count": revision_count + 1,
            "trace": [{
                "node": "writer",
                "duration_s": round(time.time() - t0, 3),
                "output_summary": f"draft v{revision_count + 1}, {len(draft)} chars",
                "llm": llm.name,
            }],
        }

    return writer

