from __future__ import annotations

import json
import logging
import time
from typing import Callable

from src.config import settings
from src.graph.state import GraphState
from src.providers.base import LLMProvider

log = logging.getLogger(__name__)

_PROMPT = (settings.prompts_dir / "planner.txt").read_text(encoding="utf-8")


def _parse_subtasks(raw: str, fallback_query: str) -> list[str]:
    try:
        data = json.loads(raw)
        subtasks = data.get("subtasks", [])
        if subtasks:
            return [str(s) for s in subtasks][:4]
    except (json.JSONDecodeError, AttributeError):
        log.warning("planner: could not parse JSON from LLM output, falling back to raw query")
    return [fallback_query]


def make_planner_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    def planner(state: GraphState) -> dict:
        t0 = time.time()
        user_prompt = f"QUERY: {state['query']}"
        raw = llm.complete(system=_PROMPT, user=user_prompt, max_tokens=400)
        subtasks = _parse_subtasks(raw, fallback_query=state["query"])

        return {
            "subtasks": subtasks,
            "trace": [{
                "node": "planner",
                "duration_s": round(time.time() - t0, 3),
                "output_summary": f"{len(subtasks)} subtask(s): {subtasks}",
                "llm": llm.name,
            }],
        }

    return planner
