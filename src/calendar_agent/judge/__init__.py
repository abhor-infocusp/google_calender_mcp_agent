"""Centralized judge: prompts, router, server, client.

The 14B fp8 + router judge that scored 95.44% on the manual oracle (2026-05-01).
Importable from RL training, eval scripts, and the FastAPI service.
"""

from calendar_agent.judge.prompts import (
    PROMPT_VERSION,
    build_router,
    extract_verdict,
    diff_states,
)

__all__ = ["PROMPT_VERSION", "build_router", "extract_verdict", "diff_states"]
