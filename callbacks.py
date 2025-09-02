from multiprocessing.pool import MapResult
from typing import Any, Optional
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for agent actions and tool calls."""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        print("LLM starting with prompts:", prompts)

    def on_llm_end(
        self,
        response: MapResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        print("LLM finished with response:", response)
