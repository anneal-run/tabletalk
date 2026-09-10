"""Test doubles. No test loads a real model."""

from typing import Any

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Returns the next scripted AIMessage on every call and records what it was asked."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []
    bound_tools: list[str] = []

    def bind_tools(self, tools, **kwargs):  # noqa: D401 - langchain signature
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("scripted model ran out of responses")
        msg = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def keyword_embedder(vocab: list[str]):
    """Unit vectors on keyword axes, so cosine similarity is just shared keywords."""

    def embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(vocab)), dtype=float)
        for i, text in enumerate(texts):
            low = text.lower()
            for j, word in enumerate(vocab):
                if word in low:
                    out[i, j] = 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out

    return embed
