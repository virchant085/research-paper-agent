"""Universal LLM client, provider-agnostic via LiteLLM.

One interface — ``chat`` (with optional tool calling), ``structured`` (JSON
object generation) and ``embed`` (batch embeddings) — routed through
`LiteLLM <https://docs.litellm.ai/>`_ so the same code drives OpenAI, Anthropic
Claude, Google Gemini, Mistral, Groq, DeepSeek, Cohere, Azure, local Ollama, and
100+ other providers. Switch models by changing ``settings.llm_model`` /
``settings.embed_model``; provider API keys are read from the environment by
LiteLLM under their standard names.

Design rules:
- The ``litellm`` module is imported and configured **lazily** (on first use), so
  importing this module never requires any provider key.
- A process-wide singleton is returned by :func:`get_llm`.

Message / tool formats follow the OpenAI conventions:

- ``messages`` items look like ``{"role": "user"|"assistant"|"system"|"tool", ...}``.
- ``tools`` look like
  ``[{"type": "function", "function": {"name", "description", "parameters": <json-schema>}}]``.
- An assistant tool-call message carries a ``tool_calls`` list; a tool result
  message looks like ``{"role": "tool", "tool_call_id": <id>, "content": <str>}``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from backend.config import settings


# --------------------------------------------------------------------------- #
# Normalised result types
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """A single function/tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    """The normalised result of a chat completion.

    ``content`` is the assistant's text answer (``None`` when the model chose to
    call tools instead). ``tool_calls`` holds any requested tool invocations.
    """

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# JSON parsing helpers
# --------------------------------------------------------------------------- #
def _strip_code_fences(text: str) -> str:
    """Strip a leading/trailing Markdown code fence (```json ... ```)."""
    s = (text or "").strip()
    if s.startswith("```"):
        newline = s.find("\n")
        s = s[newline + 1 :] if newline != -1 else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _loads_lenient(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating code fences.

    Falls back to extracting the first ``{...}`` block if the raw string is not
    directly parseable.
    """
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


# --------------------------------------------------------------------------- #
# LiteLLM-backed universal client
# --------------------------------------------------------------------------- #
class LLMClient:
    """Provider-agnostic client backed by LiteLLM.

    All model selection flows through ``settings.llm_model`` /
    ``settings.embed_model``; provider credentials are read from the environment
    by LiteLLM.
    """

    def __init__(self) -> None:
        self._litellm: Any | None = None

    def _get(self) -> Any:
        """Lazily import and configure the ``litellm`` module on first use."""
        if self._litellm is None:
            import litellm

            # Silently drop params a given model doesn't support (e.g. some models
            # reject ``response_format`` or ``temperature``) instead of erroring.
            litellm.drop_params = True
            litellm.telemetry = False
            self._litellm = litellm
        return self._litellm

    # -- chat (with optional tool calling) ---------------------------------- #
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> ChatResult:
        litellm = self._get()
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = litellm.completion(**kwargs)
        message = response.choices[0].message

        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
            )

        return ChatResult(content=getattr(message, "content", None), tool_calls=tool_calls)

    # -- structured JSON ----------------------------------------------------- #
    def structured(self, system: str, user: str, schema: dict) -> dict:
        litellm = self._get()
        prompt = (
            f"{user}\n\n"
            "Respond with a single JSON object that conforms to this JSON schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n"
            "Output only the JSON object, no prose."
        )
        response = litellm.completion(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            # Dropped automatically by LiteLLM for models that don't support it.
            response_format={"type": "json_object"},
        )
        return _loads_lenient(response.choices[0].message.content or "{}")

    # -- embeddings ---------------------------------------------------------- #
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        litellm = self._get()
        response = litellm.embedding(model=settings.embed_model, input=texts)

        # LiteLLM returns an OpenAI-style object/dict: {"data": [{"embedding", "index"}]}.
        data = response["data"] if isinstance(response, dict) else response.data

        def _index(item: Any) -> int:
            return item["index"] if isinstance(item, dict) else getattr(item, "index", 0)

        def _vector(item: Any) -> list[float]:
            emb = item["embedding"] if isinstance(item, dict) else item.embedding
            return list(emb)

        return [_vector(item) for item in sorted(data, key=_index)]


# --------------------------------------------------------------------------- #
# Singleton factory
# --------------------------------------------------------------------------- #
_CLIENT: LLMClient | None = None


def get_llm() -> LLMClient:
    """Return the process-wide universal LLM client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT
