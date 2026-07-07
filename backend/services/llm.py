"""Provider-agnostic LLM client normalised to an OpenAI-style chat interface.

This module hides the difference between OpenAI and Google Gemini behind a small
common surface: ``chat`` (with optional tool calling), ``structured`` (JSON
object generation) and ``embed`` (batch embeddings).

Design rules:
- SDK/client construction is **lazy** (built on first use), so importing this
  module never requires API keys.
- The correct provider is chosen by ``settings.llm_provider`` and cached as a
  process-wide singleton via :func:`get_llm`.

Message / tool formats follow the OpenAI conventions:

- ``messages`` items look like ``{"role": "user"|"assistant"|"system"|"tool", ...}``.
- ``tools`` look like
  ``[{"type": "function", "function": {"name", "description", "parameters": <json-schema>}}]``.
- An assistant tool-call message looks like::

      {"role": "assistant", "content": None,
       "tool_calls": [{"id", "type": "function",
                       "function": {"name", "arguments": <json-string>}}]}

- A tool result message looks like
  ``{"role": "tool", "tool_call_id": <id>, "content": <str>}``.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
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
# JSON parsing helper
# --------------------------------------------------------------------------- #
def _strip_code_fences(text: str) -> str:
    """Strip a leading/trailing Markdown code fence (```json ... ```)."""
    s = (text or "").strip()
    if s.startswith("```"):
        # Drop the opening fence line (``` or ```json etc).
        newline = s.find("\n")
        if newline != -1:
            s = s[newline + 1 :]
        else:
            s = s[3:]
        # Drop a trailing closing fence.
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
        # Best-effort: grab the outermost brace-delimited object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


# --------------------------------------------------------------------------- #
# Abstract client
# --------------------------------------------------------------------------- #
class LLMClient(ABC):
    """Common interface every provider adapter implements."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> ChatResult:
        """Run a chat completion and return a normalised :class:`ChatResult`."""
        raise NotImplementedError

    @abstractmethod
    def structured(self, system: str, user: str, schema: dict) -> dict:
        """Return a JSON object matching ``schema`` (schema is embedded in the prompt)."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# OpenAI adapter
# --------------------------------------------------------------------------- #
class OpenAIClient(LLMClient):
    """Adapter over the official ``openai`` SDK (>=1.0 style client)."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazily build the OpenAI SDK client on first use."""
        if self._client is None:
            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Configure openai_api_key in your "
                    "environment/.env to use the OpenAI provider."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> ChatResult:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": settings.openai_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
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

        return ChatResult(content=message.content, tool_calls=tool_calls)

    def structured(self, system: str, user: str, schema: dict) -> dict:
        client = self._get_client()
        prompt = (
            f"{user}\n\n"
            "Respond with a single JSON object that conforms to this JSON schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n"
            "Output only the JSON object, no prose."
        )
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return _loads_lenient(response.choices[0].message.content or "{}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = client.embeddings.create(
            model=settings.openai_embed_model,
            input=texts,
        )
        # Preserve input order.
        items = sorted(response.data, key=lambda d: d.index)
        return [list(item.embedding) for item in items]


# --------------------------------------------------------------------------- #
# Gemini adapter
# --------------------------------------------------------------------------- #
class GeminiClient(LLMClient):
    """Adapter over Google's ``google-generativeai`` SDK."""

    def __init__(self) -> None:
        self._genai: Any | None = None

    def _get_genai(self) -> Any:
        """Lazily configure and return the ``google.generativeai`` module."""
        if self._genai is None:
            if not settings.gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Configure gemini_api_key in your "
                    "environment/.env to use the Gemini provider."
                )
            import google.generativeai as genai

            genai.configure(api_key=settings.gemini_api_key)
            self._genai = genai
        return self._genai

    # -- translation helpers ------------------------------------------------- #
    @staticmethod
    def _extract_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Pull any system messages out into a single system_instruction string."""
        system_parts: list[str] = []
        rest: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content")
                if content:
                    system_parts.append(str(content))
            else:
                rest.append(msg)
        system = "\n\n".join(system_parts) if system_parts else None
        return system, rest

    def _build_history(self, messages: list[dict]) -> list[dict]:
        """Translate OpenAI-style messages into Gemini ``contents`` history.

        Role mapping: ``assistant`` -> ``model``; ``tool`` results become a
        ``function_response`` part; assistant ``tool_calls`` become
        ``function_call`` parts.
        """
        history: list[dict] = []
        # Map tool_call_id -> function name so tool results can name their call.
        call_id_to_name: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")

            if role == "user":
                history.append(
                    {"role": "user", "parts": [{"text": str(msg.get("content") or "")}]}
                )

            elif role == "assistant":
                parts: list[dict] = []
                content = msg.get("content")
                if content:
                    parts.append({"text": str(content)})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    call_id_to_name[tc.get("id", "")] = name
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    parts.append({"function_call": {"name": name, "args": args}})
                if not parts:
                    parts.append({"text": ""})
                history.append({"role": "model", "parts": parts})

            elif role == "tool":
                call_id = msg.get("tool_call_id", "")
                name = call_id_to_name.get(call_id, msg.get("name", "tool"))
                content = msg.get("content")
                history.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": name,
                                    "response": {"result": str(content or "")},
                                }
                            }
                        ],
                    }
                )

        return history

    @staticmethod
    def _translate_tools(tools: list[dict] | None) -> list[dict] | None:
        """Translate OpenAI tool schemas into Gemini FunctionDeclarations."""
        if not tools:
            return None
        declarations: list[dict] = []
        for tool in tools:
            fn = tool.get("function", {})
            declarations.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return [{"function_declarations": declarations}]

    # -- interface ----------------------------------------------------------- #
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> ChatResult:
        genai = self._get_genai()
        system, rest = self._extract_system(messages)
        history = self._build_history(rest)
        gemini_tools = self._translate_tools(tools)

        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=system,
            tools=gemini_tools,
        )
        response = model.generate_content(
            history,
            generation_config={"temperature": temperature},
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        idx = 0
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                fn_call = getattr(part, "function_call", None)
                if fn_call and getattr(fn_call, "name", None):
                    # ``args`` behaves like a mapping; normalise to a plain dict.
                    raw_args = getattr(fn_call, "args", None)
                    arguments = dict(raw_args) if raw_args else {}
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{idx}",
                            name=fn_call.name,
                            arguments=arguments,
                        )
                    )
                    idx += 1
                else:
                    text = getattr(part, "text", None)
                    if text:
                        text_parts.append(text)

        content_str: str | None = "".join(text_parts) if text_parts else None
        return ChatResult(content=content_str, tool_calls=tool_calls)

    def structured(self, system: str, user: str, schema: dict) -> dict:
        genai = self._get_genai()
        prompt = (
            f"{user}\n\n"
            "Respond with a single JSON object that conforms to this JSON schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n"
            "Output only the JSON object, no prose."
        )
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=system,
        )
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        return _loads_lenient(getattr(response, "text", "") or "{}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        genai = self._get_genai()
        result = genai.embed_content(
            model=settings.gemini_embed_model,
            content=texts,
        )
        embeddings = result["embedding"] if isinstance(result, dict) else result.embedding

        # ``embed_content`` returns a list-of-vectors for a list input, but for a
        # single string it returns one flat vector; normalise to list-of-vectors.
        if embeddings and isinstance(embeddings[0], (int, float)):
            return [list(embeddings)]
        return [list(vec) for vec in embeddings]


# --------------------------------------------------------------------------- #
# Singleton factory
# --------------------------------------------------------------------------- #
_CLIENT: LLMClient | None = None


def get_llm() -> LLMClient:
    """Return the process-wide LLM client chosen by ``settings.llm_provider``."""
    global _CLIENT
    if _CLIENT is None:
        provider = (settings.llm_provider or "gemini").strip().lower()
        if provider == "openai":
            _CLIENT = OpenAIClient()
        elif provider == "gemini":
            _CLIENT = GeminiClient()
        else:
            raise RuntimeError(
                f"Unknown llm_provider {provider!r}; expected 'gemini' or 'openai'."
            )
    return _CLIENT
