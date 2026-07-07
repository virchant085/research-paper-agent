"""Tests for backend.services.llm.

Covers the pure JSON helpers, the singleton accessor, and the real
``LLMClient`` mapping by monkeypatching the underlying ``litellm`` module.
"""
from __future__ import annotations

import litellm

import backend.services.llm as llm_mod
from backend.services.llm import (
    ChatResult,
    LLMClient,
    ToolCall,
    _loads_lenient,
    _strip_code_fences,
    get_llm,
)


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def test_strip_code_fences_json_block():
    fenced = '```json\n{"a": 1}\n```'
    assert _strip_code_fences(fenced) == '{"a": 1}'


def test_strip_code_fences_plain_fence():
    fenced = "```\nhello\n```"
    assert _strip_code_fences(fenced) == "hello"


def test_strip_code_fences_no_fence():
    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_loads_lenient_fenced_json():
    assert _loads_lenient('```json\n{"x": 10}\n```') == {"x": 10}


def test_loads_lenient_bare_json():
    assert _loads_lenient('{"y": 20}') == {"y": 20}


def test_loads_lenient_embedded_object():
    text = 'Here is your object: {"z": 30} thanks!'
    assert _loads_lenient(text) == {"z": 30}


# --------------------------------------------------------------------------- #
# Singleton
# --------------------------------------------------------------------------- #
def test_get_llm_is_singleton():
    a = get_llm()
    b = get_llm()
    assert a is b


# --------------------------------------------------------------------------- #
# Real LLMClient mapping via litellm monkeypatch
# --------------------------------------------------------------------------- #
class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeEmbeddingItem:
    def __init__(self, embedding, index):
        self.embedding = embedding
        self.index = index


class _FakeEmbeddingResponse:
    def __init__(self, data):
        self.data = data


def test_chat_maps_content_and_tool_calls(monkeypatch):
    client = LLMClient()

    message = _FakeMessage(
        content="hello there",
        tool_calls=[
            _FakeToolCall("call_1", "search_chunks", '{"query": "x", "k": 3}'),
        ],
    )

    def fake_completion(**kwargs):
        # sanity: the client passed our messages/model through
        assert kwargs["messages"]
        assert "model" in kwargs
        return _FakeResponse(message)

    monkeypatch.setattr(litellm, "completion", fake_completion)

    result = client.chat([{"role": "user", "content": "hi"}], tools=[{"x": 1}])
    assert isinstance(result, ChatResult)
    assert result.content == "hello there"
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "call_1"
    assert tc.name == "search_chunks"
    # arguments parsed from the JSON string into a dict
    assert tc.arguments == {"query": "x", "k": 3}


def test_chat_handles_no_tool_calls(monkeypatch):
    client = LLMClient()
    message = _FakeMessage(content="just text", tool_calls=None)
    monkeypatch.setattr(litellm, "completion", lambda **kw: _FakeResponse(message))

    result = client.chat([{"role": "user", "content": "hi"}])
    assert result.content == "just text"
    assert result.tool_calls == []


def test_chat_bad_tool_arguments_become_empty_dict(monkeypatch):
    client = LLMClient()
    message = _FakeMessage(
        content=None,
        tool_calls=[_FakeToolCall("c1", "export", "not-json{{")],
    )
    monkeypatch.setattr(litellm, "completion", lambda **kw: _FakeResponse(message))

    result = client.chat([{"role": "user", "content": "hi"}], tools=[{"x": 1}])
    assert result.content is None
    assert result.tool_calls[0].arguments == {}


def test_embed_returns_vectors_in_index_order(monkeypatch):
    client = LLMClient()

    # Provided out of index order; embed() must sort by .index.
    data = [
        _FakeEmbeddingItem([0.3, 0.3], index=2),
        _FakeEmbeddingItem([0.1, 0.1], index=0),
        _FakeEmbeddingItem([0.2, 0.2], index=1),
    ]

    def fake_embedding(**kwargs):
        assert kwargs["input"] == ["a", "b", "c"]
        return _FakeEmbeddingResponse(data)

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    vectors = client.embed(["a", "b", "c"])
    assert vectors == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]


def test_embed_empty_input_short_circuits(monkeypatch):
    client = LLMClient()

    def boom(**kwargs):  # pragma: no cover - must not be called
        raise AssertionError("litellm.embedding should not be called for empty input")

    monkeypatch.setattr(litellm, "embedding", boom)
    assert client.embed([]) == []


def test_structured_parses_lenient_json(monkeypatch):
    client = LLMClient()
    message = _FakeMessage(content='```json\n{"title": "T", "year": 2020}\n```')
    monkeypatch.setattr(litellm, "completion", lambda **kw: _FakeResponse(message))

    out = client.structured("sys", "user", {"type": "object"})
    assert out == {"title": "T", "year": 2020}
