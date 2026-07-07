"""Tests for the tool-calling agent loop (backend.core.agent)."""
from __future__ import annotations

from backend.config import settings
from backend.core.agent import run_agent
from backend.models.schemas import QueryResponse
from backend.services import db
from backend.services.llm import ChatResult, ToolCall


def _card(paper_id="p1"):
    from backend.models.schemas import PaperCard

    return PaperCard(paper_id=paper_id, title=f"Title {paper_id}", year=2021)


def test_run_agent_executes_tool_then_answers(fake_llm):
    db.save_card(_card("p1"))

    # First turn: model asks to call generate_lit_table (no args).
    # Second turn: model returns a plain final answer.
    fake_llm.chat_script = [
        ChatResult(
            content=None,
            tool_calls=[ToolCall(id="c1", name="generate_lit_table", arguments={})],
        ),
        ChatResult(content="the answer", tool_calls=[]),
    ]

    resp = run_agent("give me a table")
    assert isinstance(resp, QueryResponse)
    assert resp.answer == "the answer"

    # Exactly one recorded step, and it is the generate_lit_table call.
    assert len(resp.steps) == 1
    assert resp.steps[0].tool == "generate_lit_table"
    assert resp.steps[0].arguments == {}
    # The tool actually ran and produced a preview from the real table.
    assert "Title p1" in resp.steps[0].result_preview


def test_run_agent_immediate_answer(fake_llm):
    fake_llm.chat_script = [ChatResult(content="direct answer", tool_calls=[])]
    resp = run_agent("hi")
    assert resp.answer == "direct answer"
    assert resp.steps == []


def test_run_agent_step_budget_does_not_raise(fake_llm, monkeypatch):
    # Keep the budget small so the test is quick.
    monkeypatch.setattr(settings, "agent_max_steps", 3)

    # Script always returns a tool call, so the loop hits the step budget and
    # then makes a final tools-disabled call (empty script => "final answer").
    always_tool = ChatResult(
        content=None,
        tool_calls=[ToolCall(id="c", name="generate_lit_table", arguments={})],
    )
    # Provide exactly agent_max_steps scripted tool-call turns; after that the
    # script is empty so the final forced call returns the default "final answer".
    fake_llm.chat_script = [always_tool, always_tool, always_tool]

    resp = run_agent("loop please")
    # Must return without raising; the forced final answer is the fallback.
    assert isinstance(resp, QueryResponse)
    assert resp.answer == "final answer"
    # One step recorded per loop iteration.
    assert len(resp.steps) == 3
    assert all(s.tool == "generate_lit_table" for s in resp.steps)


def test_run_agent_unknown_tool_recovers(fake_llm):
    fake_llm.chat_script = [
        ChatResult(
            content=None,
            tool_calls=[ToolCall(id="c1", name="no_such_tool", arguments={})],
        ),
        ChatResult(content="recovered", tool_calls=[]),
    ]
    resp = run_agent("q")
    assert resp.answer == "recovered"
    assert len(resp.steps) == 1
    assert "unknown tool" in resp.steps[0].result_preview.lower()
