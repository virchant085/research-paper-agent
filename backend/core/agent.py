"""The function-calling agent loop.

`run_agent` drives a bounded tool-use conversation: it hands the LLM the tool
schemas, executes whatever tools the model asks for, feeds the results back, and
repeats until the model produces a plain-text answer or the step budget runs out.

Design notes
------------
* Everything is lazy: `get_llm()` builds the SDK client on first use, so importing
  this module never requires an API key.
* Tool execution is defensive. A tool raising an exception, or the model passing
  bad/missing arguments, must NOT crash the loop — the error string is fed back to
  the model as the tool result so it can recover or explain.
* Every tool invocation is recorded as a :class:`ToolStep` for UI transparency.
"""
from __future__ import annotations

from typing import Any

from backend.config import settings
from backend.core.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from backend.models.schemas import QueryResponse, ToolStep
from backend.services.llm import get_llm

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are a research-paper assistant for TEM/Robotics literature. "
    "Answer the user's question using the provided tools to retrieve evidence "
    "from the indexed papers.\n"
    "Grounding rules (strict):\n"
    "1. Answer from retrieved source text first. Every claim drawn from a "
    "paper must carry an inline citation with its provenance, e.g. "
    "\"(a1b2c3d4e5f6 | Method p.4)\".\n"
    "2. If the retrieved text does not support a claim the user asks about, "
    "say plainly that the source does not state it — never fill the gap with "
    "a guess, and never invent citations or content.\n"
    "3. Use one name for one thing: reuse the exact terms the papers "
    "themselves use (model names, datasets, metrics); do not introduce "
    "synonyms or coin new names for the authors' concepts.\n"
    "4. When comparing papers or asked about scope, surface limitations and "
    "boundary conditions rather than papering over them.\n"
    "Call tools to gather evidence before answering; when you have enough, "
    "respond with a clear, well-structured final answer and stop calling tools."
)


def _preview(text: str, limit: int = 400) -> str:
    """Return a short, single-purpose preview of a tool result for the UI."""
    if text is None:
        return ""
    return str(text)[:limit]


def run_agent(question: str, paper_ids: list[str] | None = None) -> QueryResponse:
    """Run the bounded function-calling loop and return the agent's answer.

    Parameters
    ----------
    question:
        The user's natural-language question.
    paper_ids:
        Optional whitelist of paper ids. When provided, the system prompt asks the
        model to restrict retrieval to those papers.

    Returns
    -------
    QueryResponse
        ``answer`` holds the final text; ``steps`` records each tool call made,
        in order, with a truncated preview of its result.
    """
    llm = get_llm()

    system = _SYSTEM_PROMPT
    if paper_ids:
        system += (
            "\nRestrict retrieval to these paper_ids: "
            f"{list(paper_ids)}. Pass them to the tools' paper_id / paper_ids "
            "arguments so results stay scoped to the requested papers."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    steps: list[ToolStep] = []

    for _ in range(max(1, settings.agent_max_steps)):
        result = llm.chat(messages, tools=TOOL_SCHEMAS)

        # No tool calls => the model produced a final answer.
        if not result.tool_calls:
            return QueryResponse(answer=result.content or "", steps=steps)

        # Record the assistant's tool-call turn so the follow-up tool messages
        # attach to it correctly (OpenAI chat format requires this pairing).
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": result.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # Arguments must be a JSON *string* in the wire format.
                        "arguments": _dumps(call.arguments),
                    },
                }
                for call in result.tool_calls
            ],
        }
        messages.append(assistant_msg)

        # Execute each requested tool, feeding results (or errors) back.
        for call in result.tool_calls:
            tool_output = _invoke_tool(call.name, call.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_output,
                }
            )
            steps.append(
                ToolStep(
                    tool=call.name,
                    arguments=call.arguments if isinstance(call.arguments, dict) else {},
                    result_preview=_preview(tool_output),
                )
            )

    # Step budget exhausted: force a final text answer with tools disabled so the
    # model cannot loop further and must summarize what it has gathered.
    final = llm.chat(messages, tools=None)
    return QueryResponse(answer=final.content or "", steps=steps)


def _invoke_tool(name: str, arguments: Any) -> str:
    """Look up and call a tool, returning its string output or an error message.

    Guards against unknown tool names, non-dict argument payloads, and any
    exception the tool itself raises — in every case a human-readable error
    string is returned so the loop can continue and the model can recover.
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'."

    kwargs = arguments if isinstance(arguments, dict) else {}
    try:
        output = func(**kwargs)
    except (KeyError, TypeError) as exc:
        # Bad / missing arguments from the model.
        return f"Error calling tool '{name}': invalid arguments ({exc})."
    except Exception as exc:  # noqa: BLE001 - never let a tool crash the loop
        return f"Error while running tool '{name}': {exc}"

    return output if isinstance(output, str) else str(output)


def _dumps(obj: Any) -> str:
    """Serialize tool arguments back to a JSON string for the assistant message."""
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"
