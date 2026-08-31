"""Agentic tool-calling loop.

Runs the LLM with tool definitions; when the model emits ``tool_calls``,
executes them and feeds the results back, looping until the model answers
directly or the iteration cap is reached.
"""

import json
import logging
from collections.abc import Awaitable, Callable

from app.providers.base import BaseLLMProvider, LLMResult

logger = logging.getLogger(__name__)

ToolExecutor = Callable[[str, dict], Awaitable[str]]
ToolObserver = Callable[[str, dict], Awaitable[None]]


async def run_tool_loop(
    llm: BaseLLMProvider,
    messages: list[dict],
    model: str,
    temperature: float | None,
    tools: list[dict],
    tool_executor: ToolExecutor,
    max_iterations: int = 5,
    on_tool: ToolObserver | None = None,
) -> LLMResult:
    """Run the agentic loop until the model answers directly.

    ``messages`` is mutated in place — tool-calling turns are appended so the
    caller can inspect the final history. ``on_tool`` is awaited before each
    tool executes (e.g. to notify a streaming client). Returns the final
    ``LLMResult`` (with empty ``tool_calls`` if the model answered, or the last
    result if the iteration cap was hit).
    """
    for _ in range(max_iterations):
        result: LLMResult = await llm.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
        )

        if not result.tool_calls:
            return result

        # Model decided to call tools — record its decision and run them.
        messages.append(
            {
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": result.tool_calls,
            }
        )

        for tool_call in result.tool_calls:
            function = tool_call.get("function") or {}
            full_name = function.get("name", "")
            arguments = _parse_arguments(function.get("arguments"))
            if on_tool is not None:
                await on_tool(full_name, arguments)
            output = await _safe_execute(tool_executor, full_name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": output,
                }
            )

        logger.info("Executed %d tool call(s), continuing loop", len(result.tool_calls))

    logger.warning("Tool loop reached max_iterations=%d without a final answer", max_iterations)
    return result


def _parse_arguments(raw: str | None) -> dict:
    """Parse the model's JSON-encoded tool arguments, tolerating errors."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


async def _safe_execute(executor: ToolExecutor, full_name: str, arguments: dict) -> str:
    """Execute one tool, turning any exception into an error string for the model."""
    try:
        return await executor(full_name, arguments)
    except Exception as e:
        logger.warning("Tool '%s' failed: %s", full_name, e)
        return f"Error calling tool '{full_name}': {e}"
