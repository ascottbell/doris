"""
Codex Consultant Service

GPT-5.3-Codex via the Codex CLI (codex exec) as a consultant resource
for Claude Code and Doris. Mirrors the Gemini consultant pattern.

Uses non-interactive mode (codex exec --json) to shell out to the CLI,
which authenticates via ChatGPT account or OPENAI_API_KEY.
"""

import asyncio
import json
import shutil
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class TaskType(str, Enum):
    QUICK = "quick"              # Fast, concise response
    CODE_REVIEW = "code_review"  # Thorough code analysis
    BRAINSTORM = "brainstorm"    # Creative, divergent thinking
    VALIDATE = "validate"        # Critical analysis
    RESEARCH = "research"        # Deep, comprehensive analysis


@dataclass
class TaskConfig:
    system_prompt: str
    max_thinking: bool = False


TASK_CONFIGS: dict[TaskType, TaskConfig] = {
    TaskType.QUICK: TaskConfig(
        system_prompt="You are a quick-lookup assistant. Be concise and direct. One paragraph max.",
    ),
    TaskType.CODE_REVIEW: TaskConfig(
        system_prompt=(
            "You are a senior code reviewer. Be thorough but constructive. "
            "Point out issues, suggest improvements, and note what's done well."
        ),
        max_thinking=True,
    ),
    TaskType.BRAINSTORM: TaskConfig(
        system_prompt=(
            "You are a creative brainstorming partner. Think divergently first "
            "(many ideas), then converge on the most promising ones."
        ),
        max_thinking=True,
    ),
    TaskType.VALIDATE: TaskConfig(
        system_prompt=(
            "You are a critical validator. Check assumptions, find edge cases, "
            "spot potential issues. Be thorough and skeptical."
        ),
        max_thinking=True,
    ),
    TaskType.RESEARCH: TaskConfig(
        system_prompt=(
            "You are a deep research analyst. Provide comprehensive analysis "
            "with multiple angles, evidence, and clear reasoning. Be thorough."
        ),
        max_thinking=True,
    ),
}


def _codex_available() -> bool:
    """Check if codex CLI is installed."""
    return shutil.which("codex") is not None


def _build_prompt(query: str, context: Optional[dict], task_type: TaskType) -> str:
    """Build prompt with context injection.

    All context values are wrapped with wrap_with_scan() to prevent
    prompt injection via user-controlled data (code snippets, memories, etc.).
    """
    from security.prompt_safety import wrap_with_scan

    config = TASK_CONFIGS[task_type]
    parts = [config.system_prompt]

    if context:
        if task_type == TaskType.CODE_REVIEW and "code" in context:
            wrapped_code = wrap_with_scan(context["code"], "codex-code-review")
            parts.append(f"\n\nCode to review:\n{wrapped_code}")
        if "user_preferences" in context:
            wrapped_prefs = wrap_with_scan(context["user_preferences"], "codex-user-prefs")
            parts.append(f"\n\nUser preferences: {wrapped_prefs}")
        if "relevant_memories" in context:
            wrapped_mem = wrap_with_scan(context["relevant_memories"], "codex-memories")
            parts.append(f"\n\nRelevant context: {wrapped_mem}")
        if "project_context" in context:
            wrapped_proj = wrap_with_scan(context["project_context"], "codex-project")
            parts.append(f"\n\nProject context: {wrapped_proj}")

    parts.append(f"\n\nTask: {query}")
    return "\n".join(parts)


def _parse_jsonl_response(stdout: str) -> dict:
    """Parse codex exec --json output (JSONL) into a result dict."""
    text_parts = []
    usage = {}
    thread_id = None

    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "thread.started":
            thread_id = event.get("thread_id")

        elif event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message" and "text" in item:
                text_parts.append(item["text"])

        elif event_type == "turn.completed":
            usage = event.get("usage", {})

    return {
        "text": "\n\n".join(text_parts),
        "thread_id": thread_id,
        "usage": usage,
    }


async def _run_codex(prompt: str, timeout_seconds: int = 120) -> dict:
    """
    Run codex exec and return parsed result.

    Returns dict with success, response/error, and metadata.
    """
    if not _codex_available():
        return {
            "success": False,
            "error": "codex CLI not installed. Install with: brew install --cask codex",
        }

    try:
        proc = await asyncio.create_subprocess_exec(
            "codex", "exec", "--json", "--approval-mode", "suggest", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return {
                "success": False,
                "error": f"codex exec failed (exit {proc.returncode}): {stderr[:500]}",
            }

        parsed = _parse_jsonl_response(stdout)

        if not parsed["text"]:
            return {
                "success": False,
                "error": f"No response text in codex output. stderr: {stderr[:500]}",
            }

        return {
            "success": True,
            "response": parsed["text"],
            "thread_id": parsed["thread_id"],
            "usage": parsed["usage"],
        }

    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"codex exec timed out after {timeout_seconds}s",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"codex exec error: {str(e)}",
        }


async def consult(
    task_type: TaskType,
    query: str,
    context: Optional[dict] = None
) -> dict:
    """
    Main entry point for consulting Codex.

    Args:
        task_type: Type of task (determines system prompt and approach)
        query: The question or task
        context: Optional context to inject

    Returns:
        Dict with success, response, task_type, and metadata
    """
    prompt = _build_prompt(query, context, task_type)

    timeout = 60 if task_type == TaskType.QUICK else 180

    result = await _run_codex(prompt, timeout_seconds=timeout)
    result["task_type"] = task_type.value
    result["model"] = "gpt-5.3-codex"

    return result


# Convenience functions

async def quick(query: str, context: Optional[dict] = None) -> dict:
    """Fast lookup using Codex."""
    return await consult(TaskType.QUICK, query, context)


async def code_review(code: str, description: str = "", context: Optional[dict] = None) -> dict:
    """Get a code review from Codex."""
    ctx = context or {}
    ctx["code"] = code
    return await consult(TaskType.CODE_REVIEW, description or "Review this code", ctx)


async def brainstorm(problem: str, constraints: Optional[str] = None, context: Optional[dict] = None) -> dict:
    """Brainstorm solutions to a problem."""
    query = problem
    if constraints:
        query += f"\n\nConstraints: {constraints}"
    return await consult(TaskType.BRAINSTORM, query, context)


async def validate(claim: str, context: Optional[dict] = None) -> dict:
    """Validate a claim or assumption."""
    return await consult(TaskType.VALIDATE, claim, context)


async def research(question: str, context: Optional[dict] = None) -> dict:
    """Deep research analysis using Codex."""
    return await consult(TaskType.RESEARCH, question, context)
