"""
Gemini Consultant Service

Gemini serves as a consultant resource for Claude Code and Doris.
It's the "geeky nerdy one" — good for research, validation, and brainstorming,
but doesn't make decisions. Claude Code and Doris synthesize Gemini's input before
presenting to the user.

Key principle: Gemini gets read-only context injection, not direct memory access.
We curate what it sees.
"""

import os
import json
import asyncio
import time
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai.types import GenerateContentConfig


class TaskType(str, Enum):
    QUICK_LOOKUP = "quick_lookup"      # Gemini Flash - fast facts
    CODE_REVIEW = "code_review"        # Gemini Pro - thorough analysis
    BRAINSTORM = "brainstorm"          # Gemini Pro - creative thinking
    VALIDATE = "validate"              # Gemini Pro - critical analysis
    DEEP_RESEARCH = "deep_research"    # Deep Research API - comprehensive (async)


@dataclass
class TaskConfig:
    model: str
    thinking_level: str
    max_output_tokens: int = 8000
    temperature: float = 0.7


@dataclass
class DeepResearchResult:
    """Result from Deep Research API."""
    question: str
    report: str
    status: str
    interaction_id: str
    elapsed_seconds: float


# Task type configurations
TASK_CONFIGS: dict[TaskType, TaskConfig] = {
    TaskType.QUICK_LOOKUP: TaskConfig(
        model="gemini-3-flash-preview",
        thinking_level="low",
        max_output_tokens=2000,
        temperature=0.3,
    ),
    TaskType.CODE_REVIEW: TaskConfig(
        model="gemini-3-pro-preview",
        thinking_level="high",
        max_output_tokens=8000,
        temperature=0.2,
    ),
    TaskType.BRAINSTORM: TaskConfig(
        model="gemini-3-pro-preview",
        thinking_level="high",
        max_output_tokens=6000,
        temperature=0.8,
    ),
    TaskType.VALIDATE: TaskConfig(
        model="gemini-3-pro-preview",
        thinking_level="high",
        max_output_tokens=4000,
        temperature=0.2,
    ),
}

# Deep Research agent name
DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"


class GeminiConsultant:
    """
    Gemini as a consultant — handles research, validation, and brainstorming.
    CC and Doris orchestrate; Gemini executes.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run or os.environ.get("GEMINI_DRY_RUN", "").lower() == "true"
        self.api_key = os.environ.get("GOOGLE_API_KEY")

        if not self.api_key and not self.dry_run:
            raise ValueError("GOOGLE_API_KEY not set")

        if not self.dry_run:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            print("[Gemini] DRY RUN MODE - no API calls will be made")

    async def consult(
        self,
        task_type: TaskType,
        query: str,
        context: Optional[dict] = None
    ) -> dict:
        """
        Main entry point for consulting Gemini.

        Args:
            task_type: Type of task (determines model and approach)
            query: The question or task
            context: Optional context to inject (memories, code, etc.)

        Returns:
            Dict with response and metadata
        """
        if self.dry_run:
            return self._dry_run_response(task_type, query, context)

        if task_type == TaskType.DEEP_RESEARCH:
            return await self._deep_research(query, context)
        else:
            return await self._simple_query(query, context, task_type)

    def _dry_run_response(self, task_type: TaskType, query: str, context: Optional[dict]) -> dict:
        """Return a mock response for dry run testing."""
        config = TASK_CONFIGS.get(task_type)
        model = config.model if config else "dry-run"

        print(f"[Gemini DRY RUN] task_type={task_type.value}, query={query[:60]}...")
        if context:
            print(f"[Gemini DRY RUN] context keys: {list(context.keys())}")

        if task_type == TaskType.DEEP_RESEARCH:
            return {
                "success": True,
                "task_type": task_type.value,
                "report": f"[DRY RUN] Deep Research would analyze 50-100 sources for: {query[:100]}...\n\nThis would take 2-5 minutes and cost $2-5.",
                "interaction_id": "dry-run-12345",
                "elapsed_seconds": 0.1,
                "status": "dry_run",
            }
        else:
            return {
                "success": True,
                "response": f"[DRY RUN] Gemini {task_type.value} response for: {query[:100]}...",
                "task_type": task_type.value,
                "model": model,
            }

    async def _simple_query(
        self,
        query: str,
        context: Optional[dict],
        task_type: TaskType
    ) -> dict:
        """Handle non-research queries with standard generate_content."""
        config = TASK_CONFIGS[task_type]

        # Build prompt with context injection
        prompt = self._build_prompt(query, context, task_type)

        try:
            response = self.client.models.generate_content(
                model=config.model,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                )
            )

            return {
                "success": True,
                "response": response.text,
                "task_type": task_type.value,
                "model": config.model,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_type": task_type.value,
            }

    async def _deep_research(
        self,
        question: str,
        context: Optional[dict],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> dict:
        """
        Comprehensive research using Google's Deep Research API.

        This uses the Interactions API with the deep-research agent.
        Runs asynchronously and polls for results.

        Cost: ~$2-5 per query
        Time: 2-5 minutes typically
        Sources: 50-100+
        """
        print(f"[Gemini] Starting Deep Research: {question[:80]}...")
        start_time = time.time()

        # Build the research prompt with any context
        research_prompt = question
        if context:
            if context.get("additional_instructions"):
                research_prompt += f"\n\nAdditional context: {context['additional_instructions']}"

        try:
            # Create the interaction with Deep Research agent
            interaction = self.client.interactions.create(
                input=research_prompt,
                agent=DEEP_RESEARCH_AGENT,
                background=True,
                store=True,
            )

            interaction_id = interaction.id
            print(f"[Gemini] Deep Research started, interaction_id: {interaction_id}")

            # Poll for completion
            max_wait_seconds = 600  # 10 minute max
            poll_interval = 10  # Check every 10 seconds
            elapsed = 0

            while elapsed < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                elapsed = time.time() - start_time

                # Get current status
                interaction = self.client.interactions.get(interaction_id)
                status = interaction.status

                print(f"[Gemini] Deep Research status: {status} ({elapsed:.0f}s elapsed)")

                if on_progress:
                    on_progress(f"Research in progress... ({elapsed:.0f}s)")

                if status == "completed":
                    # Extract the report from outputs
                    report = ""
                    if interaction.outputs:
                        report = interaction.outputs[-1].text

                    elapsed_total = time.time() - start_time
                    print(f"[Gemini] Deep Research completed in {elapsed_total:.1f}s")

                    return {
                        "success": True,
                        "task_type": TaskType.DEEP_RESEARCH.value,
                        "report": report,
                        "interaction_id": interaction_id,
                        "elapsed_seconds": elapsed_total,
                        "status": "completed",
                    }

                elif status == "failed":
                    error_msg = "Deep Research failed"
                    if interaction.outputs:
                        error_msg = interaction.outputs[-1].text

                    return {
                        "success": False,
                        "task_type": TaskType.DEEP_RESEARCH.value,
                        "error": error_msg,
                        "interaction_id": interaction_id,
                        "status": "failed",
                    }

            # Timeout
            return {
                "success": False,
                "task_type": TaskType.DEEP_RESEARCH.value,
                "error": f"Deep Research timed out after {max_wait_seconds}s",
                "interaction_id": interaction_id,
                "status": "timeout",
            }

        except Exception as e:
            print(f"[Gemini] Deep Research error: {e}")
            return {
                "success": False,
                "task_type": TaskType.DEEP_RESEARCH.value,
                "error": str(e),
                "status": "error",
            }

    async def deep_research_stream(
        self,
        question: str,
        context: Optional[dict] = None
    ):
        """
        Stream Deep Research progress.

        Yields status updates as the research progresses.
        Final yield contains the complete report.
        """
        print(f"[Gemini] Starting Deep Research (streaming): {question[:80]}...")
        start_time = time.time()

        research_prompt = question
        if context and context.get("additional_instructions"):
            research_prompt += f"\n\nAdditional context: {context['additional_instructions']}"

        try:
            # Create streaming interaction
            stream = self.client.interactions.create(
                input=research_prompt,
                agent=DEEP_RESEARCH_AGENT,
                background=True,
                store=True,
                stream=True,
                agent_config={"type": "deep-research", "thinking_summaries": "auto"}
            )

            report_text = ""

            for event in stream:
                elapsed = time.time() - start_time

                if hasattr(event, 'type'):
                    if event.type == "interaction.start":
                        yield {
                            "event": "start",
                            "interaction_id": event.interaction_id if hasattr(event, 'interaction_id') else None,
                            "elapsed": elapsed,
                        }

                    elif event.type == "content.delta":
                        delta = event.delta if hasattr(event, 'delta') else ""
                        report_text += delta
                        yield {
                            "event": "progress",
                            "delta": delta,
                            "elapsed": elapsed,
                        }

                    elif event.type == "interaction.complete":
                        yield {
                            "event": "complete",
                            "report": report_text,
                            "elapsed": elapsed,
                            "success": True,
                        }
                        return

        except Exception as e:
            yield {
                "event": "error",
                "error": str(e),
                "elapsed": time.time() - start_time,
                "success": False,
            }

    def _build_prompt(
        self,
        query: str,
        context: Optional[dict],
        task_type: TaskType
    ) -> str:
        """Build a prompt with appropriate context injection."""

        role_context = {
            TaskType.QUICK_LOOKUP: "You are a quick-lookup assistant. Be concise and direct.",
            TaskType.CODE_REVIEW: "You are a senior code reviewer. Be thorough but constructive. Point out issues, suggest improvements, and note what's done well.",
            TaskType.BRAINSTORM: "You are a creative brainstorming partner. Think divergently first (many ideas), then converge on the most promising ones.",
            TaskType.VALIDATE: "You are a critical validator. Check assumptions, find edge cases, spot potential issues. Be thorough and skeptical.",
        }

        prompt_parts = [role_context.get(task_type, "You are a helpful assistant.")]

        if context:
            from security.prompt_safety import wrap_untrusted

            if task_type == TaskType.CODE_REVIEW and "code" in context:
                wrapped_code = wrap_untrusted(context["code"], "code_to_review")
                prompt_parts.append(f"\n\nCode to review:\n{wrapped_code}")

            if "user_preferences" in context:
                wrapped_prefs = wrap_untrusted(context["user_preferences"], "user_preferences")
                prompt_parts.append(f"\n\nUser preferences: {wrapped_prefs}")

            if "relevant_memories" in context:
                wrapped_memories = wrap_untrusted(context["relevant_memories"], "relevant_memories")
                prompt_parts.append(f"\n\nRelevant context: {wrapped_memories}")

            if "project_context" in context:
                wrapped_project = wrap_untrusted(context["project_context"], "project_context")
                prompt_parts.append(f"\n\nProject context: {wrapped_project}")

        prompt_parts.append(f"\n\nTask: {query}")

        return "\n".join(prompt_parts)


# Singleton instance
_consultant: Optional[GeminiConsultant] = None


def get_consultant(dry_run: bool = False) -> GeminiConsultant:
    """Get or create the Gemini consultant singleton."""
    global _consultant
    # Check env var for dry run
    env_dry_run = os.environ.get("GEMINI_DRY_RUN", "").lower() == "true"
    effective_dry_run = dry_run or env_dry_run

    if _consultant is None:
        _consultant = GeminiConsultant(dry_run=effective_dry_run)
    return _consultant


def reset_consultant():
    """Reset singleton (useful for testing with different dry_run settings)."""
    global _consultant
    _consultant = None


# Convenience functions for common operations

async def quick_lookup(query: str, context: Optional[dict] = None) -> dict:
    """Fast lookup using Gemini Flash."""
    return await get_consultant().consult(TaskType.QUICK_LOOKUP, query, context)


async def code_review(code: str, description: str = "", context: Optional[dict] = None) -> dict:
    """Get a code review from Gemini Pro."""
    ctx = context or {}
    ctx["code"] = code
    return await get_consultant().consult(TaskType.CODE_REVIEW, description or "Review this code", ctx)


async def brainstorm(problem: str, constraints: Optional[str] = None, context: Optional[dict] = None) -> dict:
    """Brainstorm solutions to a problem."""
    query = problem
    if constraints:
        query += f"\n\nConstraints: {constraints}"
    return await get_consultant().consult(TaskType.BRAINSTORM, query, context)


async def deep_research(question: str, context: Optional[dict] = None) -> dict:
    """
    Conduct comprehensive research using Google's Deep Research API.

    This is async and takes 2-5 minutes typically.
    Consults 50-100+ sources and produces a detailed report.
    Cost: ~$2-5 per query.
    """
    return await get_consultant().consult(TaskType.DEEP_RESEARCH, question, context)


async def validate(claim: str, context: Optional[dict] = None) -> dict:
    """Validate a claim or assumption."""
    return await get_consultant().consult(TaskType.VALIDATE, claim, context)
