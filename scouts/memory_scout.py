"""
Memory Scout

Monitors the health of Doris's memory system:
- Extraction running before compaction
- Knowledge graph being populated
- Wisdom system coverage

Runs every hour to catch issues early before memories are lost.
"""

import logging
from datetime import datetime
from typing import Optional

from scouts.base import Scout, Observation, Relevance

logger = logging.getLogger("doris.scouts.memory")


class MemoryScout(Scout):
    """
    Monitor memory system health.

    Checks:
    1. Compaction never runs without extraction first
    2. Knowledge graph is being populated from extractions
    3. Wisdom system is covering state-changing tools
    """

    name = "memory-scout"

    def __init__(self):
        super().__init__()
        self._last_metrics: Optional[dict] = None

    async def observe(self) -> list[Observation]:
        """Check memory system health and report issues."""
        try:
            from session.persistent import get_memory_metrics
            from llm.brain import WISDOM_REQUIRED_TOOLS
        except ImportError as e:
            logger.warning(f"[MemoryScout] Failed to import: {e}")
            return []

        observations = []
        now = datetime.now()

        try:
            metrics = get_memory_metrics()
        except Exception as e:
            logger.error(f"[MemoryScout] Failed to load metrics: {e}")
            return [Observation(
                scout=self.name,
                timestamp=now,
                observation=f"Failed to load memory metrics: {e}",
                relevance=Relevance.HIGH,
                escalate=True,
                context_tags=["memory", "error"],
            )]

        # Check 1: Compactions without extraction
        extraction_skipped = metrics.get("compactions", {}).get("extraction_skipped", 0)
        compactions_total = metrics.get("compactions", {}).get("total", 0)

        if extraction_skipped > 0:
            # This is the critical issue the plan addresses
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=(
                    f"CRITICAL: {extraction_skipped} compactions ran without extraction. "
                    f"Memories may have been lost during session summarization. "
                    f"Check that daemon/extraction.py is being called in persistent.py._run_compaction()"
                ),
                relevance=Relevance.HIGH,
                escalate=True,
                context_tags=["memory", "extraction", "compaction", "data_loss"],
                raw_data={"extraction_skipped": extraction_skipped, "compactions_total": compactions_total}
            ))

        # Check 2: Knowledge graph not being populated
        extractions_total = metrics.get("extractions", {}).get("total", 0)
        entities_created = metrics.get("graph", {}).get("entities_created", 0)
        relationships_created = metrics.get("graph", {}).get("relationships_created", 0)

        if extractions_total > 5 and entities_created == 0:
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=(
                    f"Knowledge graph not being populated: {extractions_total} extractions "
                    f"but 0 entities created. Check memory/entity_extraction.py integration."
                ),
                relevance=Relevance.MEDIUM,
                escalate=False,
                context_tags=["memory", "graph", "entities"],
                raw_data={"extractions": extractions_total, "entities": entities_created}
            ))

        # Check 3: Extraction failures
        extraction_failures = metrics.get("extractions", {}).get("failures", 0)
        if extraction_failures > 3:
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=(
                    f"Memory extraction failing: {extraction_failures} failures. "
                    f"Check API connectivity and extraction prompt."
                ),
                relevance=Relevance.MEDIUM,
                escalate=extraction_failures > 10,
                context_tags=["memory", "extraction", "error"],
                raw_data={"failures": extraction_failures}
            ))

        # Check 4: Wisdom coverage (lower priority, informational)
        tools_covered = set(metrics.get("wisdom", {}).get("tools_covered", []))
        uncovered = WISDOM_REQUIRED_TOOLS - tools_covered
        wisdom_entries = metrics.get("wisdom", {}).get("entries_logged", 0)

        # Only report if we have enough data points and significant gaps
        if wisdom_entries > 50 and len(uncovered) > 5:
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=(
                    f"Wisdom coverage gap: {len(uncovered)} state-changing tools never logged. "
                    f"These tools won't benefit from experiential learning."
                ),
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["memory", "wisdom", "coverage"],
                raw_data={"uncovered_tools": list(uncovered), "total_entries": wisdom_entries}
            ))

        # Log summary
        if observations:
            logger.info(f"[MemoryScout] Found {len(observations)} issues")
        else:
            logger.debug("[MemoryScout] Memory system healthy")

        # Store metrics for comparison next run
        self._last_metrics = metrics

        return observations

    def should_escalate(self, observation: Observation) -> bool:
        """
        Escalate extraction/compaction issues immediately.

        Data loss from missed extraction is not recoverable.
        """
        if "compaction" in observation.context_tags and "extraction" in observation.context_tags:
            return True
        return super().should_escalate(observation)
