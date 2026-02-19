"""
Awareness Digest

Collects and manages observations from all scouts.
Provides a structured summary for injection into Doris's context.

The digest maintains:
- Recent observations (last 24 hours)
- Pending escalations
- Aggregated context by topic
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import json
from pathlib import Path

from scouts.base import Observation, Relevance


# Persistence file for digest state
DIGEST_FILE = Path(__file__).parent.parent / "data" / "awareness_digest.json"

# Safety limits for digest files loaded from disk
MAX_DIGEST_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_RAW_DATA_SIZE = 50_000  # chars per observation raw_data blob
MAX_OBSERVATIONS_ON_LOAD = 500  # hard cap on observations from file


@dataclass
class AwarenessDigest:
    """
    Collects and manages scout observations.

    Maintains a rolling window of observations and provides
    methods to query, summarize, and check for escalations.
    """

    observations: list[Observation] = field(default_factory=list)
    escalations: list[Observation] = field(default_factory=list)
    max_age_hours: int = 24
    max_observations: int = 100

    def add(self, observation: Observation) -> None:
        """
        Add an observation to the digest.

        If the observation is marked for escalation, also adds to escalations list.
        """
        self.observations.append(observation)

        if observation.escalate:
            self.escalations.append(observation)

        # Trim old observations
        self._cleanup()

    def add_many(self, observations: list[Observation]) -> None:
        """Add multiple observations."""
        for obs in observations:
            self.add(obs)

    def _cleanup(self) -> None:
        """Remove old observations and trim to max size."""
        cutoff = datetime.now() - timedelta(hours=self.max_age_hours)

        # Filter by age
        self.observations = [
            obs for obs in self.observations
            if obs.timestamp > cutoff
        ]

        # Trim to max size (keep newest)
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations:]

        # Escalations only last until cleared
        self.escalations = [
            obs for obs in self.escalations
            if obs.timestamp > cutoff
        ]

    def has_escalation(self) -> bool:
        """Check if there are pending escalations."""
        return len(self.escalations) > 0

    def get_escalations(self) -> list[Observation]:
        """Get pending escalations."""
        return list(self.escalations)

    def clear_escalations(self) -> list[Observation]:
        """Clear and return pending escalations."""
        escalations = self.escalations
        self.escalations = []
        return escalations

    def get_recent(
        self,
        hours: int = 6,
        relevance: Optional[Relevance] = None,
        scout: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> list[Observation]:
        """
        Get recent observations with optional filters.

        Args:
            hours: How far back to look
            relevance: Filter by relevance level
            scout: Filter by scout name
            tags: Filter by any matching tag

        Returns:
            Filtered list of observations, newest first
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        results = []

        for obs in reversed(self.observations):
            if obs.timestamp < cutoff:
                continue

            if relevance and obs.relevance != relevance:
                continue

            if scout and obs.scout != scout:
                continue

            if tags:
                if not any(tag in obs.context_tags for tag in tags):
                    continue

            results.append(obs)

        return results

    def get_by_topic(self) -> dict[str, list[Observation]]:
        """
        Group recent observations by topic (inferred from tags).

        Returns dict like:
        {
            "email": [...],
            "calendar": [...],
            "weather": [...],
            "other": [...]
        }
        """
        topics = {
            "email": [],
            "calendar": [],
            "weather": [],
            "time": [],
            "other": [],
        }

        for obs in self.get_recent(hours=12):
            categorized = False
            for topic in ["email", "calendar", "weather", "time"]:
                if topic in obs.context_tags or topic in obs.scout:
                    topics[topic].append(obs)
                    categorized = True
                    break
            if not categorized:
                topics["other"].append(obs)

        return topics

    def current(self) -> dict:
        """
        Return digest formatted for injection into Doris's context.

        This is the main interface for the daemon to get awareness data.
        """
        by_topic = self.get_by_topic()

        # Build summary
        summary = {
            "generated_at": datetime.now().isoformat(),
            "pending_escalations": len(self.escalations),
            "observations_24h": len(self.observations),
            "topics": {},
        }

        for topic, observations in by_topic.items():
            if not observations:
                continue

            summary["topics"][topic] = {
                "count": len(observations),
                "latest": [
                    {
                        "observation": obs.observation,
                        "relevance": obs.relevance.value,
                        "time": obs.timestamp.strftime("%H:%M"),
                    }
                    for obs in observations[:3]  # Last 3 per topic
                ]
            }

        return summary

    def format_for_prompt(self) -> str:
        """
        Format digest as text for system prompt injection.
        """
        lines = ["=== AWARENESS DIGEST ===", ""]

        if self.escalations:
            lines.append(f"⚠️ {len(self.escalations)} pending escalation(s):")
            for esc in self.escalations[:3]:
                lines.append(f"  - [{esc.scout}] {esc.observation}")
            lines.append("")

        by_topic = self.get_by_topic()

        for topic, observations in by_topic.items():
            if not observations:
                continue

            lines.append(f"{topic.title()} ({len(observations)} recent):")
            for obs in observations[:2]:
                relevance_marker = "!" if obs.relevance == Relevance.HIGH else ""
                lines.append(f"  {relevance_marker}{obs.observation}")
            lines.append("")

        if len(self.observations) == 0:
            lines.append("No recent observations.")

        lines.append("=== END DIGEST ===")
        return "\n".join(lines)

    def save(self) -> None:
        """Persist digest to disk."""
        DIGEST_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "observations": [obs.to_dict() for obs in self.observations],
            "escalations": [obs.to_dict() for obs in self.escalations],
            "saved_at": datetime.now().isoformat(),
        }

        DIGEST_FILE.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls) -> "AwarenessDigest":
        """Load digest from disk, or return empty digest.

        Validates file size, schema structure, and caps observation counts.
        """
        digest = cls()

        if not DIGEST_FILE.exists():
            return digest

        try:
            # Check file size before reading
            file_size = DIGEST_FILE.stat().st_size
            if file_size > MAX_DIGEST_FILE_SIZE:
                print(f"[Digest] File too large ({file_size} bytes), starting fresh")
                return digest

            raw = DIGEST_FILE.read_text()
            data = json.loads(raw)

            # Schema validation: top-level must be a dict with expected keys
            if not isinstance(data, dict):
                print("[Digest] Invalid schema: top-level is not a dict")
                return digest

            observations_raw = data.get("observations", [])
            escalations_raw = data.get("escalations", [])

            if not isinstance(observations_raw, list) or not isinstance(escalations_raw, list):
                print("[Digest] Invalid schema: observations/escalations must be lists")
                return digest

            # Cap number of observations loaded from file
            for obs_dict in observations_raw[:MAX_OBSERVATIONS_ON_LOAD]:
                obs = _observation_from_dict(obs_dict)
                if obs:
                    digest.observations.append(obs)

            for obs_dict in escalations_raw[:MAX_OBSERVATIONS_ON_LOAD]:
                obs = _observation_from_dict(obs_dict)
                if obs:
                    digest.escalations.append(obs)

            digest._cleanup()

        except json.JSONDecodeError as e:
            print(f"[Digest] Corrupt JSON: {e}")
        except Exception as e:
            print(f"[Digest] Error loading: {e}")

        return digest


def _observation_from_dict(data: dict) -> Optional[Observation]:
    """Reconstruct Observation from dict with size validation."""
    try:
        if not isinstance(data, dict):
            return None

        # Validate required string fields
        scout = data["scout"]
        observation = data["observation"]
        if not isinstance(scout, str) or not isinstance(observation, str):
            return None

        # Cap raw_data size to prevent memory abuse
        raw_data = data.get("raw_data")
        if raw_data is not None:
            raw_str = json.dumps(raw_data) if not isinstance(raw_data, str) else raw_data
            if len(raw_str) > MAX_RAW_DATA_SIZE:
                raw_data = None  # Drop oversized raw_data

        # Validate context_tags is a list of strings
        context_tags = data.get("context_tags", [])
        if not isinstance(context_tags, list):
            context_tags = []
        context_tags = [t for t in context_tags if isinstance(t, str)]

        return Observation(
            scout=scout,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            observation=observation,
            relevance=Relevance(data["relevance"]),
            escalate=bool(data.get("escalate", False)),
            context_tags=context_tags,
            raw_data=raw_data,
        )
    except (KeyError, ValueError, TypeError) as e:
        print(f"[Digest] Error parsing observation: {e}")
        return None


# Singleton instance
_digest: Optional[AwarenessDigest] = None


def get_digest() -> AwarenessDigest:
    """Get the global digest instance."""
    global _digest
    if _digest is None:
        _digest = AwarenessDigest.load()
    return _digest


def save_digest() -> None:
    """Save the global digest."""
    if _digest:
        _digest.save()


# For testing
if __name__ == "__main__":
    from scouts.base import Observation, Relevance

    digest = AwarenessDigest()

    # Add some test observations
    digest.add(Observation(
        scout="email-scout",
        timestamp=datetime.now(),
        observation="New email from school about picture day",
        relevance=Relevance.MEDIUM,
        escalate=False,
        context_tags=["email", "school", "child"],
    ))

    digest.add(Observation(
        scout="weather-scout",
        timestamp=datetime.now(),
        observation="Severe thunderstorm warning",
        relevance=Relevance.HIGH,
        escalate=True,
        context_tags=["weather", "severe"],
    ))

    digest.add(Observation(
        scout="calendar-scout",
        timestamp=datetime.now(),
        observation="'Dentist appointment' starts in 30 minutes",
        relevance=Relevance.MEDIUM,
        escalate=False,
        context_tags=["calendar", "reminder"],
    ))

    print("Digest summary:")
    print(json.dumps(digest.current(), indent=2))

    print("\n" + "=" * 50 + "\n")
    print("Prompt format:")
    print(digest.format_for_prompt())

    print("\n" + "=" * 50 + "\n")
    print(f"Has escalation: {digest.has_escalation()}")
    print(f"Escalations: {len(digest.get_escalations())}")
