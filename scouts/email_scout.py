"""
Email Scout

Monitors Gmail for new important emails and reports observations.
Runs every 30 minutes via the daemon scheduler.

Uses existing tools/gmail.py for email access.
"""

from datetime import datetime, timedelta
from typing import Optional
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scouts.base import HaikuScout, Observation, Relevance
from tools.gmail import scan_recent_emails, get_unread_count
from security.prompt_safety import wrap_with_scan
from memory.wisdom import should_escalate_based_on_wisdom


class EmailScout(HaikuScout):
    """
    Scout that monitors Gmail for important emails.

    Checks:
    - New important emails (starred, from contacts, keyword matches)
    - Significant changes in unread count
    - Emails that might need urgent attention

    Runs every 30 minutes by default.
    """

    name = "email-scout"

    def __init__(self):
        super().__init__()
        self._last_seen_ids: set[str] = set()
        self._last_unread_count: Optional[int] = None

    async def observe(self) -> list[Observation]:
        """
        Check Gmail for new important emails.

        Returns observations for:
        - New important emails since last check
        - Significant unread count changes
        """
        observations = []
        now = datetime.now()

        try:
            # Get important emails from last 2 hours
            # (covers our 30-min interval with buffer)
            important_emails = scan_recent_emails(hours=2, max_results=20)

            # Filter to emails we haven't seen
            new_emails = [
                e for e in important_emails
                if e['id'] not in self._last_seen_ids
            ]

            # Update seen set (keep last 100 to prevent memory bloat)
            for email in important_emails:
                self._last_seen_ids.add(email['id'])
            if len(self._last_seen_ids) > 100:
                # Remove oldest (this is approximate, but good enough)
                self._last_seen_ids = set(list(self._last_seen_ids)[-100:])

            # Create observations for new emails
            for email in new_emails:
                observation = await self._create_email_observation(email, now)
                if observation:
                    observations.append(observation)

            # Check unread count for significant changes
            current_unread = get_unread_count()
            if self._last_unread_count is not None:
                change = current_unread - self._last_unread_count
                if change >= 10:
                    observations.append(Observation(
                        scout=self.name,
                        timestamp=now,
                        observation=f"Unread email count jumped by {change} (now {current_unread})",
                        relevance=Relevance.MEDIUM,
                        escalate=False,
                        context_tags=["email", "volume"],
                    ))
            self._last_unread_count = current_unread

        except Exception as e:
            # Scout errors are HIGH/escalate — a broken scout is worth knowing about
            print(f"[{self.name}] Error scanning emails: {e}")
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=f"Error accessing Gmail: {str(e)[:100]}",
                relevance=Relevance.HIGH,
                escalate=True,
                context_tags=["scout_error", "email"],
            ))

        return observations

    async def _create_email_observation(
        self,
        email: dict,
        timestamp: datetime
    ) -> Optional[Observation]:
        """
        Create an observation for an email.

        Uses Haiku to classify relevance if the email seems potentially important.
        """
        sender = email.get('sender_name') or email.get('sender', 'unknown')
        subject = email.get('subject', '(no subject)')
        reasons = email.get('why_important', [])
        snippet = email.get('snippet', '')[:200]

        # Build description for classification with safety markers
        # Wrap + scan untrusted external content to detect prompt injection
        wrapped_sender = wrap_with_scan(sender, 'email_sender')
        wrapped_subject = wrap_with_scan(subject, 'email_subject')
        wrapped_snippet = wrap_with_scan(snippet, 'email_snippet') if snippet else ""

        description = f"Email from {wrapped_sender}\nSubject: {wrapped_subject}"
        if wrapped_snippet:
            description += f"\nPreview: {wrapped_snippet}"
        if reasons:
            # Importance signals are internal, not from the email
            description += f"\nImportance signals: {', '.join(reasons)}"

        # Classify with Haiku
        relevance, escalate, tags = await self.classify_relevance(
            description,
            context="This is a personal email account. School emails about kids are important. "
                    "Emails marked starred or from contacts are pre-filtered as important."
        )

        # Add email-specific tags
        tags = list(set(tags + ["email"]))

        # Escalate based on multiple criteria
        subject_lower = subject.lower()
        sender_lower = sender.lower()
        snippet_lower = snippet.lower()

        # Urgent keywords in subject
        urgent_keywords = ["urgent", "emergency", "asap", "immediately", "action required"]

        # School-related keywords (affects family schedule)
        school_keywords = [
            "snow day", "early dismissal", "school closed", "pickup",
            "dismissal", "canceled", "cancelled", "delayed opening",
        ]

        # VIP senders — configure via DORIS_VIP_SENDERS env var (comma-separated)
        # e.g., DORIS_VIP_SENDERS="spouse@example.com,school.org"
        import os
        vip_env = os.getenv("DORIS_VIP_SENDERS", "")
        vip_senders = [s.strip() for s in vip_env.split(",") if s.strip()]

        # Check all escalation criteria
        is_urgent = any(kw in subject_lower for kw in urgent_keywords)
        is_school_related = any(kw in subject_lower or kw in snippet_lower for kw in school_keywords)
        is_vip = any(vip in sender_lower for vip in vip_senders)
        is_high_relevance = relevance == Relevance.HIGH

        # Check wisdom for patterns from past missed escalations
        # This learns from user feedback like "that airline email should have been escalated"
        wisdom_says_escalate, wisdom_reason = should_escalate_based_on_wisdom(
            source="email",
            sender=sender,
            subject=subject,
            snippet=snippet,
        )
        if wisdom_says_escalate:
            print(f"[{self.name}] Wisdom suggests escalation: {wisdom_reason}")

        # Trust Haiku's HIGH relevance classification - if it's important enough to be HIGH,
        # it's important enough to escalate. This catches things like airline emails about
        # upcoming flights that don't match keywords but are clearly time-sensitive.
        # Also escalate if wisdom suggests it based on past missed patterns.
        if email.get('is_starred') or email.get('is_important') or is_urgent or is_school_related or is_vip or is_high_relevance or wisdom_says_escalate:
            escalate = True
            if relevance == Relevance.LOW:
                relevance = Relevance.MEDIUM

        return Observation(
            scout=self.name,
            timestamp=timestamp,
            observation=f"New email from {sender}: {subject} [msg_id:{email.get('id')}]",
            relevance=relevance,
            escalate=escalate,
            context_tags=tags,
            raw_data={
                "id": email.get('id'),
                "sender": email.get('sender'),
                "sender_name": sender,
                "subject": subject,
                "snippet": snippet,
                "why_important": reasons,
                "is_starred": email.get('is_starred', False),
                "is_contact": email.get('is_contact', False),
            }
        )


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        scout = EmailScout()
        print(f"Running {scout.name}...")

        observations = await scout.run()

        if not observations:
            print("No new observations")
        else:
            print(f"Found {len(observations)} observations:\n")
            for obs in observations:
                print(f"[{obs.relevance.value.upper()}] {obs.observation}")
                print(f"  Escalate: {obs.escalate}")
                print(f"  Tags: {obs.context_tags}")
                print()

    asyncio.run(test())
