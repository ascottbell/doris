"""
Email source - monitors Gmail for actionable emails.

Scout pattern: detect actionable emails and hand off to Doris for processing.
Scout scouts, Doris Dorises.

Checks for emails that might need calendar events, reminders, or notifications.
Focuses on family-relevant emails: school, activities, appointments.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from tools.gmail import scan_recent_emails, get_email_content
from proactive.models import ProactiveEvent
from security.prompt_safety import wrap_untrusted
from proactive.db import (
    save_event,
    is_event_processed,
    update_event_status,
    get_checkpoint,
    update_checkpoint,
)
from proactive.notifier import notify_action

EASTERN = ZoneInfo("America/New_York")

# Additional keywords that signal actionable emails (beyond gmail.py's list)
ACTIONABLE_KEYWORDS = [
    "recital", "performance", "show", "concert",
    "game", "practice", "rehearsal",
    "birthday party", "playdate", "sleepover",
    "doctor", "dentist", "appointment",
    "parent night", "open house", "curriculum night",
    "class trip", "field day",
]

# Senders that are highly likely to have actionable content
# Configure with sender keywords for emails that should always be evaluated
PRIORITY_SENDERS = [
    # "schoolname",            # School emails
    # "dancestudio",           # Kids' activity
]


def _was_wisdom_override(email: dict) -> tuple[bool, str | None]:
    """Check if this email would only pass filtering due to escalation wisdom.

    Returns (is_wisdom_override, reason). Used to track which emails were
    caught by wisdom so we can log correct escalations after successful handoff.
    """
    try:
        from memory.wisdom import should_escalate_based_on_wisdom
        return should_escalate_based_on_wisdom(
            source="email",
            sender=email.get("sender", "").lower(),
            subject=email.get("subject", "").lower(),
            snippet=email.get("snippet", "").lower(),
        )
    except Exception:
        return False, None


def monitor():
    """
    Main email monitor function.

    Called by the scheduler every 15 minutes.
    Scout pattern: detect actionable emails and hand off to Doris.
    """
    print("[email-monitor] Checking for actionable emails...")

    try:
        # Get recent important emails (last 4 hours to catch up)
        emails = scan_recent_emails(hours=4, max_results=20)

        if not emails:
            print("[email-monitor] No important emails found")
            update_checkpoint("email")
            return

        processed = 0
        handed_off = 0

        for email in emails:
            email_id = email.get("id")

            # Skip if already processed
            if is_event_processed("email", email_id):
                continue

            # Check if this email is likely actionable
            if not _is_potentially_actionable(email):
                # Still mark as processed to avoid re-checking
                event = ProactiveEvent(
                    source_type="email",
                    source_id=email_id,
                    raw_data=email,
                    status="ignored"
                )
                save_event(event)
                continue

            # Track whether wisdom was the reason this email passed filtering
            wisdom_override, wisdom_reason = _was_wisdom_override(email)

            # Get full email content
            try:
                full_email = get_email_content(email_id)
                email_data = {**email, "body": full_email.get("body", "")}
            except Exception as e:
                print(f"[email-monitor] Failed to get email content: {e}")
                email_data = email

            # Create event record
            event = ProactiveEvent(
                source_type="email",
                source_id=email_id,
                raw_data=email_data,
            )
            save_event(event)
            processed += 1

            # Hand off to Doris for processing
            success = _hand_off_to_doris(event, email_data)
            if success:
                update_event_status(event.id, "handed_off")
                handed_off += 1

                # If wisdom was the reason this email got through, log the correct escalation
                if wisdom_override:
                    try:
                        from memory.wisdom import log_escalation_correct
                        log_escalation_correct(
                            source="email",
                            description=f"Correctly escalated: {email.get('subject', '')} from {email.get('sender', '')}",
                            tags=[t for t in ["email", "wisdom-override"] if t],
                        )
                        print(f"[email-monitor] Logged correct escalation: {wisdom_reason}")
                    except Exception as e:
                        print(f"[email-monitor] Failed to log correct escalation: {e}")
            else:
                update_event_status(event.id, "handoff_failed")

        update_checkpoint("email")
        print(f"[email-monitor] Processed {processed} emails, {handed_off} handed to Doris")

    except Exception as e:
        print(f"[email-monitor] Error: {e}")
        import traceback
        traceback.print_exc()


def _hand_off_to_doris(event: ProactiveEvent, email_data: dict) -> bool:
    """
    Hand off an actionable email to Doris for processing.

    Scout scouts, Doris Dorises. We detect, she acts.
    """
    from llm.brain import chat_claude
    from proactive.executor import _create_notification
    from proactive.db import save_action
    from proactive.models import ProactiveAction

    subject = email_data.get("subject", "No subject")
    sender = email_data.get("sender_name") or email_data.get("sender", "Unknown")
    body = email_data.get("body", email_data.get("snippet", ""))[:2000]

    # Wrap untrusted email content to prevent prompt injection
    wrapped_sender = wrap_untrusted(sender, 'email_sender')
    wrapped_subject = wrap_untrusted(subject, 'email_subject')
    wrapped_body = wrap_untrusted(body, 'email_body')

    # Build a prompt for Doris to handle this email
    prompt = f"""[SCOUT HANDOFF] I found an email that looks actionable. Please review and take appropriate action.

From: {wrapped_sender}
Subject: {wrapped_subject}

{wrapped_body}

---
Guidelines:
- If this needs a calendar event, create it with correct date, time, and details
- Put any video/Zoom links in the location field so they're clickable
- Calendar events automatically get a 1-hour alert, so don't create separate reminders for them
- Only create reminders for things that don't have a specific time (like "permission slip due")
- If not actionable, just note that briefly"""

    try:
        print(f"[email-monitor] Handing off to Doris: {subject[:50]}...")

        # Call Doris to handle it
        # chat_claude is an alias for chat() which returns str (or ClaudeResponse
        # with return_usage=True). We don't need usage tracking here.
        response_text = chat_claude(message=prompt)

        # Create a notification action to tell the user what happened
        if response_text:
            # response_text is a str; extract a brief summary for notification
            summary = str(response_text)[:200]
            if len(str(response_text)) > 200:
                summary += "..."

            action = ProactiveAction(
                event_id=event.id,
                action_type="notify",
                action_data={
                    "message": f"Processed email from {sender}: {subject[:50]}",
                    "detail": summary,
                    "priority": "normal"
                },
                status="completed"
            )
            save_action(action)
            notify_action(action)

            print(f"[email-monitor] Doris handled: {subject[:50]}")
            return True
        else:
            print(f"[email-monitor] Doris returned empty response for: {subject[:50]}")
            return False

    except Exception as e:
        print(f"[email-monitor] Failed to hand off to Doris: {e}")
        import traceback
        traceback.print_exc()
        return False


def _is_potentially_actionable(email: dict) -> bool:
    """
    Quick check if an email is worth evaluating.

    This is a fast filter before we call Claude for full evaluation.
    """
    # STARRED EMAILS ALWAYS GET EVALUATED - the user starred it for a reason
    if email.get("is_starred"):
        return True

    # Gmail marked important - worth a look
    if email.get("is_important"):
        return True

    subject = email.get("subject", "").lower()
    snippet = email.get("snippet", "").lower()
    sender = email.get("sender", "").lower()
    text = f"{subject} {snippet}"

    # Check priority senders
    for priority_sender in PRIORITY_SENDERS:
        if priority_sender.lower() in sender:
            return True

    # Check actionable keywords
    for keyword in ACTIONABLE_KEYWORDS:
        if keyword.lower() in text:
            return True

    # Check for date/time patterns (rough heuristic)
    import re
    date_patterns = [
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
        r'\b\d{1,2}/\d{1,2}(/\d{2,4})?\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b.*\d{1,2}',
        r'\b\d{1,2}:\d{2}\s*(am|pm)?\b',
    ]
    for pattern in date_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Check escalation wisdom — did the user previously say emails like this should be escalated?
    try:
        from memory.wisdom import should_escalate_based_on_wisdom
        should_escalate, reason = should_escalate_based_on_wisdom(
            source="email",
            sender=sender,
            subject=subject,
            snippet=snippet,
        )
        if should_escalate:
            print(f"[email-monitor] Wisdom override: {reason}")
            return True
    except Exception as e:
        print(f"[email-monitor] Wisdom check failed (continuing without): {e}")

    return False


def check_specific_email(email_id: str) -> dict:
    """
    Manually check a specific email (for testing or manual triggering).

    Hands off to Doris for processing (scout pattern).
    Returns dict with event and handoff status.
    """
    from tools.gmail import get_email_content

    full_email = get_email_content(email_id)

    event = ProactiveEvent(
        source_type="email",
        source_id=email_id,
        raw_data=full_email,
    )
    save_event(event)

    result = {
        "email": {
            "subject": full_email.get("subject"),
            "sender": full_email.get("sender"),
            "body_preview": full_email.get("body", "")[:500],
        },
        "handoff": None
    }

    # Hand off to Doris
    success = _hand_off_to_doris(event, full_email)
    result["handoff"] = {
        "success": success,
        "status": "handed_off" if success else "handoff_failed"
    }

    return result
