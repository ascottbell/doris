"""Gmail integration for Doris.

Surfaces important emails using multiple signals:
1. Gmail's own importance markers (starred, important label)
2. Emails from people in your Google Contacts
3. Keyword matching for family-relevant stuff (school, deadlines, etc.)
"""

import base64
import html
import json
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scopes - if you change these, delete token.json to re-auth
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/contacts.readonly',
    'https://www.googleapis.com/auth/calendar',
]

CREDENTIALS_PATH = Path(__file__).parent.parent / "credentials" / "credentials.json"
TOKEN_PATH = Path(__file__).parent.parent / "token.json"


class _HTMLToText(HTMLParser):
    """Extract visible text from HTML, skipping <style> and <script> blocks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self._skip = False
        # Block-level elements get a space to prevent word concatenation
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "td"):
            self._pieces.append(" ")

    def handle_data(self, data):
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
        import re
        text = "".join(self._pieces)
        # Collapse whitespace
        return re.sub(r"\s+", " ", text).strip()


def strip_html(raw_html: str) -> str:
    """Convert HTML to plain text using a proper parser."""
    parser = _HTMLToText()
    parser.feed(raw_html)
    return parser.get_text()

# ---------------------------------------------------------------------------
# Encrypted token storage
# ---------------------------------------------------------------------------
# OAuth token is encrypted at rest using Fernet with a key derived from
# DORIS_API_TOKEN via PBKDF2-SHA256. Falls back to plaintext only in dev mode.
# ---------------------------------------------------------------------------

_SALT = b"doris-oauth-token-v1"  # Fixed salt — API token has sufficient entropy


def _get_fernet():
    """Return a Fernet/MultiFernet keyed from DORIS_API_TOKEN, or None in dev mode.

    Supports key rotation: when DORIS_API_TOKEN is comma-separated, encrypt() uses
    the first token and decrypt() tries all tokens in order.
    """
    from config import settings
    from security.crypto import get_fernet
    return get_fernet(settings.doris_api_token, _SALT)


def _read_token() -> Optional[Credentials]:
    """Read OAuth token from disk, decrypting if necessary.

    Handles three cases:
    1. Encrypted token.json (normal production) — decrypt and parse
    2. Plaintext token.json (legacy / dev mode) — parse directly, migrate to
       encrypted if an API token is available
    3. No token.json — return None (triggers OAuth flow)
    """
    if not TOKEN_PATH.exists():
        return None

    raw = TOKEN_PATH.read_bytes()
    fernet = _get_fernet()

    # Try decrypting first (normal case)
    if fernet:
        try:
            plaintext = fernet.decrypt(raw)
            data = json.loads(plaintext)
            return Credentials.from_authorized_user_info(data, SCOPES)
        except Exception:
            pass

    # Fallback: maybe it's still plaintext (legacy or dev mode)
    try:
        data = json.loads(raw)
        creds = Credentials.from_authorized_user_info(data, SCOPES)
        # Migrate to encrypted if we have a key
        if fernet and creds:
            _write_token(creds)
        return creds
    except Exception:
        return None


def _write_token(creds: Credentials) -> None:
    """Write OAuth token to disk, encrypting if an API token is available."""
    plaintext = creds.to_json().encode()
    fernet = _get_fernet()
    if fernet:
        TOKEN_PATH.write_bytes(fernet.encrypt(plaintext))
    else:
        TOKEN_PATH.write_text(creds.to_json())
    # Restrict file permissions to owner-only
    TOKEN_PATH.chmod(0o600)

# Keywords that indicate important emails for the family
# These are fallbacks - Gmail importance and contacts take priority
IMPORTANT_KEYWORDS = [
    # School stuff
    "pajama day", "picture day", "field trip", "early dismissal", 
    "school closed", "snow day", "delay", "permission slip",
    "parent teacher", "conference", "report card",
    # Deadlines
    "deadline", "due date", "expires", "last day",
    "action required", "response needed", "rsvp",
    # Medical/appointments
    "appointment confirmed", "appointment reminder",
    # Kids activities
    "practice canceled", "game canceled", "birthday party invitation",
]

# Senders to always ignore (marketing, newsletters, yourself)
# Add your own email addresses here so you don't surface your own sent mail
IGNORE_SENDERS = [
    "noreply@", "no-reply@", "newsletter@", "marketing@",
    "promotions@", "deals@",
    # Add your own email addresses below to filter out self-sent mail:
    # "you@example.com",
]

# Cache for contacts (refreshed once per session)
_contacts_cache: set[str] = None


def get_gmail_service():
    """Get authenticated Gmail service, prompting for auth if needed."""
    creds = _get_credentials()
    return build('gmail', 'v1', credentials=creds)


def get_people_service():
    """Get authenticated People API service for contacts."""
    creds = _get_credentials()
    return build('people', 'v1', credentials=creds)


def _get_credentials():
    """Get or refresh OAuth credentials."""
    creds = _read_token()

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                print(f"[Gmail] credentials.json not found at {CREDENTIALS_PATH}")
                raise FileNotFoundError(
                    "Gmail credentials not configured. Check server setup."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)

        _write_token(creds)

    return creds


def get_contacts_emails() -> set[str]:
    """Get all email addresses from Google Contacts."""
    global _contacts_cache
    
    if _contacts_cache is not None:
        return _contacts_cache
    
    service = get_people_service()
    emails = set()
    page_token = None
    
    while True:
        results = service.people().connections().list(
            resourceName='people/me',
            pageSize=1000,
            personFields='emailAddresses',
            pageToken=page_token
        ).execute()
        
        connections = results.get('connections', [])
        for person in connections:
            for email_obj in person.get('emailAddresses', []):
                email = email_obj.get('value', '').lower()
                if email:
                    emails.add(email)
        
        page_token = results.get('nextPageToken')
        if not page_token:
            break
    
    _contacts_cache = emails
    return emails


def scan_recent_emails(hours: int = 24, max_results: int = 50) -> list[dict]:
    """
    Scan recent emails and return important ones.
    
    Importance signals (in priority order):
    1. Starred by user
    2. Marked important by Gmail
    3. From someone in Contacts
    4. Contains important keywords
    
    Returns list of dicts with: subject, sender, snippet, date, why_important
    """
    service = get_gmail_service()
    contacts = get_contacts_emails()
    
    # Calculate the date filter
    after_date = datetime.now() - timedelta(hours=hours)
    query = f"after:{after_date.strftime('%Y/%m/%d')}"
    
    results = service.users().messages().list(
        userId='me',
        q=query,
        maxResults=max_results
    ).execute()
    
    messages = results.get('messages', [])
    important_emails = []
    
    for msg in messages:
        msg_data = service.users().messages().get(
            userId='me',
            id=msg['id'],
            format='metadata',
            metadataHeaders=['Subject', 'From', 'Date']
        ).execute()
        
        headers = {h['name']: h['value'] for h in msg_data.get('payload', {}).get('headers', [])}
        subject = headers.get('Subject', '(no subject)')
        sender = headers.get('From', '')
        date = headers.get('Date', '')
        snippet = msg_data.get('snippet', '')
        labels = msg_data.get('labelIds', [])
        
        sender_email = extract_email(sender).lower()
        sender_name = extract_name(sender)
        
        # Skip ignored senders
        if should_ignore(sender_email):
            continue
        
        # Check importance signals
        importance_reasons = []
        
        # Signal 1: Starred
        if 'STARRED' in labels:
            importance_reasons.append('starred')
        
        # Signal 2: Gmail's important marker
        if 'IMPORTANT' in labels:
            importance_reasons.append('marked important')
        
        # Signal 3: From a contact
        if sender_email in contacts:
            importance_reasons.append('from contact')
        
        # Signal 4: Keyword match (only if no other signals)
        if not importance_reasons:
            keyword_match = check_keywords(subject, snippet)
            if keyword_match:
                importance_reasons.append(keyword_match)
        
        if importance_reasons:
            important_emails.append({
                'subject': subject,
                'sender': sender_email,
                'sender_name': sender_name,
                'snippet': snippet[:200],
                'date': date,
                'why_important': importance_reasons,
                'id': msg['id'],
                'is_starred': 'STARRED' in labels,
                'is_important': 'IMPORTANT' in labels,
                'is_contact': sender_email in contacts,
            })
    
    # Sort: starred first, then important, then contacts, then keywords
    important_emails.sort(key=lambda x: (
        not x['is_starred'],
        not x['is_important'], 
        not x['is_contact'],
    ))
    
    return important_emails


def should_ignore(sender_email: str) -> bool:
    """Check if sender should be ignored."""
    for pattern in IGNORE_SENDERS:
        if pattern.lower() in sender_email:
            return True
    return False


def check_keywords(subject: str, snippet: str) -> Optional[str]:
    """Check if email contains important keywords. Returns the keyword or None."""
    text_to_check = f"{subject} {snippet}".lower()
    
    for keyword in IMPORTANT_KEYWORDS:
        if keyword.lower() in text_to_check:
            return f"contains '{keyword}'"
    
    return None


def extract_email(from_header: str) -> str:
    """Extract email address from From header."""
    if '<' in from_header and '>' in from_header:
        start = from_header.index('<') + 1
        end = from_header.index('>')
        return from_header[start:end]
    return from_header


def extract_name(from_header: str) -> str:
    """Extract name from From header."""
    if '<' in from_header:
        return from_header[:from_header.index('<')].strip().strip('"')
    return from_header


def get_unread_count() -> int:
    """Get count of unread emails in inbox."""
    service = get_gmail_service()
    results = service.users().messages().list(
        userId='me',
        q='is:unread in:inbox',
        maxResults=1
    ).execute()
    return results.get('resultSizeEstimate', 0)


def parse_email_query(natural_query: str) -> str:
    """
    Convert natural language to Gmail search syntax.

    Examples:
        "emails from Jane" → "from:jane"
        "emails about the school trip" → "school trip"
        "emails from last week" → "newer_than:7d"
        "emails with attachments from the school" → "has:attachment from:school"
    """
    import re

    query = natural_query.lower().strip()
    parts = []

    # Remove common filler words at the start
    query = re.sub(r'^(show me |find |get |search for |look for |any )?(my )?(emails? |messages? )?(about |regarding |concerning )?', '', query)

    # Handle time ranges FIRST (before "from" which could match "from last week")
    time_patterns = [
        (r'\b(from\s+)?last\s+(\d+)\s+days?\b', lambda m: f"newer_than:{m.group(2)}d"),
        (r'\b(from\s+)?last\s+week\b', lambda m: "newer_than:7d"),
        (r'\b(from\s+)?last\s+month\b', lambda m: "newer_than:30d"),
        (r'\b(from\s+)?past\s+(\d+)\s+days?\b', lambda m: f"newer_than:{m.group(2)}d"),
        (r'\b(from\s+)?past\s+week\b', lambda m: "newer_than:7d"),
        (r'\b(from\s+)?past\s+month\b', lambda m: "newer_than:30d"),
        (r'\bthis\s+week\b', lambda m: "newer_than:7d"),
        (r'\bthis\s+month\b', lambda m: "newer_than:30d"),
        (r'\btoday\b', lambda m: "newer_than:1d"),
        (r'\byesterday\b', lambda m: "newer_than:2d older_than:1d"),
    ]

    for pattern, replacement in time_patterns:
        match = re.search(pattern, query)
        if match:
            parts.append(replacement(match))
            query = re.sub(pattern, '', query)
            break

    # Handle "from [name/email]" - be careful not to match "from the" without a name
    from_match = re.search(r'\bfrom\s+(?!the\s+)([a-zA-Z0-9@._-]+)', query)
    if from_match:
        sender = from_match.group(1).strip()
        # Remove common suffixes
        sender = re.sub(r"'s?$", '', sender)
        parts.append(f"from:{sender}")
        query = query.replace(from_match.group(0), '')

    # Handle "from the [place]" as search term, not from:
    from_the_match = re.search(r'\bfrom\s+the\s+(\w+)', query)
    if from_the_match:
        place = from_the_match.group(1)
        parts.append(f"from:{place}")
        query = query.replace(from_the_match.group(0), '')

    # Handle "to [name/email]"
    to_match = re.search(r'\bto\s+([a-zA-Z0-9@._-]+)', query)
    if to_match:
        recipient = to_match.group(1).strip()
        parts.append(f"to:{recipient}")
        query = query.replace(to_match.group(0), '')

    # Handle attachment filter
    if re.search(r'\bwith\s+attachments?\b|\bhas\s+attachments?\b|\battachments?\b', query):
        parts.append("has:attachment")
        query = re.sub(r'\bwith\s+attachments?\b|\bhas\s+attachments?\b|\battachments?\b', '', query)

    # Handle unread filter
    if re.search(r'\bunread\b', query):
        parts.append("is:unread")
        query = re.sub(r'\bunread\b', '', query)

    # Handle starred filter
    if re.search(r'\bstarred\b', query):
        parts.append("is:starred")
        query = re.sub(r'\bstarred\b', '', query)

    # Handle important filter
    if re.search(r'\bimportant\b', query):
        parts.append("is:important")
        query = re.sub(r'\bimportant\b', '', query)

    # Clean up remaining text as search terms
    remaining = re.sub(r'\s+', ' ', query).strip()
    # Remove filler words
    filler = ('and', 'or', 'the', 'a', 'an', 'emails', 'email', 'messages', 'message')
    remaining = re.sub(r'^(and|or|the|a|an|emails?|messages?)\s*', '', remaining)
    remaining = re.sub(r'\s+(and|or|the|a|an|emails?|messages?)$', '', remaining)
    remaining = remaining.strip()

    if remaining and remaining not in filler:
        parts.append(remaining)

    return ' '.join(parts) if parts else natural_query


def search_emails(query: str, max_results: int = 10) -> list[dict]:
    """
    Search emails using Gmail search syntax.

    Args:
        query: Gmail search query (or natural language, will be converted)
        max_results: Maximum number of results to return

    Returns:
        List of dicts with: id, subject, sender, sender_name, date, snippet, has_attachments
    """
    service = get_gmail_service()

    # Convert natural language to Gmail syntax if needed
    # If query doesn't look like Gmail syntax, parse it
    if not any(op in query for op in [':', 'is:', 'has:', 'from:', 'to:', 'subject:']):
        query = parse_email_query(query)

    results = service.users().messages().list(
        userId='me',
        q=query,
        maxResults=max_results
    ).execute()

    messages = results.get('messages', [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId='me',
            id=msg['id'],
            format='metadata',
            metadataHeaders=['Subject', 'From', 'Date']
        ).execute()

        headers = {h['name']: h['value'] for h in msg_data.get('payload', {}).get('headers', [])}
        labels = msg_data.get('labelIds', [])

        # Check for attachments by looking at payload parts
        has_attachments = False
        payload = msg_data.get('payload', {})
        parts = payload.get('parts', [])
        for part in parts:
            if part.get('filename'):
                has_attachments = True
                break

        sender = headers.get('From', '')
        emails.append({
            'id': msg['id'],
            'subject': headers.get('Subject', '(no subject)'),
            'sender': extract_email(sender).lower(),
            'sender_name': extract_name(sender),
            'date': headers.get('Date', ''),
            'snippet': msg_data.get('snippet', ''),
            'has_attachments': has_attachments,
        })

    return emails


def get_email_content(message_id: str) -> dict:
    """
    Fetch the full content of an email.

    Args:
        message_id: The Gmail message ID

    Returns:
        Dict with: id, subject, sender, sender_name, date, body, attachments
    """
    service = get_gmail_service()

    msg_data = service.users().messages().get(
        userId='me',
        id=message_id,
        format='full'
    ).execute()

    headers = {h['name']: h['value'] for h in msg_data.get('payload', {}).get('headers', [])}
    sender = headers.get('From', '')

    # Extract body and attachments
    body = ""
    attachments = []
    payload = msg_data.get('payload', {})

    def extract_parts(part):
        """Recursively extract text and attachment info from message parts."""
        nonlocal body, attachments

        mime_type = part.get('mimeType', '')
        filename = part.get('filename', '')

        # Track attachments
        if filename:
            attachments.append(filename)

        # Extract text content
        if mime_type == 'text/plain' and not filename:
            data = part.get('body', {}).get('data', '')
            if data:
                decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                body = decoded
        elif mime_type == 'text/html' and not body and not filename:
            # Fall back to HTML if no plain text — use proper parser
            data = part.get('body', {}).get('data', '')
            if data:
                decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                body = strip_html(decoded)

        # Recurse into nested parts
        for subpart in part.get('parts', []):
            extract_parts(subpart)

    # Handle single-part messages
    if 'parts' not in payload:
        extract_parts(payload)
    else:
        for part in payload.get('parts', []):
            extract_parts(part)

    return {
        'id': message_id,
        'subject': headers.get('Subject', '(no subject)'),
        'sender': extract_email(sender).lower(),
        'sender_name': extract_name(sender),
        'date': headers.get('Date', ''),
        'body': body,
        'attachments': attachments,
    }


def forward_to_self(message_id: str, note: str = None) -> dict:
    """
    Forward an email to the user's own address.

    The recipient address is determined from the authenticated Gmail account.

    Args:
        message_id: The Gmail message ID to forward
        note: Optional note to include at the top

    Returns:
        Dict with 'success' bool and 'message_id' or 'error'
    """
    try:
        # Get the user's own email address from the Gmail API
        service = get_gmail_service()
        profile = service.users().getProfile(userId='me').execute()
        own_email = profile.get('emailAddress', '')

        if not own_email:
            return {'success': False, 'error': 'Could not determine own email address'}

        # Get the original email
        email = get_email_content(message_id)

        # Build the forwarded message
        forward_body = ""
        if note:
            forward_body += f"Doris forwarded this because: {note}\n\n"
            forward_body += "=" * 50 + "\n\n"

        forward_body += f"---------- Forwarded message ----------\n"
        forward_body += f"From: {email['sender_name']} <{email['sender']}>\n"
        forward_body += f"Date: {email['date']}\n"
        forward_body += f"Subject: {email['subject']}\n\n"
        forward_body += email['body']

        # Send the forward
        subject = email['subject']
        if not subject.lower().startswith('fwd:'):
            subject = f"Fwd: {subject}"

        return send_email(
            to=own_email,
            subject=subject,
            body=forward_body
        )

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }


def format_email(email: dict, include_body: bool = False) -> str:
    """
    Format an email as natural language summary.

    Args:
        email: Email dict (from search_emails or get_email_content)
        include_body: If True, include first ~200 words of body

    Returns:
        Natural language string summarizing the email
    """
    sender = email.get('sender_name') or email.get('sender', 'someone')
    subject = email.get('subject', 'no subject')

    # Truncate long subjects
    if len(subject) > 60:
        subject = subject[:57] + "..."

    # Parse date for relative time
    date_str = email.get('date', '')
    relative_time = ""
    if date_str:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
            diff = now - dt

            if diff.days == 0:
                hours = diff.seconds // 3600
                if hours == 0:
                    relative_time = "just now"
                elif hours == 1:
                    relative_time = "an hour ago"
                else:
                    relative_time = f"{hours} hours ago"
            elif diff.days == 1:
                relative_time = "yesterday"
            elif diff.days < 7:
                relative_time = f"{diff.days} days ago"
            else:
                relative_time = dt.strftime("%B %d")
        except Exception:
            pass

    # Build the voice output
    if include_body:
        body = email.get('body', '')
        if not body:
            # Maybe we only have a snippet
            body = email.get('snippet', '')

        # Truncate to ~200 words
        words = body.split()
        if len(words) > 200:
            body = ' '.join(words[:200]) + "..."

        # Clean up for voice
        body = body.replace('\n', ' ').replace('  ', ' ').strip()

        if relative_time:
            result = f"Email from {sender}, {relative_time}, about {subject}. "
        else:
            result = f"Email from {sender} about {subject}. "

        if body:
            result += f"It says: {body}"

        return result
    else:
        # Short version
        if relative_time:
            return f"Email from {sender}, {relative_time}: {subject}"
        else:
            return f"Email from {sender}: {subject}"


def send_email(to: str, subject: str, body: str) -> dict:
    """
    Send an email via Gmail.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body (plain text)

    Returns:
        dict with 'success' bool and 'message_id' or 'error'
    """
    try:
        service = get_gmail_service()

        # Create the email message
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject

        # Encode as base64
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')

        # Send it
        sent = service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()

        return {
            'success': True,
            'message_id': sent.get('id'),
        }

    except Exception as e:
        error_msg = str(e)

        # Check if it's a scope/auth issue
        if 'insufficient' in error_msg.lower() or 'scope' in error_msg.lower():
            # Delete token to force re-auth on next call
            if TOKEN_PATH.exists():
                TOKEN_PATH.unlink()
            return {
                'success': False,
                'error': 'Gmail send scope not authorized. Token deleted - please retry to re-authenticate.',
            }

        return {
            'success': False,
            'error': error_msg,
        }


def summarize_important_emails(hours: int = 24) -> str:
    """Get a natural language summary of important emails."""
    emails = scan_recent_emails(hours=hours)
    
    if not emails:
        return f"No important emails in the last {hours} hours."
    
    summary_parts = [f"Found {len(emails)} important email{'s' if len(emails) > 1 else ''} in the last {hours} hours:"]
    
    for email in emails:
        sender = email['sender_name'] or email['sender']
        reasons = ', '.join(email['why_important'])
        summary_parts.append(f"• {sender}: {email['subject']}")
        summary_parts.append(f"  ({reasons})")
    
    return '\n'.join(summary_parts)


# For testing
if __name__ == "__main__":
    import sys

    def print_usage():
        print("Usage:")
        print("  python tools/gmail.py                    - Show important emails")
        print("  python tools/gmail.py search <query>     - Search emails")
        print("  python tools/gmail.py read <message_id>  - Read full email content")
        print("  python tools/gmail.py forward <msg_id>   - Forward email to self")
        print("  python tools/gmail.py send <email>       - Send test email")
        print("  python tools/gmail.py parse <query>      - Test query parsing")
        print("\nExamples:")
        print('  python tools/gmail.py search "from:school"')
        print('  python tools/gmail.py search "emails from jane last week"')
        print('  python tools/gmail.py search "unread with attachments"')

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "search":
            if len(sys.argv) < 3:
                print("Usage: python tools/gmail.py search <query>")
                sys.exit(1)
            query = ' '.join(sys.argv[2:])
            print(f"Searching for: {query}")

            # Show what the query was parsed to
            parsed = parse_email_query(query)
            if parsed != query:
                print(f"Parsed to: {parsed}")

            print("\nResults:")
            emails = search_emails(query, max_results=10)
            if not emails:
                print("No emails found.")
            else:
                for i, email in enumerate(emails, 1):
                    voice_summary = format_email(email)
                    print(f"{i}. {voice_summary}")
                    print(f"   ID: {email['id']}")
                    if email.get('has_attachments'):
                        print("   📎 Has attachments")
                    print()

        elif command == "read":
            if len(sys.argv) < 3:
                print("Usage: python tools/gmail.py read <message_id>")
                sys.exit(1)
            msg_id = sys.argv[2]
            print(f"Reading email: {msg_id}\n")
            email = get_email_content(msg_id)
            print(f"From: {email['sender_name']} <{email['sender']}>")
            print(f"Date: {email['date']}")
            print(f"Subject: {email['subject']}")
            if email['attachments']:
                print(f"Attachments: {', '.join(email['attachments'])}")
            print(f"\n{'-' * 50}\n")
            print(email['body'][:2000])
            if len(email['body']) > 2000:
                print(f"\n... (truncated, {len(email['body'])} chars total)")
            print(f"\n{'-' * 50}")
            print("\nVoice summary:")
            print(format_email(email, include_body=True))

        elif command == "forward":
            if len(sys.argv) < 3:
                print("Usage: python tools/gmail.py forward <message_id> [note]")
                sys.exit(1)
            msg_id = sys.argv[2]
            note = ' '.join(sys.argv[3:]) if len(sys.argv) > 3 else None
            print(f"Forwarding email {msg_id} to self...")
            if note:
                print(f"Note: {note}")
            result = forward_to_self(msg_id, note=note)
            print(f"Result: {result}")

        elif command == "send":
            if len(sys.argv) < 3:
                print("Usage: python tools/gmail.py send <email>")
                sys.exit(1)
            recipient = sys.argv[2]
            print(f"Sending test email to {recipient}...")
            result = send_email(
                to=recipient,
                subject="Test from Doris",
                body="This is a test email sent from Doris."
            )
            print(f"Result: {result}")

        elif command == "parse":
            if len(sys.argv) < 3:
                print("Usage: python tools/gmail.py parse <query>")
                sys.exit(1)
            query = ' '.join(sys.argv[2:])
            parsed = parse_email_query(query)
            print(f"Input:  {query}")
            print(f"Output: {parsed}")

        elif command in ("help", "-h", "--help"):
            print_usage()

        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)

    else:
        print("Testing Gmail integration...")

        # Load contacts
        contacts = get_contacts_emails()
        print(f"Loaded {len(contacts)} contacts")

        print(f"\nUnread count: {get_unread_count()}")
        print("\nImportant emails (last 48 hours):")
        print(summarize_important_emails(hours=48))
