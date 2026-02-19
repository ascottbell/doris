"""
Email sender pattern analysis for Doris memory.

Analyzes 90 days of email to identify:
- Important senders (high engagement)
- Noise senders (can be ignored)
- Domain patterns (e.g., @schoolname.edu = school)
"""

import os
from collections import defaultdict
from datetime import datetime, timedelta


def analyze_senders(days: int = 90) -> dict:
    """
    Analyze email senders over the last N days.

    Returns:
        Dict with sender stats, domain patterns, and insights
    """
    from tools.gmail import get_gmail_service, extract_email, extract_name

    service = get_gmail_service()

    # Calculate date filter
    after_date = datetime.now() - timedelta(days=days)
    query = f"after:{after_date.strftime('%Y/%m/%d')}"

    # Fetch all emails (paginate if needed)
    all_messages = []
    page_token = None

    while True:
        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=500,
            pageToken=page_token
        ).execute()

        all_messages.extend(results.get('messages', []))
        page_token = results.get('nextPageToken')

        if not page_token or len(all_messages) >= 2000:  # Cap at 2000
            break

    print(f"[EmailPatterns] Analyzing {len(all_messages)} emails from last {days} days")

    # Track sender stats
    sender_stats = defaultdict(lambda: {
        'count': 0,
        'names': set(),
        'replied_to': 0,
        'starred': 0,
        'important': 0,
        'last_seen': None
    })

    domain_stats = defaultdict(lambda: {
        'count': 0,
        'senders': set()
    })

    # Analyze each email
    for i, msg in enumerate(all_messages):
        if i % 100 == 0:
            print(f"[EmailPatterns] Processing {i}/{len(all_messages)}...")

        try:
            msg_data = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['From', 'Date', 'In-Reply-To']
            ).execute()

            headers = {h['name']: h['value'] for h in msg_data.get('payload', {}).get('headers', [])}
            labels = msg_data.get('labelIds', [])

            from_header = headers.get('From', '')
            sender_email = extract_email(from_header).lower()
            sender_name = extract_name(from_header)

            if not sender_email:
                continue
            # Skip the user's own email (sent messages)
            # This is determined dynamically, but as a fallback we skip common patterns
            if sender_email == os.getenv("DORIS_USER_EMAIL", "").lower():
                continue

            # Update sender stats
            stats = sender_stats[sender_email]
            stats['count'] += 1
            if sender_name:
                stats['names'].add(sender_name)
            if 'STARRED' in labels:
                stats['starred'] += 1
            if 'IMPORTANT' in labels:
                stats['important'] += 1

            # Track date
            date_str = headers.get('Date', '')
            if date_str:
                stats['last_seen'] = date_str

            # Check if this is a reply (indicates engagement)
            if headers.get('In-Reply-To'):
                stats['replied_to'] += 1

            # Track domain
            if '@' in sender_email:
                domain = sender_email.split('@')[1]
                domain_stats[domain]['count'] += 1
                domain_stats[domain]['senders'].add(sender_email)

        except Exception as e:
            continue

    # Calculate engagement scores
    for email, stats in sender_stats.items():
        # Engagement = (starred * 3 + important * 2 + replied * 5) / count
        engagement = (
            stats['starred'] * 3 +
            stats['important'] * 2 +
            stats['replied_to'] * 5
        ) / max(stats['count'], 1)
        stats['engagement_score'] = round(engagement, 2)
        stats['names'] = list(stats['names'])  # Convert set to list

    # Categorize senders
    high_engagement = []
    medium_engagement = []
    noise_senders = []

    for email, stats in sender_stats.items():
        if stats['engagement_score'] >= 2 or stats['count'] >= 10:
            if stats['engagement_score'] >= 1:
                high_engagement.append((email, stats))
            else:
                medium_engagement.append((email, stats))
        elif stats['count'] >= 3 and stats['engagement_score'] < 0.5:
            noise_senders.append((email, stats))

    # Sort by engagement
    high_engagement.sort(key=lambda x: -x[1]['engagement_score'])
    noise_senders.sort(key=lambda x: -x[1]['count'])

    # Identify important domains
    important_domains = []
    for domain, stats in domain_stats.items():
        stats['senders'] = list(stats['senders'])
        if stats['count'] >= 5:
            # Check if senders from this domain are engaged with
            avg_engagement = sum(
                sender_stats[s].get('engagement_score', 0)
                for s in stats['senders']
            ) / len(stats['senders'])

            if avg_engagement >= 1:
                important_domains.append({
                    'domain': domain,
                    'count': stats['count'],
                    'avg_engagement': round(avg_engagement, 2)
                })

    important_domains.sort(key=lambda x: -x['avg_engagement'])

    return {
        'total_emails': len(all_messages),
        'unique_senders': len(sender_stats),
        'high_engagement': high_engagement[:20],  # Top 20
        'noise_senders': noise_senders[:20],  # Top 20
        'important_domains': important_domains[:10],
        'date_range': {
            'start': after_date.date().isoformat(),
            'end': datetime.now().date().isoformat()
        }
    }


def store_email_patterns():
    """Analyze email and store patterns in memory."""
    from memory.store import store_memory

    analysis = analyze_senders(days=90)
    stored = 0

    # Store high-engagement senders
    for email, stats in analysis.get('high_engagement', [])[:10]:
        name = stats['names'][0] if stats['names'] else email.split('@')[0]
        content = f"{name} ({email}) is an important email contact - {stats['count']} emails, engagement score {stats['engagement_score']}"
        store_memory(
            content=content,
            category="contact",
            subject=name,
            source="email:pattern_analysis",
            confidence=0.85
        )
        stored += 1
        print(f"[EmailPatterns] Stored: Important contact - {name}")

    # Store important domains
    for domain_info in analysis.get('important_domains', []):
        domain = domain_info['domain']

        # Generate domain insight
        if 'school' in domain or domain.endswith('.edu'):
            insight = f"Emails from {domain} are school-related - high priority"
        elif domain_info['avg_engagement'] >= 2:
            insight = f"Emails from {domain} are frequently engaged with - important domain"
        else:
            insight = f"{domain} is a commonly seen email domain"

        store_memory(
            content=insight,
            category="preference",
            subject="email",
            source="email:pattern_analysis",
            confidence=0.8
        )
        stored += 1
        print(f"[EmailPatterns] Stored: {insight}")

    # Store noise patterns
    noise_senders = analysis.get('noise_senders', [])
    if noise_senders:
        noise_list = [s[0] for s in noise_senders[:5]]
        content = f"Low-engagement email senders that can be deprioritized: {', '.join(noise_list)}"
        store_memory(
            content=content,
            category="preference",
            subject="email",
            source="email:pattern_analysis",
            confidence=0.75
        )
        stored += 1
        print(f"[EmailPatterns] Stored: Noise senders list")

    print(f"[EmailPatterns] Analyzed {analysis['total_emails']} emails, {analysis['unique_senders']} senders, stored {stored} patterns")
    return analysis


if __name__ == "__main__":
    result = store_email_patterns()

    print(f"\n=== Email Analysis Summary ===")
    print(f"Total emails: {result['total_emails']}")
    print(f"Unique senders: {result['unique_senders']}")

    print(f"\nTop engaged senders:")
    for email, stats in result['high_engagement'][:5]:
        name = stats['names'][0] if stats['names'] else email
        print(f"  {name}: {stats['count']} emails, score {stats['engagement_score']}")

    print(f"\nImportant domains:")
    for d in result['important_domains'][:5]:
        print(f"  {d['domain']}: {d['count']} emails, engagement {d['avg_engagement']}")
