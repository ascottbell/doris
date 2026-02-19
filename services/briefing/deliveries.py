"""
Delivery Tracker for morning briefing.

Scans recent emails for shipping notifications from:
- Amazon
- UPS
- FedEx
- USPS
"""

from datetime import datetime, timedelta
import re

from security.prompt_safety import wrap_with_scan


def get_expected_deliveries() -> dict:
    """
    Scan email for expected deliveries.

    Returns:
        dict with 'deliveries' list and 'summary' string
    """
    from tools.gmail import get_gmail_service, extract_email

    service = get_gmail_service()
    deliveries = []

    # Search queries for shipping emails
    shipping_queries = [
        # Amazon
        'from:ship-confirm@amazon.com newer_than:5d',
        'from:shipment-tracking@amazon.com newer_than:5d',
        # UPS
        'from:mcinfo@ups.com newer_than:5d subject:delivery',
        # FedEx
        'from:TrackingUpdates@fedex.com newer_than:5d',
        # USPS
        'from:USPSInformedDelivery@usps.gov newer_than:3d',
        'from:auto-reply@usps.com newer_than:5d',
    ]

    seen_subjects = set()

    for query in shipping_queries:
        try:
            results = service.users().messages().list(
                userId='me',
                q=query,
                maxResults=10
            ).execute()

            messages = results.get('messages', [])

            for msg in messages:
                msg_data = service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata',
                    metadataHeaders=['Subject', 'From', 'Date']
                ).execute()

                headers = {h['name']: h['value'] for h in msg_data.get('payload', {}).get('headers', [])}
                subject = headers.get('Subject', '')
                from_addr = headers.get('From', '')
                date_str = headers.get('Date', '')

                # Skip duplicates
                if subject in seen_subjects:
                    continue
                seen_subjects.add(subject)

                # Determine carrier
                carrier = "Unknown"
                if 'amazon' in from_addr.lower():
                    carrier = "Amazon"
                elif 'ups' in from_addr.lower():
                    carrier = "UPS"
                elif 'fedex' in from_addr.lower():
                    carrier = "FedEx"
                elif 'usps' in from_addr.lower():
                    carrier = "USPS"

                # Extract delivery date if possible
                delivery_date = None
                today_patterns = [
                    r'arriving today',
                    r'delivered today',
                    r'out for delivery',
                ]
                tomorrow_patterns = [
                    r'arriving tomorrow',
                    r'expected.*tomorrow',
                ]
                date_patterns = [
                    r'arriving (\w+day, \w+ \d+)',
                    r'delivery.*(\w+day, \w+ \d+)',
                    r'expected.*(\w+day)',
                ]

                subject_lower = subject.lower()

                if any(re.search(p, subject_lower) for p in today_patterns):
                    delivery_date = "today"
                elif any(re.search(p, subject_lower) for p in tomorrow_patterns):
                    delivery_date = "tomorrow"
                else:
                    for pattern in date_patterns:
                        match = re.search(pattern, subject_lower)
                        if match:
                            delivery_date = match.group(1)
                            break

                # Extract item description for Amazon
                item = None
                if carrier == "Amazon":
                    # Try to extract item name from subject
                    item_match = re.search(r'shipped[:\s]+(.+?)(?:\s+arriving|\s+estimated|\s*$)', subject, re.IGNORECASE)
                    if item_match:
                        item = item_match.group(1).strip()[:50]

                deliveries.append({
                    "carrier": carrier,
                    "subject": wrap_with_scan(subject[:100], "delivery_subject"),
                    "item": wrap_with_scan(item, "delivery_item") if item else None,
                    "expected": delivery_date,
                    "date_received": date_str
                })

        except Exception as e:
            print(f"[Deliveries] Error with query '{query}': {e}")
            continue

    # Dedupe and sort by expected date
    today_deliveries = [d for d in deliveries if d.get("expected") == "today"]
    tomorrow_deliveries = [d for d in deliveries if d.get("expected") == "tomorrow"]
    other_deliveries = [d for d in deliveries if d.get("expected") not in ["today", "tomorrow", None]]
    unknown_deliveries = [d for d in deliveries if d.get("expected") is None]

    sorted_deliveries = today_deliveries + tomorrow_deliveries + other_deliveries + unknown_deliveries

    # Build summary
    if not deliveries:
        summary = "No packages expected."
    elif today_deliveries:
        summary = f"{len(today_deliveries)} package(s) arriving today"
        if tomorrow_deliveries:
            summary += f", {len(tomorrow_deliveries)} tomorrow"
    elif tomorrow_deliveries:
        summary = f"{len(tomorrow_deliveries)} package(s) expected tomorrow"
    else:
        summary = f"{len(deliveries)} package(s) in transit"

    return {
        "deliveries": sorted_deliveries[:10],  # Cap at 10
        "count": len(deliveries),
        "today_count": len(today_deliveries),
        "tomorrow_count": len(tomorrow_deliveries),
        "summary": summary
    }


def format_summary() -> str:
    """Format deliveries as natural language."""
    data = get_expected_deliveries()

    if data["count"] == 0:
        return "No packages expected."

    parts = []

    if data["today_count"] > 0:
        if data["today_count"] == 1:
            parts.append("You have 1 package arriving today")
        else:
            parts.append(f"You have {data['today_count']} packages arriving today")

        # Mention items if Amazon
        today = [d for d in data["deliveries"] if d.get("expected") == "today"]
        amazon_items = [d["item"] for d in today if d.get("item")]
        if amazon_items:
            parts.append(f"including {amazon_items[0]}")

    if data["tomorrow_count"] > 0:
        if parts:
            parts.append(f"and {data['tomorrow_count']} tomorrow")
        else:
            parts.append(f"{data['tomorrow_count']} package(s) expected tomorrow")

    other_count = data["count"] - data["today_count"] - data["tomorrow_count"]
    if other_count > 0 and not parts:
        parts.append(f"{other_count} package(s) in transit")

    return ". ".join(parts) + "." if parts else "No packages expected."


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Checking for deliveries...")
    result = get_expected_deliveries()

    print(f"\nDeliveries ({result['count']} total)")
    print("=" * 50)
    print(f"Today: {result['today_count']}")
    print(f"Tomorrow: {result['tomorrow_count']}")

    for d in result["deliveries"]:
        print(f"\n[{d['carrier']}] Expected: {d['expected'] or 'Unknown'}")
        print(f"  {d['subject']}")
        if d.get("item"):
            print(f"  Item: {d['item']}")

    print(f"\nFormatted: {format_summary()}")
