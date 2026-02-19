"""
macOS Contacts integration for Doris.
Uses pyobjc to access the system Contacts framework.
"""

import Contacts
from typing import Optional


def _get_contact_store() -> Contacts.CNContactStore:
    """Get the shared contact store."""
    return Contacts.CNContactStore.alloc().init()


def _request_access() -> bool:
    """Request access to contacts. Returns True if granted."""
    store = _get_contact_store()
    # Check current authorization status
    status = Contacts.CNContactStore.authorizationStatusForEntityType_(
        Contacts.CNEntityTypeContacts
    )
    return status == Contacts.CNAuthorizationStatusAuthorized


def lookup_contact(name: str) -> Optional[dict]:
    """
    Look up a contact by name.

    Args:
        name: Name to search for (first, last, or full name)

    Returns:
        dict with 'name', 'phones', 'emails' or None if not found
    """
    if not _request_access():
        return None

    store = _get_contact_store()

    # Keys we want to fetch
    keys = [
        Contacts.CNContactGivenNameKey,
        Contacts.CNContactFamilyNameKey,
        Contacts.CNContactNicknameKey,
        Contacts.CNContactPhoneNumbersKey,
        Contacts.CNContactEmailAddressesKey,
    ]

    # Create predicate for name search
    predicate = Contacts.CNContact.predicateForContactsMatchingName_(name)

    try:
        contacts, error = store.unifiedContactsMatchingPredicate_keysToFetch_error_(
            predicate, keys, None
        )
        if error:
            return None
    except Exception:
        return None

    if not contacts or len(contacts) == 0:
        return None

    # Take the first match
    contact = contacts[0]

    # Extract phone numbers
    phones = []
    for phone in contact.phoneNumbers():
        label = phone.label() or ""
        # Clean up label (remove _$!<>!$_ wrapper)
        if label.startswith("_$!<"):
            label = label[4:-4]
        number = phone.value().stringValue()
        phones.append({"label": label, "number": number})

    # Extract emails
    emails = []
    for email in contact.emailAddresses():
        label = email.label() or ""
        if label.startswith("_$!<"):
            label = label[4:-4]
        address = email.value()
        emails.append({"label": label, "email": str(address)})

    # Build full name
    given = contact.givenName() or ""
    family = contact.familyName() or ""
    full_name = f"{given} {family}".strip() or name

    return {
        "name": full_name,
        "phones": phones,
        "emails": emails,
    }


def get_phone_number(name: str, prefer_mobile: bool = True) -> Optional[str]:
    """
    Get the best phone number for a contact.

    Args:
        name: Contact name to look up
        prefer_mobile: If True, prefer mobile/iPhone numbers

    Returns:
        Phone number string or None
    """
    contact = lookup_contact(name)
    if not contact or not contact["phones"]:
        return None

    phones = contact["phones"]

    if prefer_mobile:
        # Look for mobile/iPhone first
        for phone in phones:
            label = phone["label"].lower()
            if "mobile" in label or "iphone" in label:
                return phone["number"]

    # Return first available
    return phones[0]["number"]


def get_email(name: str) -> Optional[str]:
    """
    Get the primary email for a contact.

    Returns:
        Email string or None
    """
    contact = lookup_contact(name)
    if not contact or not contact["emails"]:
        return None

    return contact["emails"][0]["email"]


# Common name mappings for family — configure with your own aliases
# Example: {"wife": "Jane", "mom": "Susan"}
FAMILY_ALIASES = {
    # "wife": "Jane",
    # "husband": "John",
}


def resolve_recipient(name: str) -> Optional[str]:
    """
    Resolve a name or alias to a phone number for messaging.

    Handles:
    - Family aliases (mom, wife, etc.)
    - Direct contact name lookup

    Args:
        name: Name or alias to resolve

    Returns:
        Phone number string or None
    """
    # Check for family aliases
    lookup_name = FAMILY_ALIASES.get(name.lower().strip(), name)

    # Look up in Contacts
    return get_phone_number(lookup_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        name = " ".join(sys.argv[1:])
        print(f"Looking up: {name}")

        contact = lookup_contact(name)
        if contact:
            print(f"\nFound: {contact['name']}")
            print("\nPhones:")
            for p in contact["phones"]:
                print(f"  {p['label']}: {p['number']}")
            print("\nEmails:")
            for e in contact["emails"]:
                print(f"  {e['label']}: {e['email']}")

            print(f"\nBest phone: {get_phone_number(name)}")
        else:
            print("Not found")
    else:
        print("Usage: python tools/contacts.py <name>")
        print("\nNo name provided. Try:")
        print("  python tools/contacts.py John")
        print("\nNote: Contacts access must be granted in System Settings > Privacy > Contacts")
