"""
Seed initial family memories into Doris memory store.
Run: python -m memory.seed
"""

from memory.store import store_memory, get_all_active, delete_memory

# Family memories to seed — replace with your own family details
# These are generic examples showing the expected format
FAMILY_MEMORIES = [
    # Example child
    {
        "content": "Alex is 10 years old, birthday is March 15th",
        "category": "family",
        "subject": "Alex",
        "metadata": {"birthday": "03-15", "birth_year": 2016}
    },
    {
        "content": "Alex is in 4th grade",
        "category": "family",
        "subject": "Alex",
    },

    # Example child
    {
        "content": "Emma is 6 years old, birthday is July 20th",
        "category": "family",
        "subject": "Emma",
        "metadata": {"birthday": "07-20", "birth_year": 2019}
    },
    {
        "content": "Emma loves art and animals",
        "category": "family",
        "subject": "Emma"
    },

    # Example spouse
    {
        "content": "Jane is the user's spouse",
        "category": "family",
        "subject": "Jane"
    },

    # Example pet
    {
        "content": "Buddy is the family dog, needs monthly flea medicine",
        "category": "family",
        "subject": "Buddy",
        "metadata": {"reminder": "monthly flea medicine"}
    },

    # Example second home
    {
        "content": "The family has a vacation home upstate",
        "category": "home",
        "subject": "Vacation Home"
    },
    {
        "content": "Vacation home: check pipes when cold weather expected",
        "category": "home",
        "subject": "Vacation Home",
        "metadata": {"type": "maintenance"}
    },
]


def clear_all_memories():
    """Clear all existing memories (use carefully)."""
    memories = get_all_active()
    for mem in memories:
        delete_memory(mem['id'])
    print(f"Cleared {len(memories)} memories")


def seed_memories():
    """Seed initial family memories."""
    created = []
    for mem in FAMILY_MEMORIES:
        mem_id = store_memory(
            content=mem["content"],
            category=mem["category"],
            subject=mem.get("subject"),
            source="bootstrap",
            metadata=mem.get("metadata")
        )
        created.append(mem_id)
        print(f"Created: {mem['subject']} - {mem['content'][:40]}...")

    print(f"\nSeeded {len(created)} memories")
    return created


if __name__ == "__main__":
    print("Clearing existing memories...")
    clear_all_memories()

    print("\nSeeding family memories...")
    seed_memories()

    print("\nCurrent memories:")
    for mem in get_all_active():
        print(f"  [{mem['category']}] {mem['subject']}: {mem['content'][:50]}...")
