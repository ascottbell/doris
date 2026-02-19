"""
Kids Corner for morning briefing.

Fun content for kids:
- Fun fact of the day
- Kid-friendly joke or riddle
- This day in history (age-appropriate)
"""

import random
from datetime import datetime

# Pre-loaded fun facts (fallback if API fails)
FUN_FACTS = [
    "Octopuses have three hearts and blue blood!",
    "Honey never spoils - archaeologists found 3,000-year-old honey in Egyptian tombs that was still good!",
    "A group of flamingos is called a 'flamboyance'!",
    "Butterflies taste with their feet!",
    "Cows have best friends and get stressed when separated!",
    "The shortest war in history lasted 38 minutes - between Britain and Zanzibar!",
    "Bananas are berries, but strawberries aren't!",
    "Dolphins sleep with one eye open!",
    "A day on Venus is longer than its year!",
    "Sharks have been around longer than trees!",
    "Sloths can hold their breath longer than dolphins - up to 40 minutes!",
    "The Eiffel Tower grows about 6 inches taller in summer because metal expands in heat!",
    "Koalas have fingerprints almost identical to humans!",
    "There are more stars in the universe than grains of sand on Earth!",
    "Elephants are the only animals that can't jump!",
]

# Pre-loaded jokes (fallback)
JOKES = [
    ("Why don't scientists trust atoms?", "Because they make up everything!"),
    ("What do you call a fish without eyes?", "A fsh!"),
    ("Why did the cookie go to the doctor?", "Because it was feeling crummy!"),
    ("What do you call a dinosaur that crashes their car?", "Tyrannosaurus Wrecks!"),
    ("Why can't you give Elsa a balloon?", "Because she'll let it go!"),
    ("What do you call a sleeping dinosaur?", "A dino-snore!"),
    ("Why did the teddy bear say no to dessert?", "Because she was already stuffed!"),
    ("What do you call a dog that does magic tricks?", "A Labracadabrador!"),
    ("Why are ghosts bad at lying?", "Because you can see right through them!"),
    ("What do you call a bear with no teeth?", "A gummy bear!"),
    ("Why did the banana go to the doctor?", "Because it wasn't peeling well!"),
    ("What do you call cheese that isn't yours?", "Nacho cheese!"),
]

# This day in history - kid-friendly events by month/day
HISTORY_EVENTS = {
    (1, 1): "On this day in 1863, Abraham Lincoln signed the Emancipation Proclamation to free enslaved people!",
    (1, 4): "On this day in 1958, the first artificial satellite Sputnik fell back to Earth!",
    (1, 14): "On this day in 1878, Alexander Graham Bell demonstrated the telephone to Queen Victoria!",
    (1, 15): "On this day in 1929, Martin Luther King Jr. was born - he grew up to be a hero for civil rights!",
    (1, 27): "On this day in 1880, Thomas Edison got his patent for the electric light bulb!",
    (2, 2): "Groundhog Day! Punxsutawney Phil comes out to predict if spring is coming!",
    (2, 12): "On this day in 1809, Abraham Lincoln was born - he became the 16th President!",
    (7, 4): "Happy Independence Day! On this day in 1776, America declared independence!",
    (7, 20): "On this day in 1969, Neil Armstrong became the first person to walk on the Moon!",
    (12, 17): "On this day in 1903, the Wright Brothers made the first airplane flight - only 12 seconds long!",
}


def get_kids_content() -> dict:
    """
    Get fun content for kids' morning.

    Returns:
        dict with 'fun_fact', 'joke', 'history', and 'summary'
    """
    today = datetime.now()
    month_day = (today.month, today.day)

    # Fun fact - try Claude for fresh one, fallback to pre-loaded
    fun_fact = None
    try:
        from llm.api_client import call_claude
        from llm.providers import resolve_model
        response = call_claude(
            messages=[{
                "role": "user",
                "content": "Give me one fun, amazing fact that would delight kids ages 5-8. Just the fact, no preamble. Make it about animals, space, nature, or how things work."
            }],
            source="briefing-kids",
            model=resolve_model("mid"),
            max_tokens=100,
        )
        fun_fact = response.text.strip()
    except:
        fun_fact = random.choice(FUN_FACTS)

    # Joke
    joke_q, joke_a = random.choice(JOKES)
    joke = {
        "question": joke_q,
        "answer": joke_a,
        "full": f"{joke_q} {joke_a}"
    }

    # This day in history
    history = HISTORY_EVENTS.get(month_day)
    if not history:
        # Try to generate one
        try:
            from llm.api_client import call_claude
            from llm.providers import resolve_model
            response = call_claude(
                messages=[{
                    "role": "user",
                    "content": f"What's one interesting, kid-friendly thing that happened on {today.strftime('%B %d')} in history? Just 1 sentence, suitable for ages 5-8."
                }],
                source="briefing-kids",
                model=resolve_model("mid"),
                max_tokens=80,
            )
            history = response.text.strip()
        except:
            history = None

    return {
        "fun_fact": fun_fact,
        "joke": joke,
        "history": history,
        "date": today.strftime("%B %d")
    }


def format_summary() -> str:
    """Format kids content for voice - formatted as natural language."""
    content = get_kids_content()

    parts = ["Here's something fun for the kids."]

    # Fun fact
    if content.get("fun_fact"):
        parts.append(f"Fun fact: {content['fun_fact']}")

    # Joke (with pause for answer)
    if content.get("joke"):
        joke = content["joke"]
        parts.append(f"Here's a joke: {joke['question']}")
        parts.append(f"The answer is... {joke['answer']}")

    # History
    if content.get("history"):
        parts.append(content["history"])

    return " ".join(parts)


def get_just_joke() -> str:
    """Get just a joke for quick requests."""
    joke_q, joke_a = random.choice(JOKES)
    return f"{joke_q} ... {joke_a}"


def get_just_fact() -> str:
    """Get just a fun fact."""
    content = get_kids_content()
    return content.get("fun_fact", random.choice(FUN_FACTS))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Kids Corner")
    print("=" * 50)

    content = get_kids_content()

    print(f"\nDate: {content['date']}")
    print(f"\nFun Fact: {content['fun_fact']}")
    print(f"\nJoke: {content['joke']['question']}")
    print(f"Answer: {content['joke']['answer']}")
    if content['history']:
        print(f"\nHistory: {content['history']}")

    print(f"\n\nVoice version:")
    print(format_summary())
