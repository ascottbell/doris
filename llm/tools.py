"""
Tool definitions for Doris.

All tools are defined as canonical ToolDef objects. LLM providers convert
these to their wire format internally (e.g., Anthropic format, OpenAI format).

Adding a new tool:
    1. Add a dict to _RAW_TOOLS below
    2. If it's state-changing, add it to WISDOM_REQUIRED_TOOLS
    3. Implement the handler in the appropriate tools/ module
    4. Add the dispatch entry in brain.py's execute_tool()
"""

from llm.types import ToolDef


# --- Raw tool definitions (converted to ToolDef at module load) ---

_RAW_TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current time and date",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather conditions and forecast for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location to get weather for (e.g., 'NYC', 'San Francisco', 'Paris'). Defaults to current location if not specified."
                },
                "when": {
                    "type": "string",
                    "description": "Time period: 'now' for current conditions, 'tomorrow' for tomorrow's forecast, 'week' for 7-day forecast",
                    "enum": ["now", "tomorrow", "week"]
                }
            },
            "required": []
        }
    },
    {
        "name": "get_calendar_events",
        "description": "Get calendar events for a time period or specific date",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Time period or specific date. Can be: 'today', 'tomorrow', 'this week', 'weekend', a day name like 'monday' (next occurrence), or a specific date like 'February 2', 'Feb 2', '2025-02-02', 'January 15 2025' (past or future dates supported)"
                }
            },
            "required": ["period"]
        }
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event. Supports recurring events.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Event title"
                },
                "date": {
                    "type": "string",
                    "description": "Date - prefer explicit format 'January 27' or '2026-01-27' when available. Natural language like 'tomorrow' or 'monday' only when no specific date is given. IMPORTANT: If source says 'Tuesday, January 27', use 'January 27' (the explicit date), not just 'Tuesday'."
                },
                "time": {
                    "type": "string",
                    "description": "Start time (e.g., '2pm', '14:00')"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Duration in minutes (default 60)"
                },
                "location": {
                    "type": "string",
                    "description": "Event location"
                },
                "recurrence": {
                    "type": "string",
                    "enum": ["daily", "weekly", "weekdays", "biweekly", "monthly", "yearly"],
                    "description": "Recurrence pattern. 'weekdays' means Monday-Friday."
                },
                "recurrence_end": {
                    "type": "string",
                    "description": "End date for recurring events (natural language like 'December 31' or 'in 3 months')"
                }
            },
            "required": ["title", "date"]
        }
    },
    {
        "name": "move_calendar_event",
        "description": "Move/reschedule an existing calendar event to a new date and/or time",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The ID of the event to move (get this from get_calendar_events)"
                },
                "new_date": {
                    "type": "string",
                    "description": "New date in natural language (e.g., 'tomorrow', 'monday', 'january 28')"
                },
                "new_time": {
                    "type": "string",
                    "description": "New start time (e.g., '2pm', '14:00'). If not specified, keeps the original time."
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "New duration in minutes. If not specified, keeps the original duration."
                }
            },
            "required": ["event_id", "new_date"]
        }
    },
    {
        "name": "delete_calendar_event",
        "description": "Delete a calendar event",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The ID of the event to delete (get this from get_calendar_events)"
                }
            },
            "required": ["event_id"]
        }
    },
    {
        "name": "create_reminder",
        "description": "Create a reminder",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What to be reminded about"
                },
                "when": {
                    "type": "string",
                    "description": "When to remind (e.g., 'tomorrow at 3pm', 'in 2 hours', 'next Monday')"
                }
            },
            "required": ["task"]
        }
    },
    {
        "name": "list_reminders",
        "description": "List current reminders",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "send_imessage",
        "description": "Send an iMessage/text message to someone",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Who to message (contact name from address book)"
                },
                "message": {
                    "type": "string",
                    "description": "The message to send"
                }
            },
            "required": ["recipient", "message"]
        }
    },
    {
        "name": "read_imessages",
        "description": "Read recent text messages, optionally filtered by contact",
        "input_schema": {
            "type": "object",
            "properties": {
                "contact": {
                    "type": "string",
                    "description": "Filter by contact name (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of messages to return (default 10)"
                }
            },
            "required": []
        }
    },
    {
        "name": "check_email",
        "description": "Check for important or recent emails",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "Look back this many hours (default 24)"
                },
                "search": {
                    "type": "string",
                    "description": "Optional search query"
                }
            },
            "required": []
        }
    },
    {
        "name": "daily_briefing",
        "description": "Get a morning briefing with weather, calendar, emails, and reminders",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "control_music",
        "description": "Control Apple Music - play, pause, skip, or search for music",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["play", "pause", "next", "previous", "current"]
                },
                "query": {
                    "type": "string",
                    "description": "Search query for play action (e.g., 'jazz', 'Taylor Swift', 'workout playlist')"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "store_memory",
        "description": "Store a fact or piece of information to remember",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "What to remember"
                },
                "subject": {
                    "type": "string",
                    "description": "Who or what this is about (e.g., a person's name, 'work', 'health')"
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "search_memory",
        "description": "Search stored memories and facts",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    },
    # --- Knowledge Graph Tools ---
    {
        "name": "graph_entity_create",
        "description": "Create an entity in the knowledge graph (person, place, project, event, concept)",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name (e.g., 'Alice', 'Trattoria Roma', 'MyProject')"
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["person", "place", "project", "event", "concept"],
                    "description": "Type of entity"
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional additional data (e.g., {\"role\": \"colleague\", \"company\": \"Acme\"})"
                }
            },
            "required": ["name", "entity_type"]
        }
    },
    {
        "name": "graph_entity_search",
        "description": "Search for entities in the knowledge graph by name",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (name or partial name)"
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["person", "place", "project", "event", "concept"],
                    "description": "Optional filter by type"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "graph_relationship_add",
        "description": "Add a relationship between entities in the knowledge graph",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Subject entity name (will find or create)"
                },
                "predicate": {
                    "type": "string",
                    "description": "Relationship type (e.g., 'works_on', 'married_to', 'lives_in', 'has_email')"
                },
                "object": {
                    "type": "string",
                    "description": "Object entity name or literal value"
                },
                "is_entity": {
                    "type": "boolean",
                    "description": "True if object is an entity, False if literal value (default: True)"
                }
            },
            "required": ["subject", "predicate", "object"]
        }
    },
    {
        "name": "graph_entity_profile",
        "description": "Get everything known about an entity - all relationships and related entities",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name to get profile for"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "graph_query",
        "description": "Query the knowledge graph for relationships matching a pattern",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject_type": {
                    "type": "string",
                    "description": "Filter by subject entity type"
                },
                "predicate": {
                    "type": "string",
                    "description": "Filter by relationship type"
                },
                "object_type": {
                    "type": "string",
                    "description": "Filter by object entity type"
                }
            }
        }
    },
    {
        "name": "lookup_contact",
        "description": "Look up a contact's phone number or email address from the address book",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to look up (first name, last name, or full name)"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "send_email",
        "description": "Send an email via Gmail",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body text"
                }
            },
            "required": ["to", "subject", "body"]
        }
    },
    {
        "name": "read_email",
        "description": "Read the full content of a specific email by its ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "Gmail message ID (from check_email results)"
                }
            },
            "required": ["message_id"]
        }
    },
    {
        "name": "complete_reminder",
        "description": "Mark a reminder as complete",
        "input_schema": {
            "type": "object",
            "properties": {
                "reminder_id": {
                    "type": "string",
                    "description": "The ID of the reminder to complete"
                }
            },
            "required": ["reminder_id"]
        }
    },
    {
        "name": "control_browser",
        "description": "Control Google Chrome browser - get current tab info, open URLs, search the web",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["current_tab", "open_url", "search", "list_tabs"]
                },
                "url": {
                    "type": "string",
                    "description": "URL to open (for open_url action)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search action)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web using Brave Search API. Use this for looking up current info, news, facts, product info, etc. Does NOT open a browser - returns results directly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results (default 5, max 20)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_notes",
        "description": "Search Apple Notes for notes matching a query",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term to find in notes"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_note",
        "description": "Read the full content of a specific Apple Note by title",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the note to read"
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "create_note",
        "description": "Create a new Apple Note",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title for the new note"
                },
                "body": {
                    "type": "string",
                    "description": "Content of the note"
                },
                "folder": {
                    "type": "string",
                    "description": "Folder to create the note in (optional, defaults to Notes)"
                }
            },
            "required": ["title", "body"]
        }
    },
    {
        "name": "system_info",
        "description": "Get system information: battery status, storage space, WiFi network, uptime",
        "input_schema": {
            "type": "object",
            "properties": {
                "info_type": {
                    "type": "string",
                    "description": "Type of info to get",
                    "enum": ["battery", "storage", "wifi", "all"]
                }
            },
            "required": ["info_type"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the filesystem",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file on the filesystem",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_directory",
        "description": "List files and folders in a directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "search_files",
        "description": "Search for files by name pattern in a directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to search in"
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (e.g., '*.pdf', 'report*')"
                }
            },
            "required": ["path", "pattern"]
        }
    },
    {
        "name": "run_shortcut",
        "description": "Run an Apple Shortcut automation by name",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the shortcut to run"
                },
                "input": {
                    "type": "string",
                    "description": "Optional input to pass to the shortcut"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "list_shortcuts",
        "description": "List available Apple Shortcuts",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "query_documents",
        "description": "Search and extract information from documents in Documents, Downloads, Desktop, and iCloud. Use for questions about insurance policies, vehicle registration, receipts, payments, contracts, or any stored document.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language question about the document(s)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional: specific filename pattern to search"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "notify_user",
        "description": "Send a message to the user. Messages appear in the Doris app conversation history. Default 'proactive' priority shows a normal notification banner. Use 'silent' for low-priority background syncs. Use 'emergency' only for safety/security alerts that must break through Do Not Disturb.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send (keep it brief and useful)"
                },
                "priority": {
                    "type": "string",
                    "enum": ["silent", "proactive", "emergency"],
                    "description": "Notification visibility: 'proactive' (default) shows a banner notification, 'silent' syncs app in background only, 'emergency' breaks through Do Not Disturb.",
                    "default": "proactive"
                }
            },
            "required": ["message"]
        }
    },
    # Wisdom feedback
    {
        "name": "wisdom_feedback",
        "description": "Record feedback on a recent action to help Doris learn. Use when the user says things like 'good job', 'that was wrong', 'perfect', or 'delete that'. Always use the wisdom_id from the tool result when available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wisdom_id": {
                    "type": "string",
                    "description": "The wisdom_id from the tool result (from the [wisdom_id:xxx] tag). Use this to target the exact action."
                },
                "score": {
                    "type": "integer",
                    "description": "Rating 1-5 (1=bad mistake, 3=okay, 5=perfect)",
                    "minimum": 1,
                    "maximum": 5
                },
                "notes": {
                    "type": "string",
                    "description": "What was good or bad about the action"
                },
                "action_type": {
                    "type": "string",
                    "description": "Fallback: type of action to find if wisdom_id not available (create_event, send_reminder, notify, etc.)"
                }
            },
            "required": ["score"]
        }
    },
    # Escalation miss
    {
        "name": "escalation_miss",
        "description": "Record when something should have been escalated but wasn't. Use when the user says things like 'why didn't you tell me about that email?' or 'that airline email should have been escalated'. This helps scouts learn what to escalate in the future.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "What type of item was missed (email, calendar, etc.)",
                    "enum": ["email", "calendar", "weather", "reminder"]
                },
                "description": {
                    "type": "string",
                    "description": "What was missed (e.g., 'airline email about flight tomorrow')"
                },
                "sender": {
                    "type": "string",
                    "description": "For emails, who sent it"
                },
                "subject": {
                    "type": "string",
                    "description": "For emails, the subject line"
                },
                "why_important": {
                    "type": "string",
                    "description": "Why it should have been escalated"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Categorization tags (e.g., ['travel', 'airline', 'time-sensitive'])"
                }
            },
            "required": ["source", "description"]
        }
    },
    # Service control
    {
        "name": "service_control",
        "description": "Control Doris system services via launchd. Use this to restart the server or daemon if they're not responding.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "enum": ["server", "daemon"],
                    "description": "Which service to control"
                },
                "action": {
                    "type": "string",
                    "enum": ["restart", "stop", "start", "status"],
                    "description": "Action to perform"
                }
            },
            "required": ["service", "action"]
        }
    },
    # Scout status
    {
        "name": "get_scout_status",
        "description": "Check the status of the scout system (daemon health, recent observations, escalations). Use this to understand what your scouts are seeing and if the daemon is healthy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_observations": {
                    "type": "boolean",
                    "description": "Include recent observations (default true)",
                    "default": True
                }
            }
        }
    },
]


# --- Canonical tool definitions ---

TOOLS: list[ToolDef] = [
    ToolDef(name=d["name"], description=d["description"], input_schema=d["input_schema"])
    for d in _RAW_TOOLS
]


# --- Wisdom tracking ---

# State-changing tools that benefit from wisdom tracking (experiential learning)
# These tools take actions that affect the real world - log reasoning for all of them
WISDOM_REQUIRED_TOOLS = {
    # Calendar operations
    "create_calendar_event", "move_calendar_event", "delete_calendar_event",
    # Reminders
    "create_reminder", "complete_reminder",
    # Messaging/communication
    "send_imessage", "send_email",
    # Music
    "control_music",
    # Memory and data persistence
    "store_memory", "create_note", "write_file",
    "graph_entity_create", "graph_relationship_add",
    # External actions and notifications
    "notify_user", "run_shortcut",
    # Browser control
    "control_browser",
    # Service control
    "service_control",
}

# Backward compatibility alias
WISDOM_ENABLED_TOOLS = WISDOM_REQUIRED_TOOLS


def should_log_wisdom(tool_name: str) -> bool:
    """Check if a tool should have its reasoning logged to wisdom."""
    return tool_name in WISDOM_REQUIRED_TOOLS
