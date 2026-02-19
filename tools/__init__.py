"""
Tools module for Doris.

Provides various tool implementations for the assistant.
"""

def format_tool_response(data: dict) -> str:
    """Format tool response for display."""
    if not data:
        return "No data"
    
    result = []
    for key, value in data.items():
        result.append(f"{key}: {value}")
    
    return "\n".join(result)
