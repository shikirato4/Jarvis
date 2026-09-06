"""Sanitized registry-shaped snapshots of current safe capabilities."""

REAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal.run",
            "description": "Run a non-destructive command in the approved synthetic workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "minLength": 1}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "files.read_text",
            "description": "Read a text file within the approved synthetic workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator.evaluate",
            "description": "Evaluate a basic arithmetic expression without side effects.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "minLength": 1}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]
