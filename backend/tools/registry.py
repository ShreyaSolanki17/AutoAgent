"""Tool registry — maps tool names to their execute functions.

The act_node looks up tools here by name.
Each function takes a single string input and returns a string result.
"""

from backend.tools.search import search
from backend.tools.wikipedia import lookup
from backend.tools.calculator import calculate
from backend.tools.file_io import file_read, file_write

TOOLS = {
    "search":     lambda inp: search(inp),
    "wikipedia":  lambda inp: lookup(inp),
    "calculator": lambda inp: calculate(inp),
    "file_read":  lambda inp: file_read(inp),
    "file_write": lambda inp: _parse_and_write(inp),
}


def _parse_and_write(input_str: str) -> str:
    """Parse 'filename | content' format for file_write."""
    parts = input_str.split("|", 1)
    if len(parts) == 2:
        filename = parts[0].strip()
        content = parts[1].strip()
        return file_write(filename, content)
    return file_write("output.txt", input_str)


def execute_tool(tool_name: str, tool_input: str) -> str:
    """Execute a tool by name with the given input.

    Args:
        tool_name: Name of the tool (must be in TOOLS dict).
        tool_input: Input string for the tool.

    Returns:
        Tool result string, or error message if tool not found.
    """
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'. Available: {', '.join(TOOLS.keys())}"

    try:
        return TOOLS[tool_name](tool_input)
    except Exception as e:
        return f"Error executing '{tool_name}': {e}"
