"""File reader/writer tool — sandboxed to a safe working directory.
"""

import os
from pathlib import Path

# Sandbox: agent can only read/write inside this folder
WORK_DIR = Path(__file__).resolve().parent.parent.parent / "workdir"
WORK_DIR.mkdir(exist_ok=True)


def file_read(filename: str) -> str:
    """Read a text file from the work directory.

    Args:
        filename: Name of the file (e.g. "notes.txt").

    Returns:
        File contents, or error message.
    """
    try:
        filepath = WORK_DIR / filename

        # Security: prevent path traversal outside workdir
        if not filepath.resolve().is_relative_to(WORK_DIR.resolve()):
            return "Error: Path traversal not allowed"

        if not filepath.exists():
            return f"Error: File '{filename}' not found in workdir"

        return filepath.read_text(encoding="utf-8")

    except Exception as e:
        return f"Error reading '{filename}': {e}"


def file_write(filename: str, content: str) -> str:
    """Write text to a file in the work directory.

    Args:
        filename: Name of the file (e.g. "output.txt").
        content: Text content to write.

    Returns:
        Success or error message.
    """
    try:
        filepath = WORK_DIR / filename

        # Security: prevent path traversal outside workdir
        if not filepath.resolve().is_relative_to(WORK_DIR.resolve()):
            return "Error: Path traversal not allowed"

        filepath.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{filename}'"

    except Exception as e:
        return f"Error writing '{filename}': {e}"
