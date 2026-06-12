"""DuckDuckGo search tool — free, no API key required.
"""

from duckduckgo_search import DDGS


def search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Number of results to return (default 5).

    Returns:
        Formatted string with search results, or error message.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            formatted.append(f"{i}. {title}\n   {body}\n   Source: {href}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {e}"
