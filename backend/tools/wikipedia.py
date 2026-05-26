"""Wikipedia lookup tool — fetches topic summaries from Wikipedia.
"""

import wikipedia


def lookup(query: str) -> str:
    """Look up a topic on Wikipedia.

    Args:
        query: Topic to search for (e.g. "capital of France").

    Returns:
        Summary text from Wikipedia, or error message.
    """
    try:
        # Check if page exists via search first
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return f"No Wikipedia page found for: {query}"

        page_title = search_results[0]
        page = wikipedia.page(page_title, auto_suggest=False)

        # Return first 3 paragraphs (enough for most factual questions)
        sections = page.content.split("\n\n")
        summary = "\n\n".join(sections[:3])

        # Also include the auto-summary (first paragraph)
        if not summary.strip():
            summary = page.summary

        return f"Title: {page.title}\n\n{summary}"

    except wikipedia.DisambiguationError as e:
        # Ambiguous query — return options so the agent can retry
        options = e.options[:5]
        return f"Ambiguous query '{query}'. Did you mean: {', '.join(options)}?"

    except wikipedia.PageError:
        return f"No Wikipedia page found for: {query}"

    except Exception as e:
        return f"Wikipedia error: {e}"
