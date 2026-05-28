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
        # Search for the top matching page title
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return f"No Wikipedia page found for: {query}"

        # Try to fetch the best-matching page
        page = None
        for candidate_title in search_results:
            try:
                page = wikipedia.page(candidate_title, auto_suggest=False)
                break
            except (wikipedia.PageError, wikipedia.DisambiguationError):
                continue

        if page is None:
            return f"No Wikipedia page found for: {query}"

        # Build the response: include top 3 search results and the page summary
        other_results = [r for r in search_results if r != page.title]
        result_text = f"Title: {page.title}\n\n{page.summary}"

        # Include first 2 paragraphs of content for more detail
        sections = page.content.split("\n\n")
        if len(sections) > 1:
            detail = "\n\n".join(sections[:2])
            result_text += f"\n\n{detail}"

        if other_results:
            result_text += f"\n\nOther possible pages: {', '.join(other_results[:2])}"

        return result_text

    except wikipedia.DisambiguationError as e:
        options = e.options[:5]
        return f"Ambiguous query '{query}'. Did you mean: {', '.join(options)}?"

    except wikipedia.PageError:
        return f"No Wikipedia page found for: {query}"

    except Exception as e:
        return f"Wikipedia error: {e}"
