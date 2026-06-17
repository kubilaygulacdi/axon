from ddgs import DDGS
from langchain_core.tools import tool

MAX_RESULTS = 10


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information.

    Use this tool when:
    - You need up-to-date information beyond your training cutoff
    - User asks about recent events, news, or releases
    - You need to verify a fact you're uncertain about

    Do NOT use when:
    - Answer is already in context
    - Question is about general/stable knowledge (e.g. 'what is a for loop')

    Args:
        query: Search query. Be specific and concise (3-6 words ideal).
        max_results: Number of results to return, 1-10. Defaults to 5.
    """
    n = max(1, min(max_results, MAX_RESULTS))
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=n))
    except Exception as e:
        return f"Error: web search failed ({type(e).__name__}: {e})."

    if not results:
        return f"No results for '{query}'."

    rendered = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "(no title)")
        url = r.get("href", "")
        snippet = r.get("body", "").strip()
        rendered.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n\n".join(rendered)
