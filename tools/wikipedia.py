"""Wikipedia tool — searches and retrieves article content via WikipediaRetriever."""

from langchain.tools import tool
from langchain_community.retrievers import WikipediaRetriever

_retriever = WikipediaRetriever(
    wiki_client="",
    top_k_results=1,
    doc_content_chars_max=10_000,
)


@tool
def fetch_wiki_data(query: str) -> str:
    """Fetch the Wikipedia article that best matches the query.

    Searches Wikipedia and returns the full text of the top result,
    capped at 10 000 characters.  Returns "No data found" when no
    article matches.

    Args:
        query: Search term or article title (e.g. "Python programming language").
    """
    results = _retriever.invoke(query)
    if results:
        return results[0].page_content
    return "No data found"
