"""Research notes tool — persists a finding to the local knowledge base."""

from langchain.tools import tool


@tool
def save_research_note(note: str, topic: str) -> str:
    """Persist a research note to the local knowledge base (test.txt).

    Overwrites the file on each call, so use this for the final,
    most relevant finding per session rather than accumulating notes.

    Args:
        note:  The content to save (a summary, key fact, or quote).
        topic: A short label identifying what the note is about.
    """
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write(f"Topic: {topic} | Content: {note}")
    return f"Note saved — Topic: {topic} | Content: {note}"
