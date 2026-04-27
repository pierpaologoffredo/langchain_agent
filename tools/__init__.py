"""Tools package — exposes every agent tool from a single import point.

Usage:
    from tools import ALL_TOOLS                          # full list
    from tools import get_weather, fetch_wiki_data       # individual tools
"""

from .weather import get_weather
from .wikipedia import fetch_wiki_data
from .notes import save_research_note

# Convenience list for passing to create_deep_agent / create_agent
ALL_TOOLS = [get_weather, fetch_wiki_data, save_research_note]

__all__ = ["get_weather", "fetch_wiki_data", "save_research_note", "ALL_TOOLS"]
