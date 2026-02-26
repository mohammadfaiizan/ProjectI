"""
Tools: tool.name, tool.description - Auto-generated metadata.

Syntax:
  tool.name -> function name
  tool.description -> from docstring (shown to LLM)
"""

from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for information. Use when you need current facts."""
    return f"Results for: {query}"


# search_web.name -> "search_web"
# search_web.description -> "Search the web for information. Use when..."
