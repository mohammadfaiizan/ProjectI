"""
Tools: Tool args - Type hints define schema for LLM.

Syntax: def tool(a: int, b: str, c: List[str]) -> str:

LLM receives schema: a (integer), b (string), c (array of strings)
Use Field for more control: param: str = Field(description="...")
"""

from langchain_core.tools import tool
from typing import List


@tool
def summarize_items(items: List[str], max_words: int = 50) -> str:
    """Summarize a list of items. max_words limits output length."""
    return " ".join(items)[:max_words * 5]
