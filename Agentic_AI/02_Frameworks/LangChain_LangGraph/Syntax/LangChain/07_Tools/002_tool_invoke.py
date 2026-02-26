"""
Tools: tool.invoke() - Execute tool with input.

Syntax: result = tool.invoke(input_dict)

Input: Dict[str, Any] - keys = parameter names
  Example: {"a": 2, "b": 3} for def add(a: int, b: int)

Output: Tool's return value
"""

from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


result = add.invoke({"a": 2, "b": 3})  # -> 5
