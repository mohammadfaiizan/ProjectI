"""
Tools: @tool decorator - Define a tool for agents.

Syntax:
  @tool
  def my_tool(param: type) -> return_type:
      '''Description for LLM. Use when...'''
      return result

Docstring = description shown to LLM (be specific about when to use)
Type hints on params = schema for tool_calls
"""

from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers. Use when user asks for sum or addition."""
    return a + b
