"""
Tools: Tool() - Create tool from function (alternative to @tool).

Syntax: tool = Tool(name="...", description="...", func=my_func)

Note: @tool is preferred. Tool() for programmatic creation.
"""

from langchain_core.tools import Tool


def multiply(a: int, b: int) -> int:
    return a * b


multiply_tool = Tool(
    name="multiply",
    description="Multiply two numbers",
    func=multiply,
)
