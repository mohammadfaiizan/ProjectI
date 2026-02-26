"""
LangGraph: ToolNode - Prebuilt node for executing tool calls.

Syntax: from langgraph.prebuilt import ToolNode
        tool_node = ToolNode(tools)

Input: tools - List[BaseTool]
Output: Node that executes tool_calls from last AIMessage.

Use in ReAct: agent node returns AIMessage with tool_calls,
ToolNode executes them and returns ToolMessage.
"""

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [add]
tool_node = ToolNode(tools)
# workflow.add_node("tools", tool_node)
# tool_node.invoke(state) -> executes tool_calls, returns updated state
