"""
LangGraph: Cyclic edge - Loop back to previous node.

Syntax: workflow.add_edge("tools", "agent")

Creates cycle: agent -> tools -> agent (for ReAct loop).
Execution continues until conditional edge routes to END.
"""

from typing import Literal
from langgraph.graph import StateGraph, END


# ReAct pattern: agent -> (tools or END), tools -> agent
# workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
# workflow.add_edge("tools", "agent")  # Loop back
