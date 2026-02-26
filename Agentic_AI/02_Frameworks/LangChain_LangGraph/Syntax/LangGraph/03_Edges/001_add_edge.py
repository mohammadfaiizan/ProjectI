"""
LangGraph: workflow.add_edge() - Direct edge between nodes.

Syntax: workflow.add_edge(source, target)

Input:
  source: str - source node name
  target: str - target node name or END

Always follows this path after source completes.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


class MyState(TypedDict):
    counter: int


workflow = StateGraph(MyState)
# workflow.add_node("a", node_a)
# workflow.add_node("b", node_b)
workflow.add_edge("a", "b")      # a -> b
workflow.add_edge("b", END)     # b -> END
