"""
LangGraph: workflow.add_conditional_edges() - Route based on state.

Syntax: workflow.add_conditional_edges(
    source,
    path_map_fn,
    path_map={"key": "target_node"},
)

Input:
  source: str - node that triggers routing
  path_map_fn: Callable[[State], str] - returns key for path_map
  path_map: dict - maps key -> target node name
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END


class MyState(TypedDict):
    next_action: str


def route_fn(state: MyState) -> Literal["path_a", "path_b", "end"]:
    if state["next_action"] == "a":
        return "path_a"
    elif state["next_action"] == "b":
        return "path_b"
    return "end"


# workflow.add_conditional_edges("router", route_fn, {
#     "path_a": "node_a",
#     "path_b": "node_b",
#     "end": END,
# })
