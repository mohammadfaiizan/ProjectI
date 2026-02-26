"""
LangGraph: Route function - Returns target node key.

Syntax: def route_fn(state: State) -> Literal["a", "b", "c"]:
            if condition(state): return "a"
            return "b"

Input: state - current graph state
Output: str - key used in path_map (must match add_conditional_edges path_map)
"""

from typing import TypedDict, Literal
from langchain_core.messages import BaseMessage


class ReActState(TypedDict):
    messages: list


def should_continue(state: ReActState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"
