"""
LangGraph: workflow.set_entry_point() - Set starting node.

Syntax: workflow.set_entry_point(node_name)

Input: node_name: str - name of node where execution starts
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


class MyState(TypedDict):
    counter: int


def start_node(state: MyState) -> MyState:
    return state


workflow = StateGraph(MyState)
workflow.add_node("start", start_node)
workflow.set_entry_point("start")
workflow.add_edge("start", END)
