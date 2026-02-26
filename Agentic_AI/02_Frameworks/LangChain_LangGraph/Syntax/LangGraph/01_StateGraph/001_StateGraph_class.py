"""
LangGraph: StateGraph - Main class for graph-based workflows.

Syntax: workflow = StateGraph(StateSchema)

Input: State schema (TypedDict class)
Output: StateGraph instance

StateGraph manages nodes, edges, and execution flow.
"""

from typing import TypedDict
from langgraph.graph import StateGraph


class MyState(TypedDict):
    messages: list
    counter: int


workflow = StateGraph(MyState)
# workflow.add_node(...)
# workflow.add_edge(...)
# app = workflow.compile()
