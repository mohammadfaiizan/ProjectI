"""
LangGraph: workflow.add_node() - Register a node.

Syntax: workflow.add_node(name, node_function)

Input:
  name: str - node identifier
  node_function: Callable[[State], State] - receives state, returns updated state

Node function signature: def node(state: State) -> State:
"""

from typing import TypedDict
from langgraph.graph import StateGraph


class MyState(TypedDict):
    counter: int


def increment_node(state: MyState) -> MyState:
    return {"counter": state["counter"] + 1}


workflow = StateGraph(MyState)
workflow.add_node("increment", increment_node)
