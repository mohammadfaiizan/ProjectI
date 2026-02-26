"""
LangGraph: app = workflow.compile() - Compile to runnable.

Syntax: app = workflow.compile()
        app = workflow.compile(checkpointer=memory)

Input: checkpointer: BaseCheckpointSaver | None
Output: CompiledGraph - has invoke, stream, get_state, etc.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


class MyState(TypedDict):
    counter: int


workflow = StateGraph(MyState)
# ... add nodes and edges ...
app = workflow.compile()
# app.invoke(initial_state)
# app.stream(initial_state)
