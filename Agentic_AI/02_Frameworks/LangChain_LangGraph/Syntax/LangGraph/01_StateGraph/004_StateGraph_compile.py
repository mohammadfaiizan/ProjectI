"""
LangGraph: workflow.compile() - Compile graph to executable.

Syntax: app = workflow.compile()
        app = workflow.compile(checkpointer=memory)

Input: checkpointer: BaseCheckpointSaver | None - for persistence
Output: CompiledGraph - runnable with invoke, stream
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


class MyState(TypedDict):
    counter: int


workflow = StateGraph(MyState)
# ... add nodes and edges ...
# app = workflow.compile()
# app = workflow.compile(checkpointer=MemorySaver())
