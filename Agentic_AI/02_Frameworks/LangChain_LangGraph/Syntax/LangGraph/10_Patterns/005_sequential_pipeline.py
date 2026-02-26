"""
LangGraph: Sequential pipeline - Linear chain of nodes.

Syntax: workflow.add_edge("a", "b")
        workflow.add_edge("b", "c")
        workflow.add_edge("c", END)

Simple linear flow: a -> b -> c -> END
Each node runs in sequence.
"""

from langgraph.graph import StateGraph, END

# workflow.set_entry_point("research")
# workflow.add_edge("research", "write")
# workflow.add_edge("write", "review")
# workflow.add_edge("review", "finalize")
# workflow.add_edge("finalize", END)
