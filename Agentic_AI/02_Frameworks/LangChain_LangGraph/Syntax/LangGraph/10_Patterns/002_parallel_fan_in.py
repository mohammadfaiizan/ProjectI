"""
LangGraph: Parallel fan-in - Multiple branches to one node.

Syntax: workflow.add_edge("branch_a", "combine")
        workflow.add_edge("branch_b", "combine")
        workflow.add_edge("branch_c", "combine")

Combine node receives state after ALL branches complete.
State is merged (keys from all branches).
"""

from langgraph.graph import StateGraph, END

# workflow.add_node("combine", combine_node)
# workflow.add_edge("branch_a", "combine")
# workflow.add_edge("branch_b", "combine")
# workflow.add_edge("branch_c", "combine")
# workflow.add_edge("combine", END)
# combine_node receives merged state from all branches
