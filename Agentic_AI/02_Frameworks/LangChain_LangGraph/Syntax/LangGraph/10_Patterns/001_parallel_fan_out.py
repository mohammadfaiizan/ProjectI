"""
LangGraph: Parallel fan-out - One node to multiple branches.

Syntax: workflow.add_edge("input", "branch_a")
        workflow.add_edge("input", "branch_b")
        workflow.add_edge("input", "branch_c")

Multiple edges from same source = parallel execution.
All branches run (order may vary).
"""

from langgraph.graph import StateGraph

# workflow.add_node("input", input_node)
# workflow.add_node("branch_a", branch_a_node)
# workflow.add_node("branch_b", branch_b_node)
# workflow.set_entry_point("input")
# workflow.add_edge("input", "branch_a")
# workflow.add_edge("input", "branch_b")
# workflow.add_edge("branch_a", "combine")
# workflow.add_edge("branch_b", "combine")
