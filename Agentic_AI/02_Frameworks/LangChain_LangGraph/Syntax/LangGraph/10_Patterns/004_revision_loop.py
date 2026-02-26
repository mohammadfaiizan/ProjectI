"""
LangGraph: Revision loop - Conditional retry until pass.

Structure: write -> review -> conditional(revise | finalize)
          revise -> review (loop)

Review node sets passed_review. Route: pass -> finalize, fail -> revise.
"""

from langgraph.graph import StateGraph, END


# def should_revise(state): return "finalize" if state["passed_review"] else "revise"
# workflow.add_conditional_edges("review", should_revise, {"revise": "revise", "finalize": "finalize"})
# workflow.add_edge("revise", "review")  # Loop back
# workflow.add_edge("finalize", END)
