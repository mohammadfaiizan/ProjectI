"""
LangGraph: END - Terminal node for graph completion.

Syntax: workflow.add_edge("last_node", END)

Import: from langgraph.graph import END

When execution reaches END, graph completes and returns final state.
"""

from langgraph.graph import END

# workflow.add_edge("finalize", END)
# Execution stops, result is returned
