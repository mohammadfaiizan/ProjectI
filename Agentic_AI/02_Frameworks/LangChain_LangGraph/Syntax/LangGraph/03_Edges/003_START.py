"""
LangGraph: START - Entry point constant (optional).

Syntax: workflow.add_edge(START, "first_node")

Import: from langgraph.graph import START

Alternative to set_entry_point when using add_edge for entry.
START is the implicit source before set_entry_point node.
"""

from langgraph.graph import START

# workflow.add_edge(START, "first_node")
# Equivalent to: workflow.set_entry_point("first_node")
