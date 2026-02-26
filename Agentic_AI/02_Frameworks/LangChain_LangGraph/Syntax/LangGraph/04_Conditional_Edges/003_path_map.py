"""
LangGraph: path_map - Map route function output to target nodes.

Syntax: {"route_key": "target_node_name", ...}

Keys = return values from route function
Values = node names to transition to
"""

from langgraph.graph import END

# workflow.add_conditional_edges("agent", should_continue, {
#     "tools": "tools",   # route_fn returns "tools" -> go to "tools" node
#     "end": END,        # route_fn returns "end" -> go to END
# })
