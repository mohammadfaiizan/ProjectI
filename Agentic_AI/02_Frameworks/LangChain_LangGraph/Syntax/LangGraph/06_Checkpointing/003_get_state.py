"""
LangGraph: app.get_state() - Get current state from checkpoint.

Syntax: state = app.get_state(config)

Input: config with thread_id
Output: StateSnapshot with values, next, metadata
"""

# app = workflow.compile(checkpointer=memory)
# config = {"configurable": {"thread_id": "1"}}
# snapshot = app.get_state(config)
# snapshot.values -> current state dict
# snapshot.next -> pending nodes
