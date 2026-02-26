"""
LangGraph: config with thread_id - For checkpointing and resume.

Syntax: config = {"configurable": {"thread_id": "unique-id"}}

Use with invoke/stream when using checkpointer.
Thread ID groups checkpoints for same conversation/session.
Enables resume from checkpoint.
"""

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)
# config = {"configurable": {"thread_id": "user-123"}}
# result = app.invoke(initial_state, config)
# Later: resume with same config to continue from checkpoint
