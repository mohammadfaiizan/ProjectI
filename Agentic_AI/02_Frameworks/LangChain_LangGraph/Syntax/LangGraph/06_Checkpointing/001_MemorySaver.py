"""
LangGraph: MemorySaver - In-memory checkpoint storage.

Syntax: memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

Stores checkpoints in memory. Use for development, single-process.
For production: use SqliteSaver or other persistent checkpointer.
"""

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)
# Enables: resume, get_state, history
