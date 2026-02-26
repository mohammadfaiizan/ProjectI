"""
LangGraph: compile(checkpointer=) - Enable persistence.

Syntax: app = workflow.compile(checkpointer=memory)

Input: checkpointer - MemorySaver(), SqliteSaver(...), etc.
Output: CompiledGraph with checkpoint support

Enables: resume from interrupt, get_state, time travel
"""

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)
# config = {"configurable": {"thread_id": "1"}}
# app.invoke(state, config)
