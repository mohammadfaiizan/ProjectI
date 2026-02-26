"""
LangGraph: Human-in-the-loop - Interrupt for human input.

Use interrupt_before or interrupt_after on node.
When reached, execution pauses. Resume with app.invoke(human_input, config).

Requires checkpointer for persistence.
"""

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# workflow.add_node("approval", human_approval_node)
# app = workflow.compile(checkpointer=MemorySaver())
# app = app.compile(interrupt_before=["approval"])
# Or: interrupt_after=["generate"]
# When interrupted: app.get_state(config).next has pending nodes
# Resume: app.invoke({"approval_status": "approved"}, config)
