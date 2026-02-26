"""
LangGraph: Supervisor routing - Route to specialist agents.

Structure: supervisor -> conditional -> specialist_a | specialist_b | specialist_c
Supervisor decides which specialist based on state (e.g. task_type).
"""

from typing import Literal
from langgraph.graph import StateGraph, END


# def route_to_agent(state): return state["current_agent"]
# workflow.add_conditional_edges("supervisor", route_to_agent, {
#     "coder": "coder",
#     "writer": "writer",
#     "analyst": "analyst",
# })
# workflow.add_edge("coder", END)
# workflow.add_edge("writer", END)
# workflow.add_edge("analyst", END)
