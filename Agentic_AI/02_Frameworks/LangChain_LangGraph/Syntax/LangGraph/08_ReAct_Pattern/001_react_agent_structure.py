"""
LangGraph: ReAct agent structure - Agent + Tool loop.

Structure:
  agent -> conditional(tools | end) -> tools -> agent (loop)

Agent node: LLM with bind_tools, returns AIMessage
Conditional: tools_condition or custom (tool_calls? -> tools : end)
Tool node: ToolNode(tools)
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition


# workflow = StateGraph(ReActState)
# workflow.add_node("agent", llm_node)
# workflow.add_node("tools", ToolNode(tools))
# workflow.set_entry_point("agent")
# workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", "end": END})
# workflow.add_edge("tools", "agent")
