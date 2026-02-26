"""
LangGraph: tools_condition - Prebuilt route for tool calls.

Syntax: from langgraph.prebuilt import tools_condition
        workflow.add_conditional_edges("agent", tools_condition)

Returns "tools" if last message has tool_calls, else "end".
Use with ToolNode for ReAct: agent -> tools_condition -> tools or END.
"""

from langgraph.prebuilt import tools_condition

# workflow.add_conditional_edges("agent", tools_condition, {
#     "tools": "tools",
#     "end": END,
# })
# tools_condition(state) -> "tools" | "end" based on tool_calls
