"""
Agents: agent_scratchpad - MessagesPlaceholder for tool results.

Syntax: MessagesPlaceholder(variable_name="agent_scratchpad")

Prompt needs this for agent to see tool results in next turn.
Populated with format_to_openai_function_messages(intermediate_steps).
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You have tools."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
# agent_scratchpad = format_to_openai_function_messages(intermediate_steps)
# invoke({"input": "What is 5+7?", "agent_scratchpad": agent_scratchpad})
