"""
Memory: memory.load_memory_variables() - Get history for prompt.

Syntax: variables = memory.load_memory_variables(inputs)

Input: inputs - dict (often {})
Output: {memory_key: List[BaseMessage] or str}
  return_messages=True -> List[BaseMessage]
  return_messages=False -> str
"""

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)
memory.save_context({"input": "Hi"}, {"output": "Hello!"})
variables = memory.load_memory_variables({})
# variables["chat_history"] -> [HumanMessage(...), AIMessage(...)]
