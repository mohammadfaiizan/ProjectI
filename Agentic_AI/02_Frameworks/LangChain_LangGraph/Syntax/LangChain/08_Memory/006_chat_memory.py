"""
Memory: memory.chat_memory - Direct access to message history.

Syntax:
  memory.chat_memory.add_user_message("...")
  memory.chat_memory.add_ai_message("...")
  memory.chat_memory.messages -> List[BaseMessage]
"""

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_user_message("Hello")
memory.chat_memory.add_ai_message("Hi there!")
# memory.chat_memory.messages
