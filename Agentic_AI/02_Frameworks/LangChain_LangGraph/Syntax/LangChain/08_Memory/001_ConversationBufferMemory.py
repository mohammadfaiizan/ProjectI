"""
Memory: ConversationBufferMemory - Store all messages.

Syntax: memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

Stores full conversation. Use for short chats.
"""

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input",
    output_key="output",
)
