"""
Memory: ConversationTokenBufferMemory - Keep messages within token limit.

Syntax: memory = ConversationTokenBufferMemory(
    memory_key="chat_history",
    max_token_limit=500,
)

Trims oldest messages when limit exceeded.
"""

from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=500,
)
