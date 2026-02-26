"""
Memory: memory.save_context() - Save a conversation turn.

Syntax: memory.save_context(input_dict, output_dict)

Input: input_dict - {"input": "user message"} (key matches input_key)
       output_dict - {"output": "assistant message"} (key matches output_key)
"""

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)
memory.save_context(
    {"input": "My name is Alice"},
    {"output": "Nice to meet you, Alice!"},
)
memory.save_context(
    {"input": "What's my name?"},
    {"output": "Your name is Alice."},
)
