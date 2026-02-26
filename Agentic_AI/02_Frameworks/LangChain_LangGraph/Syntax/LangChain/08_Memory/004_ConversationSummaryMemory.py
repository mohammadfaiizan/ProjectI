"""
Memory: ConversationSummaryMemory - Summarize long conversations.

Syntax: memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

When buffer gets long, summarizes to save tokens.
Use for long-running chats.
"""

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True,
)
