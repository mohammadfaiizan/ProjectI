"""
Prompts: prompt.format_messages() - Get list of messages.

Syntax: messages = prompt.format_messages(**kwargs)

Input: Keyword arguments for variables
  Example: format_messages(question="Hi")

Output: List[BaseMessage] - ready for model.invoke()

"""

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{question}"),
])
messages = prompt.format_messages(question="What is Python?")
# messages -> [SystemMessage(...), HumanMessage(...)]
