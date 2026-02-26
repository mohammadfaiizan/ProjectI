"""
Models: llm.batch() - Process multiple inputs in parallel.

Syntax: responses = llm.batch(list_of_messages)

Input: List[str] or List[List[BaseMessage]]
  Each element is one request

Output: List[AIMessage]
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
inputs = [
    [HumanMessage(content="Summarize: Python")],
    [HumanMessage(content="Summarize: Java")],
]
responses = llm.batch(inputs)
# responses[0].content, responses[1].content
