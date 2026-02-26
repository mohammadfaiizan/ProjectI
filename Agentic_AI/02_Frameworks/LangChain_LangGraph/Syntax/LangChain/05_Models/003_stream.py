"""
Models: llm.stream() - Stream tokens as generated.

Syntax: for chunk in llm.stream(messages): ...

Input: Same as invoke (str or List[BaseMessage])
Output: Iterator[AIMessage] - chunks with .content
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
for chunk in llm.stream([HumanMessage(content="Count 1 to 5")]):
    print(chunk.content, end="", flush=True)
