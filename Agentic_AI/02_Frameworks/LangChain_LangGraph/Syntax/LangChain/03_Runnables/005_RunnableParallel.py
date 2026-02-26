"""
Runnables: RunnableParallel - Run multiple chains in parallel.

Syntax: RunnableParallel(key1=chain1, key2=chain2)

Input: dict (same input passed to ALL chains)
Output: {"key1": result1, "key2": result2}
"""

from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain1 = ChatPromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser()
chain2 = ChatPromptTemplate.from_template("Translate to French: {text}") | llm | StrOutputParser()

parallel = RunnableParallel(summary=chain1, french=chain2)
# parallel.invoke({"text": "AI is cool"}) -> {"summary": "...", "french": "..."}
