"""
LCEL: chain.stream() - Stream tokens as they are generated.

Syntax: for chunk in chain.stream(input): ...

Input: dict (same as invoke)

Output: Iterator - yields chunks (str or AIMessageChunk depending on chain)
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("Write a haiku about {topic}")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    | StrOutputParser()
)

for chunk in chain.stream({"topic": "the ocean"}):
    print(chunk, end="", flush=True)
