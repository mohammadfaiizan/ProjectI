"""
LCEL: Pipe operator (|) - Compose runnables into a chain.

Syntax: chain = prompt | model | output_parser

Input: dict with keys matching prompt template variables.
  Example: {"text": "hello"}
Output: depends on last component (e.g. str from StrOutputParser)
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
parser = StrOutputParser()

chain = prompt | model | parser
# chain.invoke({"text": "Hello"}) -> str
