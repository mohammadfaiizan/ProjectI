"""
LCEL: chain.batch() - Process multiple inputs in parallel.

Syntax: results = chain.batch(inputs)

Input: List[dict]
  - Each dict has same structure as invoke input
  - Example: [{"text": "a"}, {"text": "b"}]
  - Type: List[Dict[str, Any]]

Output: List[Any] - one result per input
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("Summarize: {text}")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)

inputs = [
    {"text": "Python is a programming language."},
    {"text": "Machine learning uses data."},
]
results = chain.batch(inputs)  # -> [str, str]
