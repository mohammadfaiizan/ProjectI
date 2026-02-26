"""
LCEL: Multi-step chain with dict branching.

Syntax: Use dict to pass output of one step as input to next.
  {"key": chain1} | prompt_using_key | model | parser

Input: {"input": str} (or keys matching first step)
Output: Final result
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

extract_prompt = ChatPromptTemplate.from_template("Extract topic from: {input}")
explain_prompt = ChatPromptTemplate.from_template("Explain briefly: {topic}")

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
parser = StrOutputParser()

chain = (
    {"topic": extract_prompt | model | parser}
    | explain_prompt
    | model
    | parser
)
# chain.invoke({"input": "Quantum computing uses qubits"}) -> str
