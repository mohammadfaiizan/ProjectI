"""
Runnables: RunnableBranch - Conditional routing.

Syntax: RunnableBranch(
    (condition1, chain1),
    (condition2, chain2),
    default_chain,  # no condition
)

Input: dict
  - Passed to each condition: condition(input) -> bool
  - First True wins, that chain's result is returned
  - If none True, default_chain runs

Output: Result from selected chain
"""

from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
math_chain = ChatPromptTemplate.from_template("Solve: {input}") | llm | StrOutputParser()
general_chain = ChatPromptTemplate.from_template("Answer: {input}") | llm | StrOutputParser()

def is_math(x: dict) -> bool:
    return "+" in x.get("input", "") or "calculate" in x.get("input", "").lower()

router = RunnableBranch(
    (lambda x: is_math(x), math_chain),
    general_chain,
)
# router.invoke({"input": "What is 2+2?"}) -> math_chain result
# router.invoke({"input": "Capital of France?"}) -> general_chain result
