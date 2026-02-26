"""
LCEL: chain.invoke() - Synchronous execution.

Syntax: result = chain.invoke(input)

Input: dict
  - Keys must match template variables in the prompt
  - Example: {"text": "hello", "language": "French"}
  - Type: Dict[str, Any]

Output: Final result from the chain (type depends on last component)
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("Say hi to {name}")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)

result = chain.invoke({"name": "Alice"})  # -> str
