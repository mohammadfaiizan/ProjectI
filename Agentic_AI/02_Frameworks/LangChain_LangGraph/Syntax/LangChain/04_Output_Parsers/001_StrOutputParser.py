"""
Output Parsers: StrOutputParser - Return raw string.

Syntax: parser = StrOutputParser()

Input: AIMessage (from LLM)
Output: str (message.content)

Use: When you need plain text, no structure
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

parser = StrOutputParser()
chain = (
    ChatPromptTemplate.from_template("Say hi to {name}")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | parser
)
# chain.invoke({"name": "Alice"}) -> "Hello Alice!" (str)
