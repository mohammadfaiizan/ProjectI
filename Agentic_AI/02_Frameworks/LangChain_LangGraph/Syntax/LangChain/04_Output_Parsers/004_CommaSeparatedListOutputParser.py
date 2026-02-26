"""
Output Parsers: CommaSeparatedListOutputParser - Parse comma-separated list.

Syntax: parser = CommaSeparatedListOutputParser()

Input: AIMessage (expects "a, b, c" format)
Output: List[str]
"""

from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

parser = CommaSeparatedListOutputParser()
chain = (
    ChatPromptTemplate.from_template("List 3 {topic} separated by commas")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | parser
)
# chain.invoke({"topic": "colors"}) -> ["red", "blue", "green"]
