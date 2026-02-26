"""
Output Parsers: JsonOutputParser - Parse JSON output.

Syntax: parser = JsonOutputParser()

Input: AIMessage (expects valid JSON in content)
Output: dict
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

parser = JsonOutputParser()
chain = (
    ChatPromptTemplate.from_template("Return JSON with 'answer' and 'confidence'. Query: {query}")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | parser
)
# chain.invoke({"query": "What is 2+2?"}) -> {"answer": "4", "confidence": 1.0}
