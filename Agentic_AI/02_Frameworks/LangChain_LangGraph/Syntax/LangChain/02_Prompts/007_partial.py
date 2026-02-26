"""
Prompts: .partial() - Pre-fill some variables.

Syntax: prompt = prompt.partial(format_instructions="...")

Use when some variables are fixed (e.g. from parser.get_format_instructions()).
Invoke only needs remaining variables.

Input: dict with keys for variables NOT in partial()
  Example: partial(format_instructions="...") -> invoke only needs {"description": "..."}
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(description="Name")


parser = PydanticOutputParser(pydantic_object=Person)
prompt = ChatPromptTemplate.from_template(
    "{format_instructions}\n\nExtract from: {text}"
).partial(format_instructions=parser.get_format_instructions())
# prompt.invoke({"text": "John is 30"}) - format_instructions already set
