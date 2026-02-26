"""
Output Parsers: parser.get_format_instructions() - Get format string for prompt.

Syntax: instructions = parser.get_format_instructions()

Returns: str describing expected output format (for LLM)
Use with .partial(): prompt.partial(format_instructions=parser.get_format_instructions())
"""

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(description="Name")
    age: int = Field(description="Age")


parser = PydanticOutputParser(pydantic_object=Person)
instructions = parser.get_format_instructions()
# -> JSON format description with schema for LLM
