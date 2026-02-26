"""
Output Parsers: OutputFixingParser - Fix malformed output with LLM.

Syntax: OutputFixingParser.from_llm(parser=base_parser, llm=llm)

Wraps another parser. If parse fails, asks LLM to fix the output.
Output: Same as base parser
"""

from langchain_core.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(description="Name")


base_parser = PydanticOutputParser(pydantic_object=Person)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
# If LLM returns malformed JSON, fixing_parser asks LLM to fix it
