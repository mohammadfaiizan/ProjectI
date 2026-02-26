"""
Output Parsers: PydanticOutputParser - Parse into Pydantic model.

Syntax: parser = PydanticOutputParser(pydantic_object=YourModel)

Input: AIMessage (expects JSON in content)
Output: YourModel instance

YourModel: Pydantic BaseModel with Field(description=...) for each field
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age")
    city: str = Field(description="City")


parser = PydanticOutputParser(pydantic_object=Person)
prompt = ChatPromptTemplate.from_template(
    "Extract from: {text}\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())
chain = prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | parser
# chain.invoke({"text": "John is 30, lives in NYC"}) -> Person(name="John", age=30, city="NYC")
