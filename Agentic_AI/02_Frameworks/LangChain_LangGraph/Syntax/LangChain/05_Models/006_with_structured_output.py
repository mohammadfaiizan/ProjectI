"""
Models: llm.with_structured_output() - Get Pydantic output directly.

Syntax: structured_llm = llm.with_structured_output(YourModel)

Input: Same as invoke (messages)
Output: YourModel instance (no separate parser needed)
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


class Answer(BaseModel):
    value: int = Field(description="Numeric answer")
    unit: str = Field(description="Unit if applicable")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(Answer)
# structured_llm.invoke("What is 2+2?") -> Answer(value=4, unit="")
