"""
LCEL: chain.ainvoke() - Asynchronous invocation.

Syntax: result = await chain.ainvoke(input)

Input: dict (same as invoke)

Output: Same as invoke (final result)
"""

import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("Translate to Spanish: {text}")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)


async def main():
    result = await chain.ainvoke({"text": "Hello"})
    return result


# asyncio.run(main())
