"""
LCEL: chain.astream() - Asynchronous streaming.

Syntax: async for chunk in chain.astream(input): ...

Input: dict (same as invoke)

Output: AsyncIterator - yields chunks
"""

import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("Count 1 to 5: {topic}")
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | StrOutputParser()
)


async def main():
    async for chunk in chain.astream({"topic": "numbers"}):
        print(chunk, end="", flush=True)


# asyncio.run(main())
