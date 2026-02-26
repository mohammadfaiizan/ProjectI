"""
Runnables: RunnablePassthrough.assign() - Add computed keys to input.

Syntax: RunnablePassthrough.assign(key=lambda x: x["input"].upper())

Input: dict
Output: dict with original keys + new keys from assign
  Each callable receives input dict, returns value for its key
"""

from langchain_core.runnables import RunnablePassthrough

runnable = RunnablePassthrough.assign(
    word_count=lambda x: len(x.get("text", "").split()),
    uppercase=lambda x: x.get("text", "").upper(),
)
# runnable.invoke({"text": "hello world"}) -> {"text": "hello world", "word_count": 2, "uppercase": "HELLO WORLD"}
