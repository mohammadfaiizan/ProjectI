"""
Runnables: RunnablePassthrough - Pass input through unchanged.

Syntax: runnable = RunnablePassthrough()

Input: Any dict
Output: Same dict (unchanged)

Use: Pass input to next step without modification
"""

from langchain_core.runnables import RunnablePassthrough

runnable = RunnablePassthrough()
# runnable.invoke({"x": 1, "y": 2}) -> {"x": 1, "y": 2}
