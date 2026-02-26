"""
Prompts: prompt.invoke() - Get prompt value with variable substitution.

Syntax: prompt_value = prompt.invoke(input_dict)

Input: dict with keys matching all template variables
  Example: {"question": "What is AI?"}

Output: PromptValue (use .to_string() or pass to chain)
"""

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{question}"),
])
prompt_value = prompt.invoke({"question": "What is 2+2?"})
# prompt_value.to_messages() -> list of messages
# prompt_value.to_string() -> str
