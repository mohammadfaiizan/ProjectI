"""
Prompts: PromptTemplate.format() - Format template with variables.

Syntax: formatted_str = template.format(var1=value1, var2=value2)

Input: Keyword arguments (var1=..., var2=...)
  All {variable} placeholders must be provided

Output: str - formatted prompt string
"""

from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("Translate '{word}' to {language}")
formatted = template.format(word="hello", language="French")
# -> "Translate 'hello' to French"
