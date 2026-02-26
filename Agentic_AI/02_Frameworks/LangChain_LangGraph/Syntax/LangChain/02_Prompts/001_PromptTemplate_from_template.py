"""
Prompts: PromptTemplate.from_template()

Syntax: template = PromptTemplate.from_template("text {var1} {var2}")

Input to invoke/format: dict with keys matching {placeholders}
  Example: {"var1": "hello", "var2": "world"}
"""

from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("Translate '{word}' to {language}")
# template.invoke({"word": "hello", "language": "Spanish"})
# template.format(word="hello", language="Spanish")
