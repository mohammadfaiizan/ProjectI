"""
Models: temperature - Control randomness.

Syntax: temperature=0 (deterministic) to 2 (creative)

  temperature=0: Same input -> same output, factual
  temperature=0.7: Balanced, some variation
  temperature=1.0+: More creative, varied
"""

from langchain_openai import ChatOpenAI

llm_deterministic = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_creative = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
