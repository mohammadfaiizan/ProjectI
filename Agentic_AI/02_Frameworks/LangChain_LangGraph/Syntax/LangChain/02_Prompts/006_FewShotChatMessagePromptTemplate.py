"""
Prompts: FewShotChatMessagePromptTemplate - Include examples in prompt.

Syntax: FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([...]),
    examples=[{"input": "x", "output": "y"}, ...],
)

Input: dict with keys from example_prompt + "input" for new query
  Example: {"input": "angry"}
"""

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {"input": "happy", "output": "joyful"},
    {"input": "sad", "output": "melancholic"},
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Give a synonym."),
    few_shot,
    ("human", "{input}"),
])
# final_prompt.invoke({"input": "angry"})
