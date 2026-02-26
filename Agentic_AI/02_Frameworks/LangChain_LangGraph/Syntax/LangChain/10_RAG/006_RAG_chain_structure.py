"""
RAG: RAG chain structure - context + question -> LLM.

Syntax: {"context": retriever | format_fn, "question": RunnablePassthrough()} | prompt | llm | parser

Input: str (question) - passed through as "question"
  context comes from retriever
Output: str (answer)
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n{context}\n\nQuestion: {question}"
)
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | ChatOpenAI(...)
#     | StrOutputParser()
# )
# rag_chain.invoke("What is Python?") -> str
