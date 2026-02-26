"""
LangSmith: client.create_dataset() - Create a dataset.

Syntax: dataset = client.create_dataset(name="...", description="...")

Returns Dataset. Add examples for evaluation.
"""

from langsmith import Client

client = Client()
# dataset = client.create_dataset(
#     dataset_name="qa-test",
#     description="Q&A evaluation set"
# )
