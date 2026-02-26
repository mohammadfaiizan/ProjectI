"""
LangSmith: client.create_example() - Add example to dataset.

Syntax: client.create_example(inputs={...}, outputs={...}, dataset_id=...)

inputs: input dict. outputs: expected output (for evaluation).
"""

from langsmith import Client

client = Client()
# client.create_example(
#     inputs={"question": "What is 2+2?"},
#     outputs={"answer": "4"},
#     dataset_id="dataset-uuid"
# )
