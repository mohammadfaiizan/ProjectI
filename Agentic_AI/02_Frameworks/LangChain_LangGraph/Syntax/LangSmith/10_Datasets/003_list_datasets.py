"""
LangSmith: client.list_datasets() - List datasets.

Syntax: datasets = client.list_datasets()

Returns iterator of Dataset objects.
"""

from langsmith import Client

client = Client()
# for dataset in client.list_datasets():
#     print(dataset.name, dataset.id)
