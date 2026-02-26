"""
LangSmith: Client - Synchronous API client.

Syntax: from langsmith import Client
        client = Client()

Parameters: api_key, url - optional overrides for env vars
"""

from langsmith import Client

client = Client()
# client = Client(api_key="...", url="https://api.smith.langchain.com")
