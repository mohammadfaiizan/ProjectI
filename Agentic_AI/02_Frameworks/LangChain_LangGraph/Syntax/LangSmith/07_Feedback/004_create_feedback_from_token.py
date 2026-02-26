"""
LangSmith: create_feedback_from_token() - Feedback without run_id.

Syntax: client.create_feedback_from_token(token=..., key="...", score=...)

Use when you have a share token (from shareable link) not run_id.
"""

from langsmith import Client

client = Client()
# client.create_feedback_from_token(
#     token="share-token-from-url",
#     key="helpfulness",
#     score=0.9
# )
