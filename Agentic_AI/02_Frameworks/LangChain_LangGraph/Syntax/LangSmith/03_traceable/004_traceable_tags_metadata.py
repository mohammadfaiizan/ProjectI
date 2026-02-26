"""
LangSmith: @traceable(tags=[...], metadata={...}) - Add tags and metadata.

Syntax: @traceable(tags=["beta", "v2"], metadata={"version": "1.0"})

Tags: List[str] - filter/group in LangSmith UI
Metadata: Dict - arbitrary key-value pairs
"""

from langsmith.run_helpers import traceable


@traceable(tags=["production", "critical"], metadata={"version": "2.0"})
def process_order(order_id: str) -> dict:
    return {"status": "processed"}
