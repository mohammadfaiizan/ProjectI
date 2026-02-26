"""
LangGraph: NotRequired - Optional state keys.

Syntax: key: NotRequired[type]

Keys not in initial state can be added by nodes.
Use for keys that are only set after certain nodes run.
"""

from typing import TypedDict
from typing_extensions import NotRequired


class WorkflowState(TypedDict):
    topic: str
    research: NotRequired[str]
    draft: NotRequired[str]
    review_feedback: NotRequired[str]
    revision_count: int
