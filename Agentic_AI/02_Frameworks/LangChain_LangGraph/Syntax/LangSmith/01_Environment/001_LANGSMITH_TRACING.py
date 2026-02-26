"""
LangSmith: LANGSMITH_TRACING - Enable tracing.

Syntax: os.environ["LANGSMITH_TRACING"] = "true"

Set before importing LangChain/LangSmith. Enables automatic tracing.
For v2: LANGSMITH_TRACING_V2='true'
"""

import os

os.environ["LANGSMITH_TRACING"] = "true"
# Or for v2: os.environ["LANGSMITH_TRACING_V2"] = "true"
