"""
LangSmith: trace(name, run_type, ...) - trace parameters.

Syntax: trace(name="...", run_type="chain", tags=[], metadata={})

Same parameters as @traceable. Use for ad-hoc blocks.
"""

from langsmith.run_helpers import trace


with trace("fetch_data", run_type="retriever", tags=["api"]) as run:
    data = ["item1", "item2"]
    # run.end(outputs={"count": len(data)})  # optional: explicitly set outputs
