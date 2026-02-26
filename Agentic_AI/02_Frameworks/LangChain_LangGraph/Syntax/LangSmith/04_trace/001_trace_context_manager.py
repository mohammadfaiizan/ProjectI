"""
LangSmith: trace - Context manager for tracing a block.

Syntax: with trace("block_name"): ...

Manages a LangSmith run for the block. Sync and async support.
"""

from langsmith.run_helpers import trace


with trace("my_operation") as run:
    result = "some result"
    # run is the RunTree for this block
