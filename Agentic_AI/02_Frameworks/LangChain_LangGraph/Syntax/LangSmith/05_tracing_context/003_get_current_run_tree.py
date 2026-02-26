"""
LangSmith: get_current_run_tree() - Get current RunTree.

Syntax: run_tree = get_current_run_tree()

Returns RunTree or None. Use .to_headers() for distributed tracing.
"""

from langsmith.run_helpers import get_current_run_tree

# run_tree = get_current_run_tree()
# if run_tree:
#     headers = run_tree.to_headers()  # Pass to next service
