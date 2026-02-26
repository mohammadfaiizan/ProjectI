"""
LangSmith: evaluate() - Evaluate a target on dataset.

Syntax: from langsmith import evaluate
        results = evaluate(target, data="dataset-name")

target: callable or runnable. data: dataset name or examples.
"""

from langsmith import evaluate

# def my_chain(input): return {"output": "..."}
# results = evaluate(my_chain, data="my-dataset")
