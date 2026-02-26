"""
LangSmith: run_on_dataset() - Run chain on dataset (LangChain).

Syntax: from langchain.smith import run_on_dataset
        run_on_dataset(dataset_name, llm_or_chain_factory, ...)

Evaluates chain against dataset. Logs to LangSmith.
"""

# from langsmith import Client
# from langchain.smith import run_on_dataset, RunEvalConfig
#
# client = Client()
# run_on_dataset(
#     client,
#     "my-dataset",
#     lambda: my_chain,
#     evaluation=RunEvalConfig(evaluators=["qa"])
# )
