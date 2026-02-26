"""
LangSmith: LANGSMITH_API_KEY - API key for LangSmith service.

Syntax: os.environ["LANGSMITH_API_KEY"] = "your-api-key"

Required for hosted LangSmith. Get from smith.langchain.com.
"""

import os

os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."
# Or use .env file with python-dotenv
