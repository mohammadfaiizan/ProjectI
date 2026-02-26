"""
LangSmith: wrap_anthropic() - Add tracing to Anthropic client.

Syntax: from langsmith import wrap_anthropic
        client = wrap_anthropic(Anthropic())

Patches Anthropic client. Supports tool calls, streaming.
"""

from anthropic import Anthropic
from langsmith import wrap_anthropic

client = wrap_anthropic(Anthropic())
# All client.messages.create() calls are traced
