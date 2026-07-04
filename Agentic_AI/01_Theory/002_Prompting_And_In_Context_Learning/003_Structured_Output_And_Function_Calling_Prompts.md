# Structured Output and Function Calling Prompts

## Why Structured Output Is a Harder Problem Than It Looks

A language model is, at its core, a sampler over free-form text — at each step it produces a
probability distribution over the entire vocabulary and draws the next token from it. Nothing about
that mechanism inherently understands "this token must be a comma because we're inside a JSON array"
or "this field must be one of exactly three enum values." When you ask a model in plain prose to
"return JSON," you are relying entirely on the model's learned behavior — reinforced by instruction
tuning on many JSON-shaped examples — to *choose* tokens that happen to form valid JSON, with no
external mechanism preventing it from choosing a stray trailing comma, an unescaped quote inside a
string, or an extra sentence of commentary before the opening brace. For low-stakes or
human-reviewed use this is often good enough. For a production pipeline where a `json.loads()` call
downstream will throw on any deviation, "usually valid" is not a durable engineering guarantee, and
this chapter is about closing that gap using techniques that operate at different layers: the prompt
itself, the decoding process, and the surrounding application code.

It helps to separate three distinct guarantees that are often conflated under the single phrase
"structured output":

1. **Syntactic validity** — the output parses as JSON (or YAML, or XML) at all.
2. **Schema conformance** — the parsed object has the right keys, the right types, and satisfies constraints like enums or required fields.
3. **Semantic correctness** — the values themselves are actually correct given the input, not just correctly shaped.

Prompting alone can improve all three, but only decoding-level constraints can *guarantee* the first
two, and nothing can guarantee the third except validation against ground truth or a
human/second-model review. Knowing which of the three problems you actually have determines which
tool below is the right one to reach for.

## JSON Mode: Guaranteeing Syntax, Not Schema

"JSON mode," as exposed by most major providers, is a decoding-time constraint: the sampling process
is restricted so that whatever tokens the model emits, the result is guaranteed to be syntactically
valid JSON. Under the hood this typically works by tracking the grammar state of a JSON parser
alongside generation and masking out (setting to zero probability) any next token that would produce
a syntactically invalid continuation — an unmatched brace, a stray backslash, a comma in the wrong
place. This is a real guarantee, not a hint, because it operates on the logits themselves rather
than on the model's willingness to comply with an instruction.

```python
from openai import OpenAI
import json

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract the person's name, age, and city from the text. Respond in JSON."},
        {"role": "user", "content": "Maria is 34 and lives in Lisbon."},
    ],
    response_format={"type": "json_object"},
    temperature=0,
)

result = json.loads(response.choices[0].message.content)  # guaranteed not to raise on syntax
```

The critical limitation is that JSON mode says nothing about *which* JSON you get. The model could
satisfy JSON-mode's syntactic constraint while still returning `{"result": "Maria, 34, Lisbon"}`
instead of the three separate fields you wanted, or omitting a required field entirely, or using
`"thirty-four"` instead of `34`. JSON mode solves problem (1) from the list above; it does nothing
for problem (2). This is why JSON mode prompts still need a clearly specified schema in prose or by
example — the constraint at the decoding layer and the specification at the prompt layer are
complementary, not substitutes for each other.

## Schema-Constrained and Grammar-Based Generation

The stronger guarantee — schema conformance, not just syntactic validity — comes from constraining
decoding against an actual schema (a JSON Schema, a Pydantic model, or a formal grammar) rather than
just "valid JSON in general." The mechanism is the same idea taken further: at each generation step,
the set of tokens that would violate the schema (wrong type, disallowed enum value, a key not
present in the schema, a missing required field at the point the object is about to close) is masked
out of the sampling distribution. Open-source libraries such as Outlines and Guidance implement this
by compiling a schema or a regular grammar into a finite-state constraint that is checked at every
decoding step; OpenAI's "Structured Outputs" feature (as distinct from plain JSON mode) and
Anthropic's tool-use-based structured extraction provide managed versions of the same idea through
their hosted APIs.

```python
from pydantic import BaseModel
from typing import Literal

class SupportTicket(BaseModel):
    category: Literal["billing", "bug", "feature_request", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str

# With an API that supports schema-constrained "strict" structured outputs,
# the model is decoding-time restricted to only ever produce JSON matching this schema —
# it cannot emit a category outside the Literal set, cannot omit a required field,
# and cannot produce a wrong type for `priority`.
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Customer says they were billed twice and it's urgent."}],
    response_format=SupportTicket,  # provider-specific: pass schema, not just {"type": "json_object"}
)
ticket = SupportTicket.model_validate_json(response.choices[0].message.content)
```

It is worth being clear about what this still does not solve: schema conformance is not semantic
correctness. A schema-constrained model can still confidently assign `category="bug"` to a ticket
that is actually a billing issue, because the constraint only prunes *impossible* tokens, not
*wrong-but-schema-valid* ones. Constrained decoding also has a real practical cost — because a
valid-but-poor early token choice cannot be un-chosen, overly aggressive grammar constraints can
occasionally paint the model into a corner where the only schema-valid continuation is an awkward or
lower-quality one compared to what it would have naturally produced with more freedom. In practice
this is a minor concern for well-designed schemas but is worth knowing about if you see a slight
quality regression after switching from prose instruction to hard schema constraints on a task with
genuinely ambiguous correct answers.

## How a Model Decides to Call a Tool vs. Respond Directly

Function calling (tool use) is a special case of structured output: instead of asking the model to
produce an arbitrary JSON object, you give it a menu of named functions, each with a schema for its
arguments, and the model must decide, for a given turn, whether to emit a normal text response or
emit a structured "call this function with these arguments" object instead. This decision is itself
a *learned* behavior from fine-tuning — the model has seen many training examples of conversations
where a tool call was the appropriate response and many where plain text was — and the
prompt-engineering lever you actually control is the tool *descriptions*, the framing of the
surrounding conversation, and (on most APIs) an explicit `tool_choice` setting.

The tool's `description` field is not documentation for a human reader; it is training-time-shaped
natural language that directly steers when the model decides that tool is relevant. A vague
description ("gets data") gives the model little signal about when to use it and competes poorly
against other tools with similarly vague descriptions; a precise description that states exactly
what the tool does, what kind of query it's appropriate for, and what it is *not* for measurably
reduces both under-triggering (the model answers from its own knowledge when it should have called
the tool, e.g., inventing a stock price instead of calling `get_stock_price`) and over-triggering
(the model calls a tool unnecessarily for something it could and should just answer directly).

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_stock_price",
            "description": (
                "Returns the latest real-time stock price for a given ticker symbol. "
                "Use this whenever the user asks for a CURRENT or TODAY's price. "
                "Do not use this for historical prices, price predictions, or general "
                "company information — those are not supported by this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}
                },
                "required": ["ticker"],
            },
        },
    }
]
```

Notice the explicit negative scoping in that description ("do not use this for historical
prices..."). This is one of the highest-leverage, most underused patterns in tool-calling prompt
design: when you have two tools that are semantically adjacent (a current-price tool and a
historical-price tool, a "search documents" tool and a "search the web" tool), the model's confusion
between them is usually resolved not by making either description more verbose in isolation but by
explicitly stating the boundary between them in both descriptions.

Most function-calling APIs also expose a `tool_choice` parameter that overrides the model's own
judgment about whether to call a tool at all:

- `auto` — the model decides whether a tool call or a direct text response is appropriate; this is the right default for open-ended assistants.
- `required` (or `any`) — forces the model to call some tool on this turn, useful when you have deliberately routed a request into a stage where you know a tool call is necessary and you don't want the model second-guessing that.
- a specific named tool — forces exactly that tool to be called, useful for deterministic pipeline stages where you already know which function is needed and are just using the model to fill in arguments from natural language.
- `none` — suppresses tool calls entirely, useful for a final "summarize the results in plain language" turn where you specifically don't want another round of tool use.

Constraining `tool_choice` rather than relying purely on the model's own judgment is often the
single most effective fix for erratic tool-triggering behavior in production, because it removes an
entire class of decision from the model at the point in your pipeline where you already know the
answer.

For multi-step tool orchestration — a task that requires calling several tools in sequence,
potentially using the output of one to decide the input to the next — the prompt-level lever is
making the expected workflow explicit rather than assuming the model will infer the right ordering
purely from tool descriptions. A short instruction like "first look up the customer's account, then
check their subscription status, and only then decide whether to issue a refund" measurably improves
ordering reliability compared to leaving four available tools with only their individual
descriptions and trusting the model's own planning.

```python
def run_with_tools(client, user_message: str, tools: list, tool_impls: dict, model="gpt-4o") -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content  # model chose to respond directly, no more tools needed

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = tool_impls[call.function.name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": str(result),
            })
        # loop again: give the model the tool results and let it decide the next step
```

## Practical Tips for Reducing Malformed Output in Production

**Set temperature to 0 (or near-0) for anything structured.** Structured extraction and
tool-argument generation are not creative tasks — you almost always want the single most likely,
most consistent output rather than diversity, and low temperature substantially reduces spurious
formatting deviations independent of any schema constraint.

**Prefer schema-constrained generation over prompt-only instructions whenever the API supports it.**
A hard guarantee at the decoding layer removes an entire category of production bugs that no amount
of prompt wording can fully eliminate, because prompt compliance is probabilistic and decoding
constraints are not.

**Keep schemas flat and shallow where the task allows it.** Deeply nested optional objects with many
interacting fields are harder for a model to fill in correctly and harder for constrained-decoding
libraries to handle efficiently; where the underlying task allows it, prefer a flatter schema, even
if it means a slightly less "elegant" data model, in exchange for materially higher fill accuracy.

**Use enums instead of free-text fields wherever the space of valid values is closed.** A `category`
field constrained to a `Literal`/enum cannot be schema-violated and cannot drift in spelling or
casing across calls the way a free-text field can ("Bug" vs "bug" vs "software bug").

**Always validate and retry with the error fed back to the model, rather than silently failing.**
Even with schema constraints, a stray provider bug, a truncated response due to a `max_tokens`
cutoff, or a wrapped API without strict-mode support can still produce invalid output; a
validate-then-retry loop that shows the model *its own previous output and the specific validation
error* consistently recovers far better than a generic "please try again."

```python
from pydantic import BaseModel, ValidationError
import json

def get_structured_output(client, prompt: str, schema: type[BaseModel], model="gpt-4o", max_retries=3):
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
            return schema.model_validate(data)  # raises ValidationError on schema mismatch
        except (json.JSONDecodeError, ValidationError) as e:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That output was invalid: {e}. Return corrected JSON matching the schema exactly.",
            })
    raise RuntimeError(f"Failed to get valid structured output after {max_retries} attempts")
```

**Give one or two concrete examples of correctly filled output, especially for schemas with subtle
formatting requirements** — a date format, a specific rounding convention, a particular way of
handling `null` versus omission. This is the few-shot technique from chapter 1 applied specifically
to pin down formatting edge cases that a schema alone cannot express (a JSON Schema can say a field
is a string; it cannot easily say "must be in `YYYY-MM-DD` format" as reliably as one clear example
can).

**Separate the "does the tool need to be called" decision from the "what are the correct arguments"
decision when reliability matters most.** For high-stakes actions (issuing a refund, sending an
external email, executing a trade), it is common in production systems to use the model's tool-call
proposal only as a *draft*, then run a separate, stricter validation pass — schema validation,
business-rule checks, and often human confirmation — before the tool actually executes, rather than
trusting model output to directly trigger irreversible side effects. This pattern, of treating
LLM-proposed structured output as an untrusted draft that is validated before use, comes up again
from the security angle in the final chapter of this series, where it is one of the core defenses
against a model being manipulated into producing a malicious but schema-valid tool call.
