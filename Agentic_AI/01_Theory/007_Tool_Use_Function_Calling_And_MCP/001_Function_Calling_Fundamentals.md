# Function Calling Fundamentals

## Why Function Calling Exists

A language model, no matter how large, is fundamentally a text-in, text-out system. It has no clock, no filesystem, no network socket, and no memory of anything that happened after its training data was collected. Left to its own devices, if you ask it "what's the weather in Austin right now" it will do one of two things: refuse, or confidently make something up that sounds plausible. Neither is acceptable in a production system. Function calling (also called tool use) is the mechanism that closes this gap. It gives the model a way to say, in a structured, machine-parseable way, "I need you, the calling program, to go run this specific operation with these specific inputs, and give me back the result so I can continue."

The important word there is "structured." Long before official function-calling APIs existed, people tried to get this behavior out of models by prompting them to emit something like `ACTION: search("weather in Austin")` in plain text and then writing regex to scrape it out of the response. This works occasionally and breaks constantly — models add stray punctuation, wrap the call in markdown, explain what they're about to do in the middle of the "action" line, or forget the exact syntax you asked for. Native function calling support, introduced by OpenAI in mid-2023 and quickly matched by Anthropic, Google, and others, moves this out of prompt-engineering folklore and into the model's actual training and decoding process. The model is fine-tuned to emit calls in a fixed, constrained format — usually a JSON object with a function name and arguments — and in many implementations the decoding process itself is constrained (via grammar-based or schema-based decoding) so that syntactically invalid output isn't even possible to sample. That reliability is the entire value proposition. If you can't trust the shape of the output, you can't build software around it.

It's worth being precise about what the model is and isn't doing. The model never executes anything. It has no access to your database, your filesystem, or the internet. What it does is pattern-match the user's request against a list of tool definitions you provided, decide that one of them is relevant, and generate a block of structured data describing which one and with what arguments. That's it. All of the actual power — and all of the actual risk — lives in your code, which receives that structured request and decides whether and how to act on it. This separation is not an implementation detail; it's the core safety property of the whole paradigm. The model proposes, your code disposes.

## The End-to-End Round Trip

Concretely, a single function-calling interaction is a loop with a fixed number of stages, and it's worth internalizing this loop because every framework (LangChain, LlamaIndex, the raw OpenAI/Anthropic SDKs, custom agent loops) is just a variation on it:

1. **Tool declaration.** You send the model a list of tools it's allowed to use. Each tool has a name, a natural-language description, and a JSON Schema describing its parameters. This travels alongside the conversation messages on every single request — the model doesn't "remember" what tools it has between calls; you re-declare them every time.
2. **Model decision.** The model reads the conversation and the tool list and decides, based on its training, whether the user's request is best served by calling a tool, and if so which one. This decision is made the same way the model decides which words to say next — it's not a separate classifier bolted on the side, it's the same autoregressive generation process, just constrained to a schema when it chooses the "call a tool" path.
3. **Structured call emission.** Instead of (or alongside) natural-language text, the model emits one or more tool-call objects: a name and a JSON blob of arguments. Critically, this is the end of the model's turn. The API response comes back to your code with a stop reason like `tool_calls` (OpenAI) or `tool_use` (Anthropic) instead of the normal "I'm done talking" reason.
4. **Local execution.** Your code inspects the tool call, looks up the corresponding real function, parses/validates the arguments, and executes it. This is ordinary code — a database query, an HTTP call, a subprocess, whatever you wired up. The model has no visibility into this step at all.
5. **Result injection.** You take whatever your function returned (a string, a JSON object, an error message) and append it back into the conversation as a new message with a special role (`tool` in OpenAI's API, a `tool_result` content block inside a `user` message in Anthropic's API), tagged with the ID of the specific call it's answering.
6. **Continuation.** You send the updated message list — original conversation, the model's tool-call message, and your tool-result message — back to the model for another turn. The model now has the tool's output in context and can either produce a final natural-language answer or decide it needs to call another tool.

Steps 2 through 6 can repeat many times in a single user-facing turn — this is what people mean by an "agent loop." A single user question like "compare this quarter's revenue to last quarter's and email the finance team" might trigger three or four sequential tool calls before the model finally has enough information to write a natural-language summary and stop.

Here is that loop written out plainly, using the OpenAI-style API shape (Anthropic's is structurally identical, just with different field names):

```python
import json
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather conditions for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Austin, TX'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit for the response"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def get_weather(city: str, unit: str = "fahrenheit") -> dict:
    # In reality this calls a weather API. Mocked here for illustration.
    return {"city": city, "temp": 89, "unit": unit, "condition": "sunny"}

AVAILABLE_FUNCTIONS = {"get_weather": get_weather}

def run_agent_turn(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": user_message},
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(message)

        # Stop condition: the model produced a final answer, no tool calls.
        if not message.tool_calls:
            return message.content

        # Otherwise, execute every requested tool call and feed results back.
        for call in message.tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)
            fn = AVAILABLE_FUNCTIONS.get(fn_name)

            if fn is None:
                result = {"error": f"Unknown tool requested: {fn_name}"}
            else:
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": fn_name,
                "content": json.dumps(result),
            })
        # Loop back around: send the updated messages, let the model continue.

print(run_agent_turn("What's the weather in Austin, in Celsius?"))
```

A few details in that snippet matter more than they look. The `messages.append(message)` line appends the *model's own tool-call message* back into the transcript before you do anything else — the API requires the assistant message that contains the tool call to be present in history when you later supply the tool result, otherwise the `tool_call_id` you reference won't resolve to anything. The `tool_call_id` on the result message is how the model correlates a result with the specific call it made; this matters enormously once there are multiple calls in flight (more on that below). And the loop has no hard iteration cap in this minimal version — in production code you always bound it, because a model that keeps deciding it needs "just one more" tool call can spin forever and burn tokens and money.

## The Model Deciding vs. Your Code Forcing

Every function-calling API exposes a control knob — commonly called `tool_choice` — that determines how much freedom the model has in deciding whether to call a tool at all. This is a genuinely important axis of design, not a minor parameter, because it changes the reliability characteristics of your whole system.

**`auto`** is the default and the one used in the example above: the model looks at the conversation and the available tools and decides for itself whether a tool call is warranted, and if so, which tool and with what arguments. This is what you want for open-ended assistants where sometimes the answer is "just talk," and sometimes it's "look something up first." The risk is that the model can get this wrong in both directions — calling a tool when a direct answer would have suffced (wasting latency and money), or, more dangerously, answering directly from its own (possibly stale or hallucinated) knowledge when it really should have checked a live source.

**Forced/required tool use** is when your code removes that choice entirely. OpenAI exposes this as `tool_choice={"type": "function", "function": {"name": "..."}}` to force one specific function, or `tool_choice="required"` to force *some* tool call (any of them) rather than a free-text answer. This is the right move whenever you are using the "tool calling" mechanism not to give the model optional capabilities, but to force it to emit structured data — for example, using a single fake "tool" purely as a way to get guaranteed JSON output matching a schema, a pattern that predates dedicated structured-output APIs and is still common where those aren't available. It's also useful in a classification or routing step: "given this support ticket, call `categorize_ticket` with the correct category" — you don't want the model free-associating a prose answer here, you want it to always emit the structured call.

**`none`** disables tool calling for that turn entirely, even if tools are declared. This shows up in agent loops where you want to force a final natural-language summary after tool results have already come back — you pass the same tool list (for API consistency) but set `tool_choice="none"` so the model is compelled to synthesize a text answer instead of chaining into yet another tool call.

The practical lesson here is that "the model decides" is a spectrum, not a boolean. A well-built agent uses `auto` for genuinely open-ended reasoning steps, and switches to forced tool choice for steps where you already know structurally what needs to happen and just need the model to fill in the specific arguments (which city, which record ID, which category). Relying on `auto` everywhere and hoping the model always makes the right call is how you end up with flaky, hard-to-debug agent behavior — an agent that sometimes forgets to check inventory before promising a delivery date is exhibiting exactly the kind of nondeterminism that forced tool choice exists to eliminate.

## Parallel and Multi-Tool Calls

Early function-calling implementations only allowed one tool call per model turn: the model calls a function, waits for the result, then decides what to do next, strictly sequentially. This is correct when calls are genuinely dependent — you need the output of `search_flights` before you can call `book_flight` — but it's wasteful when calls are independent. If a user asks "what's the weather in Austin and in Denver," there is no reason to make two full round trips to the model; both `get_weather` calls can be issued in the same turn.

Modern APIs support this directly: a single assistant turn can contain multiple tool-call objects, each with its own ID, and your code is expected to execute all of them (in parallel if they're independent, which is usually true) and return all of their results — each tagged with its own `tool_call_id` — before the next model turn.

```python
import concurrent.futures

def execute_all_tool_calls(tool_calls, available_functions):
    """Execute every tool call from a single model turn, in parallel."""

    def run_one(call):
        fn_name = call.function.name
        fn_args = json.loads(call.function.arguments)
        fn = available_functions.get(fn_name)
        try:
            result = fn(**fn_args) if fn else {"error": f"unknown tool {fn_name}"}
        except Exception as e:
            result = {"error": str(e)}
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": fn_name,
            "content": json.dumps(result),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        # Order of results doesn't matter here — each is tagged by tool_call_id,
        # so the model can match them up regardless of the order they're appended.
        tool_messages = list(pool.map(run_one, tool_calls))

    return tool_messages
```

Two things go wrong if you get this pattern wrong. First, if you execute the calls sequentially when they're I/O-bound and independent (two separate HTTP calls to a weather API, say), you pay their latencies additively instead of concurrently — a needless slowdown for the user. Second, and more subtly, if you fail to return a result for *every* `tool_call_id` the model emitted — say one of three calls threw an unhandled exception and your code silently dropped it — most APIs will reject the next request outright, because the protocol requires a matching tool-result message for every tool-call ID in the preceding assistant turn. The defensive pattern is to always produce a result message for every call, even if that "result" is just a serialized error, exactly as the `run_one` function above does with its `try/except`.

Parallel tool calling introduces one more decision point worth flagging: not everything that arrives together should actually run concurrently. If the model (incorrectly, or because you didn't disable parallel calls) emits both `read_file("config.json")` and `delete_file("config.json")` in the same turn, blindly parallel-executing them is a correctness and safety hazard — you'd rather detect that these two calls touch the same resource and either serialize them or reject the batch and ask the model to reconsider. Most production tool-execution layers add a dependency/conflict check between "fan out and run everything in parallel" and "actually call the function," precisely to guard against this. The tool-calling protocol gives you the mechanism for concurrency; whether it's *safe* to use in a given batch is a judgment your execution layer has to make, not something the API decides for you.

## What "the Result" Actually Looks Like

A subtlety that trips up a lot of first implementations: the content you put into the tool-result message is not free-form. It becomes part of the model's next input, and the model has never seen your internal data structures — it only sees whatever string (or content block) you serialize into that message. This means the quality of your final answer is bottlenecked by how well you format tool output for model consumption, not just by whether the tool "worked."

A few practical rules follow from this. Keep results compact — dumping a 500-row raw SQL result set into a tool message wastes context window and often confuses the model about which rows matter; summarize or truncate before returning, and tell the model you truncated. Prefer structured JSON over free text when the downstream consumer is the model itself, since the model parses structured data more reliably than it "reads" a paragraph. And always distinguish success from failure explicitly in the payload (`{"success": false, "error": "..."}` rather than just stuffing an exception's string into a field that normally holds data) — models are quite good at recognizing an explicit error field and adjusting their next action (retrying, trying an alternative tool, or apologizing to the user) when you make the failure legible, and quite bad at inferring failure from a malformed or unexpectedly-shaped success payload.

## Provider Differences Worth Knowing Cold

Interviewers frequently probe whether you've actually built against more than one vendor's API, because the surface-level concept is identical but the wire format differs enough to trip up naive ports between them.

OpenAI represents a tool call as a `tool_calls` array on the assistant message, each entry with an `id`, a `function.name`, and a `function.arguments` string (note: a *string* containing JSON, which you must `json.loads` yourself — it is not already a parsed object). Results go back as separate messages with `"role": "tool"` and a `tool_call_id` field.

Anthropic represents a tool call as a `tool_use` content block embedded inside the assistant message's `content` array (a message can mix plain text and one or more `tool_use` blocks), where `input` is already a parsed JSON object, not a string. Results go back not as a new role, but as a `user`-role message containing one or more `tool_result` content blocks, each referencing a `tool_use_id`. Anthropic also signals the reason for stopping via `stop_reason == "tool_use"` at the top level of the response, whereas OpenAI signals it via `finish_reason == "tool_calls"` on the choice.

These differences are exactly why most teams that need to be provider-agnostic build a thin normalization layer that converts each vendor's tool-call representation into one internal shape before their agent loop touches it, and converts back when constructing the result message. Trying to write agent-loop logic that branches on provider throughout is a maintenance trap.

## Streaming and Tool Calls

Most production chat interfaces stream tokens to the user as they're generated rather than waiting for a complete response, and tool calling has to coexist with that. The complication is that a streamed tool call doesn't arrive as one clean JSON object — it arrives as a sequence of incremental deltas, the same way streamed text arrives as a sequence of token fragments, and your code has to accumulate those fragments before the arguments are valid, parseable JSON.

```python
def consume_streamed_tool_call(stream) -> list[dict]:
    """Accumulate incremental tool-call deltas from a streaming response
    into complete, parseable tool-call objects."""
    accumulated = {}  # index -> {"id": ..., "name": ..., "arguments": ""}

    for chunk in stream:
        delta = chunk.choices[0].delta
        if not delta.tool_calls:
            continue
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            entry = accumulated.setdefault(idx, {"id": None, "name": None, "arguments": ""})
            if tc_delta.id:
                entry["id"] = tc_delta.id
            if tc_delta.function and tc_delta.function.name:
                entry["name"] = tc_delta.function.name
            if tc_delta.function and tc_delta.function.arguments:
                entry["arguments"] += tc_delta.function.arguments  # concatenate fragments

    # Only now, after the stream is fully drained, is entry["arguments"] valid JSON.
    return [
        {"id": e["id"], "name": e["name"], "arguments": json.loads(e["arguments"])}
        for e in accumulated.values()
    ]
```

The practical implication is that you cannot execute a tool call incrementally as its arguments stream in — you must wait for the stream (or at least that portion of it) to finish before `json.loads` will succeed, since a half-received argument string is simply invalid JSON. Some UIs paper over this by streaming a "the assistant is using a tool..." indicator to the user based on the presence of a `tool_calls` delta, without attempting to interpret the partial arguments themselves, which keeps the experience responsive without requiring you to solve incremental JSON parsing.

## Managing Conversation History Across a Multi-Step Tool Loop

A tool-calling agent's message list grows quickly — every tool call adds an assistant message and every result adds a tool message, and a task requiring six sequential tool calls has added twelve messages before the model produces a single word the user actually reads. Left unmanaged, this both wastes context window and, less obviously, can degrade quality: a very long, tool-result-heavy history buries the user's original request under noise the model has to re-attend to on every subsequent turn.

The two mitigations used in practice are truncation and summarization. Truncation is the blunt instrument: drop the oldest tool-call/tool-result pairs once the history exceeds a token budget, keeping the system prompt and the most recent exchanges intact — acceptable when older tool results are unlikely to be relevant again, risky when a later step needs to refer back to something established early in the sequence. Summarization is more careful: periodically ask the model (or a separate, cheaper model) to compress a run of completed tool interactions into a short natural-language summary that replaces the verbose original messages, preserving the substance of what was learned while discarding the raw payload. Whichever strategy you use, one rule is non-negotiable regardless of provider: you can never truncate half of a tool-call/tool-result pair — removing a tool-result message while leaving its corresponding assistant tool-call message in history (or vice versa) leaves a dangling `tool_call_id` reference that most APIs will reject outright on the next request. Truncation logic has to treat a call and its result as a single atomic unit.

## A Worked Multi-Tool Trace

It's useful to see what the message list actually looks like after a two-step tool interaction, since the abstract description of "messages get appended" is easier to reason about with a concrete trace. Given the user asks "What's the weather in Austin, and is that warmer than Denver?", a full trace might look like this by the time the model produces its final answer:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant with tool access."},
    {"role": "user", "content": "What's the weather in Austin, and is that warmer than Denver?"},

    # Turn 1: model requests both cities in parallel (see next section)
    {"role": "assistant", "tool_calls": [
        {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "Austin"}'}},
        {"id": "call_2", "function": {"name": "get_weather", "arguments": '{"city": "Denver"}'}},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": '{"city": "Austin", "temp": 89}'},
    {"role": "tool", "tool_call_id": "call_2", "content": '{"city": "Denver", "temp": 71}'},

    # Turn 2: model has both results, needs no further tools, produces final text
    {"role": "assistant", "content": "Austin is 89°F and Denver is 71°F, so Austin is warmer by 18 degrees."},
]
```

Notice that the comparison itself ("is Austin warmer") required no dedicated `compare_temperatures` tool at all — once both raw numbers were in context, the model handled the arithmetic and the natural-language synthesis directly. This is a useful design heuristic in its own right: build tools for the things a model is *unreliable* at doing itself (fetching live external state, in this example) and let the model handle composition and reasoning over the results in plain text, rather than reflexively building a tool for every sub-step of a task.

## Common Pitfalls in the Round Trip

The failure mode people hit most often when first wiring this up is forgetting that the assistant's tool-call message must be preserved in the message history sent back with the tool result — dropping it (because it "isn't useful," being just a tool call with no visible text) breaks the `tool_call_id` linkage and the API will error or hallucinate context. A second common mistake is treating `tool_choice="auto"` results as reliable in situations that actually require determinism — if your business logic assumes a tool was called and it wasn't (the model just answered in prose because it thought it already knew the answer), you need an explicit check for `message.tool_calls` being present, not an assumption baked into downstream code. A third is unbounded loops: without an iteration cap and without detecting when the model requests the exact same tool call twice in a row with identical arguments, an agent can burn through budget in a corrective loop that never converges. All three of these are addressed by treating the round trip as a state machine with explicit stop conditions, not as an ad hoc while-loop you got working once against a happy-path example.
