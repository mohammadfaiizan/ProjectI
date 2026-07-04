# Tool Schema Design Best Practices

## Why Schema Design Is the Real Bottleneck

Once you've built the mechanical round trip — declare tools, let the model call one, execute it, feed the result back — the thing that actually determines whether your agent works reliably in production has almost nothing to do with that plumbing. It's whether the model consistently picks the *right* tool, fills in *correct* arguments, and doesn't invent parameters or entire tools that don't exist. All three of these are governed by how you write the tool schema: its name, its description, its parameter definitions, and how it sits alongside the other tools you've exposed. This is, in practice, a prompt engineering problem wearing a JSON Schema costume. The model has no access to your code, your docstrings, or your mental model of what the tool does — it only sees the name and description strings you wrote and the shape of the parameters object. Every ambiguity you leave in that text is an ambiguity the model has to resolve by guessing, and it will guess wrong a meaningful fraction of the time, especially under load with dozens of tools competing for attention.

This matters more, not less, as systems mature. A demo with three tools and a friendly system prompt will look reliable no matter how sloppy the schemas are, because there's little for the model to confuse. Once you're at twenty, fifty, or a hundred tools — which is normal for an enterprise agent wired into several internal systems — small ambiguities compound, and the difference between an agent that works 95% of the time and one that works 99.5% of the time is almost entirely down to schema hygiene.

## Naming Tools So the Model Doesn't Have to Guess

The name of a tool is the single strongest signal the model uses to route a request, often more than the description, because it's the token sequence closest to where the model has to commit to a decision. Vague or generic names — `process`, `handle_request`, `do_action`, `manage_data` — force the model to lean entirely on the description to disambiguate, and descriptions get skimmed less carefully than names when there are many tools in the list. Prefer names that are specific verbs plus specific nouns: `get_current_weather` rather than `weather`, `create_calendar_event` rather than `calendar`, `search_customer_orders_by_email` rather than `search`. The extra verbosity is not wasted tokens; it's disambiguating signal that saves you far more tokens in failed calls and retries than it costs to declare.

Consistency of naming convention across your whole tool set also matters more than which specific convention you pick. If half your tools are `snake_case` and half are `camelCase`, or some start with a verb and others start with a noun (`get_weather` next to `weather_lookup`), the model has to work harder to build a mental map of "how tools in this system are named," and that extra cognitive load shows up as selection errors. Pick one convention, document it, and enforce it — many teams do this with a lint step that runs over their tool registry before deployment, checking name patterns the same way a style linter checks code.

Namespacing becomes essential once tools come from more than one subsystem. If you have a CRM integration and a ticketing integration that both expose a `search` tool, the model has no way to know which `search` does what without reading both descriptions closely under time pressure — and it will occasionally pick the wrong one. Prefixing by domain (`crm_search_contacts`, `ticketing_search_issues`) turns an ambiguous decision into an unambiguous one at the naming level, before the model even has to reason about semantics.

## Writing Descriptions That Actually Disambiguate

A tool description should answer three questions a competent human engineer would ask before calling this function for the first time: what does it do, when should I use it (and, implicitly, when should I *not*), and what does it hand back. Most under-specified tool descriptions answer only the first.

Compare two descriptions for the same function:

```python
# Weak: describes the mechanism, not the intent or boundaries.
{
    "name": "get_customer",
    "description": "Gets a customer.",
    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
}

# Strong: disambiguates scope, input format, and what's returned.
{
    "name": "get_customer_by_id",
    "description": (
        "Look up a single customer's account details using their internal "
        "customer ID (format: CUST-XXXXXX). Returns name, email, subscription "
        "tier, and account status. Does NOT return order history — use "
        "get_customer_orders for that. Use this when the user references a "
        "specific customer ID directly; use search_customers_by_email if you "
        "only have an email address."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Internal customer ID, formatted CUST-XXXXXX",
                "pattern": "^CUST-\\d{6}$"
            }
        },
        "required": ["customer_id"]
    }
}
```

The strong version does several things the weak one doesn't. It states an explicit input format (`CUST-XXXXXX`) so the model doesn't try to pass a raw name or email into a field that expects an ID. It explicitly rules out an adjacent capability ("does NOT return order history"), which is often more valuable than describing what the tool *does* do, because the primary failure mode with overlapping tools is the model assuming one tool does more than it does. And it cross-references the sibling tool to use in the alternate case, effectively encoding a decision tree directly into the text the model reads at selection time, rather than hoping the model infers the boundary on its own.

Parameter-level descriptions deserve the same care as the top-level one. `"description": "The location"` tells the model almost nothing about what a valid value looks like — a city name? A ZIP code? Latitude/longitude? Compare to `"description": "City and state, e.g. 'Austin, TX'. Do not include ZIP codes."` — this closes off an entire class of malformed calls before they happen, because it gives the model a concrete pattern to imitate. Wherever a parameter has a natural finite set of valid values, use an `enum` in the schema rather than describing the options in prose; enums are enforced structurally (in many implementations, via constrained decoding) rather than merely suggested, which is a categorically stronger guarantee than "please only use one of: draft, sent, archived" in a description string that the model might still deviate from.

## Overlapping Tools Are a Silent Failure Mode

The most damaging schema-design mistake in real systems is not a missing description — it's two or more tools whose purposes genuinely overlap, so that a well-formed request could plausibly go to either one. This happens organically as a tool catalog grows: someone adds `search_documents` for a new integration, not realizing there's already a `search_knowledge_base` that does nearly the same thing against a different backend. The model isn't being unreliable when it picks the "wrong" one in this situation — from its point of view, given the information in the descriptions, both are equally valid, and it's essentially guessing.

The fix is not a smarter model; it's disambiguating the schemas so the tools are no longer functionally interchangeable from the description's point of view. That can mean genuinely narrowing what each tool covers ("`search_knowledge_base` searches internal help-center articles only; `search_documents` searches the user's own uploaded files only"), merging the two tools into one with a parameter that selects the target corpus, or, if the overlap is truly unavoidable, adding explicit disambiguation language to both descriptions pointing at each other. What you should never do is ship two overlapping tools and hope the model's judgment saves you — audit your tool catalog periodically for exactly this kind of semantic collision, the same way you'd audit an API surface for redundant endpoints.

## How Many Tools Is Too Many

There's a real, measurable degradation in tool-selection accuracy as the number of tools presented in a single request grows, independent of how well each individual schema is written. Every tool definition — name, description, full parameter schema — is injected into the model's context on every single call, which means a large tool catalog is not free even when unused: it consumes context budget, and past a certain count (in practice, the effective ceiling for reliable auto-selection tends to sit somewhere in the range of a few dozen tools for most current models, well below the hundreds some enterprise catalogs eventually reach) the model starts making more selection errors purely from the sheer number of superficially plausible options in front of it, in much the same way a human faced with fifty similar-looking menu items takes longer and orders wrong more often than one faced with five.

The standard mitigation is **dynamic tool selection**: instead of sending your entire tool catalog on every request, retrieve only the subset likely to be relevant to the current user turn, typically via embedding similarity between the query and each tool's name-plus-description, and send only the top-k candidates.

```python
import numpy as np

class ToolRetriever:
    """Select a small, relevant subset of tools for a given query,
    instead of sending the entire tool catalog on every call."""

    def __init__(self, tools: list[dict], embed_fn):
        self.tools = tools
        self.embed_fn = embed_fn
        self._tool_embeddings = [
            self.embed_fn(f"{t['function']['name']}: {t['function']['description']}")
            for t in tools
        ]

    def select(self, user_query: str, top_k: int = 8) -> list[dict]:
        query_emb = self.embed_fn(user_query)
        scores = [
            self._cosine_sim(query_emb, tool_emb)
            for tool_emb in self._tool_embeddings
        ]
        ranked = sorted(zip(scores, self.tools), key=lambda x: x[0], reverse=True)
        return [tool for _, tool in ranked[:top_k]]

    @staticmethod
    def _cosine_sim(a, b) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
```

This pattern buys you two things simultaneously: better selection accuracy (fewer, more relevant candidates per call) and lower cost and latency (less schema text sent per request). The trade-off is added system complexity — an embedding index to maintain, and a new failure mode where the retriever itself excludes the correct tool from the candidate set, which is invisible to the model (it simply never sees the tool it needed). Production systems mitigate this with a hybrid approach: a small set of "always available" core tools sent unconditionally, plus a dynamically retrieved set layered on top for domain-specific capabilities.

## Common Failure Modes and Why They Happen

**Wrong tool selected.** This is almost always a naming or description ambiguity problem, as covered above, but it also happens when the user's request is genuinely underspecified and multiple tools could serve it — in which case the fix isn't a better schema, it's the model asking a clarifying question before acting, which you encourage via system-prompt instructions ("if the request is ambiguous between two tools, ask which one applies before calling anything") rather than schema changes.

**Malformed arguments.** The model emits a tool call, but the JSON doesn't parse, or a required field is missing, or a value is the wrong type (a string where a number was expected). This is more common with smaller or less capable models, and with parameter schemas that are themselves poorly typed — a `"type": "string"` field that's actually expected to hold a number encourages the model to treat it loosely. The fix is defense in depth: use the most specific JSON Schema types and constraints you can (`"type": "integer", "minimum": 1` rather than a loosely described string), validate every incoming call against the schema in your execution layer before running anything, and return a clear, structured error back to the model rather than letting an exception propagate — that error message becomes the model's only signal for how to correct itself on the next attempt.

```python
from jsonschema import validate, ValidationError

def safe_execute(tool_name: str, arguments: dict, schema: dict, fn) -> dict:
    """Validate arguments against the declared schema before ever calling
    the real function. This is the single most effective guard against
    malformed-argument failures reaching production code."""
    try:
        validate(instance=arguments, schema=schema)
    except ValidationError as e:
        return {
            "success": False,
            "error": f"Invalid arguments for {tool_name}: {e.message}",
            "path": list(e.path),
        }

    try:
        result = fn(**arguments)
        return {"success": True, "data": result}
    except TypeError as e:
        # Arguments matched the schema shape but not the function signature —
        # usually means the schema and the function have drifted apart.
        return {"success": False, "error": f"Argument mismatch: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Hallucinated parameters.** The model invents a parameter that was never declared in the schema (commonly because it appeared in a similar tool elsewhere in the conversation, or because the model is pattern-matching against training data where a similarly-named function took that argument). Setting `"additionalProperties": false` on the parameters schema doesn't stop the model from *emitting* the extra field, but it does let your validation layer reject the call cleanly instead of silently passing an unexpected kwarg into your function (which, depending on your function signature, might raise a `TypeError` anyway, or worse, might silently succeed if the function accepts `**kwargs`). The deeper fix is usually that the hallucinated parameter reveals a gap the model is trying to fill — if the model keeps inventing a `region` field for `get_weather`, that's a signal the tool genuinely needs a `region` field, not that the model is malfunctioning.

**Hallucinated tool calls entirely.** The model calls a tool name that was never declared at all. This is rarer with major providers (constrained decoding usually prevents it structurally) but shows up more with self-hosted or fine-tuned models that emit function calls as free text you parse yourself. Always check the tool name against your registry before doing anything else, and treat an unknown name as a hard error returned to the model, not a silent no-op.

## Testing Tool Schemas Like You'd Test an API

Because schema quality is empirical — you genuinely cannot tell by inspection alone whether a description is "clear enough" for a model to route correctly — treat tool schemas as a testable artifact with the same rigor you'd apply to API contract tests, not as static configuration you write once and forget.

A minimal but effective test harness runs a battery of representative user utterances against the live model with the current tool catalog, and asserts on which tool (if any) got called and whether the arguments were well-formed:

```python
import json

TOOL_SELECTION_TEST_CASES = [
    {
        "query": "What's the weather like in Denver right now?",
        "expected_tool": "get_current_weather",
        "expected_args_subset": {"city": "Denver"},
    },
    {
        "query": "Will it rain in Denver next week?",
        "expected_tool": "get_weather_forecast",  # distinct from current-conditions tool
        "expected_args_subset": {"city": "Denver"},
    },
    {
        "query": "Tell me a joke about weather.",
        "expected_tool": None,  # should NOT trigger any tool call
    },
]

def run_tool_selection_suite(client, tools, test_cases, model="gpt-4o"):
    failures = []
    for case in test_cases:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": case["query"]}],
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        called = message.tool_calls[0].function.name if message.tool_calls else None

        if called != case["expected_tool"]:
            failures.append({
                "query": case["query"],
                "expected": case["expected_tool"],
                "got": called,
            })
            continue

        if called and "expected_args_subset" in case:
            args = json.loads(message.tool_calls[0].function.arguments)
            for key, val in case["expected_args_subset"].items():
                if args.get(key) != val:
                    failures.append({
                        "query": case["query"],
                        "arg_mismatch": {key: (val, args.get(key))},
                    })

    return failures
```

Running this suite is how you catch the "two tools quietly overlap" problem *before* it ships, rather than discovering it from a support ticket weeks later. Because model outputs are non-deterministic even at low temperature, a mature version of this harness runs each case several times and flags anything below a chosen pass-rate threshold, rather than treating a single run as pass/fail — a schema might route correctly 9 times out of 10 and still be worth hardening. It's also worth running this suite against every model you might deploy on, not just the one you developed against: schema wording that disambiguates cleanly for one model's training can still be genuinely ambiguous for another, and "works on GPT-4o" is not the same claim as "works on the model we'll actually be running in production six months from now."

Finally, treat regressions in this suite as blocking. Tool schemas tend to accumulate small edits over time — someone tweaks a description to fix one reported misfire, without checking whether that wording change now confuses a *different*, previously-reliable case. A schema test suite that runs in CI on every change to the tool catalog is the only reliable way to catch that kind of whack-a-mole regression before it reaches users.

## Documenting Side Effects, Idempotency, and Cost

A category of information tool descriptions frequently omit, and models frequently need, is what happens when a tool is called *more than once*, or what happens if it's called and the result never makes it back to the model (a timeout, a dropped connection). A read-only tool like `get_weather` is trivially safe to retry — calling it twice with the same arguments just does the same lookup twice, no harm done. A tool like `create_order` is not: retrying it blindly after an ambiguous timeout can create two orders instead of one. Models given no information about this distinction cannot be expected to reason about it correctly, especially in an agent loop that includes automatic retry logic on tool failures.

The fix is to say so directly in the description, in language that both the model and a human reviewer can act on:

```python
{
    "name": "create_order",
    "description": (
        "Create a new order for the given items. This action is NOT idempotent — "
        "calling it twice creates two separate orders, even with identical arguments. "
        "If a previous call to this tool timed out or returned an unclear result, "
        "call check_order_status with the customer's email first to verify whether "
        "an order was already created before calling this again."
    ),
    "parameters": { "...": "..." }
}
```

For tools that are naturally idempotent — because the underlying operation is a database upsert keyed on a client-supplied idempotency token, for instance — say that explicitly too ("safe to retry with the same `idempotency_key`"), because a model that has been trained to be cautious about side effects will otherwise sometimes hesitate or ask for confirmation on an operation that was actually perfectly safe to retry, degrading the experience for no safety benefit. The same logic extends to cost and latency: a tool description that mentions "this call may take up to 30 seconds and queries a paid third-party API" gives the model (and, indirectly, the system prompting around it) a basis for deciding whether to use the tool speculatively or only when clearly necessary, versus a tool description that gives no signal at all and gets called as casually as a free, instant one.

## Required vs. Optional Parameters and Sensible Defaults

Marking a parameter `required` is a stronger commitment than it looks. Every required field is a field the model must produce a value for on every call, even when it's genuinely uncertain — and an uncertain model asked to supply a required field it lacks good information for will do one of two unhelpful things: invent a plausible-sounding value (a form of hallucination directly caused by the schema, not by the model being unreliable in the abstract), or refuse to call the tool at all because it can't satisfy the schema. The corrective is to keep the required set to the genuine minimum needed for the call to be meaningful, and to give every optional parameter both a sensible default and a description that explains what happens if it's omitted, rather than leaving the model to guess whether omission is even acceptable:

```python
{
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search terms."},
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return. Defaults to 10 if omitted.",
            "default": 10,
            "minimum": 1,
            "maximum": 50
        },
        "date_range": {
            "type": "string",
            "description": (
                "Restrict results to this time window, e.g. 'past_week', 'past_year'. "
                "Omit entirely to search with no date restriction."
            )
        }
    },
    "required": ["query"]
}
```

Here `max_results` has a numeric default and an explicit bound, so a model that doesn't care about pagination simply omits it rather than guessing a number, and `date_range`'s description tells the model precisely what omitting it means (no restriction) rather than leaving that ambiguous. This kind of precision measurably reduces both malformed calls and unnecessary follow-up turns where the model asks the user a clarifying question it didn't actually need to ask.

## Anti-Patterns Worth Naming Explicitly

A short list of concrete anti-patterns tends to be more actionable in review than general principles, because they're easy to grep a tool catalog for:

- **The "god tool."** A single tool with a `mode` or `action` string parameter that switches between entirely unrelated behaviors (`{"action": "search"}` vs `{"action": "delete"}` on the same tool). This defeats per-action risk classification (Chapter 4's permission gating needs to know which specific operation a call represents) and gives the model a much harder disambiguation problem than separate, clearly-named tools would.
- **The bare passthrough.** A tool that's a thin wrapper exposing an entire underlying API's flexibility through one free-text parameter (`{"raw_query": "..."}` handed directly to a query engine). This looks efficient to build but pushes all of the reliability burden onto the model's ability to generate a complex, correct query string from scratch, with no schema-level guardrails at all.
- **Silent truncation.** A tool that quietly cuts off long inputs or outputs without saying so in the response. The model has no way to know its result is incomplete and will confidently synthesize an answer from partial data unless the truncation is explicitly flagged in the payload (`"truncated": true`).
- **Inconsistent error shapes.** Some tools in the catalog return `{"error": "..."}` on failure, others raise and let the framework serialize an exception string, others return a partial success object with an embedded `null`. A model has to learn a per-tool convention for recognizing failure, which is exactly the kind of inconsistency that produces the "model didn't notice the tool failed and hallucinated anyway" failure mode. Standardize one result envelope shape across every tool in the catalog.

## Versioning a Tool Catalog

Tool schemas are contracts the model has implicitly learned to expect, in the sense that few-shot examples, cached prompts, and any fine-tuning done against your tool set all encode assumptions about the current shape of each schema. Changing a parameter's name or type in place, without warning, can silently degrade performance for reasons that are hard to trace back to the schema change, especially if you're also relying on prompt caching (which keys on the exact tool definitions sent) — a schema edit invalidates the cache for every session using it. The safer pattern for a breaking change is to introduce a new tool name (`search_customers_v2`) alongside the old one, migrate callers and any cached prompts deliberately, and retire the old name only after confirming nothing still depends on it, rather than editing a widely-used schema's shape in place and hoping downstream consumers adapt silently. Non-breaking additions — a new optional parameter, a clarified description — are safe to make in place, but it's worth re-running the schema test suite described above after any change, breaking or not, since "non-breaking" from a JSON Schema diff perspective doesn't guarantee "non-breaking" from a model-behavior perspective.
