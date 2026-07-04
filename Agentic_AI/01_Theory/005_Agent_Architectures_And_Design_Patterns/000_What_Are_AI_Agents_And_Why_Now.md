# What Are AI Agents, and Why Now

## Starting From the Simplest Case: A Single LLM Call

Before defining "agent," it helps to be precise about what it is *not*. The simplest way to use a large language model is a single, stateless call: you send a prompt, the model returns text, and the interaction is over. Nothing the model says changes what happens next except in the sense that a human reads the output and decides what to do with it.

```python
def summarize(document: str, llm) -> str:
    prompt = f"Summarize the following document in three sentences:\n\n{document}"
    return llm.generate(prompt)
```

This is enormously useful, but it is not agentic. The control flow is entirely outside the model. The model has no way to say "actually, I need more information before I can answer," no way to check its own work, and no way to take an action in the world beyond producing text. If the summary is wrong, nothing in this function will ever find out.

Most of what got called "AI-powered features" between 2020 and 2022 — classification endpoints, autocomplete, single-turn Q&A over a fixed context — are instances of this pattern. They are LLM *calls*, not LLM *agents*. Understanding why the industry moved from calls to agents, and what specifically changed to make that move viable, is the right place to start before touching any specific architecture like ReAct or Plan-and-Execute.

## The Defining Property: A Loop With the Model Inside It

An agent exists the moment you take the output of a model call, feed it back into the system as an input to *another* model call, and let the model itself decide when that loop should stop. The model is no longer just answering a question; it is participating in a decision about what to do next, based on the results of what it already did. That is the entire conceptual leap, and everything else — tools, memory, planning, multi-agent orchestration — is elaboration on top of it.

The canonical shape of this loop is usually described with four verbs: **perceive, reason, act, observe**.

- **Perceive**: the agent receives something to work with — a user's goal, a new message, a webhook payload, or the result of its own previous action. This is the input to the current cycle of the loop.
- **Reason**: the model examines the current state (the goal, the history so far, anything retrieved from memory) and decides what should happen next. This is the step that a plain LLM call also does, but here it is decision-making about the *process*, not just about the final answer.
- **Act**: the agent does something that has an effect outside of just generating text — it calls a function, queries a database, hits an API, writes a file, or sends a message. This is what separates an agent from a chatbot that only reasons in the open: the reasoning is allowed to cash out into a side effect.
- **Observe**: the result of that action is captured and fed back in as new perception for the next iteration of the loop.

```python
def agent_loop(goal: str, llm, tools: dict, max_steps: int = 8) -> str:
    history = [{"role": "user", "content": goal}]

    for step in range(max_steps):
        # REASON: the model looks at everything so far and decides
        # whether to call a tool or produce a final answer.
        response = llm.generate(history, tools=list(tools.values()))

        if response.is_final_answer:
            return response.text

        # ACT: execute whatever the model decided to call.
        tool_call = response.tool_call
        result = tools[tool_call.name](**tool_call.arguments)

        # OBSERVE: fold the result back into the conversation state,
        # which becomes the PERCEIVE input for the next iteration.
        history.append({"role": "assistant", "content": response.raw})
        history.append({"role": "tool", "name": tool_call.name, "content": str(result)})

    return "Gave up after max_steps without a final answer."
```

Notice what is absent from this function: there is no branch that says "if the task is type X, do step 1, then step 2, then step 3." The sequence of actions is not written by a programmer at design time; it is decided by the model at run time, one step at a time, based on what actually happened in previous steps. That is the operational definition of agency in this context — not consciousness, not free will, just *control flow that the model determines dynamically rather than control flow the programmer determines statically*.

This is also why the loop is sometimes called the OODA loop (Observe-Orient-Decide-Act) in older agent literature, or sense-think-act in robotics. The names differ, but the shape is identical: a cycle where perception feeds reasoning, reasoning produces action, and action produces new perception.

## Why This Was Not Practical Until Recently

The idea of a loop like this is not new — it goes back to 1990s intelligent-agent research and even earlier to cybernetics and control theory. What changed between roughly 2022 and 2024 is that the "reason" step became reliable and cheap enough, at scale, for the loop to actually work in production rather than as a research demo. Three specific model capabilities had to mature.

### 1. Reliable, structured tool use (function calling)

For the agent loop above to work, the model has to be able to express "call this specific function, with these specific arguments, in this specific format" reliably enough that a program can parse it without the parse itself becoming the dominant source of failure. Early attempts at this asked the model to emit free-form text like `Action: search("weather in London")` and used regular expressions to extract it — which is exactly what you see in early ReAct-style implementations. This is fragile: the model might phrase the action slightly differently, forget the exact tool name, or wrap it in explanatory prose that breaks the regex.

What changed is that model providers started training models specifically to emit structured, schema-validated function calls (OpenAI's function calling API in mid-2023, followed by equivalent capabilities from Anthropic, Google, and others). Instead of parsing prose, the calling code receives a JSON object that has already been validated against a declared schema — the tool name and its arguments are a first-class part of the model's output format, not something inferred from natural language. This turned tool selection from "usually works, occasionally needs a fallback regex and a retry" into an engineering-grade interface. Agents are only as reliable as their weakest link, and before native function calling, that weakest link was routinely the *parsing*, not the reasoning.

### 2. Context windows long enough to hold a working history

An agent loop accumulates state — every thought, every tool call, every observation gets appended to the context that the model sees on the next iteration. A model with a 4K or 8K token context window (the norm through 2022) runs out of room after a handful of tool calls, especially if any tool returns a large payload like a document, a database result set, or a stack trace. That forces awkward summarization or truncation strategies that lose information the agent may need later.

The jump to context windows in the 100K–1M token range (GPT-4-128k, Claude's 200K and later 1M-token contexts, Gemini 1.5's million-token window) meant an agent could run for dozens of steps, ingest whole documents or codebases as observations, and still have room to reason over the full history. This did not just make longer tasks *possible* — it changed the character of what tasks were worth attempting with an agent at all, because the cost of "just include everything" fell dramatically relative to the cost of building careful retrieval and compression systems for a small window.

### 3. Reasoning models that can plan and self-correct mid-task

The reasoning step in the loop needs to do more than pattern-match a response; it needs to weigh whether a tool's output actually answers the sub-question it was meant to answer, decide whether to try something else, and know when to stop. Base instruction-tuned models from 2022–2023 could imitate reasoning traces reasonably well (chain-of-thought) but were prone to committing to a plan early and rationalizing subsequent evidence to fit it, rather than genuinely updating on new information.

The generation of models optimized explicitly for extended reasoning — trained with reinforcement learning on verifiable multi-step problems, and given room to "think" with extended internal deliberation before answering (OpenAI's o1/o3 line, Claude's extended thinking modes, DeepSeek-R1) — made the reasoning step of the loop qualitatively more robust. These models are markedly better at catching an inconsistency between what they expected a tool to return and what it actually returned, and at deciding to backtrack instead of forging ahead. This does not eliminate the need for external verification (see the chapter on Reflexion and self-correction for why), but it raised the baseline quality of the "R" in the perceive-reason-act-observe loop enough that agentic workflows became worth the operational cost.

None of these three alone would have been sufficient. A model with a huge context window but unreliable function calling still fails on parsing. A model with perfect function calling but an 8K window can't hold a multi-step trace. A model with both but weak multi-step reasoning will call tools in the wrong order and never notice. It is the conjunction of all three, arriving within roughly an 18-month window, that made "agent" go from a research term to a production architecture pattern.

## Agents vs. Traditional Deterministic Software

Traditional software is built around the idea that the programmer enumerates the paths the program can take. An `if/elif/else` chain, a state machine with explicitly drawn transitions, a workflow engine with a fixed DAG of steps — in every one of these, a human decided, ahead of time, what the possible sequences of operations are. The program's job at runtime is to pick among a finite, pre-specified set of paths based on input.

An agent inverts this. The developer does not enumerate the sequence of steps; the developer provides a goal, a set of tools, and a model capable of reasoning about which tool to use and when. The actual sequence — which tool gets called, in what order, how many times, with what arguments — is synthesized at runtime by the model, and in general it is not something you could have fully enumerated in advance because it depends on values only known once execution starts (what a search actually returns, whether an API call fails, what a user says in a follow-up).

| Dimension | Traditional Software | AI Agent |
|---|---|---|
| Control flow | Fixed at design time by a programmer | Synthesized at run time by the model |
| Handling novel input | Requires new code for each new case | Generalizes from a natural-language goal and tool descriptions |
| Failure mode | Crashes or falls into a defined error path | May "succeed" with a subtly wrong answer, or loop |
| Debuggability | Step through code, deterministic replay | Non-deterministic; requires tracing prompts/completions |
| Cost per operation | Near zero (CPU cycles) | Proportional to tokens consumed across every step |
| Latency | Milliseconds | Seconds to minutes, depending on step count |
| Behavior under identical input | Identical output every time | Can vary run to run |

This is not a claim that agents are strictly "better" — it is a different point on the flexibility/reliability trade-off curve. A traditional program that resolves a well-specified business rule (calculate tax, validate a form, route a support ticket by category using a fixed taxonomy) will be faster, cheaper, and more predictable than an agent doing the same thing, and should usually remain traditional code. Agents earn their cost when the *shape* of the task cannot be fully specified in advance — when the right sequence of steps genuinely depends on intermediate results that are only knowable at run time, or when the input space is too unstructured (free-form natural language, unpredictable documents, ambiguous user intents) to enumerate a finite set of handling branches.

A practical litmus test: if you can draw the complete flowchart of every path the task could take before you start, you probably don't need an agent — a state machine or a plain script will be more reliable and cheaper. If drawing that flowchart would require branches for combinations of conditions you cannot anticipate, that is the signal an agent's dynamic control flow is earning its keep.

## Agents vs. Chatbots and "AI Assistants"

The word "agent" gets applied loosely in marketing, so it is worth being precise about how it differs from a chatbot or a single-turn assistant, because the difference is not just vibes — it maps onto concrete architectural features.

A chatbot, in the classical sense, maps a user turn to a response, usually using rules, retrieval, or a narrow classifier. It has no memory beyond the current session (often not even that), it does not decide to take actions on its own, and if it fails, it falls back to a scripted response ("I'm sorry, I didn't understand that") rather than trying an alternative strategy.

An LLM-based assistant — the kind of system that shipped as "AI chat" in many products through 2023 — is a step up: it can hold a natural conversation, use context from earlier in the session, and sometimes call a single tool per turn (e.g., a weather lookup or a calculator). But the loop still terminates after one exchange; the assistant does not decide, on its own initiative, to chain five tool calls together to accomplish something the user only stated as a goal, and it does not evaluate its own output against that goal before returning it.

An agent adds three things on top of that: autonomy over multi-step execution (it decides how many actions to take and in what order, without a human approving each one), the ability to evaluate its own progress against the original goal and adjust course (replan, retry with different parameters, ask for clarification only when genuinely stuck), and typically some form of persistent memory that outlives a single request (so it can recall a prior failure, a user preference, or an intermediate result days later).

| Property | Chatbot | LLM Assistant | AI Agent |
|---|---|---|---|
| Response scope | One scripted reply per turn | One reasoned reply per turn, maybe one tool call | Autonomous multi-step execution toward a goal |
| Plans ahead | No | Rarely | Yes, explicitly or implicitly |
| Tool use | None | Limited, single-shot | Extensive, chained, conditional |
| Self-evaluation | None | None | Often — checks output against the goal |
| Memory | None or session-only | Session context | Short-term + persistent long-term memory |
| What the user provides | An exact query | A question or request | A goal; the agent determines the steps |

A useful way to phrase the difference: with a chatbot or assistant, the user is doing the planning (they decide to ask for the weather, then separately decide to ask for a packing list). With an agent, the user states an outcome ("plan my trip"), and the agent performs the planning itself, including deciding what information it needs and how to get it.

## Why This Matters for How You'll Read the Rest of This Series

Once you accept that the defining feature of an agent is "a loop where the model's own output determines the next action," a huge amount of what looks like a fragmented landscape of agent frameworks and named patterns collapses into variations on a single theme: different ways of structuring that loop.

ReAct (the next chapter) is the loop written as literally as possible — reason, act, observe, repeat, with no separation between deciding *what* to do next and doing it. Plan-and-Execute separates the reasoning into an upfront planning pass and a more mechanical execution pass, trading some adaptiveness for more predictability. Reflexion adds a second loop on top of the first, where the agent reasons not just about the task but about its own prior failures. Graph-based and state-machine architectures make the loop's structure explicit and inspectable rather than leaving it implicit inside a single prompt. And multi-agent systems are what happens when you have more than one such loop running, coordinating with each other.

None of these are unrelated inventions. They are all engineering responses to the same underlying tension: the agent loop is powerful because the model decides the control flow dynamically, but that same dynamism is what makes agents expensive, slow, and occasionally unpredictable compared to code a human wrote by hand. Every pattern in this series is a different way of trading off flexibility against cost, latency, and reliability — and the right choice depends on where a specific task sits on that spectrum. Keeping that framing in mind will make the rest of this series read less like a list of frameworks to memorize and more like a coherent design space to reason about.

## A Concrete Walkthrough: the Same System, Three Requests

Abstract definitions of "autonomy" are easy to nod along to and hard to actually apply. It helps to watch one agent handle three requests of increasing difficulty and notice exactly where the loop from the first section does more or less work.

**Request 1: "What's 47 times 89?"** A well-built agent will route this to a calculator tool in a single perceive-reason-act-observe cycle: reason that this needs a tool rather than mental arithmetic, call it, observe `4183`, and stop. There is technically a loop here, but it only ever executes once. This is worth noting because it means "is this an agent" is not about how many steps something takes — a single-step tool call, orchestrated by a model that *decided* a tool was needed, is still agentic in the sense this chapter defined; a five-step hardcoded pipeline that never lets the model choose anything is not.

**Request 2: "What's the weather in the three cities I'm visiting next week, and should I pack a coat?"** Now the loop actually iterates. The agent has to reason that it first needs to know which three cities are being visited — information not present in the request — before it can even call a weather tool. A well-built agent will notice this gap and either ask a clarifying question or check a connected calendar/itinerary tool if one is available, rather than guessing three arbitrary cities. This is the perceive step doing real work: recognizing that the current information is *insufficient* to act, which a single LLM call has no mechanism to express — it would simply generate its best guess and move on.

**Request 3: "Our support ticket volume spiked 40% this week — figure out why and draft a response plan."** Here the number of steps is genuinely unknown in advance: the agent might need to query a ticket database, cluster tickets by category, cross-reference against a recent product deploy log, check social media for an emerging complaint, and only then synthesize a plan — and which of these it actually does, and in what order, depends entirely on what each step turns up. This is the case that most clearly could not be handled by traditional deterministic code, because the branching factor of "what to check next given what you just found" is not something a human could exhaustively pre-code.

Laying these three side by side makes the point precisely: the *architecture* is identical in all three cases (the same loop), but the number of iterations the loop actually performs, and how much reasoning happens at each perceive step, scales naturally with the task's real complexity. This is the practical payoff of building on a loop the model controls rather than a fixed script — the same system handles a trivial request cheaply and a complex one thoroughly, without needing a human to have anticipated which category a given request would fall into.

## Cost and Latency Numbers Worth Internalizing

Because agents get their flexibility by spending LLM calls where traditional code would spend nothing, it's worth having rough orders of magnitude in mind when deciding whether an agent is the right tool for a given problem, rather than reasoning about this purely in the abstract.

| Operation | Typical latency | Typical cost per call |
|---|---|---|
| Traditional API call / database lookup | 10-200 ms | Effectively $0 |
| Single LLM call (no tools, short prompt) | 0.5-3 s | Fractions of a cent to a few cents |
| One agent loop iteration (reasoning + one tool call) | 1-5 s | A few cents |
| A full agent task (5-15 iterations) | 10 s - 2 min | $0.05-$1+ |
| A multi-agent workflow (several agents, each iterating) | 1-10 min | $0.50-$10+ |

These numbers move quickly as models and infrastructure improve — prompt caching, smaller specialized models for routine steps, and faster inference have already pushed per-call costs down significantly since 2023 — but the *relative ordering* is durable: each additional loop iteration costs roughly a full LLM call's worth of latency and money, which is the direct, mechanical reason why every architecture in this series cares so much about minimizing unnecessary steps, bounding iteration counts, and choosing the cheapest pattern that satisfies the task's actual requirements rather than defaulting to the most flexible one.

## Common Misconceptions Worth Retiring Early

**"Agents are just chatbots with more steps."** The step count is a symptom, not the definition. A single-step tool call orchestrated by model judgment is agentic; a ten-turn scripted conversation flow is not, no matter how many turns it has. The defining property is *who decides the next step* — the model, at run time, versus a person, at design time.

**"More autonomy is always better."** Autonomy is a dial, not a virtue to maximize. An agent that can send emails, transfer money, or delete records without any checkpoint is not "more agentic" in a good way — it is an agent with an unmanaged blast radius. Production systems deliberately dial autonomy down at the exact points where a mistake is expensive (see the human-in-the-loop discussion in the graph-based agents chapter later in this series) and dial it up where exploration is cheap and reversible.

**"If it uses an LLM, it's an agent."** A single well-crafted prompt that classifies an email into one of five categories is an LLM application, not an agent — there is no loop, and the model's output does not determine what happens *next* beyond that one classification. Reserve the term for systems where the model's own output feeds back in as a determinant of subsequent action.

**"Agents will always be more capable than a good traditional program at the same task."** For any task narrow and stable enough to fully specify in advance, a traditional program will typically be faster, cheaper, and more reliable, full stop — the agent's advantage is specifically in handling the input and task variability that a traditional program's fixed logic cannot.

## A Practical Checklist Before Reaching for an Agent

Before building an agent for a given problem, it's worth working through a short set of questions, because the answers point fairly decisively toward "agent," "single LLM call," or "traditional code":

1. Can I enumerate, right now, every path this task could take? If yes, write a script or a state machine instead.
2. Does the task require more than one distinct piece of information or action to complete, where the second depends on the result of the first? If no, a single LLM call (with at most one tool call) is sufficient — you don't need a loop.
3. Is the cost of an occasional wrong or incomplete answer tolerable, given the latency and dollar cost of retries and multi-step execution? If the task requires near-100% correctness and there is no cheap way to verify the agent's output, budget heavily for the verification/reflection patterns covered later in this series, or reconsider whether an agent is appropriate at all.
4. Does the input space genuinely vary in ways that would require an unreasonable number of hand-written branches to cover? If the input space is actually narrow despite looking open-ended (e.g., "natural language" that in practice only ever expresses five intents), a classifier plus five scripted handlers may outperform a full agent on cost and latency while matching it on quality.

Answering these honestly for a specific task is a better predictor of whether an agent is the right architecture than any amount of reasoning about agents in the abstract — and it is the same discipline that reappears, at a finer grain, in the final chapter of this series when choosing *which* agentic pattern (or combination of patterns) to apply once you've decided an agent is warranted at all.

## A Brief Historical Arc, in Prose

It's worth knowing roughly how the field arrived here, not as trivia but because each earlier era solved a real problem and left a real limitation behind that the next era addressed — and understanding what was actually missing at each stage makes it clearer why "agent" specifically means what it means today rather than being an arbitrary label.

Rule-based systems from the 1960s through the 1980s — chatbots like ELIZA, expert systems like MYCIN — encoded human expertise as explicit IF-THEN rules and inference engines. They could be impressively convincing within a narrow domain (MYCIN's bacterial-infection diagnoses were reportedly competitive with human specialists), but they were brittle by construction: every rule had to be hand-authored, and behavior outside the anticipated rule set was undefined or absurd. The limitation these systems left behind was a total inability to generalize to phrasing or situations their authors hadn't explicitly anticipated.

The 1990s and 2000s saw academic "intelligent agents" research formalize properties like autonomy, reactivity, and proactivity, and produce architectures like BDI (Belief-Desire-Intention) that gave agents explicit internal representations of what they believed, wanted, and intended to do. This era contributed the conceptual vocabulary much of today's agent terminology still borrows from, but it mostly ran on symbolic reasoning and hand-crafted world models, which meant it inherited rule-based systems' brittleness in any domain involving open-ended natural language — a BDI agent could reason cleanly about explicitly modeled beliefs, but there was no good way to get "the customer seems frustrated" into a belief state without another layer of brittle hand-coded interpretation.

Deep learning through the 2010s solved perception at scale — models could classify images, transcribe speech, and eventually (with the transformer architecture from 2017 onward) model natural language with enough fidelity to hold a coherent conversation. This closed the perception gap that had limited earlier eras, but through approximately 2022 these models were still fundamentally single-shot: a classifier classifies, a language model completes a prompt, and neither has any built-in mechanism for deciding to take a sequence of self-directed actions toward a goal.

What arrived in 2023, and is the direct subject of this chapter, is the closing of the last gap: models capable enough at structured tool use, long-context reasoning, and multi-step planning that the perceive-reason-act-observe loop from the opening section became something you could actually build a reliable product on, rather than a research curiosity. Seen this way, "agentic AI" isn't a sudden invention so much as the point at which several decades-old ideas (autonomy loops, explicit goal decomposition, tool augmentation) finally had a reasoning engine — the modern LLM — capable of running them dependably.

## Terminology Quick Reference

A handful of terms recur throughout this series with specific meanings that are worth fixing precisely before moving on, since sloppy use of these terms is a common source of confusion in both interviews and design discussions.

| Term | Precise meaning in this series |
|---|---|
| Agent | A system in which model output determines subsequent action within a loop, rather than a human or fixed script determining it |
| Tool / function call | A structured, schema-validated invocation of external code that the model can request as part of its output |
| Agent loop | The perceive-reason-act-observe cycle that repeats until a goal is met or a stopping condition is hit |
| Context window | The maximum number of tokens a model can attend to in a single call — the hard ceiling on how much loop history can be retained without summarization |
| Grounding | Anchoring a model's claims to a verifiable external source (a tool result, a retrieved document) rather than parametric memory alone |
| Autonomy | The degree to which the agent, rather than a human, decides the next action — a dial, not a binary property |
| Orchestration | The logic that determines how multiple steps, tools, or agents are sequenced, parallelized, or routed |

Keeping these definitions crisp is what makes it possible to precisely answer questions like "is this an agent or just an LLM feature" or "how much autonomy does this system actually have" rather than falling back on vibes — both of which come up constantly in both system design discussions and technical interviews on this material.
