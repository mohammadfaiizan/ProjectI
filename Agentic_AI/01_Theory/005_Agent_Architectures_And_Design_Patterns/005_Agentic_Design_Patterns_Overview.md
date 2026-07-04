# Agentic Design Patterns: A Synthesis

## Why a Taxonomy, and Why This One

The preceding chapters each went deep on one architecture — ReAct, Plan-and-Execute, Reflexion, graph-based control flow. In production, though, agents are rarely a pure instance of exactly one of these; they combine several of them, and the combinations follow recognizable shapes. This chapter steps back and organizes the field around the four-pattern taxonomy that Andrew Ng popularized in his 2024 "Agentic Design Patterns" series — **reflection, tool use, planning, and multi-agent collaboration** — not because it's the only valid way to slice the space, but because it has become the shared vocabulary the industry actually uses in design discussions, and because each of the four maps cleanly onto material already covered in this series, letting us focus here on how they *compose* rather than re-deriving each one from scratch.

It's worth being upfront about what this taxonomy is and isn't. It is not four mutually exclusive architectures you pick one of; it's four largely independent *capabilities* an agentic system can have, and most non-trivial production agents have more than one of them simultaneously, layered on top of each other. Treating them as capabilities rather than competing designs is the key to using this taxonomy well.

## The Four Patterns, Briefly Revisited

**Reflection** is the capability for an agent to evaluate and improve its own output before finalizing it, or to learn from a completed attempt to do better on the next one. This is the territory of Chapter 3 (Reflexion and self-correction): a generate-critique-refine loop within a single episode, and/or an episodic memory of past mistakes that informs future attempts. The core lesson from that chapter carries directly into system design here — reflection is a valuable but bounded capability, and its value is gated by whether the critique step has access to a genuine external signal (test execution, retrieval against a source of truth, a human) rather than just the model re-examining its own output with the same blind spots that produced it.

**Tool use** is the capability for an agent to extend what it can perceive and affect beyond generating text — querying a database, calling an API, executing code, searching the web. This underlies essentially every architecture in this series; ReAct is, structurally, "reasoning interleaved with tool use," and even a pure planning system is only useful insofar as its plan steps eventually bottom out in tool calls that touch the real world. Tool use is less a standalone "pattern" you add on top of an agent and more a prerequisite substrate — an agent with no tools can only ever produce text, which limits it to tasks fully solvable from what the model already knows or from what's in its context window.

**Planning** is the capability to decompose a goal into a structured sequence (or graph) of steps before, or concurrently with, executing them — Chapter 2's Plan-and-Execute territory, generalized. Not every agent needs explicit planning: for tasks where the next step is always obvious from the immediate situation, a pure reactive loop (ReAct, Chapter 1) is planning "compressed" into single-step lookahead, decided fresh at each turn rather than committed to upfront. Explicit planning earns its cost when a task has enough structure to benefit from committing to a decomposition, especially when steps can be parallelized or when the plan itself needs to be reviewable before execution — as covered in Chapter 2's discussion of steerability.

**Multi-agent collaboration** is the capability to split a task across multiple distinct LLM-driven roles — each with its own prompt, tools, and (often) its own instance of one or more of the other three patterns — rather than asking one agent to hold the entire task in a single context and a single reasoning process. This is the pattern most distinct from the other three in kind rather than degree: reflection, tool use, and planning are all things a *single* agent can do; multi-agent collaboration is fundamentally about *decomposing responsibility* across separate reasoning processes, which introduces its own concerns (coordination, communication protocol, conflicting outputs) that don't arise within a single agent no matter how sophisticated its internal loop.

## How They Compose: A Worked Example

Consider a research-and-report agent tasked with "produce a competitive analysis of three companies in the same market." Walking through how the four patterns show up in a well-built version of this system illustrates composition better than describing it abstractly.

The system's outer structure is a **plan**: decompose into "research company A," "research company B," "research company C," and "synthesize into a report," with the three research subtasks flagged as independent and therefore parallelizable — exactly the Plan-and-Execute structure from Chapter 2. Each of those three research subtasks is not carried out by the planning agent itself but delegated to a separate **sub-agent**, giving us **multi-agent collaboration** nested inside the plan: three "Researcher" agent instances running concurrently, each scoped to one company, followed by a distinct "Writer" agent instance that only sees their outputs, not their internal reasoning. Within each Researcher sub-agent, the actual work of finding information is a bounded **ReAct loop making tool calls** — search the web, follow a promising link, extract facts from a page, decide if enough has been gathered — which is the **tool use** pattern doing the concrete work inside one node of the outer plan. Finally, before the Writer's report is returned to the user, a **reflection** step checks the draft against the three research summaries for unsupported claims or contradictions, and — if the system has a way to detect this (e.g., a citation-checking pass against the retrieved sources) — the finding gets fed back for a revision pass rather than being asked to self-approve.

```python
def competitive_analysis(companies: list[str], llm, tools) -> str:
    # PLANNING: decompose into independent research tasks + a synthesis step
    plan = {
        "research": companies,          # independent -> parallelizable
        "synthesize": "after all research completes",
    }

    # MULTI-AGENT: one Researcher sub-agent per company, run concurrently
    research_results = run_in_parallel([
        lambda c=c: researcher_agent(c, llm, tools) for c in plan["research"]
    ])

    # (each researcher_agent internally runs a bounded ReAct loop = TOOL USE)

    draft = writer_agent(research_results, llm)          # a distinct agent role

    # REFLECTION: verify the draft against sources before returning it
    critique = citation_check(draft, research_results)   # external signal, not self-judgment
    if not critique["all_claims_supported"]:
        draft = writer_agent(research_results, llm, feedback=critique)

    return draft
```

Notice that nothing about this example required inventing a new architecture — it is planning, multi-agent delegation, tool-using ReAct sub-loops, and an externally-anchored reflection pass, wired together according to where each pattern's strengths line up with a part of the task. This is the normal shape of a mature production agent: not a single named pattern implemented purely, but a composition where each pattern is applied at the grain of the sub-problem it's actually good at.

## A Framework for Choosing Which Patterns to Apply

Rather than memorizing "use pattern X for task type Y," it's more durable to reason from a small number of task characteristics that each pattern responds to directly. For a given task (or sub-task, since real systems apply this recursively), ask the following.

**Is the sequence of steps knowable in advance, or does it depend on intermediate results?** If knowable in advance and the steps have real dependency structure worth exploiting (some can run in parallel, or a human should review the plan before execution begins), lean toward explicit **planning**. If genuinely not knowable in advance — each step's necessity or shape depends on what the previous step revealed — lean toward a reactive loop (**ReAct**) instead, and resist the temptation to force a plan onto a task that doesn't have one, which Chapter 2 covered as the most common misapplication of Plan-and-Execute.

**Does the task require touching the world beyond generating text?** If yes — and almost every task that matters commercially does, even if only "look something up" — the agent needs **tool use**, and the design question becomes less "should we use tools" and more "how tightly should tool calls be interleaved with reasoning" (ReAct-style, one at a time) versus "how much can be batched into a plan's steps" (Plan-and-Execute style).

**Is output quality high-stakes enough to justify an extra verification pass, and — critically — is there an external signal available to anchor that pass?** If the task has a checkable ground truth (tests pass, a schema validates, a retrieved source confirms a claim), add a **reflection** step anchored to that signal; it will reliably catch a real class of errors. If the only available "check" is asking the same model to grade its own work with no new information, be honest that this buys much less than it appears to (Chapter 3's central finding), and consider whether the budget is better spent on a stronger single generation pass, better tools, or routing to a human, rather than an unanchored self-critique loop that mostly adds latency and cost.

**Does the task decompose into genuinely distinct roles or expertise areas, or is it one continuous line of reasoning that would only get confused by being split up?** Multi-agent decomposition helps when sub-tasks benefit from different tool access, different prompting/expertise framing, or different context windows that shouldn't be polluted with each other's noise (a legal-review agent and a code-generation agent genuinely benefit from not sharing a context, and from having narrowly scoped tools each). It actively hurts when the task's steps are tightly coupled and each depends on subtle context from the others — splitting such a task across agents just adds a lossy communication bottleneck (each agent can only pass along what it thinks to summarize for the next one) where a single agent with the full context would have reasoned about it directly. Multi-agent systems and their specific orchestration patterns (hierarchical, peer-to-peer, debate) get fuller treatment in the dedicated multi-agent chapters of this series; the point to take from this framework is narrower — decompose across agents when the sub-problems are genuinely separable and benefit from isolation, not by default because "multi-agent" sounds more sophisticated than a single well-built agent.

**What does control-flow structure need to look like operationally?** If the system needs checkpointing, human-in-the-loop review at specific points, or the ability to audit exactly which path an execution took, express the composition of whichever patterns apply as an explicit **graph** (Chapter 4) rather than nesting them as implicit loops inside one another. This is orthogonal to the other four questions — it's a question about how you *implement* whatever combination of reflection, tool use, planning, and multi-agent structure you've decided the task needs, not a fifth pattern to weigh against the others.

## A Compact Decision Table

| Task characteristic | Pattern to reach for | Chapter |
|---|---|---|
| Next step depends on results not yet known | Reactive loop (ReAct) | 1 |
| Steps knowable upfront, some parallelizable, or need pre-execution review | Explicit planning | 2 |
| Any interaction with the world beyond text | Tool use | 1, and throughout |
| Checkable ground truth exists (tests, schema, retrieval) | Reflection anchored to that signal | 3 |
| Only signal available is the same model self-judging | Skip or heavily discount reflection; invest elsewhere | 3 |
| Sub-tasks need distinct expertise, tools, or isolated context | Multi-agent decomposition | 6+ (dedicated chapters) |
| Need pause/resume, human review, or execution-path auditing | Explicit graph structure | 4 |

## The Meta-Lesson

The single biggest mistake this taxonomy helps avoid is architecture-first design: picking a fashionable pattern (usually "let's make it multi-agent," or "let's add a reflection loop") because it sounds sophisticated, rather than starting from the task's actual characteristics — its dependency structure, its need for tools, whether quality can be externally checked, whether it decomposes into distinct expertise — and letting those characteristics dictate the minimum composition of patterns that satisfies them. Every pattern covered across this series adds cost: more LLM calls, more latency, more surface area for a subtle bug in the control flow, more moving parts to trace when something goes wrong in production. The engineering discipline this series has been building toward is not "know all the patterns" but "know precisely which cost each pattern buys you which benefit, so you add only the ones a given task actually needs, and no more."

## How Popular Frameworks Map Onto These Patterns

It's easy to mistake framework choice for architecture choice, so it's worth being explicit that the four patterns in this chapter are conceptual and every major framework is really just a different set of ergonomics for expressing combinations of them, not a fundamentally different underlying model of what an agent is.

**LangGraph** is, structurally, an implementation of the graph-based control-flow model from Chapter 4 — nodes, edges, conditional routing, cycles, and checkpointing as first-class citizens — and it is commonly used as the substrate *underneath* the other three patterns rather than as a competing pattern: a planning agent, a multi-agent handoff, and a reflection loop can all be expressed as specific graph shapes on top of it. **CrewAI** and similar "role-based" frameworks foreground the multi-agent collaboration pattern specifically, providing built-in abstractions for defining named agent roles, their tools, and how tasks are delegated and handed off between them, typically with a lighter-weight, more implicit notion of control flow underneath than LangGraph's explicit graph. **AutoGen** similarly centers multi-agent collaboration but leans further into agents communicating with each other through open-ended conversation turns rather than a fixed workflow graph, which trades some structural predictability for more flexible, emergent coordination — closer in spirit to the peer-to-peer multi-agent patterns covered in the dedicated multi-agent chapters of this series. The **OpenAI Agents SDK** (successor to the Assistants API's agent-like features) bundles native tool use and a "handoff" primitive for delegating between specialized agents, again implementing tool use and multi-agent collaboration as the two most foregrounded patterns, with planning and reflection left to be composed by the developer on top.

The practical implication: choosing a framework is not the same decision as choosing an architecture. It's entirely possible to build a Plan-and-Execute system with heavy reflection inside CrewAI's role-based abstractions, or a purely reactive ReAct-style agent inside LangGraph by simply defining a two-node cycle. What each framework actually optimizes for is the ergonomics of expressing *one* of the four patterns particularly well — which is worth knowing when picking a framework for a specific system, but shouldn't be confused with the underlying architectural decision this series has been building a vocabulary for.

## A Second Worked Example: When Composition Goes Wrong

It's just as instructive to see a case where applying more patterns made a system worse, because the failure mode is common and specific. Consider a support-ticket triage agent that was built with all four patterns stacked on by default: a planning phase that decomposed "resolve this ticket" into sub-steps, three specialized sub-agents (a "classifier" agent, a "resolver" agent, and a "responder" agent) handed off sequentially, and a reflection pass that critiqued the responder's draft before sending it — all wired together regardless of ticket complexity.

For the 80% of tickets that were simple password-reset or order-status requests, this pipeline added four to six LLM calls and ten to twenty seconds of latency to answer a question a single tool-calling LLM call could have resolved directly. Worse, the reflection pass — running with no external signal, just the responder's own model re-reading its draft — occasionally "corrected" a perfectly fine response into a more hedged, less useful one, exhibiting exactly the sycophantic-drift failure mode described in Chapter 3, because there was nothing external for it to check against and it was invoked unconditionally on every ticket rather than only on the ones where added scrutiny was likely to be worth its cost.

The fix was not to remove any pattern wholesale but to make each pattern's application conditional on the task characteristics this chapter's framework asks about: a cheap upfront classifier (not itself a heavyweight agent) routed simple, well-understood ticket types straight to a single tool-calling response, reserving the full plan/multi-agent/reflection pipeline for tickets that were genuinely ambiguous or novel enough to need it, and the reflection step was re-anchored to check the response against the account/order data actually retrieved (an external signal) rather than asking the responder to grade its own tone and completeness. This is the taxonomy's real payoff in practice — not a checklist of patterns to include, but a discipline for deciding, per task and even per request, which of them are actually earning their cost right now.

## Anti-Patterns: Recognizable Mistakes in Applying These Patterns

Beyond the over-composition case above, a handful of specific mistakes recur often enough across production agent systems to be worth naming individually, because each one is a plausible-sounding idea that turns out to backfire in a specific, predictable way.

**Over-engineering the control flow for a task that doesn't need it.** Wrapping a task that is genuinely a single tool call in a full graph with multiple nodes, a planning phase, and a reflection pass adds latency and failure surface for no corresponding benefit — the four-question framework earlier in this chapter exists specifically to catch this before it's built, not after.

**Under-engineering a task that has real structure.** The opposite mistake: treating every task as a single flat ReAct loop, including ones with obvious independent sub-parts that could be parallelized via planning, or obvious verification opportunities that could be checked via reflection. This shows up as agents that are technically "agentic" but needlessly slow and expensive relative to what the task's actual structure would allow.

**Ignoring error handling at pattern boundaries.** Each pattern introduces a new boundary where something can silently go wrong — a sub-agent in a multi-agent handoff returning malformed output that the next agent doesn't validate, a planner producing a plan with a dependency cycle that the executor doesn't check for, a reflection critique that's parsed with a fragile regex that silently defaults to "acceptable" on a parse failure. Every pattern-composition boundary needs its own explicit validation; assuming the previous stage "did its job correctly" is how quiet correctness regressions creep into a multi-pattern system.

**Uncontrolled iteration counts.** Reflection loops, replanning loops, and multi-agent back-and-forth all have the same latent risk: a bound that was fine on a simple case can allow far more iterations (and far more cost) than intended on a harder or malformed case. Every loop introduced by any of these four patterns needs an explicit, tested upper bound, not an implicit assumption that the model will naturally converge quickly.

**No monitoring of pattern-specific metrics.** A system that only logs overall success/failure per request cannot tell whether a regression came from the planner, a specific sub-agent, or the reflection step — the same point made about ReAct evaluation in Chapter 1 applies at the level of composed systems: instrument each pattern's contribution separately (plan validity rate, per-sub-agent success rate, reflection accept/reject rate) so a regression can be localized quickly rather than requiring a full trace-by-trace investigation every time something degrades.

## A Short Set of Interview-Style Questions and How to Answer Them

**"When would you choose ReAct over Plan-and-Execute?"** When the correct sequence of actions cannot be determined without seeing intermediate results — exploratory, diagnostic, or open-ended tasks where committing to an upfront plan would just be guessing. Plan-and-Execute wins when the task decomposes predictably regardless of what intermediate steps find, especially when some of those steps are independent and can run in parallel, or when a human needs to review the strategy before any action is taken.

**"Why doesn't asking a model to double-check its own answer reliably catch errors?"** Because the same reasoning process (and the same underlying knowledge gaps) that produced the error is being asked to evaluate it — there is no new information introduced by re-reading, so shared blind spots simply pass the check. Reliable self-correction requires an external signal: executable verification, retrieval against a source of truth, a different model, or a human.

**"What does an explicit agent graph buy you that an implicit reasoning loop doesn't?"** Three concrete things: a legible, inspectable structure you can checkpoint and resume from (critical for long-running or human-in-the-loop workflows), a natural place to insert monitoring and validation at defined boundaries rather than inside opaque generated text, and the ability to express cycles and fan-out/fan-in parallelism as structural properties rather than as ad hoc code wrapped around a single model call.

**"How would you decide whether a task needs a multi-agent architecture at all?"** Check whether the task's sub-parts genuinely benefit from isolation — different tools, different expertise framing, or context that shouldn't bleed between them. If the sub-parts are tightly coupled and each needs deep context from the others to do its job well, splitting them into separate agents usually just adds a lossy communication bottleneck; a single, well-scoped agent with full context often outperforms an artificially decomposed multi-agent version of the same task.

**"What's the single biggest practical risk in composing several of these patterns together?"** Cost and latency compounding silently — each added pattern (a planning pass, a reflection pass, a multi-agent handoff) adds LLM calls that are individually reasonable but collectively expensive, especially if applied unconditionally rather than gated on the actual complexity of the request, which is exactly the failure mode in the support-ticket triage example above.

## Mapping the Whole Series Onto One Table

It's useful, as a final synthesis, to see every architecture covered across this series laid out against the same set of axes at once — not to re-explain any one of them, but to make the relationships between them visible in a single glance.

| Architecture | Primary pattern(s) it embodies | Control-flow decided | Best-fit task shape | Chapter |
|---|---|---|---|---|
| Single LLM call | None (not agentic) | Fully outside the model | Well-specified, single-turn tasks | 0 |
| ReAct | Tool use, implicit planning | Continuously, one step at a time | Exploratory / unpredictable-structure tasks | 1 |
| Plan-and-Execute | Planning, tool use | Upfront, revised on explicit triggers | Decomposable tasks with real dependency structure | 2 |
| Reflexion / reflection loops | Reflection | Same as the base architecture it wraps | Recurring tasks with a genuine external outcome signal | 3 |
| Graph-based agents | Structural scaffolding for any of the above | Explicit, as a data structure | Long-running, auditable, or human-in-the-loop workflows | 4 |
| Multi-agent systems | Multi-agent collaboration | Distributed across roles | Tasks that decompose into genuinely separable expertise | 5 (and dedicated chapters) |

Reading this table by row shows what each chapter added on top of the single-LLM-call baseline; reading it by the "control-flow decided" column shows the real throughline of the entire series — every step from a single call to a full multi-agent graph is a different answer to the same question of *when and how much control to hand to the model versus fix in advance*, which is exactly the lens the decision framework earlier in this chapter is built around.

## Closing Framing for the Series

Across these six chapters, the throughline has been the same single idea approached from different angles: an agent is a system where the model's own output determines what happens next, and every named pattern — ReAct, Plan-and-Execute, Reflexion, graph-based control flow, and the four-pattern taxonomy in this chapter — is a specific, disciplined way of shaping how much control to hand the model at a given point, and what scaffolding (structure, verification, delegation, persistence) to wrap around that control to make the resulting system fast enough, cheap enough, and reliable enough for the task at hand. Interview-level fluency in this material is less about being able to recite what ReAct or Reflexion stands for, and much more about being able to look at an unfamiliar task, correctly identify which of these tensions it actually presents, and justify a minimal architecture that addresses exactly those tensions and no others.
