# Agentic and Tool Use Benchmarks

Every benchmark in files 001 and 002 shares a structural property: the model receives a self-contained prompt and produces a single response (or, for SWE-bench, a single patch), and that response is graded in isolation.

Agentic benchmarks break that structure. The model takes an *action*, observes an *environment's response* to that action, and must decide on a *next* action, repeating over many steps, often with partial observability, irreversible side effects, and a success criterion defined by end *state* rather than by matching a reference string. This is a different measurement problem, not just a harder version of the same one.

This file covers four benchmarks that each target a different slice of that problem: WebArena (browsing a real-feeling website), OSWorld (operating a full desktop OS), tau-bench (multi-turn tool use under a user and a policy), and GAIA (general-purpose multi-step tool orchestration toward a factoid answer). It closes with why agentic evaluation is structurally harder to keep reliable and uncontaminated than static QA — a point worth understanding mechanically, not just asserting.

## WebArena

**Citation:** Zhou, Xu, Zhou, Chen, Zhang, Zhou, Wang, Zhu, Neubig, Bisk, Fried, Yao, "WebArena: A Realistic Web Environment for Building Autonomous Agents," 2023 (CMU).

### The capability gap it was designed to expose

Before WebArena, most "can an LLM use a web browser" evaluation was either purely synthetic (simplified toy web pages built specifically to be easy to parse) or relied on the live, uncontrolled internet — real websites that change, get rate-limited, show different content per session, or contain destructive real-world side effects like actually placing a paid order.

WebArena's contribution is a set of **self-hosted, fully functional, reproducible** websites:

- An e-commerce store
- A Reddit-like forum
- A GitLab-style code-hosting/collaboration platform
- A map/navigation tool
- A content-management admin panel

Each is a real, complex web application (not a scripted mock), running in a sandboxed environment that every evaluation run gets an identical, resettable copy of. Against these sites, WebArena poses 812 tasks phrased as natural user intents — "find the cheapest laptop bag under $50 with at least a 4-star rating and add it to my cart," "open an issue on this repository referencing the bug described in this screenshot's context" — requiring the agent to actually navigate: click, type into forms, follow links, scroll, handle multi-page flows and stateful carts/sessions, rather than answer a question about a static page.

### Evaluation mechanics

Because there is no single canonical "correct string" for most of these tasks — many correct action sequences can reach the same valid end state, and several tasks have genuinely open-ended correct answers — WebArena uses **task-specific functional correctness checkers** that inspect the actual end state of the environment or the content of the agent's final response. This might mean checking whether an item with the right attributes is actually present in the cart, whether a specific database record was actually created with the right fields, or doing a fuzzy/programmatic match against an acceptable answer set for informational tasks.

This is a meaningfully more expensive evaluation-infrastructure investment than a benchmark that only needs an exact-match string comparator: every task requires bespoke verification logic written against that task's specific environment state, which is part of why agentic benchmarks are more costly to build and maintain than static QA sets (elaborated below).

### Results and what they revealed

At introduction, GPT-4-based agents — using a fairly standard ReAct-style observe-think-act loop over the page's accessibility tree / DOM — succeeded on roughly 14% of tasks. This is strikingly low relative to how capable GPT-4 looked on knowledge and reasoning benchmarks at the same time, and it is the clearest evidence that "answering questions about text" and "successfully operating a real interactive interface toward a multi-step goal" are different, only loosely correlated skills.

Subsequent work has pushed success rates up considerably through better agent scaffolding (more reliable DOM-to-text serialization, retry/self-correction loops, better action grounding) as much as through base-model improvement — itself a recurring theme in agentic evaluation, where reported numbers conflate "base model capability" and "scaffold engineering quality" far more than single-turn benchmarks do.

### Known weaknesses

The environments, while realistic, are still a fixed, finite set of sandboxed sites — an agent (or its developers, iterating against the public leaderboard) can overfit to the quirks of these specific five sites' DOM structure and interaction patterns in a way that would not transfer to arbitrary real websites. This is precisely the property WebArena was trying to move past relative to toy environments, but has not fully escaped.

The task set is also static once published, and the sandboxed sites' HTML/task specifications are public, so agent trajectories, solution walkthroughs, and even scaffold code tuned specifically to these tasks are increasingly present on the open web — a distinct contamination vector, described in more detail below.

## OSWorld

**Citation:** Xie, Zhang, Chen, Wang, Yu, Zhu, Cheng, Yang, Zhu, Yang, Fried, Zhou, et al., "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments," 2024.

### The capability gap it was designed to expose

WebArena tests browser-based interaction; OSWorld extends the same idea to the **entire desktop operating system**: 369 tasks spanning real Ubuntu and (in an extended setting) Windows environments, requiring the agent to operate actual, unmodified consumer/professional applications — a terminal, LibreOffice Writer/Calc/Impress, GIMP, VS Code, a file manager, Chrome — including cross-application workflows such as extracting a figure from a PDF, editing it in GIMP, and pasting it into a slide deck.

This is done via the same input modalities a human would use: mouse clicks/drags at pixel coordinates or accessibility-tree element targets, keyboard input, and screenshots (or an accessibility-tree text representation) as the agent's observation of current state. This is a strictly larger action and observation space than WebArena's: a browser DOM is at least structured and consistently parseable, while a full desktop's screen is a raster image, or an accessibility tree that varies wildly in quality and completeness across different applications, many of which have poor or nonexistent accessibility metadata.

This requires the agent to visually or structurally *ground* an intended action ("click the export button") into an actual clickable coordinate or element reference — itself an open, actively researched sub-problem (GUI grounding).

### Evaluation mechanics

Like WebArena, OSWorld uses task-specific execution-based checkers, but here the checkers usually inspect **final file-system or application state** — did the correct file get saved with the right content at the right path, did the spreadsheet cell contain the right formula/value, did the system setting actually change — rather than a live webpage's DOM. This requires an even heavier sandboxing and snapshotting infrastructure than WebArena: a full, resettable VM snapshot per task rather than a resettable web-app database.

### Results

Reported human performance on OSWorld tasks is around 72% — humans do not get everything right either, since some tasks are genuinely fiddly or ambiguous even for a competent human operator. Early GUI-agent baselines using strong contemporary multimodal LLMs scored well under 15%, one of the largest human-model gaps reported for any benchmark discussed in this document.

This is strong evidence that computer-use as a general skill — as opposed to narrow, well-instrumented API-based tool use — was, and to a real extent still is, a substantially unsolved problem even for models that are simultaneously near-ceiling on knowledge benchmarks. This gap is the direct empirical motivation behind the "computer use" agent products released by frontier labs since 2024 (Anthropic's computer-use capability in Claude, OpenAI's analogous operator-style agents) — OSWorld-style evaluation is closer to what those products actually need to be good at than any single-turn text benchmark.

### Known weaknesses

Task diversity, while broad relative to WebArena, is still finite and application-specific. Strong performance on OSWorld's specific set of LibreOffice/GIMP/terminal tasks does not guarantee generalization to arbitrary other software, and different applications have wildly different accessibility-metadata quality — an agent's measured performance can depend heavily on which applications happen to expose good structured element information versus requiring pure pixel-level visual grounding, a confound that is more about the software ecosystem than about the agent's underlying reasoning capability.

Nondeterminism is also a bigger practical problem than in WebArena: real desktop applications have loading times, animations, occasional crashes, and version-dependent UI layouts, all of which introduce environment noise into a single rollout that a purely text-based benchmark never has to contend with. This means OSWorld numbers benefit even more from multi-seed averaging than WebArena's do, and are correspondingly more expensive to evaluate rigorously.

## tau-bench (τ-bench)

**Citation:** Yao, Su, et al. (Sierra), "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains," 2024.

### The capability gap it was designed to expose

WebArena and OSWorld both evaluate an agent acting alone against an environment. tau-bench instead targets the specific structure of a customer-service-style agent: a **three-party interaction** among the LLM agent, a simulated human user (itself played by an LLM prompted with a private goal/persona the agent does not see up front), and a set of backend tools/APIs (e.g., look up a reservation, process a refund, modify a booking) that the agent must call correctly.

The two domains released are airline (booking changes, cancellations, refunds under airline policy) and retail (order modification, returns, exchanges under store policy). The realistic complication tau-bench is built to surface: the user's initial request is often underspecified or slightly wrong about their own situation, mirroring how real customers describe their problem. The agent must ask clarifying questions or verify details via tools rather than assuming.

The harder, policy-adherence axis: the agent is given a written policy document — e.g., "refunds over $200 require verifying date of purchase is within 30 days," "cannot change a booking within 24 hours of departure without a fee" — that it must actually follow even when a straightforward reading of the user's request would suggest otherwise. The agent has to balance being helpful to the user against correctly enforcing business rules it was given as context, not as training.

### The pass^k reliability metric

Beyond simple single-attempt success rate, tau-bench reports **pass^k**: the probability that an agent succeeds on a given task across **k independent trials**, all under the same starting task and policy but with fresh (independently sampled) user-simulator behavior and agent stochasticity each trial.

This is deliberately not the same quantity as code-benchmark pass@k (file 002), which asks "does at least one of k attempts succeed" — a best-of-k, most-optimistic framing appropriate when a human or system can pick the best of several generated candidates. tau-bench's pass^k instead asks "how often does the agent succeed reliably across repeated independent attempts at the same task" — a worst-case/consistency framing, because a customer-service agent that resolves a given class of ticket correctly 90% of the time on average but fails unpredictably the other 10% is a materially worse deployed product than the raw single-trial success rate alone would suggest.

```python
# tau-bench-style pass^k: probability of succeeding on ALL k independent trials
# (contrast with code-benchmark pass@k: probability of succeeding on AT LEAST ONE of k)
def pass_hat_k(trial_outcomes: list[list[bool]], k: int) -> float:
    """trial_outcomes[i] = list of pass/fail booleans across independent trials
    of task i (all trials drawn under the same task/policy, fresh user-sim each time)."""
    per_task = []
    for outcomes in trial_outcomes:
        n = len(outcomes)
        if n < k:
            raise ValueError("need at least k independent trials per task")
        from math import comb
        c = sum(outcomes)  # number of successful trials
        if c < k:
            per_task.append(0.0)
        else:
            per_task.append(comb(c, k) / comb(n, k))
    return sum(per_task) / len(per_task)
```

### A representative policy-conflict scenario

A concrete illustration of what tau-bench is actually probing: a simulated airline customer says "I need to change my flight to tomorrow, my current one departs in 18 hours." The airline policy document given to the agent states that same-day and next-day changes within 24 hours of departure incur a $150 fee unless the customer has a specific premium status tier. The user, in their initial message, doesn't mention their status tier at all. A correct agent must recognize the policy is conditional on information it doesn't yet have, query the customer's account via a tool call to check status tier, and only then either apply or waive the fee and execute the change — never simply trusting the user's framing of their own request, and never simply proceeding without checking the policy-relevant fact first. An agent that skips the status check and either always applies or always waives the fee is exhibiting exactly the failure mode tau-bench's policy-adherence design targets.

### Results and weaknesses

Reported results across both domains show meaningful drop-off from pass@1-style single-trial success to pass^k as k grows, even for frontier models — underscoring that reliability under repetition is a distinct and currently weaker capability than one-shot task success, a finding with direct deployment relevance since production agent systems are evaluated over large volumes of repeated, structurally similar tickets, not single showcase examples.

Weaknesses: the user simulator is itself an LLM, meaning tau-bench's difficulty and realism are partly bottlenecked by how well that simulator plays a believably underspecified, sometimes-confused human — a weak or unrealistic user simulator could make the benchmark either artificially easy (an overly cooperative simulated user) or artificially hard/unfair (an inconsistent or contradictory simulated user), and this simulator-quality confound is hard to fully audit. The domains (airline, retail) are also narrow relative to the full space of real tool-using-agent deployments, so generalization of tau-bench results to other agentic domains is an assumption, not a demonstrated fact.

## GAIA

**Citation:** Mialon, Fourrier, Swift, Wolf, LeCun, Scialom (Meta AI / Hugging Face), "GAIA: A Benchmark for General AI Assistants," 2023.

### The capability gap it was designed to expose

GAIA targets the gap between "a model that knows a lot" and "a model that can actually orchestrate tools — web search, code execution, file/document/multimedia parsing — across multiple steps to derive an answer it could not have produced from parametric knowledge alone."

Its 466 questions are constructed so that the *final answer* is a short, unambiguous, exact-match-checkable string (a number, a name, a short phrase), deliberately avoiding the open-ended-grading problem that a "did the agent do the task correctly" checker for WebArena/OSWorld/tau-bench requires. The *path* to that answer typically requires several dependent tool-use steps: e.g., finding a specific fact buried in a linked document, cross-referencing it against a number extracted from an image or a spreadsheet, and computing a derived value from both.

GAIA questions are organized into three difficulty levels, with Level 1 solvable with relatively few tool-use steps and Level 3 requiring long, multi-step tool-orchestration chains with many opportunities for a single wrong intermediate step to derail the final answer.

### Evaluation mechanics

Because answers are designed up front to be short and unambiguous, GAIA can use simple exact-match or light normalization (case-insensitive, minor formatting tolerance) scoring on the final answer string — reusing the cheap-grading advantage of a factoid-QA benchmark while still requiring genuinely agentic behavior to reach that answer. This is GAIA's specific niche relative to WebArena/OSWorld/tau-bench, all of which need bespoke state-based checkers because their tasks don't reduce to a short final string.

### Results

Reported human accuracy on GAIA is high, around 92%, reflecting that these questions, while requiring tool orchestration, are not intended to be beyond ordinary human research-assistant competence given time and tool access. Early GPT-4-plus-plugins-style agents were reported far below that — commonly cited figures put early tool-augmented GPT-4 agents in the range of roughly 15-30% depending on level and scaffold.

As with the other benchmarks in this file, subsequent frontier agent systems (better retrieval/browsing tools, more robust code-execution sandboxes, improved multi-step planning) have pushed scores up substantially since GAIA's introduction. Flagged as approximate/self-reported: some 2024-2025 frontier agent stacks have reported scores approaching or exceeding the original human baseline on GAIA's public leaderboard, though these figures should be treated cautiously given the contamination and leaderboard-gaming concerns discussed next.

### A representative GAIA-style question

To illustrate the "short final answer, long tool-use path" design concretely: "What is the population difference between the two cities mentioned in the abstract of the third paper cited in [a specific linked reference document], according to each city's most recently published census figure?" Answering this requires opening the linked document, identifying the third citation, retrieving that paper's abstract, extracting two city names, searching for each city's census figure, and computing a subtraction — five or six dependent tool-use steps chained together — while the graded output is just a single number. A single wrong step anywhere in that chain (misidentifying the third citation, retrieving a stale census figure) produces a wrong final number indistinguishable, from the grader's point of view, from a purely incorrect approach.

### Weaknesses

The exact-match-on-a-short-final-answer design, while cheap to grade, means GAIA cannot distinguish between an agent that reached the answer via a genuinely correct and generalizable multi-step process versus one that got lucky, guessed from partial information, or exploited some shortcut unrelated to real tool orchestration. This is a generic risk shared with GSM8K/MATH/AIME in file 002, but arguably more acute here since the *point* of GAIA is to measure process, not just outcome, and the scoring mechanism only ever looks at outcome.

The benchmark is also small (466 questions) and, since its introduction, has an active public leaderboard with submitted agent trajectories — meaning solved trajectories and even discussion of specific questions' answers plausibly leak onto the web over time, a contamination dynamic that is structurally different from (and in some ways easier to trigger accidentally than) static-QA contamination, discussed next.

## A sketch of what a WebArena-style functional checker actually looks like

To make the "bespoke, task-specific checker" point concrete rather than abstract, a simplified sketch of what verifying a WebArena shopping-cart task might involve:

```python
def check_add_to_cart_task(env_state: dict, target_product_attrs: dict) -> bool:
    """env_state: a snapshot of the sandboxed e-commerce site's database state
    after the agent's episode ends. target_product_attrs: the task's success
    criteria, e.g. {"category": "laptop bag", "max_price": 50, "min_rating": 4}."""
    cart_items = env_state.get("cart", [])
    for item in cart_items:
        if (item["category"] == target_product_attrs["category"]
                and item["price"] <= target_product_attrs["max_price"]
                and item["rating"] >= target_product_attrs["min_rating"]):
            return True
    return False
```

Notice this checker has to encode the task's success criteria as executable logic against the environment's actual internal state schema — it is not reusable across tasks the way a single `exact_match(prediction, reference)` function is reusable across every question in MMLU. A benchmark with 812 WebArena-scale tasks needs, in the limit, 812 pieces of logic like this, each specific to its task's environment state and success condition, each capable of having its own bugs (e.g., an off-by-one on `<=` vs `<` for the price threshold that silently fails a correct agent, or that passes an incorrect one).

## Why agentic benchmarks are harder to keep reliable and uncontaminated than static QA

This is worth stating as a set of distinct, compounding mechanisms rather than a single vague "agents are complex" claim:

1. **Environments drift; static text does not.** A live website, desktop application, or backend API changes over time — software gets updated, UI layouts change, APIs deprecate fields — meaning a golden trajectory recorded at benchmark-construction time can silently stop working, and a failure at evaluation time can reflect environment drift rather than agent incapability. WebArena and OSWorld mitigate this by self-hosting frozen environment snapshots rather than pointing at the live internet, which is itself a nontrivial engineering commitment.
2. **Grading requires bespoke, state-based checkers, not string comparison.** Every task needs its own verification logic written against that specific environment's internal state. This verification code is itself software that can have bugs, edge cases, or unintended strictness/looseness, and unlike an exact-match string comparator, it is generally not something a third party can easily audit by inspection alone.
3. **Larger action spaces produce more rollout-to-rollout nondeterminism.** A single wrong click, a misread screenshot, or a slightly different phrasing to a simulated user can send an entire multi-step trajectory down an unrecoverable path. Meaningful comparison requires averaging over many independent rollouts per task, multiplying evaluation cost relative to a benchmark where one forward pass produces one gradable answer.
4. **Contamination takes a different, harder-to-detect shape.** Static-QA contamination is usually "did the model see this exact question-and-answer pair during pretraining" — checkable, in principle, via n-gram overlap against training data (see `../05_Evaluation_Methods/` for that methodology). Agentic-benchmark contamination is more often "did the model (or its developers, via RL rollouts or scaffold engineering) see a *solution trajectory* or *discussion of this specific environment's quirks*." This kind of contamination doesn't require the exact question text to appear anywhere — only that the *environment* and *task family* be public and popular enough to attract writeups, which is almost guaranteed for any benchmark that gets attention.
5. **Scaffold-versus-model-capability confounding is worse here than anywhere else in this document.** Every number in this file is a joint function of the underlying model and the surrounding agent harness. Two papers reporting different numbers for what is nominally the same model are frequently actually comparing different scaffolds.
6. **Construction cost is much higher, so benchmarks are smaller and refresh more slowly.** Building a single WebArena- or OSWorld-style task requires standing up and validating a real, functioning environment plus a bespoke checker plus a hand-verified golden trajectory — orders of magnitude more labor per item than writing a multiple-choice question. This is the direct reason these benchmarks (a few hundred to low thousands of tasks) are so much smaller than MMLU's ~16,000 questions, and why refreshing them to stay ahead of saturation or contamination happens on a much slower cadence.

## Common interview framings worth preparing for

A few ways this material tends to get probed in a staff-level interview setting, worth anticipating:

- **"Which of these benchmarks would you use to evaluate a coding agent product?"** — none of the four directly; you'd reach for SWE-bench Verified (file 002) for repository-scale code tasks, and possibly a WebArena-style approach only if the coding agent also needs to operate a browser-based IDE or ticketing system as part of its workflow. Knowing that this file's four benchmarks are not the right tool for that specific question, and why, is itself a signal of real fluency.
- **"Why did OSWorld report such a low early-agent success rate relative to GAIA's?"** — the honest answer is that they are not measuring comparable difficulty on a single shared scale; OSWorld's action space (raw GUI control across arbitrary real applications) is a fundamentally harder perception-and-grounding problem than GAIA's (orchestrating a small number of well-behaved, API-shaped tools), so the two percentages are not on commensurable footing even though both are called "success rate."
- **"If you had to add a fifth benchmark to this file, what capability gap would it target?"** — a reasonable answer is long-horizon autonomy with minimal human oversight (an agent operating over hours or days on a loosely specified goal, checking in only occasionally) — none of the four benchmarks here test sustained autonomy at that timescale; all are bounded, single-session tasks completed in one continuous interaction.

## A note on how these benchmarks are typically combined in practice

In practice, no single team runs only one of these four — a frontier lab's internal agentic-evaluation suite typically draws on several simultaneously, precisely because each targets a different point on the axis described above and none alone is a complete picture. A common pattern: GAIA or a GAIA-like factoid-with-tool-use suite as a cheap, frequently-run regression check (since its grading is cheap); WebArena or a similar browser-based suite run less frequently as a heavier, more infrastructure-intensive check specifically for browser-operating product features; tau-bench-style multi-trial evaluation specifically for any customer-facing conversational-agent product, given its direct relevance to reliability-under-repetition; and OSWorld-style full-computer-use evaluation reserved for product surfaces that actually expose computer-use capability, given how expensive it is to run at scale. Treating these four as a menu to select from based on what capability a given product surface actually needs, rather than as a single monolithic "agentic eval suite" to run uniformly, is itself part of a mature evaluation practice in this space.

## Quick-reference comparison

| Benchmark | Environment | Scoring mechanism | Reported human baseline | Reported early-agent gap |
|---|---|---|---|---|
| WebArena | 5 self-hosted realistic websites | Task-specific state/functional checkers | Not the focus (task-design baseline) | GPT-4 agents ~14% success |
| OSWorld | Real Ubuntu/Windows desktop + apps | File-system/app-state checkers | ~72% | Early agents <15% |
| tau-bench | Simulated user + policy + tools (airline/retail) | Success + pass^k reliability | Not the focus | Large pass@1-to-pass^k drop-off |
| GAIA | Open web/tools/documents | Exact-match on short final answer | ~92% | Early GPT-4+plugins ~15-30% |

## How these benchmarks relate to each other along a single axis

It is useful to keep a mental ordering of these four by "how much of the uncontrolled real world does the agent have to correctly model":

1. **GAIA** — the environment is the open web and a set of well-behaved tools (search, code execution); the hard part is orchestration and multi-hop derivation, not perceiving or acting within a messy interactive interface.
2. **WebArena** — a real, stateful, but self-hosted and bounded web application; the agent must handle genuine UI interaction (clicking, forms, multi-page flows) but within a fixed, structured DOM.
3. **tau-bench** — adds a second intelligent party (the simulated user) and an explicit policy-constraint layer on top of tool use, making the evaluation as much about social/policy reasoning under ambiguity as about raw tool orchestration.
4. **OSWorld** — the largest action and observation space of the four: an entire real desktop OS with pixel-level visual grounding requirements and the least structured, least consistent observation format (screenshots and inconsistent accessibility trees across arbitrary applications).

This ordering is also, roughly, an ordering of evaluation cost and infrastructure burden — GAIA's checkers are the cheapest to write (exact-match on a string), OSWorld's are the most expensive (full VM snapshotting, file-system-state inspection, handling real application nondeterminism).

## What none of these four benchmarks test

Worth stating explicitly, since interviewers sometimes probe for what's missing rather than what's present: none of these four benchmarks evaluate an agent's behavior under adversarial or malicious tool outputs (a tool call returning manipulated content designed to hijack the agent's subsequent behavior — a live, actively researched risk category sometimes called indirect prompt injection, distinct from the jailbreak techniques covered in file 005 that target the model's own safety training directly). None of the four evaluate cost-awareness or efficiency trade-offs (an agent that solves a task correctly but uses ten times the tool calls or tokens a more efficient approach would need scores identically to an efficient one, under every metric described in this file). And none of the four evaluate multi-agent coordination (several agents working together, or one agent supervising another) as its own capability, even though production agent systems increasingly use exactly that pattern. These are reasonable gaps to name if asked "what would you build next" in this space.

## Synthesis

The four benchmarks in this file are best understood as probing four different points on the same underlying axis — how much of the real world must the agent correctly model and act within, beyond the text of the prompt itself. This runs from a single realistic-but-sandboxed web app (WebArena), to an entire desktop OS with pixel-level visual grounding demands (OSWorld), to a three-party social/policy-constrained interaction with a reliability metric that specifically targets repeatability (tau-bench), to a general tool-orchestration-to-a-factoid-answer setting that keeps the cheap grading of static QA while forcing genuinely agentic behavior to get there (GAIA).

None of them are close to saturated the way MMLU or GSM8K are — human-agent gaps remain large across all four as of this writing. This is itself informative: it suggests that whatever capability gains produced near-ceiling performance on knowledge and math benchmarks have not transferred anywhere near as completely to grounded, multi-step, real-environment task execution, and that this gap — not incremental knowledge-benchmark gains — is where a large share of current frontier-lab research effort is concentrated.
