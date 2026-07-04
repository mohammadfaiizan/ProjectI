# Designing a Coding Agent

## Why This Question Shows Up in Interviews

"Design a coding agent" has become one of the canonical system-design prompts for senior AI
engineers, for the same reason "design a URL shortener" became canonical for backend engineers a
decade ago: it is small enough to reason about in forty-five minutes but rich enough to expose every
hard problem in the field. A coding agent has to read and understand a large, messy codebase it has
never seen before; decide what to change; produce an edit that actually applies cleanly; verify the
edit didn't break anything; and do all of this inside a tight budget of tokens, wall-clock time, and
dollars, while a human is trusting it not to quietly wreck their repository. Every one of those
clauses maps to a real subsystem, and every subsystem has at least one non-obvious trade-off. This
chapter walks through the design the way you'd walk an interviewer through it: start from what the
system actually needs to do, sketch the components and how data flows between them, then spend most
of the time on the decisions that don't have a clean right answer.

It helps to be concrete about what "coding agent" means here. We are not designing a single
autocomplete call that suggests the next few tokens in an editor. We are designing something closer
to an autonomous pair programmer: given a natural-language task ("fix this failing test," "add rate
limiting to the /login endpoint," "migrate this module from callbacks to async/await"), the agent
should be able to explore the repository on its own, form a plan, make multi-file edits, run the
test suite, read the failures, and iterate — with as little hand-holding as possible, but with an
easy path for a human to step in when something looks wrong.

## Clarifying Requirements First

Before drawing boxes, it's worth stating the requirements explicitly, because they change the design
a lot depending on the answers.

- **Repository scale**: are we operating on a 2,000-line side project or a 10-million-line monorepo with thousands of packages? This single variable determines whether "just put the whole repo in context" is even conceivable.
- **Autonomy level**: is this a CLI tool a developer runs and watches, a CI bot that opens pull requests unattended, or something in between? This determines how much guardrail and approval machinery you need.
- **Latency and cost budget**: is a single task allowed to take 30 seconds, or is a 20-minute agentic run with dozens of tool calls acceptable because it replaces an hour of human work? Budgets shape how aggressively you can retry, how big a model you can afford to call at each step, and how much redundant verification you can build in.
- **Blast radius of a mistake**: can a bad edit merge straight to production, or does it sit behind CI and code review? The answer determines how much you invest in sandboxing and pre-merge verification versus relying on the human safety net that already exists.
- **Tooling surface**: does the agent only need read/edit/run-tests access, or does it also need to hit package registries, cloud consoles, or production data? Every additional capability is both a productivity multiplier and a new failure mode.

For the rest of this chapter, assume the harder end of these: a large, unfamiliar monorepo, a mix of
supervised (developer-in-the-loop) and semi-autonomous (CI-triggered) usage, and a cost/latency
budget that allows a multi-step loop but not an unbounded one.

## High-Level Architecture

At the center of the system is an **agent loop** — a controller that alternates between calling an
LLM to decide the next action and executing that action against the real world (the filesystem, a
shell, a test runner). Everything else in the design exists to feed that loop good information and
to keep it from doing damage.

```
                     +-----------------------------+
                     |         Orchestrator         |
                     |  (agent loop / state machine)|
                     +---------------+---------------+
                                     |
      +------------------+----------+----------+------------------+
      |                  |                     |                  |
      v                  v                     v                  v
+-----------+   +-----------------+   +-----------------+   +-------------+
| Context   |   |  Edit / Diff    |   |  Sandbox /      |   |  Human      |
| Retrieval |   |  Application    |   |  Test Execution |   |  Gate       |
| (repo map,|   |  (patch format, |   |  (containers,   |   |  (approve / |
|  search,  |   |  apply + lint)  |   |  resource caps) |   |   reject)   |
|  embeddings|   +-----------------+   +-----------------+   +-------------+
+-----------+
      |                  |                     |                  |
      +------------------+----------+----------+------------------+
                                     |
                                     v
                         +-----------------------+
                         |  Working memory /      |
                         |  scratchpad / plan     |
                         +-----------------------+
```

The orchestrator receives a task, asks the context-retrieval subsystem for the minimum set of files
and symbols relevant to that task, feeds that plus the task into the LLM, receives back either a
plan, a tool call, or a proposed diff, routes risky actions through a human gate, applies safe
edits, runs tests in a sandbox, and feeds the results back into the loop. This repeats until the
task is done, the agent gives up, or a budget (time, cost, or iteration count) is exhausted. Each of
the four boxes deserves its own discussion.

## Component 1: Context Retrieval Over the Repository

The single biggest constraint on a coding agent is that even the largest context windows (hundreds
of thousands of tokens) are minuscule compared to a real codebase. A monorepo with 10 million lines
of code is on the order of 100M+ tokens — you cannot "just paste the repo in." So the first real
engineering problem is: given a task description, how do you find the handful of files (usually well
under 1% of the repo) that are actually relevant?

There are three complementary retrieval strategies, and production systems use all three together
rather than picking one:

**Structural/symbolic search.** Build an index of the repo's symbol graph — function and class
definitions, call sites, import relationships — using a parser (tree-sitter is the common choice
because it's fast and language-agnostic). When the task mentions a function name, file path, or
error message, you can jump directly to the relevant definitions and their callers/callees. This is
precise and cheap, but only works when the query has a literal anchor to latch onto.

**Semantic/embedding search.** For queries phrased in natural language ("where do we validate user
emails?"), you embed chunks of the codebase (typically function- or class-sized chunks, not
arbitrary line windows, so that a chunk is a coherent unit) into a vector store, and embed the query
the same way, then retrieve nearest neighbors. This handles fuzzy, conceptual queries that symbolic
search can't, at the cost of occasional irrelevant matches and the operational overhead of keeping
an index fresh as the repo changes.

**Repo map / architectural summary.** Independent of any specific query, it helps to maintain a
compact, always-in-context summary of the repo's shape: directory structure, key modules, and a
one-line description of what each does. This gives the model a mental map so it can decide *where to
look* even before retrieval runs, and it's cheap to keep updated incrementally.

```python
class RepoContextBuilder:
    """Assembles the minimal context window for a coding task."""

    def __init__(self, symbol_index, vector_index, repo_map, token_budget=12_000):
        self.symbol_index = symbol_index
        self.vector_index = vector_index
        self.repo_map = repo_map
        self.token_budget = token_budget

    def build(self, task_description: str, mentioned_paths: list[str]) -> str:
        chunks = []

        # 1. Always include the compact architectural map (cheap, high value)
        chunks.append(("repo_map", self.repo_map.summary(), 500))

        # 2. Anchor on any file/symbol explicitly named in the task or a
        #    failing stack trace -- these are near-certain to be relevant
        for path in mentioned_paths:
            content = self.symbol_index.read_file(path)
            chunks.append((path, content, estimate_tokens(content)))

        # 3. Fill remaining budget with semantically relevant chunks,
        #    ranked by similarity and de-duplicated against what's
        #    already included
        remaining = self.token_budget - sum(c[2] for c in chunks)
        for hit in self.vector_index.search(task_description, top_k=20):
            if remaining <= 0:
                break
            if hit.path in {c[0] for c in chunks}:
                continue
            chunks.append((hit.path, hit.text, hit.tokens))
            remaining -= hit.tokens

        return "\n\n".join(f"# {path}\n{text}" for path, text, _ in chunks)
```

A subtlety that matters in practice: retrieval quality degrades the agent's *entire* run, not just
one step, because a bad initial context leads to a bad plan, which leads to edits in the wrong
files, which the agent then has to discover and undo several iterations later — burning cost and
time. This is why it's worth over-investing in retrieval relative to how flashy it feels: it's the
highest-leverage component in the whole system.

## Component 2: Edit and Diff Application

Once the model decides what to change, it has to communicate that change in a format that can be
applied deterministically and safely. There are three broad approaches, with real trade-offs:

**Full file rewrite** — the model outputs the entire new content of a file. This is the most
reliable for the model to produce (no need to reason about line numbers or context matching) but is
token-expensive for large files and risks silently dropping unrelated content the model "forgot" was
there.

**Unified diff / patch format** — the model outputs a `diff -u`-style patch with context lines, and
the system applies it with a patcher. This is compact and mirrors what developers already review in
pull requests, but LLMs are notoriously bad at getting line numbers and exact whitespace right, so
patches frequently fail to apply cleanly against the real file.

**Search-and-replace blocks** — the model outputs an anchor (a short, unique snippet of existing
code) and the replacement text, and the system finds the anchor in the file and swaps it. This
sidesteps the line-number problem entirely and is the most robust in practice, provided the anchor
is required to be unique — if the same snippet appears twice, the edit is rejected rather than
silently applied to the wrong occurrence.

```python
class SearchReplaceEditor:
    def apply(self, file_path: str, search: str, replace: str) -> None:
        original = read_file(file_path)
        occurrences = original.count(search)

        if occurrences == 0:
            raise EditError(
                f"Anchor not found in {file_path}. The model likely has a "
                f"stale view of the file -- refresh context and retry."
            )
        if occurrences > 1:
            raise EditError(
                f"Anchor matches {occurrences} locations in {file_path}; "
                f"refusing ambiguous edit. Ask for a larger, unique anchor."
            )

        updated = original.replace(search, replace)
        write_file(file_path, updated)

        # Verify the edit didn't break syntax before committing to it
        if not syntax_is_valid(file_path, updated):
            write_file(file_path, original)  # roll back
            raise EditError(f"Edit produced invalid syntax in {file_path}")
```

Whichever format you choose, treat edit application as a step that can fail and must report a
*structured, actionable* error back into the loop — "anchor not found" is something the model can
recover from by re-reading the file; a bare exception traceback is not. This is the general
principle that recurs throughout agent design: every tool should fail in a way that gives the model
(or a human) enough signal to correct course, not just a signal that something went wrong.

## Component 3: The Test Execution Sandbox

An agent that edits code but never runs anything is just a very expensive autocomplete. The value of
a coding agent comes from closing the loop: make a change, run the tests, read the result, adjust.
That means the system needs a place to execute untrusted, LLM-generated code and shell commands
safely and repeatably.

The sandbox needs to provide process isolation (a container or microVM, not the host machine — the
agent will, sooner or later, run `rm -rf` on the wrong path or install a malicious-looking
dependency it hallucinated), resource limits (CPU, memory, and disk quotas, since a bad change can
trigger an infinite loop or a runaway memory allocation in the code under test), a hard wall-clock
timeout per command (test suites can hang, and a hung sandbox is a stuck agent), and network policy
(usually deny-by-default, with narrow allowlists for package registries, because unrestricted
network access from an autonomous agent is both a security risk and a way for it to silently depend
on external state).

```python
class SandboxRunner:
    def __init__(self, image: str, cpu_limit=2, mem_limit_mb=2048, timeout_s=120):
        self.image = image
        self.cpu_limit = cpu_limit
        self.mem_limit_mb = mem_limit_mb
        self.timeout_s = timeout_s

    def run(self, command: list[str], workdir: str) -> "ExecResult":
        container = start_container(
            image=self.image,
            mounts={workdir: "/workspace"},
            cpu=self.cpu_limit,
            memory_mb=self.mem_limit_mb,
            network="restricted",  # allowlist only
        )
        try:
            proc = container.exec(command, cwd="/workspace", timeout=self.timeout_s)
            return ExecResult(
                exit_code=proc.exit_code,
                stdout=truncate(proc.stdout, max_chars=8_000),
                stderr=truncate(proc.stderr, max_chars=4_000),
                timed_out=proc.timed_out,
            )
        finally:
            container.destroy()  # always clean up, even on failure
```

Two practical details separate a demo sandbox from a production one. First, **ephemeral, disposable
environments**: spin up fresh for each run (or checkpoint and restore) rather than reusing a
long-lived container, so a corrupted dependency install in run 3 doesn't silently poison run 4.
Second, **output truncation with signal preservation**: a failing test suite can produce megabytes
of stack traces, and the tempting move is to dump it all back into the model's context — but that
blows the token budget and often buries the one relevant assertion failure under noise. It's usually
better to run failures through a lightweight parser (test-framework-aware, e.g., parsing pytest or
Jest output) that extracts just the failing test names, the assertion diff, and the top frame of
each traceback, and pass *that* structured summary to the model, with the option to fetch full
output for one specific test if needed.

## Component 4: The Iteration Loop

The loop itself is a state machine, not a single prompt. A reasonable shape is: **plan → act →
observe → reflect → repeat**, with explicit exit conditions.

```python
def run_coding_agent(task: str, repo, max_iterations=15, max_cost_usd=2.00):
    context = RepoContextBuilder(...).build(task, mentioned_paths=extract_paths(task))
    history = [SystemMsg(AGENT_INSTRUCTIONS), UserMsg(task), UserMsg(context)]
    spent = 0.0

    for i in range(max_iterations):
        response = llm.call(history)
        spent += response.cost_usd
        action = parse_action(response)  # tool call, edit, or "done"

        if action.type == "done":
            return finalize(action, tests_passed=run_full_suite(repo))

        if action.risk_level == "high" and not action.pre_approved:
            decision = human_gate.request_approval(action)
            if decision != "approve":
                history.append(ToolResultMsg(action.id, "rejected by reviewer"))
                continue

        result = execute(action, repo)  # edit, shell command, or test run
        history.append(ToolResultMsg(action.id, result.summary()))

        if spent >= max_cost_usd:
            return finalize_partial(history, reason="cost budget exhausted")

    return finalize_partial(history, reason="iteration budget exhausted")
```

The exit conditions matter as much as the happy path. Without a hard cap on iterations and cost, a
stuck agent — one that keeps making a syntactically valid but semantically wrong fix, watches the
same test fail, and tries a slightly different wrong fix — will loop until someone notices the bill.
A good design treats "give up gracefully and hand back a clear summary of what was tried" as a
first-class success outcome, not a failure to be hidden.

## Trade-off: How Much Autonomy to Grant

This is the crux of the interview question, and there's no universally correct answer — the right
point on the spectrum depends on blast radius and the cost of a human reviewing versus the cost of a
mistake. It helps to think of autonomy as a dial with roughly four settings:

At the low end, the agent proposes a diff and does nothing until a human approves it — essentially a
very good autocomplete for pull requests. This is safe but slow, and it doesn't scale past a handful
of tasks a day per reviewer. One level up, the agent can freely read files and run tests
(side-effect-free actions) but must get approval before writing to disk or running arbitrary shell
commands — this is a good default for most "agent in the developer's terminal" products, because
reads and test runs are reversible and edits are not. Higher still, the agent can edit and run tests
freely inside a sandbox and only requires approval to open a pull request or merge — this is
appropriate when the sandbox is fully isolated from anything that matters, so the worst case is
wasted compute, not damage. At the top end, fully autonomous end-to-end (edit, test, open PR, and
even merge on green CI) is reserved for narrow, low-risk, well-tested task categories — dependency
bumps, lint fixes, codemods with strong test coverage — where the acceptance criteria are
unambiguous and automatically checkable.

The practical recommendation to give in an interview is not "pick one level," but "make autonomy a
per-action, configurable property, driven by risk classification of the action itself" — deleting a
file, touching authentication code, or modifying CI configuration should always route through a
stricter gate than editing a docstring, regardless of which global autonomy mode the product is in.
This is discussed in depth as a general pattern in the chapter on human-in-the-loop design.

## Trade-off: Context Management at Scale

A second axis worth spending interview time on is what happens as repository size grows from "fits
in one context window" to "genuinely enormous." At small scale, you can be generous — include whole
files, don't worry much about retrieval precision. At monorepo scale, several techniques become
necessary rather than optional.

**Hierarchical summarization**: instead of retrieving raw file contents for everything the model
might plausibly need, maintain pre-computed summaries at the file and module level (a paragraph
describing what a file does and its public interface), and only expand to full source for files the
agent has decided are directly relevant. This is the codebase equivalent of an index versus a
full-text scan.

**Context compaction mid-run**: a long-running agentic session accumulates tool outputs, failed
attempts, and intermediate reasoning in its history, and that history itself will eventually exceed
the context window. The standard fix is periodic compaction — summarizing the older parts of the
transcript into a compact "what has been tried and learned so far" note, while keeping recent turns
verbatim — rather than a hard truncation that silently drops information the model still needs (like
"I already tried approach X and it failed because Y").

**Staged retrieval with re-querying**: don't assume the first retrieval call gets everything right.
Let the agent issue follow-up searches ("show me callers of this function," "find the config file
for this service") as part of the loop, the same way a human engineer greps around after an initial
read. This trades a bit of latency for a large improvement in correctness, since it turns retrieval
from a single guess into an interactive process.

**Cost-aware model routing**: not every step needs the most capable (and most expensive) model.
Classifying which files are relevant, or checking whether a diff applied cleanly, can run on a
small, fast model; the actual planning and code-generation steps justify a larger model. This is
explored in more depth in the chapter on scalability and cost trade-offs, but it's worth flagging
here because context management and model routing compound — a well-retrieved, tightly-scoped
context lets you get away with a cheaper model for more of the loop.

## Failure Modes Worth Naming Explicitly

A strong answer proactively names what goes wrong, rather than waiting to be asked. The most common
failure modes for coding agents are: **stale context**, where the agent edits based on a file
snapshot that a previous step already changed, producing an anchor-not-found error or, worse,
silently clobbering the earlier edit; **cascading bad edits**, where an incorrect first change
causes a chain of "fixes" that compound the damage rather than correcting it — this is why a hard
iteration cap and a "revert and re-plan from scratch" fallback are not optional extras; **test-suite
blind spots**, where the agent optimizes for "tests pass" and finds a degenerate solution (deleting
the failing test, or hardcoding the expected output) that is technically green but wrong — mitigated
by explicitly instructing the model that modifying tests to make them pass is a high-risk action
requiring approval, and by diffing test files separately in review; and **prompt injection from
repository content**, where a malicious comment, README, or dependency file contains text crafted to
hijack the agent's instructions once it's pulled into context — mitigated by treating all repository
content as untrusted data rather than instructions, and by keeping the system prompt's authority
explicit and non-overridable by content encountered during retrieval.

## Putting Rough Numbers On It

Interviewers generally want to see that you can reason quantitatively, not just architecturally. A
representative back-of-envelope for a single "fix this failing test" task: context retrieval
assembles roughly 8-15K tokens of repository context; the planning/generation call runs on a
frontier model at maybe 20-40K tokens of input (context plus conversation history) and a few
thousand tokens of output; the loop typically takes 3-8 iterations before either succeeding or
giving up; each iteration includes one LLM call and one sandbox execution (tests typically take 5-60
seconds depending on suite size). That puts a single task at roughly 100-300K tokens total and 1-5
minutes of wall clock time, at a cost on the order of a dollar or less on current frontier-model
pricing — cheap compared to a human engineer's time, but expensive enough that an unbounded retry
loop across thousands of CI-triggered tasks per day can become a real line item, which is exactly
why the cost cap in the iteration loop above is not a defensive afterthought but a core design
requirement.

