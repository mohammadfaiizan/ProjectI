# Consensus, Debate, And Voting Patterns

## Why Run The Same Problem Through Multiple Agents At All

Everything covered so far in this chapter has been about dividing *different* work across multiple agents — decomposition, delegation, distinct roles. This chapter is about a different move entirely: running the *same* question through multiple agents (or the same agent multiple times) and combining their answers, on the premise that the combination is more reliable than any single run. This is worth being skeptical about by default, because it is also the easiest multi-agent pattern to over-apply — it multiplies cost and latency in exchange for a quality gain that is real but bounded, and there are entire classes of problems where it buys almost nothing. Understanding exactly why and when it helps is more valuable than the mechanics of implementing it, which are comparatively simple.

The underlying statistical intuition is genuine: an LLM's output for a non-trivial question is a sample from a distribution over possible answers, shaped by the randomness in decoding and the model's own uncertainty about ambiguous or hard sub-decisions within the answer. If errors across multiple independent samples (or multiple differently-prompted agents) are not perfectly correlated, aggregating across them — by vote, by debate-driven revision, or by a judge picking the best one — averages out some of that idiosyncratic error, in the same way an ensemble of classifiers or the "wisdom of crowds" effect works in traditional ML and in human forecasting. The catch, and it is a real one, is the qualifier "not perfectly correlated": if you run the same model, at the same temperature, with the same prompt, five times, on a question where the model has one consistent blind spot or misconception, all five samples will tend to share that blind spot, and voting across them won't fix it — you've paid 5x the cost for close to 1x the diversity of perspective. Real gains require genuine diversity: different prompts, different roles, different models, or an interaction structure (like debate) that forces agents to actually engage with each other's reasoning rather than just resampling the same distribution independently.

## Multi-Agent Debate

Debate structures the interaction explicitly: multiple agents each produce an initial answer to the same question, then see each other's answers and reasoning, and are asked to critique, defend, or revise their own position in light of what the others said, repeating for a small number of rounds before a final answer is extracted. The mechanism that makes this more than expensive resampling is that each agent is now conditioned on a genuinely different, adversarial input each round — its own prior answer plus a critique of it, or a competing answer that contradicts it — which pushes it to actually engage with counter-evidence rather than simply resampling from the same unconditional distribution over answers.

```python
class DebateAgent:
    def __init__(self, name, llm, stance_prompt=""):
        self.name = name
        self.llm = llm
        self.stance_prompt = stance_prompt

    def answer(self, question: str, other_answers: list[dict] | None = None) -> str:
        if not other_answers:
            return self.llm.generate(f"{self.stance_prompt}\nQuestion: {question}\nAnswer with reasoning.")

        others_text = "\n\n".join(
            f"{a['agent']} argued:\n{a['answer']}" for a in other_answers
        )
        return self.llm.generate(f"""
        {self.stance_prompt}
        Question: {question}
        Your previous answer is below, along with what other agents argued.
        Critically evaluate their reasoning. If they raise a point that
        changes your view, update your answer and say so explicitly.
        If you still disagree, explain specifically why their reasoning
        is wrong rather than just restating your position.

        Other agents' answers:
        {others_text}

        Revise your answer:
        """)


def run_debate(agents: list[DebateAgent], question: str, rounds: int = 3) -> list[dict]:
    round_answers = [{"agent": a.name, "answer": a.answer(question)} for a in agents]

    for _ in range(rounds - 1):
        next_round = []
        for agent in agents:
            others = [a for a in round_answers if a["agent"] != agent.name]
            revised = agent.answer(question, others)
            next_round.append({"agent": agent.name, "answer": revised})
        round_answers = next_round

    return round_answers


def judge_debate(llm, question: str, final_answers: list[dict]) -> str:
    transcript = "\n\n".join(f"{a['agent']}: {a['answer']}" for a in final_answers)
    return llm.generate(f"""
    Question: {question}
    Final positions after debate:
    {transcript}

    Considering the strength of each argument (not just which position is
    more common), determine the best answer and explain why.
    """)
```

### Deciding When Debate Has Converged

A fixed round count, as used in `run_debate` above, is the simplest termination rule, but it wastes calls on easy questions that converge in one round and cuts off hard questions that would have benefited from a fourth or fifth round. A more efficient approach checks for actual convergence — stop as soon as agents stop changing their positions, rather than always running the full budget.

```python
def run_debate_until_convergence(agents: list[DebateAgent], question: str, max_rounds: int = 5) -> list[dict]:
    round_answers = [{"agent": a.name, "answer": a.answer(question)} for a in agents]

    for round_num in range(max_rounds - 1):
        next_round = []
        for agent in agents:
            others = [a for a in round_answers if a["agent"] != agent.name]
            revised = agent.answer(question, others)
            next_round.append({"agent": agent.name, "answer": revised})

        # Convergence check: if nobody's position changed materially,
        # further rounds are unlikely to help and just burn tokens.
        unchanged = all(
            _same_conclusion(prev["answer"], curr["answer"])
            for prev, curr in zip(round_answers, next_round)
        )
        round_answers = next_round
        if unchanged:
            break

    return round_answers


def _same_conclusion(previous: str, current: str) -> bool:
    # In practice this is itself a small LLM call ("do these two answers
    # reach the same conclusion, ignoring phrasing?") or a comparison on
    # a structured field if answers are constrained to a schema.
    return previous.strip() == current.strip()
```

This mirrors the classical Delphi method from forecasting — expert panelists give an estimate, see the anonymized group's estimates, revise, and the process stops once further rounds stop changing the group's answer — and it's a good reference point because Delphi studies are also where the "shared bias doesn't get fixed by more rounds" limitation was first well documented: if every panelist has access to the same flawed information, rounds of revision converge the group but don't necessarily converge it on the truth.

Debate earns its cost on questions that are genuinely contestable and where reasoning quality, not just factual recall, determines the right answer — open-ended judgment calls, ambiguous requirement interpretation, evaluating trade-offs where reasonable people (or models) could initially disagree. It earns its cost less on factual lookup questions, where an agent that's simply wrong about a fact usually won't be argued out of it by another agent that's equally uncertain, and where retrieval or tool use (letting an agent check a source) is a far cheaper and more reliable fix than another round of debate. A practical failure mode worth watching for is "agreement collapse without correctness" — two agents built on the same base model, given the same weak evidence, often converge to agreement quickly simply because they're similarly persuadable, not because they've actually found the correct answer; debate reduces variance, but it does not reliably fix a shared systematic bias, which is the same limitation resampling has, just partially mitigated rather than eliminated.

## Ensemble Voting

Voting is the simpler, cheaper sibling of debate: instead of iterative back-and-forth, you generate multiple independent answers (from multiple agents, multiple prompts, or repeated sampling of one agent) and combine them with a fixed aggregation rule rather than further LLM reasoning. This is the multi-agent analogue of self-consistency prompting and classical ensemble methods, and it is usually cheaper than debate because it's a single round rather than several.

### Self-Consistency: The Degenerate, Cheapest Case

Before reaching for multiple distinct agents, it's worth naming the simplest possible version of this pattern explicitly, because it is often good enough on its own: self-consistency sampling runs a *single* agent multiple times at a non-zero temperature on the same prompt and takes a majority vote over the independent samples, with no separate agent identities, roles, or prompts involved at all. This is the cheapest way to buy some of the variance-reduction benefit of ensembling, because it needs no additional prompt engineering or role design — just re-running the same call.

```python
def self_consistency_vote(llm, prompt: str, n_samples: int = 5, temperature: float = 0.8) -> dict:
    samples = [llm.generate(prompt, temperature=temperature) for _ in range(n_samples)]
    counts = Counter(samples)
    winner, count = counts.most_common(1)[0]
    return {"winner": winner, "support": count, "n_samples": n_samples, "samples": samples}
```

Self-consistency is the right first thing to try when you suspect an error is a stochastic reasoning slip (the model occasionally drops a step in a multi-step calculation) rather than a systematic gap, and it should generally be tried and measured *before* building out a multi-agent debate or voting pipeline with distinct roles, because it's a fraction of the engineering cost and, on many tasks, captures most of the achievable benefit. Multi-agent debate and role-diverse voting are justified once you've established that self-consistency alone plateaus — which happens exactly when the errors are correlated across samples because they stem from the model's fixed viewpoint rather than from sampling noise, at which point you need genuine diversity (different prompts, different roles, different models), not just more samples from the same distribution.

### Sourcing Real Diversity Instead of Resampling

The value of any ensemble is bounded by how independent its members' errors actually are, so it's worth being deliberate about where diversity comes from rather than assuming that "multiple agents" automatically implies "multiple perspectives."

```python
class DiverseEnsemble:
    """Combines genuinely different sources of variation: different
    underlying models, different role framings, and different retrieved
    context — rather than N calls to the same model with the same prompt."""

    def __init__(self, members: list[dict]):
        # each member: {"name": ..., "llm": ..., "role_prompt": ..., "retriever": Optional[...]}
        self.members = members

    def collect_answers(self, question: str) -> list[dict]:
        answers = []
        for member in self.members:
            context = member["retriever"](question) if member.get("retriever") else ""
            prompt = f"{member['role_prompt']}\nContext: {context}\nQuestion: {question}"
            answers.append({"agent": member["name"], "answer": member["llm"].generate(prompt)})
        return answers
```

Mixing model families (a strong general-purpose model alongside a domain-tuned one), mixing role framings (a "skeptical reviewer" framing alongside a "steelman advocate" framing of the same question), and mixing information access (one agent with live retrieval, one without) all introduce real, independent sources of variation that a majority vote or a judge can meaningfully exploit — this is worth the modest extra setup cost precisely because it's what makes the difference between an ensemble that actually reduces error and one that just multiplies the bill for the same answer five times over.

```python
from collections import Counter


class VotingAggregator:
    def __init__(self, agents):
        self.agents = agents

    def majority_vote(self, question: str) -> dict:
        votes = {agent.name: agent.answer(question) for agent in self.agents}
        counts = Counter(votes.values())
        winner, count = counts.most_common(1)[0]
        return {"winner": winner, "support": count, "total": len(votes), "votes": votes}

    def weighted_vote(self, question: str, weights: dict[str, float]) -> dict:
        votes = {agent.name: agent.answer(question) for agent in self.agents}
        tallies: dict[str, float] = {}
        for agent_name, vote in votes.items():
            tallies[vote] = tallies.get(vote, 0.0) + weights.get(agent_name, 1.0)
        winner = max(tallies, key=tallies.get)
        return {"winner": winner, "tallies": tallies}

    def ranked_vote(self, question: str, options: list[str]) -> dict:
        # Borda count: each agent ranks all options; points awarded by rank position.
        scores = {opt: 0 for opt in options}
        for agent in self.agents:
            ranking = agent.rank(question, options)
            for position, option in enumerate(ranking):
                scores[option] += len(options) - position
        return {"winner": max(scores, key=scores.get), "scores": scores}
```

Majority voting is the right default when answers naturally fall into a small number of discrete categories (a classification decision, a yes/no judgment, picking among a fixed set of options) and independent samples give genuine variance to average out. Weighted voting matters when you have prior evidence that some agents (or some models) are more reliable than others for this specific type of question — a specialist model's vote should count for more than a generalist's on its home turf, and that weighting can be learned from historical accuracy rather than fixed arbitrarily. Ranked/Borda voting is useful when the output space isn't a clean single "correct answer" but a preference ordering — picking the best of several draft options where "best" is a judgment call agents might rank differently even if none is flatly wrong.

Voting has a structural weakness debate doesn't: when the space of possible free-text answers is large (as opposed to a fixed small set of options), literal string-matching votes rarely coincide even when the answers are substantively the same, because two correct answers phrased differently look like two different "votes." The common fix is to use an LLM as a semantic clustering step before counting — grouping answers by substantive agreement rather than exact text match — or to use an LLM-as-judge that picks the best single answer from the set rather than tallying votes at all, which shifts you from "voting" toward "best-of-n with a judge," a closely related but distinct pattern worth knowing by name since it shows up constantly in evaluation and RLHF-adjacent tooling.

```python
def best_of_n_with_judge(llm, question: str, candidates: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
    choice = llm.generate(f"""
    Question: {question}
    Candidate answers:
    {numbered}

    Pick the single best candidate by index. Consider correctness,
    completeness, and clarity. Respond with only the index number.
    """)
    return candidates[int(choice.strip())]
```

### Rough Cost Comparison

It helps to have concrete orders of magnitude in mind rather than reasoning about "more calls" in the abstract. For a task where a single well-prompted agent call costs roughly one unit of tokens and latency:

| Approach | Relative LLM calls | Relative latency | When it pays off |
|---|---|---|---|
| Single agent, single pass | 1x | 1x | Default baseline; most tasks |
| Single agent + tool verification (tests, schema check, calculator) | 1-2x | 1-1.5x | Anything with a checkable correct answer |
| Self-consistency (5 samples, majority vote) | 5x | ~1x (parallelizable) | Stochastic slips on ambiguous or multi-step reasoning |
| Diverse ensemble (3-5 distinct agents/models, single round + judge) | 4-6x | ~1.2x (parallel generation, one extra judge call) | Judgment calls where genuine diversity is available |
| Multi-agent debate (3 agents, 3 rounds) | 9-12x | 3x (rounds are sequential) | High-stakes, contestable, non-interactive workloads |

The latency column matters as much as the call-count column: self-consistency and diverse ensembles parallelize cleanly (all samples can be requested concurrently), so their latency overhead is much smaller than their cost overhead, while debate is inherently sequential across rounds and pays for that in wall-clock time as well as tokens. This is often the deciding factor in practice — a 5x cost increase that stays within one round-trip's latency is a much easier sell for an interactive product than a 3x latency increase on top of a 10x cost increase.

### Judge Selection And Self-Preference Bias

Whether you're picking a winner from a debate or from an ensemble of independently generated candidates, the choice of judge matters as much as the choice of ensemble members, and a common, easy-to-miss mistake is letting one of the ensemble's own members also act as the judge. Models (like people) exhibit measurable self-preference bias — a model asked to judge a set of candidate answers that includes one it wrote itself tends to rate its own answer more favorably than an independent judge would, even when explicitly instructed to be objective, simply because its own phrasing and reasoning style matches what it considers natural or correct. The practical fix is structural rather than a matter of better prompting: use a judge that did not generate any of the candidates, ideally a different model entirely, and if using an LLM-as-judge at all, consider randomizing the order in which candidates are presented (since position bias — favoring whichever candidate appears first — is a second, independent bias worth guarding against) and stripping any metadata that would let the judge infer which candidate came from which source.

```python
def judge_without_bias(judge_llm, question: str, candidates: list[str]) -> int:
    import random
    order = list(range(len(candidates)))
    random.shuffle(order)  # guard against position bias
    shuffled = [candidates[i] for i in order]

    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(shuffled))
    choice = judge_llm.generate(f"""
    Question: {question}
    Candidates (order is randomized and carries no meaning):
    {numbered}
    Pick the best by index only, based on correctness and completeness.
    """)
    shuffled_index = int(choice.strip())
    return order[shuffled_index]  # map back to the original candidate index
```

This matters most exactly when the stakes are high enough to justify an ensemble in the first place — if you're already paying the cost of running multiple agents to reduce error on a high-stakes decision, an uncorrected self-preference or position bias in the judging step can quietly erase most of the quality gain the ensemble was built to capture, while still looking, from the outside, like a working multi-agent quality process.

## When The Extra Cost Is (And Isn't) Worth It

The honest framing is that debate and voting are a form of test-time compute spending: you are trading tokens, latency, and dollars for a probabilistic improvement in answer quality, and that trade needs to be evaluated against the alternative of simply spending those same resources on a single stronger pass — a bigger/better model, a more careful single prompt, retrieval augmentation, or letting one agent use tools to verify its own work. In many cases, that alternative wins outright: giving a single capable agent access to a calculator, a code execution sandbox, or a search tool to check its own claim is usually cheaper and more reliable than having three agents debate the claim without any of them having a way to actually verify it — debate improves reasoning quality, it does not manufacture facts nobody in the ensemble actually has access to.

The pattern earns its cost under a fairly specific set of conditions. First, when the task is genuinely high-stakes and errors are expensive relative to the extra inference cost — a medical triage summary, a legal risk assessment, a large financial decision — where even a modest reduction in error rate is worth several times the token cost. Second, when the question is inherently ambiguous or judgment-based rather than fact-lookup, so there is real reasoning variance across samples for debate or diverse ensembling to average out, rather than a single fact that's either known or not. Third, when you can source genuine diversity cheaply — different models (e.g., one call to a strong general model, one to a domain-fine-tuned model, one to a different vendor's model) rather than five calls to the same model at the same temperature, since diversity, not sample count, is what drives the benefit. Fourth, when latency is not on the critical interactive path — batch or asynchronous workloads (offline report generation, overnight reconciliation jobs) can absorb the multi-round latency of debate far more easily than a chat interface where a user is waiting on the response.

Conversely, the pattern is usually not worth it for well-defined tasks with a checkable correct answer (code that can be tested, math that can be verified, structured extraction that can be schema-validated) — here, a single agent with a verification loop (run the tests, check the schema, retry on failure) achieves a similar or better reliability gain at a fraction of the cost of running the whole pipeline multiple times and voting, because the verification is deterministic and exact rather than probabilistic and approximate. It is also usually not worth it when the base model is simply too weak for the task domain — no amount of voting or debate among instances of a model that lacks the requisite knowledge or reasoning capability will produce a correct answer none of them individually had the ingredients to reach; in that situation, upgrading to a stronger single model is strictly more effective than ensembling weaker ones. A reasonable heuristic to apply before reaching for debate or voting: ask whether the errors you're worried about come from *stochastic reasoning slips* (a good model occasionally missing a step, worth averaging out) or from a *systematic capability gap* (the model doesn't know something or can't do something at all, which averaging cannot fix) — the former justifies the multi-agent spend, the latter calls for a better model, better tools, or better retrieval instead.

```python
def should_use_ensemble(task_profile: dict) -> bool:
    """A rough decision heuristic, not a substitute for measuring
    actual quality/cost trade-offs on your own eval set."""
    high_stakes = task_profile.get("error_cost", "low") in ("medium", "high")
    is_judgment_task = task_profile.get("has_single_verifiable_answer", False) is False
    latency_tolerant = task_profile.get("interactive", True) is False
    capability_sufficient = task_profile.get("base_model_capability", "adequate") != "insufficient"

    return high_stakes and is_judgment_task and latency_tolerant and capability_sufficient
```

The right process, in production, is empirical rather than dogmatic: measure quality with and without the ensemble on a representative evaluation set, measure the incremental cost and latency, and only keep the ensemble if the quality delta clears a bar that justifies the multiplier — treat "add more agents" as a hypothesis to test against a single well-engineered agent, not an assumption to design in from the start.
