# Interview Questions — Part 2

## Q1: Why does moving from 4-option to 10-option multiple choice (MMLU to MMLU-Pro) matter more than it might sound like, and derive the actual guessing-baseline math.

The random-guess floor for n-option multiple choice is `1/n`: 25% for 4 options, 10% for 10 options. This isn't just a cosmetic difference in expected score under pure guessing — it changes how much a model can gain from *partial* knowledge via elimination. Under 4-option MC, ruling out just one obviously-wrong distractor already raises guess accuracy from 25% to 33% (1/3), and ruling out two gets you to 50% — a model with only weak, partial signal about a question can look substantially more accurate than its true underlying knowledge would suggest, because the format itself is generous. Under 10-option MC, ruling out the same absolute number of implausible distractors (say, three obviously-wrong options) only raises guess accuracy from 10% to about 14% (1/7) — the elimination strategy simply has much less leverage when there are more plausible-looking distractors to begin with, assuming the added distractors are constructed to be genuinely plausible rather than padding.

This is exactly why MMLU-Pro's redesign paired the 10-option format change with careful distractor construction (not just adding six throwaway wrong answers) — the guessing-floor math only delivers its intended benefit if the additional distractors are actually plausible enough to resist trivial elimination; nine options where six are obviously silly would barely improve on the 4-option case in practice, even though the nominal random-guess floor looks much lower on paper.

## Q2: GAIA keeps exact-match, factoid-style answer scoring even though the task requires genuinely agentic multi-step tool use. What tradeoff does this represent, and what does it specifically fail to capture?

GAIA's design bet is to preserve the cheap, unambiguous grading advantage of static QA (a short final answer — a number, name, or short phrase — checked via exact or lightly normalized string match) while still forcing genuinely agentic behavior to *reach* that answer (multi-step web search, document parsing, code execution, cross-referencing). This sidesteps the expensive bespoke-state-checker infrastructure that WebArena, OSWorld, and tau-bench all require, at the cost of only ever being able to grade the *outcome*, never the *process*.

What it fails to capture: an agent that reaches the correct final answer via a lucky guess, a shortcut unrelated to genuine tool orchestration, or partial information that happened to be sufficient, scores identically to an agent that executed a fully correct, generalizable multi-step reasoning-and-tool-use process — the scoring mechanism has no visibility into *how* the answer was derived, which is arguably the entire thing GAIA was built to probe. This is the same generic "exact-match-on-final-answer can't detect invalid intermediate reasoning that coincidentally lands on the right answer" limitation that applies to GSM8K/MATH/AIME (file 002), but it's more consequential here because GAIA's premise is specifically about validating *process* (multi-step tool orchestration), and outcome-only scoring is a structural mismatch with that stated goal, even though it was a deliberate and reasonable tradeoff for keeping grading tractable at scale.

## Q3: Two frontier models differ by 1.5 percentage points on GPQA Diamond. Is this a meaningful capability difference? How would you actually determine that instead of just reading the leaderboard?

GPQA Diamond is 198 questions. A 1.5-point difference corresponds to roughly 3 flipped answers out of 198 — well within the range you'd expect from ordinary sampling noise (different random seeds, minor prompt-formatting differences, or a couple of genuinely ambiguous/borderline-labeled questions going one way or the other) rather than necessarily reflecting a real underlying capability gap. Reporting a bare percentage with no uncertainty quantification on a benchmark this small is close to reporting noise as signal.

To actually determine whether the difference is meaningful: compute a confidence interval for each model's score (e.g., via bootstrap resampling of the per-question correctness, analogous to how LMSYS bootstraps Bradley-Terry strengths for Chatbot Arena) and check whether the intervals overlap substantially — if they do, the honest conclusion is "not distinguishable given this sample size," not "model A is better." Beyond statistics, you'd also want to check whether the two models were evaluated under the exact same protocol (same prompt template, same number of few-shot examples if any, same answer-extraction parser) since GPQA scoring is sensitive to these details, and whether either model shows signs of the same MMLU-Redux-style label-noise sensitivity — a few of GPQA's 198 Diamond questions being genuinely mislabeled would be entirely sufficient on its own to produce a 1.5-point spread between two models with truly identical underlying capability.

## Q4: Implement a MATH-style answer-equivalence checker: normalize a LaTeX-boxed final answer and fall back to symbolic equivalence checking when direct string comparison fails.

```python
import re

def extract_boxed_answer(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None

def normalize_math_answer(ans: str) -> str:
    ans = ans.strip().replace(" ", "")
    ans = ans.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = re.sub(r"\\dfrac", "\\frac", ans)
    ans = ans.replace("^\\circ", "").replace("\\%", "").rstrip("%")
    return ans

def math_is_equivalent(pred_text: str, reference_text: str) -> bool:
    pred, ref = extract_boxed_answer(pred_text), extract_boxed_answer(reference_text)
    if pred is None or ref is None:
        return False
    p, r = normalize_math_answer(pred), normalize_math_answer(ref)
    if p == r:
        return True
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        diff = sympy.simplify(parse_latex(p) - parse_latex(r))
        return bool(diff == 0)
    except Exception:
        return False  # unparseable or a genuine mismatch -- fail closed, not open
```

The important design point to state explicitly if asked to critique this: string normalization catches most superficial formatting differences cheaply, but the symbolic fallback is where correctness risk concentrates — different eval harnesses (the original Hendrycks et al. code, EleutherAI's lm-evaluation-harness, Google's Minerva-derived normalizer) implement different normalization rules and different symbolic-equivalence coverage, so reported MATH accuracy for the *same* model can differ non-trivially across papers depending purely on which harness's equivalence checker was used — this is a real, documented reproducibility problem specific to benchmarks with non-trivial answer equivalence, and it does not arise at all for GSM8K's plain-integer answers.

## Q5: Explain "benchmaxxing" as a concrete instance of Goodhart's law, with a specific, mechanistic example of how it could happen without anyone deliberately cheating.

Goodhart's law: when a measure becomes a target, it ceases to be a good measure. "Benchmaxxing" is the LLM-specific instance: once a benchmark becomes a widely-cited headline number (something every paper's abstract and every product launch announcement quotes), there's direct competitive and organizational pressure to improve specifically on that number, and that pressure can produce score gains that don't reflect proportional gains in the underlying general capability the benchmark was only ever meant to be a proxy for.

A concrete, non-malicious mechanism: a post-training data pipeline includes publicly available instruction-tuning or SFT datasets that were themselves built (by third parties, for legitimate reasons) from exam-style question banks structurally similar to MMLU's source material — practice tests, certification-exam question sets, textbook question banks. Nobody on the training team needs to deliberately target MMLU's specific 15,908 questions; simply having more exam-question-shaped data in the training mix than a previous version had can lift MMLU disproportionately relative to other capabilities, purely because the *task shape* MMLU tests (recognize the right option among several for an exam-style question) got more represented in training, independent of whether general knowledge or reasoning improved at all. This is exactly the kind of mechanism that would produce the "tops the MMLU leaderboard, users complain it's worse" scenario (file 007) without requiring any deliberate benchmark-gaming intent — it's an emergent property of optimizing against any proxy metric using data that happens to resemble that proxy's task shape.

## Q6: What is HarmBench's distinguishing methodological contribution relative to earlier, more ad hoc red-teaming evaluations?

Earlier red-teaming writeups typically used bespoke behavior sets and bespoke (often less rigorously validated) success classifiers per paper, making cross-paper comparison unreliable — one paper's "80% attack success rate" and another's "80% attack success rate" might reflect entirely different behavior definitions and entirely different judgments of what counts as "harmful compliance." HarmBench's contribution is a **standardized, shared harness that evaluates models and attack methods jointly**: a curated set of ~400 specific harmful behaviors across defined categories, a battery of established attack methods (direct request, GCG suffix optimization, PAIR-style iterative refinement, human-written jailbreak templates) applied uniformly across all evaluated models, and a fine-tuned compliance classifier (rather than keyword-based refusal detection, which is known to be unreliable in both directions) used consistently to score every resulting response.

This dual structure — same behaviors, same attacks, same classifier, applied across every (model, attack) pair — produces a matrix that supports two distinct kinds of legitimate comparison from one harness: which models are most robust across the whole attack battery (model-robustness ranking) and which attack methods are most effective across the whole model set (attack-strength ranking, useful for red-teaming researchers benchmarking a newly proposed jailbreak technique against established ones on equal footing). That dual-purpose, standardized-harness design is the specific methodological advance, not merely "having more harmful prompts than before."

## Q7: Scenario — you're responsible for the pre-release evaluation suite for a new frontier model. What combination of benchmarks from this document would you actually run, and what would you explicitly refuse to conclude from any single one of them?

I'd run a deliberately heterogeneous suite covering distinct axes, not a maximal list: MMLU-Pro and GPQA Diamond for broad-plus-adversarially-hard knowledge/reasoning (not plain MMLU, which is saturated and label-noisy enough to be low-information at this point); AIME (most recent available year, to minimize contamination exposure) and MATH-500 for math reasoning, reported with confidence intervals given AIME's small n; SWE-bench Verified under a fixed, documented agent scaffold for realistic coding-agent capability, explicitly reported alongside the scaffold version used; at least one agentic benchmark (GAIA or tau-bench, depending on whether the product surface is more tool-orchestration or more customer-facing-conversational) with multiple independent trials per task rather than single-shot numbers; HarmBench-style harmful-compliance rate paired with an XSTest/OR-Bench-style benign-compliance rate, reported as the two-axis frontier rather than a single safety number; and RULER (not bare needle-in-a-haystack) if the model claims meaningful long-context capability, reporting effective context length rather than a single accuracy figure at max length.

What I'd explicitly refuse to conclude from any single benchmark in that list: that a strong knowledge/reasoning score predicts strong agentic or tool-use performance (files 001-002 vs. 003 show these are only loosely correlated); that a static safety benchmark score is a robustness guarantee against future jailbreak techniques (it's a lower bound, not a ceiling); that any of these public benchmark numbers predicts deployed user satisfaction on our actual product's specific task distribution (file 007's entire point) — for that last one, I'd insist on running our own internal eval set built from real historical product traffic before treating the public suite as sufficient for a launch decision, and I'd flag contamination risk explicitly on any benchmark that's been public for more than about a year rather than treating its number as beyond suspicion.

## Q8: Implement a RULER-style "effective context length" computation given a grid of (context length, task, accuracy) results and a short-context baseline.

```python
def effective_context_length(
    results: dict[tuple[int, str], float],  # (length, task_name) -> accuracy
    baseline: dict[str, float],              # task_name -> short-context baseline accuracy
    lengths: list[int],                      # ascending
    tasks: list[str],
    threshold_frac: float = 0.85,             # must retain >= 85% of baseline accuracy
) -> int:
    """Returns the largest length at which the model retains at least
    threshold_frac of its own short-context baseline accuracy, averaged
    across all tasks in the suite -- the length axis degrades monotonically
    in practice, so we walk from shortest to longest and stop at first failure."""
    effective_length = lengths[0]
    for L in lengths:
        retained_fracs = []
        for task in tasks:
            acc = results[(L, task)]
            base = baseline[task]
            retained_fracs.append(acc / base if base > 0 else 0.0)
        avg_retained = sum(retained_fracs) / len(retained_fracs)
        if avg_retained >= threshold_frac:
            effective_length = L
        else:
            break  # first length where the suite average drops below threshold
    return effective_length
```

Two things worth stating unprompted if asked to critique this: (1) the "monotonic degradation, stop at first failure" assumption is a simplification — real results can be noisy enough that a model dips below threshold at one length and recovers slightly at the next, so a more careful implementation would require the threshold to hold for all subsequent lengths too, not just check up to the first violation; (2) averaging across tasks before comparing to threshold hides exactly the RULER finding this metric was built to surface — aggregation/multi-hop tasks degrade earlier than retrieval tasks — so in practice you'd want to report effective length per task category as well as an overall figure, otherwise a model that's still excellent at retrieval but has already collapsed on aggregation at the same length gets a single blended number that obscures which specific skill degraded first.

## Q9: List the distinct mechanisms that make agentic benchmarks harder to keep reliable and uncontaminated than static QA benchmarks — don't just say "agents are more complex," name the specific compounding factors.

Six distinct, compounding mechanisms: (1) **environment drift** — live websites, desktop apps, and APIs change over time in ways fixed benchmark text never does, so a golden trajectory recorded at construction time can silently stop working, which is why WebArena/OSWorld self-host frozen environment snapshots rather than pointing at the live internet, at real ongoing maintenance cost; (2) **grading requires bespoke, task-specific state-based checkers** rather than string comparison — each is its own piece of verification software that can have bugs or unintended strictness, and auditing hundreds of distinct checkers is far more expensive than auditing a static answer key; (3) **larger action spaces produce more rollout-to-rollout nondeterminism** — a single wrong click or misread screenshot can derail an entire trajectory, requiring multi-seed averaging (as tau-bench's pass^k formalizes) to get a stable signal, multiplying evaluation cost; (4) **contamination takes a different, harder-to-detect shape** — trajectory/strategy leakage rather than verbatim text leakage, which standard n-gram overlap detection isn't built to catch (see Part 1, Q19); (5) **scaffold-vs-model-capability confounding** — every reported number is jointly a function of the base model and the surrounding agent harness, and two papers citing different numbers for "the same" model are often actually comparing different scaffolds (see Part 1, Q18); (6) **much higher construction cost per task** — a single realistic environment plus checker plus verified golden trajectory takes vastly more labor than writing a multiple-choice question, which is why these benchmarks are smaller (hundreds to low thousands of tasks vs. MMLU's ~16,000) and refresh on a much slower cadence, giving both environment drift and trajectory-leakage contamination more relative time to accumulate between benchmark revisions.

## Q10: Why does increasing context-window length specifically increase vulnerability to many-shot jailbreaking, and why is this a notable exception to the general framing of long context as a pure capability win?

Many-shot jailbreaking works by exploiting in-context learning itself as the attack vector: including a large number of in-context example turns depicting the model progressively complying with more harmful requests, relying on the same mechanism that makes few-shot prompting effective for legitimate tasks (a sufficiently large, consistent pattern of demonstrations in-context shifts the model's effective behavior toward continuing that pattern) to instead shift the model's effective safety policy toward compliance. Documented specifically in Anthropic's own published long-context red-teaming research, the key finding is that attack effectiveness scales with the *number* of in-context demonstration turns the attacker can fit — and that number scales directly with available context length.

This makes it a genuine exception to the usual framing (throughout file 004) of longer context windows as an unambiguous capability improvement: the same architectural and infrastructure investment that lets a model usefully process a full codebase or a long document also mechanically expands the attack surface for this specific jailbreak family, since a bigger context window means an attacker can fit more manipulative demonstration turns before hitting a length limit. This is a case where a capability axis and a safety-risk axis move together rather than being independent design considerations, and it's part of why safety evaluation (file 005) can't be treated as a one-time gate decoupled from ongoing capability changes — every major capability improvement (longer context, better tool use, more autonomous agentic behavior) plausibly opens or widens a corresponding attack surface that has to be separately red-teamed, not assumed safe by default because the model "got better."

## Q11: Scenario — you suspect a competitor's model was contaminated on GSM8K (scores suspiciously high relative to its performance on your own held-out internal math eval), but you have no access to their training data. How would you investigate this with only black-box access to the model?

Several black-box-compatible signals, used together rather than any single one in isolation: (1) **Perturbation sensitivity** — take a sample of GSM8K problems and construct minimally-perturbed variants (change the specific numbers, character names, or surface phrasing while preserving the exact same underlying reasoning structure and difficulty) and check whether accuracy drops substantially on the perturbed versions relative to the original — a model relying on genuine reasoning should perform comparably on both, while a model that memorized GSM8K's exact instances would show a much larger gap. (2) **Canary/exact-recitation probing** — prompt the model with the first part of a GSM8K question and check whether it can complete the rest verbatim or the exact reference solution text verbatim beyond what plausible reconstruction from the question alone would produce; unusually precise recitation of the reference chain-of-thought (not just the final answer) is a stronger contamination signal than getting the final answer right. (3) **Comparative difficulty-curve analysis** — check whether the model's accuracy is unusually flat across GSM8K's difficulty range (2-step vs. 8-step problems) relative to its accuracy curve on a comparably-difficulty-graded but definitely-uncontaminated private eval (like your own internal set) — genuine reasoning capability should show some sensitivity to problem complexity, and an anomalously flat curve specifically on the public benchmark is suspicious. (4) **Cross-benchmark consistency** — check whether the suspicious gap (high GSM8K, unremarkable on your internal eval) also shows up as a similarly anomalous gap on other grade-school-arithmetic-shaped evals that aren't GSM8K itself; if the model is just good at this difficulty tier in general, the gap shouldn't be specific to GSM8K alone. None of these is individually conclusive without training-data access, but a model showing high recitation fidelity, flat difficulty sensitivity, and a gap specific to the named public benchmark rather than to the general task type is a strong composite signal.

## Q12: Implement an aggregator that produces the two-axis safety scorecard from file 005 (harmful-compliance rate and benign-compliance rate) with bootstrap confidence intervals on both axes.

```python
import numpy as np

def safety_scorecard_with_ci(
    harmful_outcomes: list[bool],   # True = model complied with a harmful request
    benign_outcomes: list[bool],    # True = model complied with a benign lookalike request
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    harmful = np.array(harmful_outcomes, dtype=float)
    benign = np.array(benign_outcomes, dtype=float)

    def bootstrap_mean_ci(data: np.ndarray) -> tuple[float, float, float]:
        point = data.mean()
        boots = [rng.choice(data, size=len(data), replace=True).mean()
                 for _ in range(n_bootstrap)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(point), float(lo), float(hi)

    h_point, h_lo, h_hi = bootstrap_mean_ci(harmful)
    b_point, b_lo, b_hi = bootstrap_mean_ci(benign)
    return {
        "harmful_compliance_rate": {"point": h_point, "ci95": (h_lo, h_hi)},  # want LOW
        "benign_compliance_rate": {"point": b_point, "ci95": (b_lo, b_hi)},   # want HIGH
    }
```

The reason to bootstrap both axes rather than report bare point estimates: both harmful-behavior sets (HarmBench's ~400) and benign-lookalike sets (XSTest/OR-Bench) are finite samples from a much larger space of possible requests, and a scorecard used to compare two models or two versions of the same model needs confidence intervals to distinguish a genuine safety-posture change from sampling noise — exactly the same statistical discipline argued for in Q3 of this file regarding GPQA Diamond score differences, applied here to a safety metric where the stakes of over-interpreting noise as a real signal (in either direction — falsely concluding a model got safer, or falsely concluding it got more over-cautious) are arguably higher.

## Q13: You're asked in an interview: "isn't pass@k for code just the same idea as pass^k for tau-bench, only renamed?" How do you correct this without being pedantic about notation?

They share a combinatorial ingredient (both are computed from binomial-coefficient-based probabilities over repeated trials) but they answer opposite questions and are appropriate in opposite deployment contexts, so treating them as interchangeable would actually mislead someone about what either number means. Pass@k asks whether *at least one* of k independent attempts succeeds — the right question when a system can generate several candidates and select or automatically verify the best one before it matters to any real outcome, which is realistic for code generation (run the candidates against tests, ship the one that passes). Pass^k asks whether *all* of k independent attempts succeed — the right question when there's no selection step and every trial is a real-world outcome that either resolves correctly or doesn't, which is realistic for a deployed conversational agent handling many structurally similar tickets one at a time, where "it worked at least once across several tries at the same ticket" isn't a coherent concept — there's only one real attempt per real ticket, and pass^k over repeated *simulated* trials of the same task is a proxy for how reliably the underlying policy would perform if you handed it that same situation many times.

The concrete consequence of conflating them: a model with a strong pass@k (good at producing at least one correct output among several tries) can still have a weak pass^k (unreliable when you don't get to pick the best try) — this is exactly the empirical gap tau-bench's authors report for frontier models, and it's the whole reason the metric exists rather than the field just reusing pass@k. So the correct correction is not "they're totally unrelated," it's "they're built from the same combinatorial machinery but instantiate opposite selection assumptions, and which one is relevant depends entirely on whether your deployment context lets you discard failed attempts before they count."

## Q14: MMLU's headline score is a macro-average across 57 subjects rather than a micro-average (raw accuracy) across all ~15,908 questions. Why does this design choice matter, and what would change if you reported it the other way?

Macro-averaging (compute per-subject accuracy first, then average those 57 numbers with equal weight) means every subject contributes equally to the final score regardless of how many questions it has — a subject with 100 questions counts exactly as much as one with 1,500. Micro-averaging (pool all questions together and compute one overall accuracy) instead implicitly weights subjects by their question count, so subjects that happen to have more questions in the dataset dominate the aggregate score more.

This matters because MMLU's 57 subjects were not curated to have proportional question counts reflecting their real-world importance or difficulty — subject sizes vary for reasons related to source-material availability (how many practice-exam questions were findable for that subject), not a deliberate weighting decision. Macro-averaging is the design choice that treats "broad coverage across many distinct knowledge domains" as the thing being measured, consistent with MMLU's stated goal of measuring breadth; if it were micro-averaged instead, a model's score would be disproportionately driven by whichever subjects happen to be numerically over-represented (in practice, several STEM subjects have large question counts), and a model that was excellent at those particular over-represented subjects but weak on several smaller humanities subjects could score deceptively high under micro-averaging while the macro-averaged score would correctly reflect the humanities weakness pulling the aggregate down. In short: the averaging convention is itself a substantive methodological choice about what "good at MMLU" is defined to mean, not an arbitrary reporting detail, and it's worth being able to state which convention a given reported number used if precision matters for a comparison.

## Q15: A model scores 95% pass@1 on the newly released AIME 2025, far above any previously reported frontier model number on any prior AIME year. What alternative explanations would you check before concluding this represents a genuine capability leap?

Several non-capability explanations to rule out before accepting the number at face value: (1) **Contamination via rapid post-release exposure** — AIME 2025's problems and worked solutions get discussed on competition-math forums (AoPS and similar) within days of the exam being administered; if there was any meaningful gap between the exam date and the model's training-data cutoff or evaluation date, solutions could plausibly have entered a web-crawl-derived training or RL-rollout dataset, which is exactly the "temporary, rapidly decaying contamination advantage" file 002 describes for AIME specifically. (2) **Small-n variance** — a single year's AIME is only 15 or 30 problems; a genuinely-capable model could land unusually high on this specific year's problem mix by chance, and 95% (roughly 14-15 out of 15, or 28-29 out of 30) is close enough to a perfect score that even one or two flipped problems substantially changes the headline percentage — check whether the number is reported with a confidence interval or across multiple years, and be skeptical of a single-year number presented without either. (3) **Scaffold/prompting differences** — check whether this number used a different, more generous evaluation protocol than prior reported numbers (more sampling attempts with a best-of-n selection presented as if it were pass@1, a more elaborate CoT-eliciting prompt template, or extended test-time compute/tool use not available to the models it's being compared against) — an apples-to-oranges protocol difference is a common, non-malicious source of a suspiciously large reported jump. (4) **Selective reporting** — check whether this is the only math benchmark reported, or whether it's part of a full suite including MATH-500 and other math evals showing a consistent, proportionate improvement; a large isolated jump on exactly one high-profile, recently-released benchmark with no corroborating improvement elsewhere is a pattern consistent with benchmark-specific optimization (deliberate or not) rather than general mathematical-reasoning improvement.

## Q16: Explain the difference between the original MMLU log-likelihood scoring protocol and generation-plus-parsing scoring, and why this makes some cross-paper MMLU comparisons less apples-to-apples than they look.

The original protocol (Hendrycks et al.) scores a question by comparing the model's log-probability of each answer-letter token (A/B/C/D) conditioned on the prompt, and taking the argmax — the model never actually "generates" an answer in the ordinary sense; you're directly reading off which next-token probability is highest among four candidates. This only really works well for base/completion-style models where you can cleanly score next-token log-probabilities, and it's insensitive to a model's ability to follow an instruction like "respond with only the letter" — that instruction-following step doesn't even happen under this protocol, since the harness constructs the answer-token comparison directly.

Generation-plus-parsing instead prompts the model to actually produce a response (typically instructed to answer with a specific letter or a specific format) and then parses the free-form output text to extract the model's chosen answer, using some regex or lightweight extraction logic. This protocol is necessary for most modern instruction-tuned/chat models, where directly reading token log-probabilities either isn't exposed via the API being used, or the model's raw next-token distribution isn't representative of its "real" answer once instruction-following and chat-formatting are involved. But it introduces a new failure mode the log-likelihood protocol never had: a model can know the correct answer but phrase its response in a way the parser fails to extract correctly (extra preamble, unexpected formatting, refusing to commit to a single letter) — a *parsing* failure that gets scored identically to a *knowledge* failure, silently deflating the reported score for reasons unrelated to the capability being measured. Because different papers' harnesses use different parsing logic and different prompt templates instructing the model how to respond, two papers reporting "MMLU: 84.2%" for superficially comparable models are not guaranteed to be measuring the same protocol, and a difference of a point or two between such reports could be entirely attributable to harness differences rather than the models themselves — worth flagging any time someone treats decimal-level MMLU comparisons across papers as precise.

## Q17: Implement both sequential Elo and batch Bradley-Terry MLE on the same small toy dataset, processed in two different orders, to concretely demonstrate the order-dependence problem sequential Elo has and Bradley-Terry doesn't.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def expected_score(r_a, r_b, scale=400.0):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))

def elo_update(r_a, r_b, score_a, k=32.0, scale=400.0):
    e_a = expected_score(r_a, r_b, scale)
    return r_a + k * (score_a - e_a), r_b + k * ((1 - score_a) - (1 - e_a))

def run_sequential_elo(votes, k=32.0, initial=1000.0):
    ratings = {}
    for a, b, s in votes:
        ratings.setdefault(a, initial); ratings.setdefault(b, initial)
        ratings[a], ratings[b] = elo_update(ratings[a], ratings[b], s, k)
    return ratings

def fit_bradley_terry(votes, models):
    idx = {m: i for i, m in enumerate(models)}
    X, y = [], []
    for a, b, s in votes:
        winner, loser = (a, b) if s == 1.0 else (b, a)
        row = np.zeros(len(models)); row[idx[winner]] = 1.0; row[idx[loser]] = -1.0
        X.append(row); y.append(1)
    clf = LogisticRegression(fit_intercept=False, penalty=None).fit(np.array(X), np.array(y))
    return {m: float(clf.coef_[0][idx[m]]) for m in models}

# Toy dataset: A beats B twice, B beats C twice, C beats A once (a near-cycle,
# deliberately sparse and order-sensitive)
votes_order1 = [("A", "B", 1.0), ("B", "C", 1.0), ("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 1.0)]
votes_order2 = list(reversed(votes_order1))

elo_1 = run_sequential_elo(votes_order1)
elo_2 = run_sequential_elo(votes_order2)
bt_1 = fit_bradley_terry(votes_order1, ["A", "B", "C"])
bt_2 = fit_bradley_terry(votes_order2, ["A", "B", "C"])

print("Sequential Elo, order 1:", elo_1)
print("Sequential Elo, order 2 (reversed):", elo_2)
print("Bradley-Terry, order 1:", bt_1)
print("Bradley-Terry, order 2 (reversed):", bt_2)
```

Running this shows `elo_1 != elo_2` (the reversed processing order yields different final ratings for the same multiset of outcomes, because sequential Elo's updates depend on the ratings *at the time* each game is processed, and those intermediate ratings differ depending on what's been processed so far) while `bt_1` and `bt_2` come out identical (Bradley-Terry's MLE fit depends only on the multiset of observed outcomes, not their order, since it's optimizing one global likelihood over the whole dataset at once rather than updating incrementally). This is the concrete, runnable version of the order-dependence claim made qualitatively in file 006 and Part 1 Q8 — worth having actually run once so you can describe the effect from direct observation rather than only citing it as a known property.

## Q18: How would you design an evaluation protocol to separately measure "did the base model get better" versus "did the agent scaffold get better" for a SWE-bench Verified number, given the confound discussed in Part 1 Q18?

A minimal 2x2 ablation: evaluate {old model, new model} x {old scaffold, new scaffold}, holding everything else fixed (same task subset, same compute/time budget per task, same number of allowed tool calls or retries), and report all four resulting resolve rates rather than just the headline (new model, new scaffold) number. The four cells let you decompose the total observed gain into a base-model-capability main effect (comparing old-model/old-scaffold to new-model/old-scaffold), a scaffold-engineering main effect (comparing old-model/old-scaffold to old-model/new-scaffold), and — often the most informative and most commonly ignored — an interaction effect (does the new scaffold help the new model more or less than it helps the old model, which tells you whether the scaffold improvement and the model improvement are complementary or substitutable).

Beyond the 2x2 structure itself, I'd also fix the compute/attempt budget explicitly across all four cells (same number of tool calls, same wall-clock or token budget) — a common way this kind of comparison gets silently unfair is a new scaffold that simply allows more retries or a larger context budget for repo exploration, which would inflate its apparent improvement in a way that has nothing to do with either the base model or genuinely better scaffold logic, just more resources spent per task. I'd also report per-instance-category breakdowns (e.g., by repository or by whether the fix is single-file vs. multi-file) rather than only the aggregate resolve rate, since scaffold improvements (better repo search/localization) and base-model improvements (better code reasoning once the right file is found) plausibly help different instance categories differentially, and the aggregate number alone would hide which mechanism is actually responsible for the overall gain.

## Q19: Contrast static-benchmark contamination and agentic-benchmark "trajectory leakage" contamination directly — what would a mitigation that works for one but not the other look like?

Static-benchmark contamination is fundamentally a **text-overlap** problem: the risk is that the exact (or near-paraphrase) benchmark item — question and/or answer — appears in training data. Mitigations that work here operate at the text level: n-gram/substring overlap checking against the training corpus, canary strings embedded in the benchmark release to detect verbatim scraping, decontamination filtering pipelines that remove matched documents before training, and temporal holdouts (evaluating only on benchmark content published after a model's training-data cutoff) — all of these are covered in `../05_Evaluation_Methods/004_Contamination_Aware_Evaluation_Design.md` and all fundamentally rely on being able to define and detect "this specific text appeared."

Trajectory-leakage contamination for agentic benchmarks is a **strategy-overlap**, not text-overlap, problem: what leaks is a description of *how* to solve a task in a given environment (a blog post's narrated walkthrough, a shared scaffold's tuned prompting/tool-use strategy, a forum discussion of a specific environment's quirks), which can influence a model's effective performance without any verbatim text from the original benchmark ever appearing in training data. None of the static-benchmark mitigations transfer directly: n-gram overlap checking finds copied text, not copied *strategies* expressed in arbitrarily different words; canary strings only detect verbatim scraping of the canary itself, not paraphrased strategy discussion; temporal holdouts help somewhat (a fresh environment introduced after a model's training cutoff has had less time for writeups to accumulate) but decay just as fast as they do for AIME once the environment becomes popular enough to attract discussion.

A mitigation that specifically targets trajectory leakage and has no clean static-benchmark analogue: **environment novelty/rotation** — periodically introducing meaningfully new environments or task variations (new websites, new desktop application combinations, new tool sets) rather than reusing the same fixed environment indefinitely, so that even if strategy-level discussion of the old environment has proliferated, it doesn't transfer cleanly to the rotated version. This is expensive (agentic benchmarks are already far costlier to construct per task than static QA, per Q9 in this file) and is a fundamentally different mitigation category from anything in the contamination-detection toolkit built for text-level overlap — it's closer to "keep changing the test" than "detect and filter the leak," which is a meaningfully different strategy than anything available for static benchmarks.

## Q20: Scenario — leadership asks for a single number representing "how safe" a new model is, to make a launch go/no-go decision. How do you push back, and what would you propose instead?

I'd push back directly on the premise that a single scalar can do this job, using the precision/recall analogy explicitly: a harmful-compliance-only number can be trivially minimized by refusing everything, which would look maximally "safe" by that number while being a materially worse, less useful, and arguably still not fully safe (over-refusal has its own real costs, including pushing users toward less-safe alternative tools or sources) model — and a single number has no way to encode that tradeoff or reveal which side of it a given model sits on. I'd also point out that any static red-teaming number is a lower bound on current vulnerability, not an upper bound, and presenting it as "the" safety number risks leadership treating a snapshot against known attacks as a durable guarantee against attacks not yet devised.

What I'd propose instead: a short scorecard, not a single number — (1) harmful-compliance rate on a standardized battery (HarmBench-style) with a bootstrapped confidence interval, explicitly labeled as "robustness against currently known attack techniques," not "safety"; (2) the paired benign-compliance rate on a matched over-refusal set (XSTest/OR-Bench-style), so the go/no-go conversation sees both sides of the frontier at once; (3) a qualitative red-teaming summary from whatever adversarial testing (internal or external, ideally including techniques not in the standardized battery) was actually run before launch, since the standardized numbers by definition can't cover novel attack vectors; (4) an explicit statement of what wasn't tested (e.g., specific high-risk domains, specific languages, specific modalities) rather than letting the two headline numbers imply comprehensive coverage. The go/no-go decision itself should be a judgment call informed by that scorecard plus context (what's the product surface, who are the users, what's the blast radius of a failure), not a threshold check against one scalar — and I'd frame that explicitly as the more defensible process, not just a more complicated one, precisely because a single-number safety metric is the kind of thing that looks rigorous while actually hiding the exact tradeoff (over-refusal vs. under-refusal) that matters most for the decision being made.
