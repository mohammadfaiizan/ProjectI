# Interview Questions — Part 1

## Q1: What does MMLU actually measure, and what are its two most well-documented, verified (not speculative) weaknesses?

MMLU (Hendrycks et al., 2021) measures breadth of factual/procedural knowledge across 57 subject areas via 4-option multiple-choice questions sourced from real exams and practice tests, scored either via log-likelihood comparison over the four answer-letter tokens (the original protocol) or via generation-plus-parsing (the common modern protocol for chat models) and aggregated as a macro-average across the 57 subjects (each subject counts equally regardless of how many questions it contributes).

Two weaknesses are documented by follow-up empirical work, not just asserted as generic multiple-choice critiques:

1. **Verified ground-truth label errors.** MMLU-Redux (Gema et al., 2024) manually re-annotated a stratified sample across all 57 subjects and found meaningful, subject-dependent error rates — some subjects (their headline example is virology) have error rates high enough that a real fraction of a model's "wrong" answers are actually correct answers scored against a bad label. This means any reported MMLU accuracy has a subject-dependent noise floor from the labels themselves, independent of model capability, which is large enough to move close model comparisons.
2. **Answer-position sensitivity.** Multiple studies have shown that shuffling which letter (A/B/C/D) holds the correct answer measurably changes accuracy on the same underlying question, and some models show a detectable prior bias toward specific letters when uncertain — evidence that log-likelihood-over-letters scoring is picking up some surface-form pattern-matching rather than purely content-driven reasoning, a confound the standard protocol does not control for.

Beyond these two, MMLU is also functionally saturated for frontier models (clustering in the high 80s, close to the paper's own ~89.8% estimated human-expert baseline), which compounds both weaknesses: once models are within a couple of points of each other, label noise and position sensitivity can dominate the remaining variance, making MMLU low-information for ranking current frontier models specifically.

## Q2: MMLU-Pro was introduced as a successor to MMLU. What specifically changed, and what is the single most important empirical result the MMLU-Pro paper reports to justify the redesign?

Three concrete changes: (1) answer options expanded from 4 to 10, dropping the random-guess floor from 25% to 10% and reducing the benefit of elimination-based guessing; (2) a filtering pipeline removed questions that weaker models answered correctly with high consistency across prompt perturbations (a proxy for "too easy" or "too pattern-matchable"), replaced/supplemented with more reasoning-intensive questions from harder sources; (3) a human-verification pass specifically targeting the kind of label-error problem MMLU-Redux later formalized for the original MMLU.

The single most important empirical result is the **chain-of-thought sensitivity gap**: removing CoT prompting (forcing a direct answer with no reasoning) costs models far more accuracy on MMLU-Pro (reported on the order of 16-33 percentage points across tested models) than on original MMLU. That's the actual evidence the redesign worked as intended — MMLU could largely be solved by pattern-matching/recall, while MMLU-Pro's questions are constructed so that skipping explicit multi-step reasoning has a real, measurable cost, meaning the benchmark is measuring something closer to reasoning-dependent problem solving rather than fact recognition.

## Q3: Explain precisely how GPQA earns the "Google-proof" claim in its name. What was actually measured, and what does the measurement not establish?

GPQA's validation methodology is the specific mechanism behind the claim, not just marketing: questions written by PhD-level subject-matter experts were given to **skilled non-expert validators** — people with strong general research ability and unrestricted internet access, but without graduate training in that specific subfield — who were given real time (30+ minutes) to search the web and answer. These validators averaged around 34% accuracy, barely above the 25% random-guess floor for 4-option multiple choice, meaning unrestricted web search gave them very little traction. Domain PhD experts answering in their own subfield, by contrast, averaged roughly 65-74%. The gap between those two numbers is the entire empirical basis for "Google-proof": it demonstrates resistance to a skilled non-expert doing ordinary web lookup *at the time of validation*.

What it does not establish: (1) resistance to a model that was pretrained on a crawl including the published answer key itself — the validation only tested human search behavior, not model memorization exposure, and once GPQA's questions and answers are public, that specific protection erodes for any model trained afterward; (2) permanence — "Google-proof" was measured once, against the search tools and indexed content available at that time, not as a durable property; (3) freedom from label errors — expert-authored doesn't guarantee expert-verified-by-independent-audit at MMLU-Redux's level of scrutiny, and graduate-level questions are exactly the kind where subtle errors are hardest to catch even by a second expert.

## Q4: Scenario — a new model tops the MMLU leaderboard on release, but within weeks users are complaining it's noticeably worse than the previous version for their actual work. Walk through how you'd investigate this.

First, resist treating "worse" as monolithic — pull a sample of actual complaints and categorize the failure mode: is it a formatting/instruction-following regression, a tone/verbosity shift, increased confident hallucination, a regression on a specific task category, or something else entirely? MMLU is multiple-choice knowledge recall; it has no mechanism to detect any of these failure modes, so a leaderboard win there is not evidence against any of them.

Second, check what actually changed in the new model's training pipeline relative to the prior version — a new RLHF/preference-optimization pass, a data-mix change, or new SFT data resembling benchmark-style questions can each explain a benchmark/reality divergence differently. If preference-tuning changed, consider the style/length-gaming dynamic documented for Chatbot Arena (verbose, confidently-formatted responses win preference votes without necessarily being more correct) — a similar dynamic in the new model's own training data could shift output style in ways that read as "worse" even if underlying correctness didn't regress.

Third, run evals that are actually close to the complaint category rather than relying on MMLU to adjudicate something it wasn't built to measure — an instruction-following eval, a domain-specific held-out set resembling your real product traffic, or a multi-turn conversation eval. If you have an internal eval suite built from your own users' historical query distribution, that is far more diagnostic than any public benchmark for a "did this get worse for our users specifically" question.

Fourth, consider contamination/benchmaxxing directly: if the MMLU gain looks disproportionately large relative to gains elsewhere, ask whether training data specifically resembling MMLU's task shape (exam-style questions, practice tests) increased in the new pipeline without a matching general-capability gain — this is the textbook Goodhart's-law explanation for exactly this symptom pattern (see file 007) and should be an explicit hypothesis you check, not just a hand-wave.

## Q5: Implement the unbiased pass@k estimator used for code-generation benchmarks like HumanEval, and explain why you can't just sample exactly k completions and check if any pass.

Sampling exactly k completions per problem and checking for any pass is a valid but high-variance, sample-inefficient estimator — it wastes information (you'd need to resample from scratch for every different k you want to report) and is noisier than necessary given how expensive LLM sampling is. The standard approach instead draws a larger fixed batch of `n >= k` samples once, counts how many `c` pass, and computes the exact probability that a random k-sized subset of those n samples contains at least one pass:

```python
import numpy as np

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: probability a random k-subset of n samples
    (c of which passed) contains at least one passing sample."""
    if n - c < k:
        return 1.0  # fewer than k failures exist, so any k-subset must include a pass
    # 1 - P(all k chosen from the (n-c) failing samples)
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def estimate_pass_at_k(results: list[list[bool]], k: int) -> float:
    """results[i]: pass/fail booleans for the n samples drawn for problem i."""
    per_problem = [pass_at_k(len(r), sum(r), k) for r in results]
    return float(np.mean(per_problem))
```

The product-form computation (`1 - prod(1 - k/i for i in range(n-c+1, n+1))`) is algebraically equivalent to `1 - C(n-c, k)/C(n, k)` but avoids computing large binomial coefficients directly, which can overflow or lose precision for realistic n (e.g., n=200). The key practical benefit of this estimator: draw one batch of n=200 samples per problem, and you can retroactively estimate pass@1, pass@10, and pass@100 from that same batch without ever redrawing samples — each `k` just changes which subset-probability you're computing over the same fixed observed data.

## Q6: What is the actual selection criterion behind BIG-Bench-Hard, and what was the paper's central empirical finding?

BBH (Suzgun et al., 2022) is not a hand-picked "seems hard" subset — it uses a mechanical selection rule: of BIG-Bench's 200+ original tasks, BBH keeps the 23 where the best available model performance at curation time fell *below* the average human rater performance reported in the original BIG-Bench paper, isolating the specific tasks with a documented, measured human-model gap rather than tasks chosen by intuition. The resulting 23 tasks skew algorithmic and multi-step-reasoning-heavy (Boolean expression evaluation, object tracking, logical deduction, Dyck-language bracket matching, multi-step arithmetic, date understanding) because that's where the original suite's gaps concentrated.

The central empirical finding is that few-shot chain-of-thought prompting closes most or all of the human-model gap on 17 of the 23 tasks, while standard few-shot prompting without explicit reasoning traces does not — at the time, this was some of the clearest task-level evidence that CoT specifically helps with multi-step, compositional reasoning rather than being a generically helpful trick, since the tasks were selected precisely for requiring that kind of reasoning. As with MMLU, BBH is now largely saturated for frontier models (often 90%+ under CoT prompting), and its synthetic/algorithmic task shape makes it more susceptible than naturalistic benchmarks to targeted exposure via structurally similar training data.

## Q7: Why is SWE-bench a meaningfully harder and different evaluation than HumanEval or MBPP, and what specifically did SWE-bench Verified fix about the original?

HumanEval/MBPP give a clean docstring and ask for a single self-contained function; SWE-bench gives a real GitHub issue and an entire existing repository (potentially hundreds of thousands of lines across many files) and asks for a patch that resolves the issue without breaking anything else. This forces capabilities function-level benchmarks abstract away entirely: localization (finding the relevant code with no docstring pointing you at it), whole-repository context integration (a fix has to respect existing conventions and invariants, not just satisfy a local spec), sometimes multi-file edits, and surviving the project's actual test infrastructure rather than a handful of purpose-written unit tests. Grading is via the real PR's `FAIL_TO_PASS` tests (must now pass) and `PASS_TO_PASS` tests (must not regress) — a much closer proxy for real software-engineering work than isolated function synthesis.

SWE-bench Verified is a 500-instance, human-annotated (professional software engineers) subset of the original 2,294 instances, specifically screening out: issues underspecified relative to what the reference PR actually did (so a differently-but-validly-fixed patch would be unfairly scored as failing); instances where passing the tests essentially required reproducing the exact reference implementation rather than any correct fix; broken or flaky execution environments that would fail even a correct patch; and issue text lacking information the reference PR's author had access to but that wasn't captured in the mined text. This doesn't change the task design, it removes noisy/unfair instances so that a failing score is much more likely to reflect a genuine model shortcoming rather than benchmark construction noise — directly analogous to what MMLU-Redux did for MMLU, but done proactively as curation rather than as a post-hoc audit.

## Q8: Implement the classic sequential Elo rating update from a stream of pairwise match outcomes, exactly as it would apply to anonymized pairwise votes in a Chatbot-Arena-style setting.

```python
def expected_score(r_a: float, r_b: float, scale: float = 400.0) -> float:
    """Probability A beats B under the logistic Elo model."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))

def elo_update(r_a: float, r_b: float, score_a: float,
                k: float = 32.0, scale: float = 400.0) -> tuple[float, float]:
    """score_a: 1.0 if A won the vote, 0.0 if A lost, 0.5 for a tie."""
    e_a = expected_score(r_a, r_b, scale)
    e_b = 1.0 - e_a
    score_b = 1.0 - score_a
    return r_a + k * (score_a - e_a), r_b + k * (score_b - e_b)

def run_sequential_elo(votes: list[tuple[str, str, float]],
                        k: float = 32.0, initial_rating: float = 1000.0) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for model_a, model_b, score_a in votes:
        ratings.setdefault(model_a, initial_rating)
        ratings.setdefault(model_b, initial_rating)
        ratings[model_a], ratings[model_b] = elo_update(
            ratings[model_a], ratings[model_b], score_a, k
        )
    return ratings
```

The update is proportional to the *surprise* of the outcome: if A was expected to win with probability 0.7 and did win, the update is small (`k * 0.3`); if A was expected to win with probability 0.7 and lost, the update is large and negative (`k * (0 - 0.7)`). This is exactly why Elo is a self-correcting online estimator that needs no global optimization — each update only uses the single just-observed game. Its key weakness, worth stating unprompted: it's order-dependent (the same set of games processed in a different order can yield different final ratings, especially early on with sparse data), which is precisely why LMSYS's actual production Chatbot Arena leaderboard does not use this sequential update — it fits a Bradley-Terry model via maximum likelihood over the whole vote history at once, which is order-invariant and supports proper bootstrap confidence intervals (see Q16 in Part 2 for the batch fit).

## Q9: Walk through the scoring-mechanics differences across GSM8K, MATH, and AIME, and explain exactly why the field escalated through all three rather than just making GSM8K's problems harder in place.

GSM8K answers are single plain integers/decimals extracted from a `####`-delimited final line and compared via simple exact match — trivial to grade automatically, but the format itself caps the achievable difficulty (grade-school arithmetic composed over a few steps), and frontier models saturated it (>95%) by around 2023. MATH's answers are LaTeX-boxed and can be mathematically equivalent while textually very different (`1/2` vs `0.5` vs `\frac{1}{2}`; `2\sqrt{3}` vs `\sqrt{12}`), so scoring requires a normalization pipeline plus a symbolic-equivalence fallback (e.g., via sympy) rather than plain string matching — a genuinely harder grading problem, and one where different eval harnesses' normalization rules produce non-trivially different reported numbers for the same model. AIME answers are, by the competition's own original design (for unambiguous human grading), bare integers in [0, 999] — as easy to grade as GSM8K, but the *problems* are genuinely competition-hard, calibrated to challenge top US high-school competitors.

The escalation wasn't "make GSM8K's problems harder in place" because GSM8K's whole task shape (grade-school word problems) has a natural difficulty ceiling — you can't make grade-school arithmetic into competition mathematics without changing the entire problem genre. MATH escalated along the *content-difficulty* axis while accepting a harder grading problem as the cost. AIME escalated along a different axis entirely: rather than engineering a new problem set, the field imported an external, real, already-existing hard-problem source specifically because its yearly cycle offers a (temporary, decaying) contamination advantage that a static benchmark like MATH — whose problems have been public and discussed since 2021 — cannot replicate. Frontier reasoning models jumping from single digits to ~80% pass@1 on AIME 2024 is the most-cited piece of evidence that RL-driven long-chain-of-thought training produces genuine gains on hard multi-step math reasoning specifically, which is exactly the capability GSM8K and MATH could no longer differentiate.

## Q10: For each of WebArena, OSWorld, tau-bench, and GAIA, state the specific capability gap it was built to expose, in one or two sentences each.

**WebArena** exposes the gap between answering questions about web content and actually operating a realistic, stateful website (clicking, form-filling, multi-page flows) toward a multi-step goal — GPT-4-based agents scored around 14% at introduction despite GPT-4 looking strong on knowledge benchmarks at the same time, showing these are only loosely correlated skills.

**OSWorld** extends the same idea to an entire desktop OS and real, unmodified applications (LibreOffice, GIMP, a terminal), adding pixel-level/accessibility-tree GUI grounding as a distinct sub-problem beyond WebArena's more structured DOM — human performance around 72% versus early agent baselines under 15% is one of the largest human-model gaps in this document.

**tau-bench** targets multi-turn, policy-constrained customer-service-style interaction among an agent, a simulated user, and backend tools, specifically probing whether an agent can follow written business-policy constraints under a realistically underspecified user request — and, via its pass^k metric, whether the agent succeeds *reliably* across repeated independent trials of the same task, not just once.

**GAIA** targets multi-step tool orchestration (web search, code execution, document/file parsing) while keeping the cheap grading of static QA by designing final answers to be short and exact-match-checkable despite requiring a genuinely agentic, multi-step derivation path — reported human accuracy around 92% versus early GPT-4-plus-tools agents in the 15-30% range shows the same knowledge-vs-orchestration gap as the other three, just measured with factoid-QA-style scoring instead of environment-state checking.

## Q11: Implement a needle-in-a-haystack evaluation harness: build a haystack of a target token length with a needle sentence inserted at a controllable depth, then evaluate retrieval accuracy across a grid of lengths and depths.

```python
def build_haystack(filler_docs: list[str], target_length_tokens: int, tokenizer) -> str:
    text = ""
    for doc in filler_docs:
        text += doc + "\n"
        if tokenizer.count_tokens(text) >= target_length_tokens:
            break
    return tokenizer.truncate_to_tokens(text, target_length_tokens)

def insert_needle(haystack: str, needle: str, depth_pct: float, tokenizer) -> str:
    tokens = tokenizer.encode(haystack)
    insert_at = int(len(tokens) * depth_pct / 100)
    new_tokens = tokens[:insert_at] + tokenizer.encode(needle) + tokens[insert_at:]
    return tokenizer.decode(new_tokens)

def run_niah_grid(model, filler_docs, needle: str, question: str, expected_answer: str,
                   lengths: list[int], depths: list[float], tokenizer) -> dict:
    results = {}
    for L in lengths:
        haystack = build_haystack(filler_docs, L, tokenizer)
        for d in depths:
            doc = insert_needle(haystack, needle, d, tokenizer)
            prompt = doc + f"\n\nQuestion: {question}\nAnswer:"
            response = model.generate(prompt)
            results[(L, d)] = expected_answer.lower() in response.lower()
    return results
```

Two design choices matter enough to call out explicitly if asked to extend this: (1) grading via substring match is fragile once the expected answer requires paraphrase-tolerant matching — a production harness typically uses an LLM judge instead, which then inherits general LLM-judge reliability concerns; (2) this only implements the single-needle case — a more informative variant (closer to what RULER does) would insert multiple needles with distinct keys and require retrieving a specified one, or all of them, which stresses discrimination/completeness rather than pure single-fact lookup, and is harder to saturate than the classic single-needle version.

## Q12: What specifically does RULER add beyond needle-in-a-haystack, and why does its "effective context length" metric matter more than a single accuracy number at max context length?

RULER decomposes "long-context ability" into four categories rather than testing one narrow skill: retrieval (single/multi-key/multi-value/multi-query needle variants), multi-hop tracing (a variable-tracking task requiring following a chain of references scattered across the document), aggregation (e.g., identifying the most frequent word across the entire input, which requires integrating the whole context rather than localizing one relevant span), and QA with distractors (real questions answered inside a context padded with substantial irrelevant text). This matters because a model can ace single-needle retrieval via a shortcut strategy ("find the one anomalous sentence, ignore the rest") that gives it no traction whatsoever on an aggregation task, which by construction has no single localized answer span — RULER's task diversity specifically prevents that shortcut from looking like general long-context competence.

The effective-context-length metric — the largest length at which performance on RULER's suite stays within a threshold of the model's own short-context baseline — matters because it directly separates two different claims that get conflated in marketing materials: "accepts N tokens as input" versus "can actually make correct use of N tokens of relevant information." RULER's headline finding is that many models advertising 128K+ context show effective context lengths far shorter than advertised once evaluated on the harder task categories — aggregation and multi-hop tasks degrade at much shorter lengths than retrieval tasks do — meaning a single accuracy number at max advertised length, or worse, a single needle-retrieval number, systematically overstates real long-context capability relative to what RULER's harder categories reveal.

## Q13: Explain the "lost in the middle" phenomenon mechanistically, and connect it explicitly to why needle-in-a-haystack tests are typically plotted with position/depth as an axis at all.

Liu et al. (2023) showed, using multi-document QA and key-value retrieval tasks, that model accuracy at using a piece of relevant information is not uniform across its position in the input — performance is typically highest when the relevant information sits near the very beginning or very end of the context, and measurably worse when it sits in the middle, producing a U-shaped accuracy curve as a function of position. This is plausibly related to how causal self-attention interacts with the length distribution of training data: tokens near the start have accumulated the least competing context to attend over, tokens near the end are what immediately precedes the model's response (so they benefit from recency the way human serial-position effects show recency benefits too), while middle tokens have neither structural advantage — though the precise mechanistic cause is still an active research question rather than something fully settled by that one paper.

This is exactly why needle-in-a-haystack's standard visualization sweeps depth (0-100%) as one axis of its heatmap in the first place — the methodology was designed, from the start, around the expectation that position within the context matters, and the depth-sweep is effectively a direct, if narrow, probe of the same U-shaped effect "lost in the middle" documents more rigorously with naturalistic multi-document tasks. RULER's aggregation and multi-hop tasks go further by requiring integration across many positions simultaneously rather than one localized needle, which should suffer even more severely from a middle-is-weak bias than single-needle retrieval does — meaning RULER's harder categories are, in effect, testing the "lost in the middle" phenomenon under a much less forgiving task structure than either classic NIAH or the original lost-in-the-middle paper's own QA tasks used.

## Q14: Why can't a benchmark that only measures "refusal rate on harmful requests" ever fully characterize a model's safety behavior?

Because refusing everything trivially minimizes harmful compliance while destroying usefulness on everything else — a model that refuses 100% of HarmBench's ~400 harmful behaviors and also refuses a large share of benign requests that merely resemble harmful ones on the surface (e.g., refusing a woodworking question because it mentions "knife") scores identically, on a harmful-only metric, to a model that refuses those same harmful behaviors while correctly answering the benign lookalikes. This is structurally the same problem as reporting only recall without precision in a classification setting: a harmful-refusal-only number can't distinguish a well-calibrated model from an indiscriminately over-cautious one.

The methodologically correct fix is to report a **paired, two-axis result**: harmful-compliance rate (want low) alongside a benign-compliance rate measured on a matched set of superficially-similar-but-actually-benign prompts (XSTest, OR-Bench), treating safety evaluation as a point on a precision/recall-style frontier rather than a single scalar. A model can be moved along that frontier by tuning refusal aggressiveness — more aggressive training pushes harmful-compliance down but typically pushes benign-compliance down too — and reporting only one axis lets a lab present a model as "safe" while omitting that it cost a rise in over-refusal, or as "helpful" while omitting a rise in harmful-compliance. Good safety evaluation reports both numbers side by side (or, where the model exposes a tunable refusal threshold, an actual curve), specifically so that kind of one-sided presentation is visible.

## Q15: Scenario — a lab reports 0% attack success rate on HarmBench for a new model. Is this model safe to deploy? What's missing from that number?

No single static red-teaming number, however good, should be read as a safety guarantee, for two independent reasons. First, 0% ASR is a statement about robustness against the *specific attack battery HarmBench currently tests* (direct requests plus known jailbreak techniques like GCG suffix optimization, PAIR-style iterative refinement, and human-written templates) — it says nothing about attacks not yet invented, and the jailbreak literature has a consistent history of new techniques (multi-turn escalation/"crescendo," encoding obfuscation, many-shot jailbreaking via long context) defeating defenses that were robust against the previously-known attack set. A static benchmark score is a lower bound on current vulnerability, not an upper bound or a certification — the correct operational practice is continuous, open-ended red-teaming, with HarmBench serving as a standardized regression floor (did we at least stay robust to all previously known attacks) rather than a safety ceiling.

Second, 0% ASR on harmful behaviors alone tells you nothing about over-refusal — this could be a well-calibrated model or an indiscriminately over-cautious one, and you cannot tell which from this number alone (see Q14). Before treating this as evidence of deployment readiness, you'd want: the paired benign-compliance rate on an XSTest/OR-Bench-style set; confirmation the compliance classifier used to compute ASR is itself reliable and not systematically missing obfuscated or creatively-phrased harmful compliance; and an honest accounting of whether the model's safety training specifically targeted HarmBench's exact behavior categories in a way that might not generalize to novel harmful requests outside that set (the same benchmark-targeting/Goodhart's-law concern that applies to capability benchmarks applies here, with higher stakes).

## Q16: Implement a Bradley-Terry model fit via logistic regression over a set of pairwise preference votes, the way LMSYS actually computes Chatbot Arena ratings (rather than sequential Elo).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def fit_bradley_terry(votes: list[tuple[str, str, float]], models: list[str]) -> dict[str, float]:
    """votes: (model_a, model_b, score_a) where score_a in {0.0, 0.5, 1.0}.
    Fits latent strengths theta such that P(i beats j) = sigmoid(theta_i - theta_j),
    via logistic regression with a +1/-1 one-hot-difference feature encoding."""
    idx = {m: i for i, m in enumerate(models)}
    X, y, w = [], [], []
    for model_a, model_b, score_a in votes:
        if score_a == 0.5:
            for winner, loser in [(model_a, model_b), (model_b, model_a)]:
                row = np.zeros(len(models)); row[idx[winner]] = 1.0; row[idx[loser]] = -1.0
                X.append(row); y.append(1); w.append(0.5)
        else:
            winner, loser = (model_a, model_b) if score_a == 1.0 else (model_b, model_a)
            row = np.zeros(len(models)); row[idx[winner]] = 1.0; row[idx[loser]] = -1.0
            X.append(row); y.append(1); w.append(1.0)
    clf = LogisticRegression(fit_intercept=False, penalty=None)
    clf.fit(np.array(X), np.array(y), sample_weight=np.array(w))
    theta = clf.coef_[0]
    return {m: float(theta[idx[m]]) for m in models}
```

The key property this buys over sequential Elo: the fit is **order-invariant** — it depends only on the full multiset of observed outcomes, not the sequence they arrived in, which directly eliminates sequential Elo's early-game order-dependence problem, and it supports proper bootstrap confidence intervals by resampling the vote log and refitting many times, which is exactly what LMSYS reports alongside its leaderboard rankings to signal that small rank differences between models with overlapping intervals aren't statistically meaningful.

## Q17: What are the three most concretely documented critiques of Chatbot Arena, and which one does LMSYS's own "style-controlled" leaderboard attempt to fix?

(1) **Voter-population bias** — votes come from a self-selected population of people who visit the Arena website, skewed toward AI-enthusiast/technical users, not representative of the general population of deployed-product users or of enterprise use cases. (2) **Prompt-distribution bias** — the prompt mix reflects whatever casual visitors happen to type (creative writing, casual QA, coding snippets, roleplay), not a curated or representative task distribution, so a model can rank highly by being excellent specifically at what casual users tend to ask while being comparatively weaker at task types the platform's voters rarely probe. (3) **Style/length gaming** — human voters are measurably influenced by response length, formatting, and confident tone independent of substantive correctness, so a model tuned (deliberately or as a side effect of RLHF preference-modeling, which has the same human-rater susceptibility) toward longer, more elaborately formatted, more assertive responses can win a disproportionate share of votes without the underlying content being better.

LMSYS's style-controlled leaderboard specifically targets critique (3): it's a regression-based adjustment that statistically controls for response length and markdown-formatting features when estimating each model's underlying Bradley-Terry strength, aiming to isolate a "quality holding style constant" ranking from the raw, style-confounded one. It does not address voter-population or prompt-distribution bias, which are more structural to how the platform collects data in the first place and are not fixable by a post-hoc statistical adjustment to the rating computation.

## Q18: Scenario — a new agentic coding scaffold pushes a model's SWE-bench Verified resolve rate up 20 points versus last quarter's number for what is nominally "the same" underlying model. Does this mean the base model got smarter?

Not necessarily, and this is exactly the scaffold-confound problem that's specific to agentic benchmarks (file 003). SWE-bench Verified numbers are a joint function of the base model's capability and the surrounding agent harness — how repository context is retrieved and serialized, what tools are available (search, test-running, iterative self-correction loops), how errors are recovered from, and how many attempts/retries the scaffold allows. A 20-point jump from a new scaffold on the *same* underlying model is entirely plausible and would demonstrate exactly that: scaffold engineering, independent of base-model capability, is a large lever on this specific benchmark's score.

To actually answer "did the base model get smarter," you'd need to hold the scaffold fixed and vary only the model (or vice versa) — report resolve rate for both the old and new model under the *same* scaffold, and separately report the old model under both the old and new scaffold, to decompose how much of the 20-point gain is attributable to each factor. Absent that ablation, the honest statement is that the reported number improved, and the two most likely explanatory buckets (base-model capability gain, scaffold-engineering gain) are confounded in a single headline figure — treating the number as pure evidence of base-model improvement, without the ablation, overstates what was actually shown, and this exact confound is one of the most common sources of inflated or misleading claims in agentic-benchmark reporting.

## Q19: Contamination detection generally relies on checking overlap between benchmark content and training data. Why does this approach work much better for static QA benchmarks than for agentic benchmarks like WebArena or GAIA?

For static QA benchmarks (MMLU, GSM8K, MATH), contamination detection is at least tractable in principle: the benchmark item is a fixed, self-contained piece of text (a question and its answer), and n-gram or substring overlap checking against a training corpus can directly ask "did this specific text, or a close paraphrase, appear in the training data" — canary strings and known-answer-key matching give an even cleaner signal when available. This methodology is covered in `../05_Evaluation_Methods/004_Contamination_Aware_Evaluation_Design.md`.

For agentic benchmarks, the thing that would need to "leak" for the benchmark's validity to be compromised is usually not the eval item's literal text — it's a **solution trajectory or scaffold strategy** for a specific environment (a blog post walking through exactly how to complete WebArena's shopping-cart task, discussion of GAIA's specific question set and how various agents solved it, or shared scaffold code tuned to a benchmark's specific task family). None of that requires any verbatim overlap with the original benchmark item's text to have happened — the model or its developers only need exposure to *discussion of how to solve tasks of this shape*, which standard n-gram-overlap contamination detection is not built to catch at all, since it's looking for text-level duplication, not strategy-level exposure. This is compounded by agentic benchmarks' construction cost being much higher (bespoke environments and checkers per task), so they're smaller and refresh far more slowly than static QA sets, giving solution writeups more relative time to accumulate and spread before the benchmark itself is ever revised.

## Q20: Implement the tau-bench pass^k reliability metric, and explain precisely how it differs from the pass@k metric used for code benchmarks — don't just say "one is k-of-k and one is 1-of-k," derive why that distinction matters for what each metric is actually trying to measure.

```python
from math import comb

def pass_hat_k(trial_outcomes: list[list[bool]], k: int) -> float:
    """tau-bench-style pass^k: probability of succeeding on ALL k of a random
    k-subset of independent trials of the same task (fresh user-sim/stochasticity
    each trial). trial_outcomes[i] = pass/fail booleans across n independent
    trials of task i, with n >= k."""
    per_task = []
    for outcomes in trial_outcomes:
        n = len(outcomes)
        if n < k:
            raise ValueError("need at least k independent trials per task")
        c = sum(outcomes)
        per_task.append(0.0 if c < k else comb(c, k) / comb(n, k))
    return sum(per_task) / len(per_task)
```

Code-benchmark pass@k asks: "of k independently sampled attempts, does *at least one* succeed?" — a best-of-k, most-optimistic framing, appropriate because a code-generation deployment can plausibly run several candidate completions and pick (or automatically test-select) the best one; it's answering "given a budget of k tries, can we get a win." tau-bench's pass^k asks the opposite-in-spirit question: "across k independent trials of the *same* task, do *all of them* succeed?" — a worst-case/consistency framing, because a deployed customer-service agent handling a large volume of structurally similar tickets doesn't get to discard the failures and keep only its best attempt; every trial is a real ticket that either gets resolved correctly or doesn't, and a stakeholder cares about *reliability across repetition*, not "did the agent succeed at least once out of several tries at the identical situation."

This distinction matters because it changes what the metric is even measuring conceptually: pass@k measures *capability ceiling under selection* (can the model produce a correct output at all, if you're allowed to pick the best of several), while pass^k measures *consistency under no selection* (does the model produce a correct output reliably, every time, with no opportunity to discard a bad rollout). A model can have a very respectable pass@1 or even pass@5 success rate while having a surprisingly low pass^5 — exactly the gap tau-bench's authors report finding for frontier models — because different independent rollouts of the same nominal task can diverge into a wrong tool call, a missed policy check, or a miscommunication with the simulated user in ways that a single-trial or best-of-k evaluation would never surface, and that gap is precisely the deployment-relevant reliability signal a production agent system needs and that pass@k structurally cannot provide.
