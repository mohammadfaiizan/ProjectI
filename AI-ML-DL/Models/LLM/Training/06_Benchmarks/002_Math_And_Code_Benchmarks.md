# Math and Code Benchmarks

Math and code benchmarks are the benchmark family that has moved fastest and been revised most often, because they are the family where "correct" has an unambiguous, mechanically checkable definition — a final numeric answer either matches or doesn't, a unit test either passes or doesn't.

That mechanical checkability is exactly what makes these benchmarks useful as reward signals for RL post-training (see the RLVR/GRPO discussion in the DeepSeek-R1 note under `../../OpenSource/008_DeepSeek_R1.md`). It is also exactly what makes them saturate fast: once a benchmark's answer format is a clean automatic check, it is also a clean automatic optimization target, and frontier labs train directly against exactly that kind of signal.

This file traces the resulting escalation — GSM8K to MATH to AIME-style competition math, and HumanEval/MBPP to SWE-bench — as two parallel instances of the same "benchmark gets solved, field moves the goalposts" dynamic that file 007 treats as a general pattern.

## GSM8K

**Citation:** Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek, Hilton, Nakano, Hesse, Schulman, "Training Verifiers to Solve Math Word Problems," OpenAI, 2021.

### What it measures

GSM8K ("Grade School Math 8K") is 8,500 human-written grade-school-level math word problems (roughly 7,500 train / 1,000 test), each requiring a short chain of elementary arithmetic operations — addition, subtraction, multiplication, division, no algebra, no calculus — typically spanning 2 to 8 reasoning steps to reach a final numeric answer.

A representative example: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" The problems are deliberately linguistically simple and grade-school in *content* — the difficulty is entirely in correctly composing several small arithmetic steps in the right order, not in mathematical sophistication.

### Scoring mechanics

Each training example's reference solution is a natural-language chain of reasoning steps ending in a numeric answer flagged with a `####` delimiter (e.g., `... #### 72`). Evaluation is **exact-match on the final number**: the model's generated solution is parsed for the number following its own final-answer delimiter, and that number is compared for exact equality against the reference answer.

There is no partial credit and no symbolic-equivalence step needed, because grade-school arithmetic answers are always a single unambiguous integer or simple decimal. This is the simplest possible case of automatic math-answer grading, and part of why GSM8K became a default early benchmark: it requires no equivalence-checking infrastructure at all, just string/numeric comparison after parsing.

```python
import re

def extract_final_answer(generation: str) -> str | None:
    # Common convention: model asked to end with "#### <answer>"
    match = re.search(r"####\s*([\-0-9.,/]+)\s*$", generation.strip())
    if match:
        return match.group(1).replace(",", "")
    return None

def gsm8k_exact_match(generation: str, reference_answer: str) -> bool:
    pred = extract_final_answer(generation)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(reference_answer)) < 1e-6
    except ValueError:
        return pred.strip() == reference_answer.strip()
```

### Saturation and known weaknesses

GSM8K is thoroughly saturated for frontier models. Models at or above roughly GPT-3.5/GPT-4 scale with chain-of-thought prompting routinely score above 90%, and current frontier models score above 95%, leaving essentially no headroom to differentiate among them.

Documented weaknesses:

1. **Contamination risk is high** given how widely GSM8K is cited, reproduced, and discussed across the open web since 2021 — it is a standard fixture in nearly every LLM paper's eval table, meaning any broad web crawl for pretraining after roughly 2022 plausibly contains verbatim or near-verbatim copies.
2. **A handful of problems have debatable reference answers**, flagged by the community as ambiguous — a smaller-scale version of the MMLU-Redux label-error finding, though GSM8K's simplicity means this is a much smaller effect than MMLU's.
3. **Exact-match on the final number cannot detect reasoning-process errors that happen to cancel out** to the right final number. A model can get several intermediate steps wrong and still score correct if the arithmetic happens to work out, which the exact-match protocol cannot detect at all — a limitation shared with MATH and AIME below.

## MATH

**Citation:** Hendrycks, Burns, Kadavath, Arora, Basart, Tang, Song, Steinhardt, "Measuring Mathematical Problem Solving With the MATH Dataset," 2021.

### What it measures and why it's harder than GSM8K

MATH is 12,500 competition-mathematics problems (train + test, roughly 7,500/5,000 split) sourced from real math competitions (AMC, AIME, and similar), spanning algebra, geometry, number theory, counting and probability, and precalculus, with five self-reported difficulty levels.

Unlike GSM8K's grade-school arithmetic, MATH problems require actual competition-mathematics technique: recognizing which identity or theorem applies, multi-step algebraic manipulation, and answers that are frequently not plain integers — fractions, radicals, intervals, ordered tuples, matrices, or symbolic expressions.

### Scoring mechanics — the genuinely hard part

Because answers can be mathematically equivalent while being textually very different (`1/2` vs `0.5` vs `\frac{1}{2}`; `2\sqrt{3}` vs `\sqrt{12}`; a fully reduced fraction vs an unreduced one), naive string exact-match badly underestimates accuracy.

MATH's standard convention has both the reference and the model's response state a final answer inside a LaTeX `\boxed{...}` command, which is then extracted and passed through a normalization-and-equivalence pipeline rather than compared as a raw string:

```python
import re

def extract_boxed_answer(text: str) -> str | None:
    # Find the contents of the last \boxed{...}, handling nested braces
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None

def normalize_math_answer(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace(" ", "").replace("\\!", "").replace("\\,", "")
    ans = ans.replace("^\\circ", "").replace("\\%", "").replace("%", "")
    ans = re.sub(r"\\left|\\right", "", ans)
    ans = re.sub(r"\\dfrac", "\\frac", ans)
    # a real pipeline continues: strip \text{}, unify \frac{a}{b} forms,
    # unify equivalent radical/decimal representations, strip trailing units
    return ans

def math_is_equivalent(pred: str, reference: str) -> bool:
    p, r = normalize_math_answer(pred), normalize_math_answer(reference)
    if p == r:
        return True
    # Fallback: symbolic equivalence via a CAS for algebraic answers that
    # survive string normalization as unequal but are mathematically equal
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        return bool(sympy.simplify(parse_latex(p) - parse_latex(r)) == 0)
    except Exception:
        return False
```

This normalization-plus-symbolic-fallback pipeline is itself a known source of scoring noise across papers. Different eval harnesses — OpenAI's, Hendrycks' original code, EleutherAI's lm-evaluation-harness, Minerva's — implement slightly different normalization rules, so reported MATH accuracy numbers are not perfectly comparable across papers unless they used the same harness. This is a subtle but real reproducibility problem specific to benchmarks with non-trivial answer equivalence, which does not arise for GSM8K's plain-integer answers.

### Saturation and weaknesses

Frontier models (GPT-4o, Claude 3.5 Sonnet, and especially reasoning-tuned models like o1 and DeepSeek-R1) score in the 90s on the commonly reported **MATH-500** subset — a 500-problem subset popularized by OpenAI's and subsequent papers' evaluation splits, distinct from the full 5,000-problem test set. DeepSeek-R1 reports roughly 97.3% on MATH-500 (self-reported, flagged as approximate).

This level of saturation for top reasoning models is exactly why the field moved to AIME-style benchmarks: MATH no longer separates frontier reasoning models from each other, even though it still separates reasoning-tuned models from non-reasoning-tuned ones of similar base scale.

Additional documented weaknesses: contamination risk (MATH problems are drawn from public competition archives extensively documented and solved online, e.g., on AoPS forums, so both problems and worked solutions are present in typical web-crawl pretraining data); difficulty-level labels are self-reported by the original curators and not independently re-validated at MMLU-Redux's level of rigor; and, as with GSM8K, exact/symbolic final-answer matching cannot detect a correct final answer reached via invalid intermediate reasoning — a nontrivial failure mode specifically for competition math, where a lucky guess or a flawed-but-coincidentally-correct derivation is more plausible than in grade-school arithmetic.

## AIME-style competition math benchmarks

### What it is and why the field needed it

AIME (American Invitational Mathematics Examination) is a real, pre-existing US high-school competition — not a benchmark purpose-built for LLM evaluation — that the field adopted as an evaluation set specifically *because* GSM8K and MATH stopped differentiating frontier reasoning models.

Format: 15 problems per exam sitting, each with an integer answer in [0, 999]. This integer-answer format is deliberate in the original competition, to make grading unambiguous for human graders too — a property that happens to make it trivially easy to grade for LLM evaluation as well, with none of MATH's equivalence-checking complexity. Two AIME sittings occur each year (AIME I and AIME II), so a given year contributes 30 problems; papers commonly report results on a specific year (e.g., "AIME 2024") or splice two years together for a slightly larger n (~60 problems) to reduce noise.

### Why this differentiates frontier models when MATH no longer does

AIME problems are genuinely harder than the median MATH problem — they are drawn from a competition specifically designed to challenge the strongest US high-school mathematics competitors (AIME is the qualifying round that feeds into the USAMO), calibrated so that even a skilled human competitor is expected to solve considerably less than all 15 in the allotted time.

Pre-reasoning-era LLMs scored in the single digits to low tens of percent on AIME. Reasoning-RL-trained models (OpenAI's o1 family, DeepSeek-R1) report pass@1 numbers in the 70-80% range on AIME 2024 — DeepSeek-R1 reports approximately 79.8% versus OpenAI o1-1217's reported approximately 79.2%, both self-reported and approximately equal within likely noise given the small n (flagged as self-reported figures from each lab's own paper, not independently cross-verified).

That jump — from single digits to roughly 80% — is the single most-cited piece of evidence in the 2024-2025 "reasoning models" wave that RL-on-verifiable-rewards training produces a real, large capability gain specifically on hard multi-step mathematical reasoning, as opposed to a superficial prompting trick.

### Scoring mechanics

Exact integer match, identical in spirit to GSM8K's exact-match but with no need for GSM8K's `####`-delimiter parsing convention, since AIME's own answer format is already a bare integer 0-999 — about as clean an automatic-grading setup as exists in this space.

### Weaknesses — a structurally different contamination dynamic

AIME's small n (15 or 30 problems per year) means per-problem noise dominates: a single flipped answer moves the reported percentage by roughly 3.3 (1/30) to 6.7 (1/15) points, so headline comparisons between models that differ by a few points are close to comparing noise unless a paper reports across multiple years or bootstraps a confidence interval — rarely done in practice.

More structurally interesting is the contamination dynamic. Because each year's AIME is a *fresh, previously nonexistent* problem set at the time it is administered, evaluating on the most recent year's exam has a genuinely lower contamination risk than a perennial, years-old dataset like MATH — *at the moment of first use*. But that advantage decays immediately: once an AIME sitting's problems and solutions are discussed on competition-math forums (which happens within days), that year's exam becomes exactly as contamination-prone as any other public dataset for any model trained afterward.

The practical consequence is that AIME-based evaluation is not a one-time fix but an ongoing treadmill. The field has to keep re-anchoring to the *next* year's fresh exam to preserve the contamination advantage — a real operational cost, since there are only two AIME sittings a year, so the supply of genuinely fresh competition-math problems at this exact difficulty and format is limited. This is why AIME-based evaluation is better described as a moving target than a settled benchmark the way MATH or GSM8K were meant to be.

## HumanEval and MBPP

**HumanEval citation:** Chen et al. (OpenAI, includes the Codex paper), "Evaluating Large Language Models Trained on Code," 2021.
**MBPP citation:** Austin et al. (Google), "Program Synthesis with Large Language Models," 2021.

### What they measure

Both are function-level code-generation benchmarks: given a natural-language problem description (and, for HumanEval, a function signature plus docstring with input/output examples), the model must generate a complete function body, which is then checked against a held-out set of unit tests.

- **HumanEval**: 164 hand-written Python problems, explicitly constructed by OpenAI to avoid overlap with existing code corpora — each problem was authored fresh rather than sourced from existing programming-problem sites, specifically to reduce contamination risk relative to scraping problems that already have public solutions online.
- **MBPP** ("Mostly Basic Programming Problems"): a larger, crowd-sourced set of about 974 problems, with a commonly used "sanitized" subset of about 427 problems that were manually reviewed and cleaned. Each has a short natural-language prompt, a reference solution, and typically 3 test cases. As the name states, MBPP problems are intentionally simpler and more basic than HumanEval's, aimed at entry-level programming ability.

### The pass@k metric, precisely

Code models are typically evaluated by sampling multiple completions per problem (since generation is stochastic) and asking whether *at least one* of k sampled attempts passes all unit tests — this is "pass@k." The naive way to estimate this would be to literally generate exactly k samples per problem and check for any pass, but that estimator has high variance and is a poor use of expensive samples.

The standard unbiased estimator, introduced in the Codex/HumanEval paper, instead generates a larger fixed number `n` of samples per problem (`n >= k`), counts how many `c` of them pass, and computes the probability that a random size-k subset of those n samples contains at least one passing sample:

```
pass@k = E_problems[ 1 - C(n - c, k) / C(n, k) ]
```

where `C(a, b)` is "a choose b," `n` is the total number of samples drawn for that problem, and `c` is the number of those `n` samples that pass all tests. Intuitively: `C(n-c, k) / C(n, k)` is the probability that a random k-subset drawn from the n samples contains *zero* passing samples (you're choosing k items from the n-c failing ones only), so one minus that is the probability the k-subset contains at least one pass.

Computing it this way lets you draw a single generous batch of n samples (e.g., n=200) per problem once, and then estimate pass@1, pass@10, pass@100, etc. from that same batch without redrawing — much more sample-efficient than separately sampling k completions per k value.

```python
import numpy as np
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k given n total samples, c of which passed."""
    if n - c < k:
        return 1.0  # not enough failures to fill a whole k-subset without a pass
    return 1.0 - comb(n - c, k) / comb(n, k)

def estimate_pass_at_k(results: list[list[bool]], k: int) -> float:
    """results[i] = list of pass/fail booleans for the n samples on problem i."""
    per_problem = [pass_at_k(len(r), sum(r), k) for r in results]
    return float(np.mean(per_problem))
```

A common numerically stable reformulation avoids computing large binomial coefficients directly:

```python
def pass_at_k_stable(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    # product form: 1 - prod_{i=n-c+1}^{n} (i-k)/i, equivalent, avoids overflow
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

### pass@1 vs pass@k in practice

Most frontier-model leaderboard numbers you see quoted (e.g., "HumanEval: 92%") are **pass@1 with a single greedy or low-temperature sample** — a simplified special case (`n=k=1`, so the formula above just reduces to whether that single sample passed) rather than the full multi-sample estimator. The full pass@k machinery matters more for research comparing sampling strategies, or for agentic coding setups that can afford multiple attempts (e.g., an agent that runs the generated code and retries on failure), where the interesting question is exactly how quickly the pass rate rises with additional attempts.

### Weaknesses and saturation

Both benchmarks are effectively saturated for frontier models — pass@1 scores above 90% on HumanEval are routine, leaving little headroom. Both benchmark suites' problems are small, self-contained, single-function tasks unlike almost all real software engineering work, which is exactly the gap SWE-bench was built to expose.

Contamination is a serious concern for MBPP in particular given its crowd-sourced, less curated construction and multi-year public presence. HumanEval was more deliberately contamination-resistant at introduction (freshly authored problems), but has been public and heavily quoted since 2021, so the same web-crawl exposure concern applies by now. Both benchmarks also test isolated, well-specified function synthesis with a clean docstring — they say very little about a model's ability to work inside an existing, large, imperfectly-documented codebase, debug someone else's code, or reconcile a fix with a broader system's existing behavior.

## SWE-bench and SWE-bench Verified

**Citation:** Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan, "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", 2023 (Princeton/Stanford). SWE-bench Verified: a curated subset released by OpenAI in collaboration with the SWE-bench authors, 2024.

### Why this is a meaningfully different, harder evaluation than function-level benchmarks

SWE-bench does not ask a model to write an isolated function against a clean docstring. Each of its 2,294 task instances is built from a *real, merged pull request* on one of 12 popular Python open-source repositories (e.g., django, scikit-learn, sympy, matplotlib, requests, astropy), where that PR resolved an actual reported GitHub issue.

The model is given the real issue text and the full repository at the commit just before the fix, and must produce a patch (typically a unified diff) that, when applied, makes the specific tests the original PR added or modified pass (`FAIL_TO_PASS` tests) while not breaking the tests that were passing before (`PASS_TO_PASS` tests — a regression check).

This forces capabilities function-level benchmarks abstract away entirely:

- **Localization** — figuring out *which files and lines* in a large, real codebase are relevant, with no docstring pointing at the right function.
- **Whole-repository reasoning** — understanding how a fix interacts with surrounding abstractions, conventions, and invariants, not just satisfying a local spec.
- **Multi-file edits**, sometimes required, whereas HumanEval/MBPP are single-function by construction.
- **Grading against a real test harness** — the model's patch has to survive the project's actual, sometimes large, sometimes slow, sometimes flaky test infrastructure.
- **Closer proxy to real engineering work** — reading an issue, finding relevant code, making a targeted fix without breaking anything else, rather than a coding-interview-style isolated exercise.

### What "Verified" specifically fixed

The original SWE-bench, being built by mining real PRs at scale with an automated pipeline, contained a meaningful amount of noisy or unfair task instances:

- Issues underspecified relative to what the reference PR actually did, so a model producing an equally valid but different fix would be graded as failing.
- Instances where `FAIL_TO_PASS` tests were only passable by essentially reproducing the exact reference implementation, not any correct solution.
- Broken, non-reproducible, or flaky execution environments (dependency resolution failures, nondeterministic tests) that would fail even a correct patch.
- Issue descriptions lacking information the reference PR's author had access to but which wasn't captured in the mined text.

SWE-bench Verified is a 500-instance subset of the original 2,294, selected via a human (professional software engineer) annotation pass that specifically screened out instances with these problems. This doesn't change the task design — it removes noisy/unfair instances so that a failing score is much more likely to reflect an actual model shortcoming, and a passing score is much more likely to reflect a genuinely correct fix. This is analogous in spirit to what MMLU-Redux did for MMLU, but executed proactively as a curated subset rather than as a post-hoc audit paper.

### Reported trajectory

(Approximate, self-reported by each lab/paper — flagged.) The original full SWE-bench was extremely hard for early models — initial reported resolve rates were in the low single digits to low tens of percent depending on the scaffold used. SWE-bench performance is heavily dependent on the surrounding agent scaffold: how the model is given tools to browse/search the repo, run tests, and iterate, not just the base model's raw capability — a methodological wrinkle where two papers reporting different numbers for "the same" underlying model are often actually reporting different scaffolds.

By 2024, Claude 3.5 Sonnet was reported around 49% resolve rate on SWE-bench Verified with a well-engineered agent scaffold. Subsequent frontier models (Claude Sonnet/Opus 4-class, GPT-4.1/o-series-based scaffolds) have been reported climbing well past that, into the 60-70%+ range through 2025 as both base-model capability and agent-scaffold engineering (better repo search, test-running loops, self-correction) improved together. Because scaffold quality is such a large confound, SWE-bench Verified numbers should always be read alongside a description of the harness used to obtain them, not treated as a pure base-model capability number the way a pass@1 HumanEval score more nearly is.

### Remaining weaknesses

Even Verified is limited to 12 Python repositories, all popular, well-maintained open-source projects with mature test suites and conventional engineering practices. It says relatively little about performance on other languages, proprietary/enterprise codebases with different conventions and much less test coverage, issues requiring cross-repository or cross-service reasoning, or genuinely novel/greenfield feature work rather than bug-fix-shaped issues — SWE-bench's issue-mining methodology naturally selects for "existing behavior is wrong, fix it" problems rather than "build this new thing from scratch," which is a real and different skill.

Contamination is also a live concern: because the source repositories and their full commit histories, issues, and PRs are public, and because SWE-bench itself is now a widely-discussed benchmark with blog posts, leaderboards, and even publicly shared successful-agent-trajectory writeups, a model's pretraining or RL-rollout data could plausibly include the exact PR that resolves a given evaluation instance, or detailed discussion of how other agents solved it — an ongoing risk that "Verified" curation does not itself mitigate, since it improves label/task quality, not contamination exposure.

## Quick-reference comparison

| Benchmark | Answer format | Grading difficulty | Ceiling status | Escalation successor |
|---|---|---|---|---|
| GSM8K | Plain integer/decimal | Trivial (exact match) | Saturated (>95%) | MATH |
| MATH | LaTeX-boxed expression | Hard (normalization + symbolic fallback) | Saturating for reasoning models (~90s on MATH-500) | AIME |
| AIME | Integer 0-999 | Trivial (exact match) | Not saturated; frontier reasoning models ~80% | Fresh yearly cycle |
| HumanEval / MBPP | Passing unit tests | Trivial (run tests) | Saturated (>90% pass@1) | SWE-bench |
| SWE-bench Verified | Patch passes FAIL_TO_PASS + PASS_TO_PASS | Moderate (real test infra, scaffold-dependent) | Not saturated; frontier agents 60-70%+ | Broader languages/repos (open problem) |

## Synthesis

Both tracks in this file exhibit the identical underlying shape: an easy, cleanly-gradable benchmark (GSM8K; HumanEval/MBPP) gets saturated, a harder version with a genuinely more complex answer-equivalence or task-realism problem replaces it (MATH; SWE-bench), and once *that* starts to saturate for the strongest models, the field either escalates difficulty further using an external, non-benchmark-specific source of hard problems that keeps naturally refreshing (AIME's yearly cycle) or invests in more careful human curation of the existing hard benchmark to remove noise (SWE-bench Verified) rather than inventing an entirely new task.

The throughline for interview purposes: math benchmarks differentiate on *answer-equivalence-checking difficulty* (integer exact-match to symbolic LaTeX equivalence to still-integer-but-much-harder-problems), while code benchmarks differentiate on *task realism* (isolated function synthesis to whole-repository issue resolution). These are two different axes of "harder," and knowing which axis a given benchmark moved along when its predecessor saturated is a good signal of genuine fluency in this space rather than memorized leaderboard trivia.
