# LLM Evaluation Metrics and Benchmarks

## Why Evaluating LLM Output Is Fundamentally Hard

Evaluating traditional software is comparatively easy: you define an expected output, run the code, and diff the result. Evaluating a large language model breaks this model at its foundation, because for almost any interesting prompt there is no single correct output. Ask an LLM to summarize a news article and there are dozens of valid summaries that differ in wording, emphasis, and length but are equally "correct." Ask it to write a function and there are many implementations that all pass the same tests. This is the central tension of LLM evaluation: the task is open-ended and the space of acceptable answers is large and fuzzy, yet you still need a number (or a small set of numbers) you can track over time, gate deployments on, and use to compare one model or prompt version against another.

This chapter works through the evaluation toolkit roughly in order of how mechanical versus how semantic each technique is — starting with exact string matching, moving through n-gram overlap metrics inherited from machine translation, then embedding-based semantic similarity, and finally LLM-as-judge, which is the technique that dominates production evaluation today because it is the only one flexible enough to score genuinely open-ended generation. Along the way we'll cover the standard public benchmarks used to compare foundation models, why those benchmarks are becoming less trustworthy over time, and how to build the one evaluation asset that actually matters for your product: a task-specific eval set grounded in your own data and failure modes.

## Exact Match and Its Narrow but Real Use Cases

Exact match is the simplest possible metric: normalize the model's output and the reference answer (lowercase, strip punctuation and whitespace, sometimes strip articles like "a"/"the") and check for byte-for-byte equality. It sounds almost too naive to be worth mentioning, but it is still the right metric for a specific and common class of task — anything with a single, structurally verifiable correct answer. Extractive question answering over a passage ("what year did X happen"), multiple-choice question answering, classification into a fixed label set, arithmetic answers, and structured extraction into a schema (extract the invoice total as a number) are all cases where exact match, or a very close variant like F1-over-tokens (used by the classic SQuAD benchmark to give partial credit for near-exact spans), is a legitimate and cheap metric.

```python
import re
import string


def normalize_answer(text: str) -> str:
    """Standard SQuAD-style normalization: lowercase, strip punctuation,
    articles, and collapse whitespace so trivial formatting differences
    don't count as mismatches."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def token_f1(prediction: str, reference: str) -> float:
    """Partial-credit overlap metric, used when exact match is too strict
    (e.g. reference is 'about 12 miles' and prediction is '12 miles')."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = {}
    for tok in pred_tokens:
        common[tok] = min(pred_tokens.count(tok), ref_tokens.count(tok))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
```

The failure mode to internalize is that exact match collapses the instant the task has any legitimate answer diversity. If you apply exact match to "summarize this email" or "write a SQL query that returns the top 5 customers," you will get a near-zero score even from a model that is producing perfectly good, functionally correct output, simply because it phrased things differently or aliased a column differently than your reference. Exact match's blindness to paraphrase is precisely the gap the next tier of metrics tries to close.

## BLEU, ROUGE, and Why N-Gram Overlap Metrics Struggle With Generative Tasks

BLEU (Bilingual Evaluation Understudy) was designed for machine translation and ROUGE (Recall-Oriented Understudy for Gisting Evaluation) was designed for summarization, and both work on the same underlying idea: count how much n-gram overlap exists between the generated text and one or more human reference texts. BLEU is precision-oriented (of the n-grams the model produced, how many appear in the reference) with a brevity penalty to stop the model from gaming the score by outputting one safe word; ROUGE is recall-oriented (of the n-grams in the reference, how many did the model reproduce), which fits summarization because you care whether the summary captured what mattered in the source.

```python
from collections import Counter


def get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(prediction: str, reference: str, n: int = 2) -> dict:
    """Simplified ROUGE-N: recall, precision, and F1 over n-gram overlap."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    pred_ngrams = get_ngrams(pred_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    overlap = sum((pred_ngrams & ref_ngrams).values())  # multiset intersection
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    precision = overlap / max(sum(pred_ngrams.values()), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"recall": recall, "precision": precision, "f1": f1}
```

These metrics were reasonable choices in an era when generation models were weaker and more likely to produce degenerate, repetitive, or badly-structured text, so surface-level overlap correlated decently with quality. That correlation has mostly broken down for evaluating modern LLM output, for a reason worth stating precisely: BLEU and ROUGE are purely lexical. They have no notion of synonymy, paraphrase, or semantic equivalence. A model that writes "the feline rested on the rug" when the reference says "the cat sat on the mat" scores badly on ROUGE despite being a perfect paraphrase, while a model that mechanically copies fifteen words from the reference but strings them together into a nonsensical or factually wrong sentence can score deceptively well. This is the exact opposite of what you want an evaluation metric to reward.

The practical consequence is that BLEU/ROUGE are now mostly useful as a cheap, fast regression signal for tasks that are inherently close to extractive (summarization where good summaries do tend to reuse source vocabulary, translation where reference translations are professionally normalized) rather than as a quality bar for genuinely generative, creative, or conversational output. If you see a paper or vendor benchmark leaning heavily on BLEU/ROUGE for evaluating a chat assistant or an agent's free-form responses, that's a signal to look more closely at what's actually being measured — it is very easy to improve a ROUGE score by fine-tuning a model to imitate reference phrasing without actually improving the substance of its answers.

## Embedding-Based Semantic Similarity Metrics

The next rung up the ladder swaps n-gram overlap for semantic overlap by embedding both the generated text and the reference into a shared vector space and measuring cosine similarity. This immediately fixes the paraphrase-blindness problem: "the feline rested on the rug" and "the cat sat on the mat" produce nearly identical sentence embeddings even though they share almost no surface tokens.

```python
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def semantic_similarity_score(prediction: str, reference: str, embed_fn) -> float:
    """embed_fn is any sentence-embedding model call, e.g. OpenAI's
    text-embedding-3-small or a local sentence-transformers model."""
    pred_vec = embed_fn(prediction)
    ref_vec = embed_fn(reference)
    return cosine_similarity(np.array(pred_vec), np.array(ref_vec))
```

A more sophisticated variant in this family is BERTScore, which does not just embed the whole sentence into one vector — it embeds every token (using contextual embeddings from a model like BERT), then greedily matches each reference token to its most similar candidate token, and averages the resulting similarities into precision, recall, and F1 numbers. This gives finer-grained credit than a single whole-sentence cosine score: a long generated answer that gets most sentences right but botches one clause won't have its single similarity score dragged down as much as with sentence-level embedding, because most of its tokens still find good matches in the reference.

The limitation to be honest about in an interview: embedding similarity measures *semantic proximity to the reference*, not *correctness*. It's entirely possible for a plausible-sounding but factually wrong answer to sit close in embedding space to the correct answer, because they're about the same topic and use similar vocabulary — "the Treaty of Versailles was signed in 1920" and "the Treaty of Versailles was signed in 1919" are extremely close in embedding space despite one being wrong. Embedding similarity is also sensitive to length and register mismatches in ways that don't track quality (a terse correct answer versus a verbose correct answer can score lower against a reference of the "wrong" length), and it says nothing about qualities like instruction-following, format compliance, or reasoning validity that don't show up as lexical or semantic content at all. It is a good complement to other metrics, particularly as a fast, cheap regression check, but it should rarely be your only signal for anything a user will actually read.

## LLM-as-Judge: The Dominant Technique for Open-Ended Evaluation

The technique that has become the backbone of production LLM evaluation is using another (usually stronger, or at least independently prompted) LLM to score or compare outputs according to a rubric written in natural language. The insight behind LLM-as-judge is simple but powerful: LLMs are already good at exactly the kind of nuanced, contextual judgment that lexical and embedding metrics can't express — "is this answer helpful," "does this response follow the requested format," "is this code correct and idiomatic," "did the assistant maintain the right tone." Rather than trying to encode that judgment into a hand-built formula, you delegate it to a model and treat its verdict as the metric.

There are two dominant modes of LLM-as-judge. **Pointwise scoring** asks the judge to rate a single output against a rubric, typically on a numeric or categorical scale. **Pairwise comparison** shows the judge two candidate outputs (for example, from two different prompt versions or two different models) for the same input and asks which one is better, optionally with a margin or a tie option. Pairwise comparison tends to produce more reliable, better-calibrated judgments than pointwise scoring, because relative judgments ("is A better than B") are cognitively easier and more consistent than absolute judgments ("is this a 7 or an 8 out of 10") — the same phenomenon shows up in human annotation, which is why preference-based data collection (as used to train reward models for RLHF) uses pairwise comparisons rather than asking annotators to assign absolute scores.

```python
import json


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator grading an AI assistant's response.

Task the assistant was given:
{task}

Assistant's response:
{response}

Grading rubric:
{rubric}

Score the response from 1 to 5 on each rubric dimension. Be strict: a 5 means
the response is essentially flawless on that dimension, a 1 means it fails
badly. Do not be swayed by confident or verbose phrasing -- judge substance.

Return ONLY valid JSON in this exact shape:
{{
  "scores": {{"<dimension>": <int 1-5>, ...}},
  "overall_score": <int 1-5>,
  "reasoning": "<2-3 sentences justifying the scores>"
}}
"""


class LLMJudge:
    def __init__(self, judge_llm, rubric: str):
        self.judge_llm = judge_llm
        self.rubric = rubric

    def score(self, task: str, response: str) -> dict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            task=task, response=response, rubric=self.rubric
        )
        # temperature=0 for reproducibility; judges should be as deterministic
        # as the underlying model allows, since we're using them as a metric
        raw = self.judge_llm.generate(prompt, temperature=0)
        return json.loads(raw)

    def pairwise_compare(self, task: str, response_a: str, response_b: str) -> dict:
        # Randomize which response is "A" vs "B" per call at a higher level
        # to cancel out positional bias -- judges systematically favor
        # whichever response they see first or second, so always run this
        # twice with the order swapped and check for agreement.
        prompt = f"""You are comparing two AI assistant responses to the same task.

Task: {task}

Response A: {response_a}

Response B: {response_b}

Which response better satisfies the task? Consider correctness, completeness,
and clarity. Respond with ONLY valid JSON:
{{"winner": "A" | "B" | "tie", "reasoning": "<one sentence>"}}
"""
        raw = self.judge_llm.generate(prompt, temperature=0)
        return json.loads(raw)


def debiased_pairwise_compare(judge: LLMJudge, task: str, resp_1: str, resp_2: str) -> str:
    """Run the comparison twice with order swapped; only trust a verdict
    if it's consistent both ways, otherwise call it a tie."""
    first_pass = judge.pairwise_compare(task, resp_1, resp_2)
    second_pass = judge.pairwise_compare(task, resp_2, resp_1)

    winner_map = {"A": resp_1, "B": resp_2}
    swapped_map = {"A": resp_2, "B": resp_1}

    w1 = winner_map.get(first_pass["winner"])
    w2 = swapped_map.get(second_pass["winner"])

    if w1 == w2 and w1 is not None:
        return w1
    return "tie"
```

LLM-as-judge is powerful, but it inherits and amplifies several biases you need to actively defend against, because an evaluation system with a hidden bias is worse than no evaluation system — it gives you false confidence. **Position bias** is the tendency of a judge to systematically prefer whichever response appears first (or second) in a pairwise prompt, independent of quality; the fix is running each comparison twice with the order swapped, as in the snippet above, and discarding or flagging disagreements. **Verbosity bias** is the well-documented tendency of LLM judges to rate longer, more elaborately formatted responses as better even when a shorter response is equally or more correct — this is worth explicitly calling out in your rubric ("do not reward length or confident tone; judge substance and correctness only") and worth spot-checking with adversarial test cases where the longer answer is deliberately wrong. **Self-preference bias** shows up when a model is used to judge its own outputs or outputs from models in the same family — it tends to rate them more favorably, which is why using a different, typically stronger model as the judge (e.g., using GPT-4-class or Claude Opus-class models to judge outputs from smaller or different models) is standard practice, and why any eval claiming "our model beats GPT-4" using GPT-4 itself as the judge deserves skepticism.

The other practical concern is judge reliability itself: before trusting an LLM judge in a pipeline, you should validate it against a small set of human-labeled examples and compute agreement (Cohen's kappa, or simple percent agreement) between the judge and human raters. A judge that only agrees with humans 60% of the time on your specific task is not a metric you should be gating deployments on, no matter how sophisticated the rubric looks. Treat the judge itself as a model that needs its own evaluation set — this is a genuinely recursive problem, and acknowledging that recursion explicitly is a strong signal in an interview that you understand the limits of the technique rather than treating LLM-as-judge as a magic oracle.

```python
def validate_judge_against_humans(judge: LLMJudge, labeled_examples: list[dict]) -> dict:
    """labeled_examples: [{"task": ..., "response": ..., "human_score": 1-5}, ...]
    Run before trusting a judge in a CI gate."""
    agreements = 0
    diffs = []
    for ex in labeled_examples:
        judge_result = judge.score(ex["task"], ex["response"])
        judge_score = judge_result["overall_score"]
        diff = abs(judge_score - ex["human_score"])
        diffs.append(diff)
        if diff <= 1:  # within one point counts as agreement
            agreements += 1

    return {
        "agreement_rate": agreements / len(labeled_examples),
        "mean_absolute_diff": sum(diffs) / len(diffs),
        "n_examples": len(labeled_examples),
    }
```

## Standard Public Benchmarks and Their Known Weaknesses

Public benchmarks exist to let the field compare models on a level playing field, and it's worth knowing the major ones and what each actually measures. MMLU (Massive Multitask Language Understanding) tests broad knowledge and reasoning across 57 subjects via multiple-choice questions, and is the most commonly cited "general knowledge" benchmark. HumanEval and its successors (MBPP, LiveCodeBench) test code generation by having the model write a function and running it against hidden unit tests — a rare case where evaluation actually is exact and objective, since code either passes the tests or it doesn't. GSM8K and MATH test grade-school and competition-level mathematical reasoning. HellaSwag and ARC test commonsense reasoning via sentence/scenario completion. MT-Bench and Chatbot Arena (now LMSYS/LMArena) evaluate conversational quality — MT-Bench via LLM-as-judge scoring of multi-turn dialogues, Chatbot Arena via large-scale crowdsourced pairwise human preference voting aggregated into an Elo rating, which makes it one of the few large-scale benchmarks driven by real human judgment rather than automated scoring.

Two systemic problems undermine essentially all of these benchmarks, and both are worth being able to explain crisply.

**Contamination** is the problem of benchmark data leaking into a model's pretraining corpus. Because frontier models are trained on enormous scrapes of the public internet, and because benchmark datasets and their answer keys are frequently posted publicly (on GitHub, in blog posts discussing them, in academic papers indexed by search engines), there is a real chance that a model has literally seen the benchmark's test questions, and sometimes their answers, during pretraining. A model that has memorized "the answer to MMLU question #4521 is C" is not demonstrating reasoning ability on that question — it's demonstrating retrieval of a memorized fact, and its benchmark score is inflated in a way that doesn't transfer to novel, unseen problems of the same type. Contamination is hard to detect reliably after the fact (you'd need to inspect the training corpus, which most labs don't disclose in full), and this is precisely why held-out, provably-novel benchmarks and continuously refreshed leaderboards (like Chatbot Arena, or benchmarks that rotate in fresh questions periodically) are trusted more than static, years-old benchmark files that have had ample time to leak into crawl data.

**Saturation** is the problem of a benchmark losing its discriminative power once models routinely score near the ceiling. MMLU scores climbed from around 40-50% for early instruction-tuned models to well above 85-90% for frontier models within a few years, at which point the benchmark stops distinguishing "very good" from "excellent" models — the remaining gap is often within the noise of the benchmark's own label-quality issues (MMLU is known to contain a nontrivial fraction of questions with debatable or outright wrong reference answers). Once a benchmark saturates, small score differences between models stop being meaningful, and continuing to lead with that benchmark in marketing materials is a signal to discount rather than a signal of genuine differentiation. This dynamic is exactly why the field keeps producing new, harder benchmarks (MMLU gave way to MMLU-Pro and GPQA, HumanEval gave way to LiveCodeBench and SWE-bench) — saturation is a treadmill, not a one-time event.

A related and more subtle problem is benchmark-target overfitting even without literal contamination: once a benchmark becomes a widely tracked leaderboard, labs have strong incentive to include benchmark-adjacent data (similar question formats, similar reasoning patterns) in their training or fine-tuning mix, which improves the benchmark score without necessarily improving general capability by the same margin. This is Goodhart's Law applied to LLM evaluation — "when a measure becomes a target, it ceases to be a good measure" — and it's a large part of why relying solely on public benchmark leaderboards to pick a model for your production use case is a mistake senior engineers should actively push back on.

## Building a Task-Specific Evaluation Set

Given everything above, the single highest-leverage evaluation asset for any real product is not a public benchmark at all — it's a small, curated, continuously growing dataset built directly from your own task and your own failure modes. The process looks like this in practice.

Start by mining real production inputs (or, pre-launch, inputs from user research and internal dogfooding) rather than inventing synthetic examples from scratch — synthetic examples written by the team building the system tend to be unconsciously biased toward cases the system already handles well. Stratify the collected examples across the dimensions that matter for your task: input length, topic/domain coverage, difficulty, and — critically — known-hard edge cases (ambiguous phrasing, adversarial inputs, multi-step requests, requests with missing information). A common anti-pattern is an eval set consisting entirely of "happy path" examples, which will show a misleadingly flat, high pass rate right up until a real failure ships to production.

For each example, decide what "correct" means and encode it in whatever form supports automated (or semi-automated) grading: a reference answer for similarity-based scoring, a set of required facts/keywords that must appear, a rubric for an LLM judge, or — best when feasible — an executable check (does the generated SQL actually return the right rows when run against a test database, does the generated code pass unit tests). Executable, deterministic checks should always be preferred over judge-based checks when the task permits them, because they remove an entire category of judge-reliability risk.

```python
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class EvalCase:
    id: str
    input: str
    reference: Optional[str] = None          # for similarity-based grading
    required_facts: list[str] = field(default_factory=list)  # for keyword/fact checks
    rubric: Optional[str] = None              # for LLM-judge grading
    check_fn: Optional[Callable[[str], bool]] = None  # for executable grading
    tags: list[str] = field(default_factory=list)  # e.g. ["edge_case", "multi_step"]


class TaskEvalSuite:
    def __init__(self, cases: list[EvalCase], judge: "LLMJudge" = None):
        self.cases = cases
        self.judge = judge

    def grade_case(self, case: EvalCase, model_output: str) -> dict:
        if case.check_fn is not None:
            return {"method": "executable", "passed": case.check_fn(model_output)}

        if case.required_facts:
            missing = [f for f in case.required_facts if f.lower() not in model_output.lower()]
            return {"method": "fact_check", "passed": not missing, "missing": missing}

        if case.rubric and self.judge:
            result = self.judge.score(case.input, model_output)
            return {"method": "llm_judge", "passed": result["overall_score"] >= 4, **result}

        raise ValueError(f"Case {case.id} has no grading method configured")

    def run(self, model_fn: Callable[[str], str]) -> dict:
        results = []
        for case in self.cases:
            output = model_fn(case.input)
            grade = self.grade_case(case, output)
            results.append({"id": case.id, "tags": case.tags, "output": output, **grade})

        pass_rate = sum(r["passed"] for r in results) / len(results)
        by_tag = {}
        for r in results:
            for tag in r["tags"]:
                by_tag.setdefault(tag, []).append(r["passed"])
        tag_pass_rates = {tag: sum(v) / len(v) for tag, v in by_tag.items()}

        return {"overall_pass_rate": pass_rate, "by_tag": tag_pass_rates, "details": results}
```

The eval set is not a one-time deliverable — it should grow every time production surfaces a real failure. The standard workflow is: a user hits a bad output, someone triages it, the offending input (with a corrected reference or rubric) gets added to the eval suite as a regression case, and the fix (prompt change, retrieval change, model change) is validated against the full growing suite before shipping. Over months this turns your eval set into a genuinely valuable, proprietary asset that captures the specific ways your specific product tends to fail — something no public benchmark can give you, because no public benchmark knows what your users actually ask for. Track pass rate over time per tag/category rather than as one aggregate number; an aggregate can stay flat while quality quietly degrades on a specific, high-value subset (a common example: overall pass rate holds steady while performance on multi-turn or non-English inputs silently regresses because those cases are underrepresented in the aggregate).

## Putting It Together: A Layered Evaluation Strategy

In production, none of these techniques are used in isolation — they're layered by cost and fidelity. Cheap, fast, purely mechanical checks (exact match, schema validation, executable tests) run on every single generation as a first-pass filter, because they're nearly free and catch a meaningful fraction of hard failures (malformed JSON, wrong data type, empty output). Embedding similarity runs as a fast regression signal in CI against a reference set, cheap enough to run on every pull request. LLM-as-judge, being the slowest and most expensive (an extra model call per evaluated example, sometimes two for debiased pairwise comparison), is reserved for the subset of quality dimensions that genuinely require semantic judgment — helpfulness, tone, reasoning soundness — and is typically run on a sampled subset in continuous production monitoring plus the full task-specific eval suite before any release. Public benchmarks, finally, are useful for one narrow purpose: initial model selection when you're choosing which foundation model to build on, and even then only as one input alongside your own task-specific eval results, never as a substitute for them.
