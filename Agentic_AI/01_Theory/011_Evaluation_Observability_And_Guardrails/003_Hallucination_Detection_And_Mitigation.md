# Hallucination Detection and Mitigation

## Why Hallucination Is Not a Bug, It's the Architecture

The single most important mental model for reasoning about hallucination is this: a large language model is a next-token predictor trained to maximize the probability of plausible continuations given its training distribution, and it has no built-in mechanism for distinguishing "this is true" from "this is a fluent, high-probability continuation of the prompt." Truth was never part of the training signal in any direct sense. During pretraining, the model is rewarded for predicting the next token in real text scraped from the internet — it learns the statistical structure of language, facts that appear frequently and consistently enough to be memorized, and stylistic patterns of confident, authoritative-sounding prose. It is never given a supervised signal that says "flag this sentence as unverified" or "output 'I don't know' when your internal confidence is low," because raw pretraining data doesn't come annotated with ground truth about the model's own future uncertainty.

This explains why hallucinations have a very specific character: they are usually fluent, confident, and internally coherent, because the model is optimizing for exactly those properties (a plausible continuation) whether or not the content is factually grounded. A hallucinated citation looks like a real citation — correct format, plausible author names, a plausible-sounding journal — because the model learned the *shape* of citations extremely well without any mechanism tying that shape to an actual verified source. This is fundamentally different from how a human expert who doesn't know an answer behaves; a human typically has some access to their own uncertainty ("I'm not sure, let me check") whereas a base LLM's fluency is nearly constant across both confident, well-grounded generations and confabulated ones, because fluency is what the training objective directly optimized, not confidence calibration.

Instruction tuning and RLHF partially address this by training models to say "I don't know" in some situations and to hedge language when appropriate, and this measurably reduces (though doesn't eliminate) hallucination rates, because now there is at least some training signal connecting expressed uncertainty to situations humans labeled as ones where uncertainty was appropriate. But this is a learned behavior pattern layered on top of the same underlying architecture, not a fix to the root cause — the model still has no direct introspective access to "did I actually know this or did I pattern-match my way to a plausible-sounding answer." This is why even the best frontier models still hallucinate on questions just past the edge of their training data or reasoning ability, and why hallucination rate scales with how far a query sits from well-represented, high-consensus training data — narrow, low-frequency, or rapidly-changing facts (a mid-tier company's Q3 revenue, a niche API's exact parameter names, an event from after the training cutoff) are hallucination hotspots precisely because the statistical signal supporting a correct answer is thin or entirely absent, and the model fills the gap with the most probable-sounding tokens rather than an honest "I don't have this information."

## The Taxonomy of Hallucination

It's worth distinguishing several distinct failure modes that get lumped under "hallucination," because they call for different mitigations. **Factual fabrication** is inventing information wholesale — a citation that doesn't exist, a statistic that was never published, a historical event that never happened. **Factual error** is stating something confidently that is simply wrong, often because the model's training data contained the error, contained conflicting information, or the fact falls just outside high-confidence territory. **Context-unfaithful generation** (sometimes called "closed-domain hallucination") happens specifically in RAG and summarization settings, where the model is given source material but generates a claim that isn't actually supported by it — this is arguably the most fixable category because, unlike the first two, there's a concrete reference (the provided context) to check the claim against, which is exactly what grounding and faithfulness checks (below) are built for. **Reasoning hallucination** happens in multi-step reasoning or agentic tasks, where each individual step looks locally plausible but an intermediate logical leap is unsupported or invalid, producing a confidently wrong conclusion built on a chain that looks rigorous. Keeping these categories distinct matters because a mitigation that works well for context-unfaithful generation (retrieval grounding) does essentially nothing for reasoning hallucination, which needs a different kind of intervention (self-consistency, verification passes) covered below.

## Detection Technique: Self-Consistency Checking

One of the cheapest and most robust hallucination signals doesn't require any external source of truth at all — it exploits the fact that a model's genuine knowledge tends to be stable under resampling, while confabulated content tends to vary. The idea, popularized under the name **SelfCheckGPT**, is to sample the same prompt multiple times at nonzero temperature and check whether the resulting answers agree with each other. If the model actually "knows" a fact, most samples will restate it consistently (perhaps with different wording but the same substance); if the model is confabulating, different samples will diverge because there was no real underlying fact anchoring the generation, just independent draws from a probability distribution over plausible-sounding completions.

```python
from collections import Counter


class SelfConsistencyChecker:
    def __init__(self, llm, n_samples: int = 5, temperature: float = 0.8):
        self.llm = llm
        self.n_samples = n_samples
        self.temperature = temperature

    def check_claim(self, prompt: str, claim: str, nli_fn) -> dict:
        """Sample n independent completions and check how many are consistent
        with the claim under test, using an NLI (natural language inference)
        model or judge to assess entailment/contradiction rather than exact
        string match, since paraphrases should count as agreement."""
        samples = [
            self.llm.generate(prompt, temperature=self.temperature)
            for _ in range(self.n_samples)
        ]

        verdicts = [nli_fn(premise=sample, hypothesis=claim) for sample in samples]
        # verdicts are one of: "entailment", "contradiction", "neutral"
        counts = Counter(verdicts)

        support_score = counts["entailment"] / self.n_samples
        contradiction_score = counts["contradiction"] / self.n_samples

        return {
            "support_score": support_score,
            "contradiction_score": contradiction_score,
            "likely_hallucinated": support_score < 0.5 and contradiction_score > 0.2,
            "samples": samples,
        }
```

The intuition scales down to something even simpler and cheaper that's usable as a first-pass filter without an NLI model: ask for N independent answers to the same factual question and check whether they agree on the key entity (the name, the number, the date). High variance across independently sampled answers to the identical question is itself a usable, cheap uncertainty signal — it doesn't tell you *which* answer (if any) is correct, but it reliably flags "this question sits in a low-confidence region for this model," which is exactly the population of outputs worth routing to a human reviewer or a more expensive verification pass rather than shipping directly.

```python
def variance_based_uncertainty(llm, question: str, n_samples: int = 5) -> dict:
    answers = [llm.generate(question, temperature=0.7) for _ in range(n_samples)]
    normalized = [a.strip().lower() for a in answers]
    most_common, count = Counter(normalized).most_common(1)[0]
    agreement_rate = count / n_samples
    return {
        "agreement_rate": agreement_rate,
        "high_uncertainty": agreement_rate < 0.6,
        "most_common_answer": most_common,
        "all_answers": answers,
    }
```

The cost trade-off is explicit and worth stating in an interview: self-consistency checking multiplies inference cost by N (you're making N calls instead of one), so it's typically reserved for high-stakes claims rather than applied to every generation — a common pattern is to run the cheap primary generation once, then trigger a self-consistency pass only for outputs that will be shown to a user without human review, or that feed into a downstream automated action.

For tasks with a single verifiable final answer — math word problems, multi-step arithmetic, anything where the answer is a discrete value rather than free text — self-consistency has a well-known, cheaper variant called **self-consistency decoding** or majority-vote sampling: sample several independent reasoning chains, extract just the final answer from each, and take the mode. This works because independent reasoning chains that happen to make the same arithmetic or logical error are much rarer than chains that each independently arrive at the correct answer via different (but individually valid) reasoning paths — errors tend to be idiosyncratic to a specific sampled chain, while correctness is comparatively more reproducible across chains.

```python
def majority_vote_answer(llm, problem: str, n_samples: int = 7) -> dict:
    """Cheaper than full NLI-based self-consistency because it only needs
    an exact-match comparison on the extracted final answer, not a semantic
    entailment judgment over full free-text responses."""
    chains = [llm.generate(f"{problem}\nThink step by step.", temperature=0.7)
              for _ in range(n_samples)]
    final_answers = [extract_final_answer(chain) for chain in chains]
    winner, votes = Counter(final_answers).most_common(1)[0]
    return {
        "answer": winner,
        "vote_share": votes / n_samples,
        "low_confidence": votes / n_samples < 0.5,
        "chains": chains,
    }


def extract_final_answer(chain: str) -> str:
    # In practice: regex for "the answer is X" / "####  X" style markers,
    # or a small parsing LLM call for less structured reasoning chains.
    import re
    match = re.search(r"(?:answer is|=)\s*([\-\d.]+)\s*$", chain.strip())
    return match.group(1) if match else chain.strip().splitlines()[-1]
```

## Detection Technique: Token-Level Uncertainty (Logprobs and Entropy)

When the model API exposes token-level log-probabilities (as OpenAI's completions and chat completions APIs do via `logprobs`), you get a much cheaper uncertainty signal than resampling, because it requires zero extra generations — the information comes for free from the single call you were making anyway. The idea is that the model's own probability distribution over the next token is itself a form of confidence estimate: a token the model assigns 98% probability to was a near-certain continuation given everything before it, while a token it assigned 30% probability to (with several other tokens close behind) reflects genuine indecision at that point in generation, and factual claims generated through a low-probability token sequence are measurably more likely to be wrong than claims generated through high-probability sequences.

```python
import math


def token_level_uncertainty(logprobs: list[float]) -> dict:
    """logprobs: per-token log-probabilities returned alongside a completion.
    Perplexity is the geometric mean of inverse token probabilities --
    a direct, cheap read on how 'surprised' the model was by its own output."""
    avg_logprob = sum(logprobs) / len(logprobs)
    perplexity = math.exp(-avg_logprob)

    # entropy proxy: how much the least-confident tokens drag down the average
    min_logprob = min(logprobs)
    low_confidence_tokens = sum(1 for lp in logprobs if lp < -2.0)  # p < ~13%

    return {
        "perplexity": perplexity,
        "min_token_logprob": min_logprob,
        "low_confidence_token_count": low_confidence_tokens,
        "flag_for_review": perplexity > 5.0 or low_confidence_tokens > 3,
    }
```

The caveat that keeps this from being a complete solution on its own: token-level probability measures the model's confidence in its *phrasing*, not necessarily in the *truth* of the underlying claim. A model can be extremely confident (low perplexity, high per-token probability) while stating a wrong fact fluently and consistently, precisely because fluent delivery of a memorized-but-wrong fact is itself a high-probability continuation — this is the same fluency-without-truth-signal problem described at the top of this chapter, just visible at the token level instead of the full-response level. Logprob-based uncertainty is best used as a cheap, always-on triage signal that routes the genuinely uncertain tail of outputs toward more expensive checks (self-consistency, retrieval verification), not as a standalone hallucination detector.

## Benchmarks for Measuring Hallucination

Several public benchmarks exist specifically to quantify hallucination rate rather than general capability, and knowing them signals familiarity with how the field measures this problem empirically rather than anecdotally. **TruthfulQA** poses questions deliberately designed to elicit common misconceptions and false beliefs that appear frequently in training data (folk wisdom, conspiracy-adjacent claims, misremembered facts), specifically to test whether a model repeats popular-but-false statements versus giving the correct, less-common answer — it's a direct test of the "high-frequency-in-training-data" failure mode where the model's next-token objective favors the statistically common (but wrong) continuation. **HaluEval** provides a large set of examples paired with both correct and deliberately hallucinated responses across QA, dialogue, and summarization tasks, letting you benchmark a hallucination *detector's* discriminative ability rather than a generator's hallucination rate. **FActScore** decomposes long-form generations (like biography generation) into atomic facts and checks each against a reliable knowledge source, producing a fine-grained precision score rather than a single pass/fail judgment for the whole response — conceptually the same atomic-claim-decomposition idea used in the `FaithfulnessChecker` above, but applied against open-world knowledge sources like Wikipedia instead of a fixed, task-specific context. Knowing these benchmarks matters less for citing them by name than for understanding what they reveal in aggregate: hallucination rate is highly task- and domain-dependent (a model can score well on TruthfulQA's common-misconception traps while still hallucinating badly on narrow, low-frequency factual queries in a specific domain), which is exactly why, as with general capability evaluation, a task-specific hallucination eval built from your own domain's actual failure patterns is worth more operationally than any public leaderboard number.

## Detection Technique: Retrieval Grounding

The most widely deployed mitigation for factual hallucination is architectural rather than a post-hoc check: give the model source material to ground its answer in — this is precisely what retrieval-augmented generation (RAG) is for, and it converts the (extremely hard) problem "does the model's internal knowledge contain the correct fact" into the (much more tractable) problem "does this specific answer follow from this specific provided text," which is a form of reading comprehension the model is quite good at when the instructions are explicit about only using the provided context.

Grounding reduces hallucination but does not eliminate it, for two reasons worth understanding precisely. First, retrieval itself can fail — if the retrieved passages don't actually contain the answer, the model faces the same choice it always faces: admit it doesn't know, or generate a plausible-sounding answer anyway, and without explicit instruction and reinforcement to prefer the former, models default to the latter, because "producing a plausible-sounding answer" remains the higher-probability continuation even when it isn't well-supported by the given context. Second, even with good retrieval, the model can still ignore or misread the provided context and revert to its parametric (pretrained) knowledge, especially when parametric knowledge is very strong/confident and conflicts with the retrieved passage — a well-documented failure mode where a model "knows" an outdated fact strongly enough that fresh, correct context in the prompt fails to override it.

```python
GROUNDED_ANSWER_PROMPT = """Answer the question using ONLY the information in
the provided context. If the context does not contain enough information to
answer confidently, say "I don't have enough information to answer this"
rather than guessing or relying on outside knowledge.

Context:
{context}

Question: {question}

Answer (cite which part of the context supports each claim):
"""


def grounded_generate(llm, question: str, retrieved_passages: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(retrieved_passages))
    prompt = GROUNDED_ANSWER_PROMPT.format(context=context, question=question)
    return llm.generate(prompt, temperature=0)
```

The instruction to explicitly cite which part of the context supports each claim is not decorative — forcing the model to produce attributions gives you a mechanically checkable structure afterward (does citation `[2]` actually contain the claim it's attached to), and it also appears to reduce hallucination somewhat on its own, likely because generating an explicit citation requirement shifts the model's generation process toward retrieval-conditioned tokens rather than free association from parametric memory. This citation-then-verify pattern is the backbone of the faithfulness check described next.

## Detection Technique: Fact-Verification Passes

A fact-verification (or faithfulness-checking) pass takes a generated answer and its source context and explicitly checks, claim by claim, whether each claim is supported. This is usually implemented as a second LLM call — sometimes the same model, sometimes a separate one — whose entire job is verification rather than generation, which tends to produce more reliable judgments than asking the generating call to self-critique in the same turn, because a fresh call isn't anchored by its own prior generation and is more willing to flag problems in it.

```python
import json


class FaithfulnessChecker:
    def __init__(self, verifier_llm):
        self.verifier_llm = verifier_llm

    def decompose_into_claims(self, answer: str) -> list[str]:
        """Break the answer into atomic, independently checkable claims --
        checking claim-by-claim is far more reliable than asking 'is this
        whole paragraph faithful' in one shot, since a single unsupported
        clause in an otherwise-grounded paragraph is easy to miss holistically."""
        prompt = f"""Break the following text into a list of atomic factual
        claims, one per sentence fragment that asserts something checkable.

        Text: {answer}

        Return JSON: {{"claims": ["claim 1", "claim 2", ...]}}
        """
        result = json.loads(self.verifier_llm.generate(prompt, temperature=0))
        return result["claims"]

    def verify_claim(self, claim: str, sources: list[str]) -> dict:
        prompt = f"""Source documents:
        {json.dumps(sources)}

        Claim: "{claim}"

        Is this claim directly supported by the source documents? Answer
        strictly based on what the sources say, not on general knowledge.

        Return JSON: {{"verdict": "supported" | "unsupported" | "contradicted",
                        "evidence": "<quote from source or null>"}}
        """
        return json.loads(self.verifier_llm.generate(prompt, temperature=0))

    def check(self, answer: str, sources: list[str]) -> dict:
        claims = self.decompose_into_claims(answer)
        verdicts = [self.verify_claim(c, sources) for c in claims]

        supported = sum(1 for v in verdicts if v["verdict"] == "supported")
        contradicted = [v for v in verdicts if v["verdict"] == "contradicted"]

        return {
            "faithfulness_score": supported / len(claims) if claims else 1.0,
            "total_claims": len(claims),
            "contradicted_claims": contradicted,
            "claim_verdicts": list(zip(claims, verdicts)),
        }
```

Decomposing into atomic claims before verifying is the detail that separates a robust faithfulness checker from a weak one — asking a judge "is this whole answer faithful to the sources, yes or no" in a single pass tends to average over problems, letting one unsupported sentence hide inside an otherwise well-grounded paragraph. Per-claim verification also gives you something actionable: you know exactly which sentence to flag, strike, or regenerate, rather than a single opaque faithfulness score for the whole response.

## Mitigation at the Prompt Level

The cheapest mitigations require no architecture change, only prompt design, and are worth exhausting before reaching for something more expensive. Explicitly instructing the model to say "I don't know" or "I'm not certain" when appropriate, and — importantly — giving concrete examples of what an acceptable "I don't know" response looks like in a few-shot prompt, measurably reduces confident fabrication, because you're supplying exactly the training-time-missing signal ("here's when hedging was the right call") in-context. Asking the model to show its reasoning before its final answer (chain-of-thought) surfaces intermediate steps that a downstream check (or a human) can inspect for unsupported leaps, even though CoT alone doesn't guarantee the stated reasoning caused the final answer. Requiring citations/attributions for every factual claim, as shown above, creates a checkable structure and nudges generation toward context-conditioned tokens. Lowering temperature reduces the variance that drives some fabrication, though it does not address fabrications the model would generate confidently even at temperature zero — low temperature reduces *stochastic* hallucination, not *systematic* hallucination rooted in wrong or missing parametric knowledge.

```python
UNCERTAINTY_AWARE_PROMPT = """Answer the question. If you are not confident
in part or all of your answer, say so explicitly rather than presenting
uncertain information as fact. Use phrases like "I believe" or "I'm not
certain, but" for anything you are not highly confident about.

Example of a good response when uncertain:
Q: What was Acme Corp's exact revenue in Q3 2023?
A: I don't have verified, up-to-date financial data for Acme Corp's Q3 2023
results. You should check their official investor relations filing for an
accurate figure.

Example of a bad response (confident fabrication):
Q: What was Acme Corp's exact revenue in Q3 2023?
A: Acme Corp reported Q3 2023 revenue of $847 million, up 12% year over year.

Question: {question}
Answer:
"""
```

## Mitigation at the Architecture Level

Beyond prompting, several architectural patterns directly target hallucination. **Retrieval augmentation**, discussed above, is the dominant architectural mitigation for factual queries with an external, checkable knowledge source. **Tool use for verifiable facts** — routing numeric computation to a calculator tool, current-events questions to a search tool, and structured data queries to a database or API rather than letting the model recall the answer from memory — eliminates an entire class of hallucination by construction, since the model is no longer generating the fact itself, only orchestrating a call to a system that can. **Constrained decoding / structured output** (JSON schema enforcement, grammar-constrained generation) doesn't reduce factual hallucination directly, but it eliminates a related failure mode — malformed or partially-fabricated structure — by making it impossible for the model to emit tokens outside a valid schema, which matters a great deal in agentic pipelines where a downstream system parses the output automatically. **Ensemble / multi-model cross-checking**, where two different models (or the same model with different prompts/contexts) independently answer the same question and disagreements are flagged for review, catches model-specific hallucination patterns that a single self-consistency check within one model might share across all its samples if the confabulation is a systematic bias in that model rather than sampling noise.

```python
def cross_model_verification(question: str, model_a, model_b) -> dict:
    """Two independently-trained models are less likely to share the exact
    same hallucination, since their confabulations arise from different
    training data and objectives rather than a common blind spot."""
    answer_a = model_a.generate(question, temperature=0)
    answer_b = model_b.generate(question, temperature=0)

    agreement_prompt = f"""Do these two answers to the same question agree
    on the key facts (even if worded differently)?

    Question: {question}
    Answer A: {answer_a}
    Answer B: {answer_b}

    Return JSON: {{"agree": true/false, "discrepancy": "<description or null>"}}
    """
    # Use a third, neutral judge to avoid biasing toward either model's phrasing
    verdict = model_a.generate_json(agreement_prompt)
    return {"answer_a": answer_a, "answer_b": answer_b, **verdict}
```

## Mitigation at the Product Level

Some of the most effective mitigations aren't technical at all — they're product decisions about how much epistemic weight to put on the model's raw output before it reaches a user. **Confidence surfacing**: showing the user which parts of an answer are grounded in a cited source versus generated from the model's general knowledge, rather than presenting a uniform block of text with equal implied authority throughout. **Friction proportional to stakes**: a low-stakes creative-writing assistant can tolerate confident fabrication far better than a medical-information or financial-advice product, and the product design — how much verification runs before output reaches the user, whether a human reviews outputs before they're acted on, whether the system requires explicit user confirmation before taking an action based on a generated fact — should scale with the cost of being wrong, not be uniform across every feature in an app. **Explicit scope boundaries**: a support bot that's instructed (and evaluated) to stay within its knowledge base and refuse to answer outside it hallucinates far less than one given license to "be helpful" about anything, because the refusal boundary gives the model an explicit low-cost escape hatch instead of forcing a choice between admitting ignorance and confabulating. **User education and framing**: labeling AI-generated content as such, and setting user expectations that verification is the user's responsibility for consequential decisions, doesn't reduce the underlying hallucination rate but does reduce the real-world harm of the hallucinations that do slip through, which is a legitimate and often underrated mitigation lever, particularly when combined with the technical mitigations above rather than as a substitute for them.

## Putting the Layers Together

None of these techniques fully solves hallucination in isolation, and that's expected given the root cause — a next-token predictor with no ground-truth signal for its own uncertainty is never going to be made perfectly truthful by any single patch. The practical, production-grade approach layers cheap, always-on mitigations (grounding via retrieval, uncertainty-aware prompting, structured output) with selective, expensive verification (fact-checking passes, self-consistency sampling, cross-model checks) triggered specifically for high-stakes outputs, and backs all of it with product-level design that limits the blast radius of whatever hallucinations still get through. The chapter on guardrails that follows covers how these detection and mitigation techniques get wired into an actual request pipeline as enforceable checks rather than best-effort prompting advice.
