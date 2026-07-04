# RAG Evaluation and Metrics

## 1. Why RAG evaluation is genuinely hard

Every earlier chapter in this section made an implicit promise: chunk more carefully, choose a better embedding model, add hybrid retrieval, rerank the candidates, try Self-RAG or GraphRAG, and the system will get better. None of that is verifiable without a way to measure "better" that is more rigorous than eyeballing a handful of outputs and deciding they look fine. Evaluation is not a nice-to-have appendix to a RAG project; it is the instrument that tells you whether every technique in Chapters 1 through 7 actually helped, by how much, and — critically — which part of the pipeline it helped.

The reason RAG evaluation resists a single obvious metric is that a RAG system fails along two structurally different axes, and a naive end-to-end check ("is the final answer correct?") cannot tell you which one broke. Consider a user asking "What is our policy on refunds for digital products purchased more than 30 days ago?" and the system produces a wrong answer. There are at least three distinct root causes that produce an identical symptom:

- **Retrieval failure.** The retriever never found the chunk that discusses digital product refund exceptions — it returned chunks about physical product returns instead — so the generator was working from incomplete or irrelevant evidence and did the best it could with what it had. The generator is innocent here; you cannot prompt-engineer your way out of context that was never retrieved.
- **Generation failure.** The correct chunk was retrieved and sitting right there in the context window, but the model ignored it, or misread it, or filled in the gap with something it recalled from pretraining that sounds plausible but is wrong. Retrieval did its job perfectly; the generator hallucinated or misused correct context anyway.
- **Both.** Retrieval returned partial, ambiguous context, and the generator compounded the problem by guessing at the missing piece instead of saying "I don't have enough information."

An end-to-end "is the answer right" score is binary noise with respect to this distinction — a 0 tells you the system failed, and nothing else. It gives you no signal on whether to invest the next sprint in a better chunking strategy, a better embedding model, or a stricter generation prompt that forces the model to abstain when context is insufficient. This is the single most important idea in this chapter, and it is why every serious RAG evaluation framework — RAGAS, TruLens, DeepEval, ARES, and the internal eval harnesses most production teams build — insists on decomposing evaluation into **retrieval metrics** (did we find the right evidence?), **generation metrics** (did we use that evidence correctly?), and only then an **end-to-end metric** (did the user get a good answer?), evaluated as three separate numbers rather than one conflated score.

Practically, this means every RAG evaluation run should report at minimum: a retrieval-quality number (precision/recall over retrieved chunks against a known-relevant set), a faithfulness number (is the generated answer grounded in what was retrieved), an answer-quality number (does the answer actually address the question), and only as a summary, an end-to-end correctness number. When faithfulness is high but end-to-end correctness is low, the fix is upstream — better retrieval or a better golden set — not a generation prompt tweak. When faithfulness is low but retrieval metrics are high, the fix is downstream — the generator is ignoring good context, which is a prompting, model-choice, or context-formatting problem. Without separating these, teams routinely spend weeks tuning prompts to fix what was actually a retrieval bug, or replacing a perfectly good embedding model to fix what was actually a generator that hallucinates regardless of what it's given.

The rest of this chapter builds up the standard toolkit for measuring both dimensions, popularized by frameworks like RAGAS (Retrieval-Augmented Generation Assessment), and then addresses the two practical problems that determine whether any of these metrics are trustworthy in the first place: where the ground truth to evaluate against comes from, and how much to trust an LLM when it is the thing doing the grading.

## 2. RAGAS-style metrics in depth

The defining idea behind the RAGAS family of metrics is that a single holistic "rate this answer 1 to 10" score from an LLM is unreliable and uninterpretable — it conflates too many judgments into one number and gives a judge model too much room to be inconsistent. The RAGAS approach instead decomposes each metric into an **atomic, checkable unit** (a factual claim, a generated question, a retrieved chunk, a ground-truth statement) and computes the metric as an aggregate over many small, mechanically verifiable judgments rather than one big subjective one. This section builds four such metrics from first principles, each with a working approximation of the real algorithm.

### 2.1 Faithfulness

Faithfulness asks a narrow, specific question: of everything the generated answer asserts, how much of it can actually be traced back to the retrieved context, as opposed to being pulled from the model's parametric memory (i.e., hallucinated, even if it happens to be true in the real world)? A faithful answer might still be a bad answer — but an unfaithful one is a RAG system failing at the one thing RAG exists to fix.

The mechanism, rather than asking an LLM "is this answer faithful, yes/no," works in two decomposed steps:

1. **Claim decomposition.** Ask an LLM to break the generated answer into a list of atomic factual statements — self-contained assertions that could each be independently true or false, with pronouns resolved and compound sentences split. "Refunds are available within 30 days for physical products, and digital products are non-refundable once downloaded" decomposes into two claims, not one.
2. **Claim verification.** For each atomic claim, ask an LLM (or a smaller, cheaper NLI — natural language inference — model fine-tuned for entailment) whether the claim is entailed by, contradicted by, or unaddressed by the retrieved context. Only claims marked "entailed" count as supported.

Faithfulness is then the simple ratio: `supported_claims / total_claims`. Decomposing first is what makes this trustworthy — asking a judge to evaluate one long multi-part answer holistically lets a single wrong sentence buried in an otherwise-good paragraph slip through unnoticed, whereas checking each atomic claim in isolation forces every assertion to earn its keep.

```python
"""
faithfulness.py

RAGAS-style faithfulness: decompose the generated answer into atomic
claims, verify each claim against the retrieved context, and score
faithfulness as the fraction of claims that are actually supported.

Dependencies: openai
    pip install openai
"""

import json
import os
from dataclasses import dataclass
from typing import List
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o-mini"


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str      # "supported" | "unsupported" | "contradicted"
    reasoning: str


def decompose_into_claims(answer: str) -> List[str]:
    """Break a generated answer into atomic, independently checkable claims."""
    prompt = f"""Break the following answer into a list of atomic factual
claims. Each claim must be a single, self-contained statement that could be
independently checked as true or false. Resolve pronouns and shared subjects
so each claim stands alone. Return ONLY a JSON array of strings, no prose.

Answer:
\"\"\"{answer}\"\"\"
"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    # Models are more reliable emitting a JSON object than a bare array;
    # ask for {"claims": [...]} and unwrap it.
    payload = json.loads(response.choices[0].message.content)
    return payload.get("claims", [])


def verify_claim_against_context(claim: str, context: str) -> ClaimVerdict:
    """Check whether a single atomic claim is entailed by the retrieved context."""
    prompt = f"""You are verifying whether a claim is supported by a given
context. Answer strictly based on the context, not on outside knowledge.

Context:
\"\"\"{context}\"\"\"

Claim:
\"\"\"{claim}\"\"\"

Is the claim explicitly supported (entailed) by the context, explicitly
contradicted by it, or not addressed by it at all? Respond with a JSON
object: {{"verdict": "supported" | "contradicted" | "unaddressed",
"reasoning": "<one sentence>"}}.
"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return ClaimVerdict(
        claim=claim,
        verdict=result.get("verdict", "unaddressed"),
        reasoning=result.get("reasoning", ""),
    )


def faithfulness_score(answer: str, retrieved_context: str) -> dict:
    """
    Returns the faithfulness ratio plus per-claim verdicts for debuggability.
    Only "supported" counts toward the numerator; "contradicted" and
    "unaddressed" both count as unfaithful, since an unaddressed claim is,
    by definition, not something the retrieved context backs up.
    """
    claims = decompose_into_claims(answer)
    if not claims:
        return {"score": 1.0, "claims": [], "note": "no factual claims found"}

    verdicts = [verify_claim_against_context(c, retrieved_context) for c in claims]
    supported = sum(1 for v in verdicts if v.verdict == "supported")
    return {
        "score": supported / len(verdicts),
        "num_claims": len(verdicts),
        "num_supported": supported,
        "claims": [v.__dict__ for v in verdicts],
    }


if __name__ == "__main__":
    context = (
        "Our refund policy allows customers to request a full refund within "
        "30 days of purchase, provided the product is unused and in its "
        "original packaging. Digital products are non-refundable once "
        "downloaded, except where required by local consumer law."
    )
    answer = (
        "You can get a full refund within 30 days if the item is unused. "
        "Digital products can also be refunded within 60 days of purchase."
    )
    result = faithfulness_score(answer, context)
    print(json.dumps(result, indent=2))
    # Expect the first claim to be "supported" and the second — a fabricated
    # 60-day digital refund window that contradicts the context — to be
    # flagged as "contradicted", dragging the faithfulness score below 1.0.
```

Two practical notes on faithfulness in production. First, using a dedicated NLI model (e.g., a cross-encoder fine-tuned on MNLI or a purpose-built factual-consistency model) for the verification step instead of a second LLM call is common when cost or latency matters, since verification is a simpler binary/ternary classification task than generation and a much smaller model handles it well. Second, faithfulness measures groundedness, not correctness — an answer can be perfectly faithful to context that is itself wrong or outdated, and faithfulness will happily score it 1.0. That's why faithfulness must be read alongside context precision/recall (is the context even good?) rather than in isolation.

### 2.2 Answer relevancy

Faithfulness alone is not sufficient — an answer can be scrupulously grounded in the retrieved context and still fail to actually address what the user asked. A generator that receives a question about digital-product refund timelines and responds with an accurate but generic summary of the entire refund policy is faithful (every sentence traces to the context) yet unhelpful, because it didn't focus on what was asked. Answer relevancy is the metric that catches evasiveness, incompleteness, and off-topic drift.

The RAGAS mechanism for this is elegantly indirect: rather than asking a judge "does this answer address the question," it works backward. Given the generated answer, prompt an LLM to generate several plausible questions that this answer would be a good, complete response to. If the answer genuinely and fully addresses the original question, the questions generated from it should closely resemble the original question. If the answer is generic, partial, or evasive, the reverse-engineered questions will drift toward something broader or different — because a vague answer is compatible with many different questions, not just the one that was actually asked. Concretely: embed the original question and each generated question, compute cosine similarity between the original and each generated one, and average — a low average similarity is a symptom of an answer that wandered from the question.

```python
"""
answer_relevancy.py

RAGAS-style answer relevancy: generate several questions the given answer
would plausibly answer, embed them alongside the original question, and
average their cosine similarity. Low similarity implies the answer drifted
from what was actually asked (evasive, incomplete, or off-topic).

Dependencies: openai, numpy
    pip install openai numpy
"""

import json
import os
import numpy as np
from typing import List
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
NUM_GENERATED_QUESTIONS = 3


def generate_questions_for_answer(answer: str, n: int = NUM_GENERATED_QUESTIONS) -> List[str]:
    """Reverse-engineer plausible questions this answer would be a good response to."""
    prompt = f"""Given the following answer, generate {n} distinct questions
that this answer would be a complete and appropriate response to. Infer the
question purely from the answer's content — do not assume any specific
original question. Return a JSON object: {{"questions": [...]}}.

Answer:
\"\"\"{answer}\"\"\"
"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,  # some diversity across generated questions is desirable
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    return payload.get("questions", [])


def embed_texts(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-10, None)


def answer_relevancy_score(original_question: str, answer: str) -> dict:
    generated_questions = generate_questions_for_answer(answer)
    if not generated_questions:
        return {"score": 0.0, "note": "could not generate comparison questions"}

    all_texts = [original_question] + generated_questions
    embeddings = embed_texts(all_texts)
    original_vec, generated_vecs = embeddings[0], embeddings[1:]

    similarities = generated_vecs @ original_vec  # both are unit-normalized
    return {
        "score": float(np.mean(similarities)),
        "generated_questions": generated_questions,
        "per_question_similarity": [float(s) for s in similarities],
    }


if __name__ == "__main__":
    question = "How long do I have to return a digital product for a refund?"

    focused_answer = (
        "Digital products are non-refundable once downloaded, except where "
        "required by local consumer law."
    )
    evasive_answer = (
        "We offer a range of policies covering shipping, warranties, and "
        "customer support to make sure you have a great experience."
    )

    print("Focused answer:", answer_relevancy_score(question, focused_answer))
    print("Evasive answer:", answer_relevancy_score(question, evasive_answer))
    # Expect the focused answer's score to sit meaningfully higher than the
    # evasive one's, since questions reverse-engineered from a generic,
    # off-topic answer won't resemble the specific original question.
```

A subtlety worth calling out: because question generation is stochastic, answer relevancy scores have some run-to-run variance, which is why production implementations generate several questions per answer (three to five) and average rather than relying on a single generated question. It's also worth noting this metric is context-agnostic by design — it only compares the answer to the question, never to the retrieved chunks — which is exactly why it complements faithfulness rather than duplicating it: faithfulness checks answer-vs-context, answer relevancy checks answer-vs-question, and a genuinely good answer needs to score well on both independently.

### 2.3 Context precision

Faithfulness and answer relevancy evaluate the generation half of the pipeline. Context precision and context recall move upstream to evaluate retrieval itself, independent of whatever the generator later does with what it was given. Context precision asks: of the chunks the retriever actually returned, what fraction were relevant, and — just as importantly — were the relevant ones ranked near the top of the list?

The "near the top" clause is what separates this from a flat precision calculation. A retriever that returns five chunks, of which one is relevant and it happens to be ranked first, is doing meaningfully better than a retriever that returns the same five chunks with the one relevant chunk buried at rank five — because most downstream consumers (the generator's context window, a reranker, a human skimming results) weight earlier positions more heavily, and many pipelines only pass the top-k after a cutoff that a low-ranked relevant chunk might not survive. RAGAS therefore computes context precision in a way that mirrors **average precision** from classic information retrieval: at each rank position, compute the precision of everything retrieved up to and including that position, but only accumulate that precision value into the final score at the positions where the item is actually relevant. This rewards a retriever for pushing relevant items early, not merely for including them somewhere in the list.

```python
"""
context_precision.py

RAGAS-style context precision: rank-weighted precision over retrieved
chunks, analogous to average precision. Relevant chunks that rank earlier
contribute more to the score than the same relevant chunk ranked later.

No external API calls needed here if relevance labels are already known
(e.g., from a golden set); an `is_relevant` LLM-judge function is included
for the common case where relevance must be inferred against the question.
"""

import json
import os
from typing import Callable, List
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o-mini"


def llm_judge_is_relevant(question: str, chunk_text: str) -> bool:
    """Ask an LLM whether a retrieved chunk is relevant to answering the question."""
    prompt = f"""Question: {question}

Retrieved passage:
\"\"\"{chunk_text}\"\"\"

Does this passage contain information that is useful for answering the
question, even partially? Respond with a JSON object: {{"relevant": true|false}}.
"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return bool(json.loads(response.choices[0].message.content).get("relevant", False))


def context_precision_at_k(
    question: str,
    ranked_chunks: List[str],
    relevance_fn: Callable[[str, str], bool] = llm_judge_is_relevant,
) -> dict:
    """
    ranked_chunks: chunk texts in the order the retriever returned them
    (index 0 = top-ranked / highest similarity).

    Computes rank-weighted precision:
        sum over relevant positions i of (precision@i) / (number of relevant chunks)
    which is exactly average precision restricted to this single query.
    """
    relevance_flags = [relevance_fn(question, chunk) for chunk in ranked_chunks]
    num_relevant_total = sum(relevance_flags)
    if num_relevant_total == 0:
        return {"score": 0.0, "relevance_flags": relevance_flags, "note": "no relevant chunks retrieved"}

    precision_sum = 0.0
    relevant_seen = 0
    for i, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            relevant_seen += 1
            precision_at_i = relevant_seen / i
            precision_sum += precision_at_i

    return {
        "score": precision_sum / num_relevant_total,
        "relevance_flags": relevance_flags,
        "num_relevant": num_relevant_total,
        "num_retrieved": len(ranked_chunks),
    }


if __name__ == "__main__":
    question = "How long do I have to return a physical product for a full refund?"

    # A retriever that ranks the relevant chunk first.
    good_ranking = [
        "Customers may request a full refund within 30 days of purchase for unused physical products.",
        "Shipping times for domestic orders are 3 to 5 business days.",
        "International orders can take 7 to 21 business days depending on customs.",
    ]
    # The same three chunks, but with the relevant one buried last.
    poor_ranking = list(reversed(good_ranking))

    print("Good ranking:", context_precision_at_k(question, good_ranking))
    print("Poor ranking:", context_precision_at_k(question, poor_ranking))
    # Both retrieve the same single relevant chunk, but the good ranking
    # should score close to 1.0 while the poor ranking scores much lower,
    # since burying the one relevant chunk at rank 3 tanks precision@3.
```

Notice that both example rankings contain exactly one relevant chunk out of three — a flat precision calculation would score them identically at 1/3. The rank-weighted version correctly separates them, which is the entire point: it measures whether the retriever's *ordering* is doing useful work, not just whether relevant material exists somewhere in the returned set.

### 2.4 Context recall

Context precision tells you whether what was retrieved is good; it says nothing about whether everything necessary was retrieved at all. Context recall closes that gap: of everything actually required to construct a correct, complete answer — as determined by a trusted reference/ground-truth answer, not by the system's own output — what fraction was present somewhere in the retrieved context?

This matters because precision and recall fail independently and require different fixes. A retriever can have perfect precision (every retrieved chunk is relevant) while still missing a second, equally necessary piece of information that a different chunk in the corpus contains — in which case precision looks great and the final answer is still incomplete or wrong, because recall is low. Low context recall is a decisive signal, because it means no improvement to the generator can fix the problem: the generator cannot cite or reason about information it was never given. The fix has to happen upstream — better chunking so the needed fact isn't split across an awkward boundary, a better embedding model, hybrid retrieval to catch a keyword the dense model missed, or simply a larger top-k.

The RAGAS mechanism decomposes the *ground-truth* answer (not the generated one) into atomic statements — the same claim-decomposition idea as faithfulness, but applied to the reference answer — and then checks whether each atomic statement can be attributed to, i.e. found or entailed within, the retrieved context. Recall is the fraction of ground-truth statements that were attributable.

```python
"""
context_recall.py

RAGAS-style context recall: decompose a trusted ground-truth answer into
atomic statements, then check how many of those statements are actually
attributable to (supported by) the retrieved context. Low recall means
necessary information was never retrieved, regardless of generation quality.

Dependencies: openai
    pip install openai
"""

import json
import os
from typing import List
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o-mini"


def decompose_ground_truth(reference_answer: str) -> List[str]:
    """Same claim-decomposition idea as faithfulness, applied to the reference answer."""
    prompt = f"""Break the following reference answer into a list of atomic
factual statements, each independently checkable. Return a JSON object:
{{"statements": [...]}}.

Reference answer:
\"\"\"{reference_answer}\"\"\"
"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content).get("statements", [])


def is_statement_attributable(statement: str, retrieved_context: str) -> bool:
    """Check whether a ground-truth statement can be found/supported in the retrieved context."""
    prompt = f"""Retrieved context:
\"\"\"{retrieved_context}\"\"\"

Statement:
\"\"\"{statement}\"\"\"

Can this statement be attributed to (i.e., is it supported by or derivable
from) the retrieved context above? Respond with a JSON object:
{{"attributable": true|false}}.
"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return bool(json.loads(response.choices[0].message.content).get("attributable", False))


def context_recall_score(reference_answer: str, retrieved_context: str) -> dict:
    statements = decompose_ground_truth(reference_answer)
    if not statements:
        return {"score": 1.0, "statements": [], "note": "no statements found in reference"}

    attributable_flags = [is_statement_attributable(s, retrieved_context) for s in statements]
    num_attributable = sum(attributable_flags)
    return {
        "score": num_attributable / len(statements),
        "num_statements": len(statements),
        "num_attributable": num_attributable,
        "detail": list(zip(statements, attributable_flags)),
    }


if __name__ == "__main__":
    reference_answer = (
        "Customers get a full refund within 30 days for unused physical "
        "products in original packaging. Digital products are non-refundable "
        "once downloaded, except where local consumer law requires otherwise."
    )

    # Retrieval that only surfaced the physical-product refund chunk.
    incomplete_context = (
        "Our refund policy allows customers to request a full refund within "
        "30 days of purchase, provided the product is unused and in its "
        "original packaging."
    )
    result = context_recall_score(reference_answer, incomplete_context)
    print(json.dumps(result, indent=2, default=str))
    # Expect roughly half the ground-truth statements to be unattributable,
    # since the digital-product exception was never retrieved at all —
    # a clean example of a low-recall retrieval failure.
```

With all four metrics in hand, a full RAGAS-style evaluation run over a batch of (question, retrieved_context, generated_answer, reference_answer) tuples produces four separate scores per example, and averaging each across the eval set gives four separate aggregate numbers rather than one blended score — which is precisely the diagnostic separation Section 1 argued for. A regression in context recall after a chunking change points you straight at the chunker; a regression in faithfulness after swapping the generation model points you straight at the model or the grounding prompt; a drop in context precision after adding more documents to the corpus points you at the retriever's ability to discriminate, not at chunk size.

## 3. Building a golden evaluation set

None of the metrics above mean anything without something to check against — context precision and recall need a notion of "which chunks are actually relevant," and answer-quality checks are far more reliable with a trusted reference answer to compare against rather than judged in a vacuum. This golden set (also called a reference set, eval set, or test set) is the single most valuable and most neglected artifact in a RAG project; teams that skip building one end up unable to tell whether any change they ship is actually an improvement.

There are two complementary sources for a golden set, and mature teams use both.

**Real user queries from logs.** Wherever a RAG system already has any production or beta traffic, mining actual user queries is the highest-fidelity source of evaluation examples available, because it reflects the real distribution of how people phrase questions, what topics they actually ask about, and what edge cases show up (typos, multi-part questions, questions that assume context from a prior turn). The catch is that real queries do not come with ground-truth relevant chunks or reference answers attached — those still have to be constructed, typically by a subject-matter expert reviewing the query, checking the corpus for the actually-relevant passages, and writing or approving a reference answer. This is labor-intensive, which is exactly why it's usually reserved for a curated subset of the highest-value or most failure-prone query patterns observed in production, rather than attempted for the entire log.

**Synthetic generation from the corpus.** For everything else — and especially for bootstrapping an eval set before any production traffic exists — the standard technique is to generate question/answer pairs directly from known chunks, which has the enormous practical advantage that the ground-truth relevant chunk is known automatically, by construction, because you generated the question from that specific chunk. Feed a chunk into an LLM, ask it to write a question that the chunk answers plus a reference answer derived only from that chunk, and tag the resulting pair with the source chunk's ID. Do this across a representative sample of the corpus and you get a large eval set with zero manual relevance labeling.

Synthetic generation is only useful if it's stratified across the query types a production system actually has to survive, not just simple factoid lookups:

- **Simple factoid** — answerable from a single chunk with a direct lookup ("What is the refund window for physical products?").
- **Multi-hop / compositional** — requires combining information from two or more chunks that may not be adjacent or from the same document ("If I bought a digital product 45 days ago in a country with strong consumer protection law, am I eligible for a refund?").
- **Comparison** — requires contrasting two entities or policies retrieved from different parts of the corpus ("How does the refund window differ between physical and digital products?").
- **Summarization / broad** — requires synthesizing across many chunks or an entire document rather than pinpointing one passage ("Summarize our entire returns and refunds policy.").
- **Out-of-scope / unanswerable** — deliberately outside what the corpus covers, and the only correct behavior is an explicit "I don't know" or "this isn't covered," never a confident guess. This category is easy to forget and disproportionately important: an eval set built only from answerable questions can never detect a system that has quietly learned to hallucinate a plausible-sounding answer whenever retrieval comes up empty, because every example in the set rewards answering. Without unanswerable examples, a regression where the model starts confidently fabricating answers to out-of-scope questions will sail through evaluation undetected.

The last, unglamorous but non-negotiable step is human review. LLM-generated questions have three characteristic failure modes that make raw synthetic output untrustworthy as ground truth: they can be trivially easy (restating the chunk almost verbatim, testing nothing about real retrieval difficulty), ambiguous (a question with multiple valid interpretations that make the "correct" answer contestable), or — most insidiously — answerable from a completely different chunk than the one used to generate them, because the corpus contains overlapping or duplicated information elsewhere. A reviewer with domain knowledge needs to at minimum spot-check the generated set, discard or fix the bad examples, and verify the tagged "source chunk" really is the only correct source (or explicitly relabel it as multi-source). Treating unreviewed synthetic examples as ground truth is a common way for a RAG evaluation pipeline to quietly measure noise instead of quality.

```python
"""
golden_set_generator.py

Synthetic golden-set generation: for each chunk, prompt an LLM to write a
question the chunk answers plus a reference answer derived only from that
chunk. The chunk_id is attached automatically, giving ground-truth
relevant-chunk labels for free. Includes basic stratification across query
types, including an explicit "unanswerable" category generated from
plausible-sounding but out-of-corpus questions.

Dependencies: openai
    pip install openai
"""

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import List
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
GENERATOR_MODEL = "gpt-4o-mini"

QUERY_TYPES = ["factoid", "multi_hop", "comparison", "summarization"]


@dataclass
class GoldenExample:
    question: str
    reference_answer: str
    source_chunk_ids: List[str]
    query_type: str
    answerable: bool
    needs_human_review: bool = True  # every synthetic example starts unreviewed


def generate_factoid_example(chunk_id: str, chunk_text: str) -> GoldenExample:
    prompt = f"""Given the passage below, write ONE specific question that
can be fully and directly answered using only this passage, plus the
correct answer derived only from this passage. Avoid questions that are
answerable by restating the passage almost verbatim — make it a genuine
question a user might ask.

Passage:
\"\"\"{chunk_text}\"\"\"

Return a JSON object: {{"question": "...", "answer": "..."}}.
"""
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    return GoldenExample(
        question=payload["question"],
        reference_answer=payload["answer"],
        source_chunk_ids=[chunk_id],
        query_type="factoid",
        answerable=True,
    )


def generate_multi_hop_example(chunk_a_id: str, chunk_a_text: str,
                                chunk_b_id: str, chunk_b_text: str) -> GoldenExample:
    prompt = f"""Given the two passages below, write ONE question that can
only be fully answered by combining information from BOTH passages
together — not from either passage alone — plus the correct combined
answer.

Passage A:
\"\"\"{chunk_a_text}\"\"\"

Passage B:
\"\"\"{chunk_b_text}\"\"\"

Return a JSON object: {{"question": "...", "answer": "..."}}.
"""
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    return GoldenExample(
        question=payload["question"],
        reference_answer=payload["answer"],
        source_chunk_ids=[chunk_a_id, chunk_b_id],
        query_type="multi_hop",
        answerable=True,
    )


def generate_unanswerable_example(chunk_id: str, chunk_text: str) -> GoldenExample:
    """
    Generate a plausible-sounding question that this chunk (and, by
    construction of the prompt, the surrounding domain) does NOT answer —
    the correct system behavior is to refuse/abstain, not guess.
    """
    prompt = f"""Given the passage below, write ONE question that sounds
like it belongs to the same domain/topic as the passage, but that this
passage does NOT contain enough information to answer. The question should
be plausible enough that a user might actually ask it, not absurd or
off-topic.

Passage:
\"\"\"{chunk_text}\"\"\"

Return a JSON object: {{"question": "..."}}.
"""
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    return GoldenExample(
        question=payload["question"],
        reference_answer="I don't have enough information to answer this.",
        source_chunk_ids=[],
        query_type="unanswerable",
        answerable=False,
    )


def build_golden_set(chunks: List[dict], unanswerable_fraction: float = 0.15) -> List[GoldenExample]:
    """
    chunks: list of {"chunk_id": str, "text": str}
    Produces a stratified set: factoids from single chunks, multi-hop pairs
    from adjacent chunks, and an unanswerable slice sized as a fraction of
    the total so the eval set can actually catch a system that never
    learned to say "I don't know".
    """
    examples: List[GoldenExample] = []

    for chunk in chunks:
        examples.append(generate_factoid_example(chunk["chunk_id"], chunk["text"]))

    for i in range(len(chunks) - 1):
        examples.append(
            generate_multi_hop_example(
                chunks[i]["chunk_id"], chunks[i]["text"],
                chunks[i + 1]["chunk_id"], chunks[i + 1]["text"],
            )
        )

    num_unanswerable = max(1, int(len(chunks) * unanswerable_fraction))
    for chunk in random.sample(chunks, k=min(num_unanswerable, len(chunks))):
        examples.append(generate_unanswerable_example(chunk["chunk_id"], chunk["text"]))

    return examples


if __name__ == "__main__":
    sample_chunks = [
        {"chunk_id": "policy-001-0", "text": (
            "Customers may request a full refund within 30 days of purchase "
            "for unused physical products in original packaging."
        )},
        {"chunk_id": "policy-001-1", "text": (
            "After 30 days, only store credit is issued, valid for 12 months "
            "from the date of issue."
        )},
        {"chunk_id": "policy-002-0", "text": (
            "Domestic orders typically arrive within 3 to 5 business days "
            "using standard shipping."
        )},
    ]

    golden_set = build_golden_set(sample_chunks)
    for example in golden_set:
        print(json.dumps(asdict(example), indent=2))
    # Every example is emitted with needs_human_review=True by design — this
    # is a reminder, not a formality: a reviewer must confirm each question
    # is unambiguous, non-trivial, and correctly attributed before it is
    # trusted as ground truth for context precision/recall scoring.
```

A practical rule of thumb: aim for a golden set with at least 100-200 reviewed examples before trusting aggregate metrics on it, spread roughly across the five query-type categories above, and re-run the full stratified generation process whenever the corpus changes substantially (a new document category, a major content rewrite) rather than letting the eval set silently drift out of sync with what the system actually indexes.

## 4. LLM-as-judge evaluation and its pitfalls

Look back at every metric implemented in Section 2: claim decomposition, entailment checking, question generation, relevance judgment, statement attribution — every single one of them ultimately asks an LLM to make a judgment call. This is not incidental; it is close to unavoidable, because the alternative (exact string matching, keyword overlap, n-gram metrics like BLEU/ROUGE) is well known to correlate poorly with actual answer quality for open-ended generation — two answers can be semantically identical with zero word overlap, or share every word while one subtly reverses the meaning. LLM-as-judge is the dominant approach in production RAG evaluation precisely because it can approximate human judgment on exactly this kind of open-ended correctness, at a fraction of the cost and latency of a human reviewer. But that power comes with well-documented, reproducible biases that a senior engineer needs to actively account for rather than discover the hard way in a postmortem.

**Position bias.** In pairwise comparisons — "which of these two answers is better, A or B" — judge models exhibit a measurable tendency to favor whichever answer is shown in a particular position, and the direction of the bias (favoring first or favoring second) varies by model and prompt. This means a naive pairwise eval that always presents the new pipeline's answer as "A" and the baseline's as "B" is measuring position preference conflated with actual quality, not quality alone.

**Verbosity bias.** Judge models tend to rate longer, more elaborated answers as higher quality independent of whether the extra length adds correct or useful information. A generator that pads every answer with restated context and hedged caveats can score better under a naive judge than a generator that gives a crisp, fully correct, and appropriately concise answer — which is the opposite of what most products actually want.

**Self-preference bias.** A model used as judge tends to rate outputs generated by its own model family more favorably than equivalent-quality outputs from a different model family, plausibly because it recognizes and prefers its own characteristic phrasing, structure, and reasoning style. This is the reason using the same model as both the RAG system's generator and its own evaluator is a genuine methodological hazard, not just a theoretical concern — you risk a system that looks like it's improving because the judge is rewarding its own stylistic fingerprint rather than genuine answer quality, especially when comparing against a competing generator model.

**Prompt sensitivity and inconsistency.** The same judge, given the same inputs, can produce different scores across repeated calls, and small, semantically irrelevant rewordings of the judging prompt (reordering criteria, rephrasing the rating scale) can shift scores meaningfully. This effect is amplified at nonzero sampling temperature, where the judge's own token-level randomness compounds with genuine ambiguity in borderline cases.

None of these biases mean LLM-as-judge should be abandoned — human evaluation at RAG-system scale is neither fast nor cheap enough to run continuously — but they do mean a judge needs to be treated as an instrument that requires calibration, not an oracle. A handful of practical mitigations address most of the risk. Replace open-ended "rate this answer from 1 to 10" prompts with structured rubrics that spell out explicit, separately-scored criteria (factual accuracy, completeness relative to the reference, appropriate refusal when unanswerable, absence of unsupported claims) — a rubric narrows the judge's degrees of freedom and makes scores more reproducible and more interpretable when they disagree with each other. Prefer reference-based grading over reference-free grading whenever a golden set with a trusted reference answer is available, since giving the judge a concrete "here is what a correct answer looks like" anchor to compare against is a fundamentally easier and more consistent task than asking it to assess quality from first principles with no ground truth in hand; reference-free grading should be reserved for the parts of the pipeline (like answer relevancy in Section 2.2) that are structurally reference-free by design. Run judges at temperature 0 for reproducibility, and for genuinely high-stakes evaluation gates (a release decision, a regression threshold that blocks deployment), sample the judge multiple times and use a self-consistency aggregate (majority vote or average) rather than trusting a single call. Periodically pull a random sample of LLM-judge scores and have a human independently score the same examples, then measure agreement with a statistic like Cohen's kappa (which corrects for chance agreement, unlike raw percent-agreement) — this is the only way to know whether a judge that looks well-calibrated in isolation is actually tracking human judgment, and it should be repeated periodically, not done once and assumed to hold forever as the system, the query distribution, and even the judge model version change. And finally, where feasible, use a different model family as judge than the one used as the RAG system's own generator, specifically to sidestep self-preference bias when comparing generator options.

## 5. Offline vs online evaluation in production

Everything covered so far — RAGAS-style metrics, a golden set, an LLM judge — constitutes **offline evaluation**: running a fixed, curated set of examples through the pipeline in a controlled setting, typically wired into CI/CD so that every pipeline change (a new chunking strategy, a different embedding model, an added reranking stage, a rewritten generation prompt) is evaluated against the golden set before it is allowed to deploy, with a regression gate — for example, "faithfulness must not drop by more than 2 points and context recall must not drop by more than 3 points relative to the current production baseline, or the deployment is blocked." Offline evaluation's defining strengths are that it's cheap to run as often as needed (on every pull request, if the golden set and judge calls are budgeted for it), fully reproducible (the same inputs produce comparable results run to run, modulo judge temperature/consistency concerns from Section 4), and fast to attribute a regression to a specific change, since it runs in isolation against a fixed set right after the change that might have caused it. Its defining weakness is equally structural: it can only ever be as good as the golden set's coverage of real query behavior, and no golden set — however carefully stratified across factoid, multi-hop, comparison, summarization, and unanswerable categories — anticipates every way real users will actually phrase questions, combine topics, or push the corpus's edges once the system is live.

**Online evaluation** is the complementary layer that observes the system under real production traffic rather than a fixed rehearsal set, and it comes in a few concrete forms. Implicit behavioral signals are the highest-volume and cheapest to collect: click-through or source-expansion behavior (did the user open the cited source, suggesting they wanted to verify or didn't trust the answer at face value), explicit thumbs-up/thumbs-down feedback attached to individual responses, the rate at which users immediately rephrase or re-ask a question (a strong proxy for dissatisfaction — nobody rephrases a question they were happy with the answer to), and session abandonment (the user leaves without any further interaction after a response, which can indicate either satisfaction or frustrated give-up, and usually needs to be interpreted alongside other signals to tell which). Explicit feedback — a user-submitted "this answer was wrong" report, a support escalation that traces back to a bad RAG response — is lower-volume but higher-signal, since it's an unambiguous, human-confirmed failure rather than an inferred one. Periodic human review of a random sample of live conversations rounds this out, catching problems that no automated signal surfaces well, such as an answer that is technically correct but delivered in a tone that damages trust.

The reason production teams need both layers rather than picking one is that they check different, complementary failure modes and neither can substitute for the other. Offline evaluation is the regression gate: it is what stands between "an engineer changed the chunking strategy" and "that change reached every user," and it is the only layer fast and cheap enough to run on every single pipeline change before deployment — you cannot wait for a week of live traffic and human review to decide whether to ship a reranker upgrade. But offline evaluation is fundamentally blind to distribution shift: if real users start asking about a new topic the golden set never anticipated, or start phrasing questions in a style the synthetic generator never produced, offline metrics stay flat and green while real user satisfaction quietly degrades. Online evaluation is what catches exactly that gap, because it is a direct read on the actual, current query distribution and actual user reactions rather than a curated rehearsal of it — and it also serves a second, equally important function as a sanity check on whether the offline metrics mean anything at all. If a pipeline change improves offline faithfulness and context recall scores but online thumbs-down rate and rephrase rate do not improve — or get worse — that divergence is a warning sign that the offline metric may be gamed, misaligned with what users actually value, or measuring an aspect of quality the golden set overweights relative to real usage. Treating an offline metric as trustworthy only for as long as it continues to track online satisfaction, and periodically checking that correlation explicitly, is what separates a genuinely useful evaluation program from an engineering team quietly optimizing a metric that stopped reflecting reality months ago.
