# Embedding Fine-Tuning and Domain Adaptation

## Why Off-the-Shelf Embeddings Sometimes Fail

General-purpose embedding models — OpenAI's `text-embedding-3` family, Cohere's embed models, open-source options like BGE, E5, or GTE — are trained on enormous, broad corpora to be good at general semantic similarity across a huge range of topics. This breadth is exactly what makes them useful defaults, and exactly what causes them to underperform in specialized domains where the notion of "similar" diverges from general-purpose semantic similarity. A legal document retrieval system needs to distinguish between contracts that differ in a single clause about liability limitation — a distinction a general embedding model, trained mostly on web text and never specifically taught that this particular clause type is the discriminating feature in this domain, may represent as nearly identical because the surrounding text is otherwise similar in surface form. A medical retrieval system needs "myocardial infarction" and "heart attack" pulled close together (a general model likely already does this reasonably well, since this pairing is common enough in general text) but also needs much finer distinctions between drug names, dosages, and specific clinical trial terminology that appear rarely enough in general pretraining data that the model never learned sharp, reliable geometry around them.

This is the central question this chapter answers: when is it worth the real engineering investment of fine-tuning an embedding model for your domain, versus accepting a strong off-the-shelf model and investing effort elsewhere (better chunking, reranking, hybrid search, prompt engineering downstream of retrieval)? Getting this judgment right is a genuinely high-value skill, because fine-tuning an embedding model is a nontrivial, multi-week investment with real risk of producing a worse model than you started with if done without care, and many teams reach for it prematurely when a cheaper intervention (a good cross-encoder reranker, in particular) would have solved the actual problem.

## Contrastive Learning: The Core Idea

Virtually all modern embedding models, whether trained from scratch or fine-tuned from a pretrained checkpoint, are trained with some form of **contrastive learning**. The core idea is simple to state: given a pair of inputs known to be semantically related (a "positive pair" — a query and its correct answer passage, two paraphrases of the same sentence, an image and its caption), push their embeddings closer together in the vector space; given inputs known to be unrelated (a "negative pair"), push their embeddings farther apart. Repeated over millions of such pairs, the model learns a general-purpose notion of "closeness in this space corresponds to semantic relatedness in the real world" — which is, definitionally, exactly the property that makes an embedding space useful for nearest-neighbor retrieval.

The reason this works better than, say, training the model to directly predict a similarity score via regression on labeled pairs is that contrastive learning doesn't require calibrated absolute similarity labels (which are expensive and often inconsistent to collect — is this pair a 0.7 or a 0.75?) — it only requires *relative* judgments (this pair is more similar than that pair), which are far easier to construct at scale, including automatically from naturally occurring data like query-click logs, question-answer pairs mined from forums, or even self-supervised signals like adjacent sentences in a document being treated as weak positive pairs.

## InfoNCE Loss

The dominant loss function for contrastive embedding training today is **InfoNCE** (Information Noise-Contrastive Estimation), and understanding its shape explains several important, sometimes counterintuitive properties of how well-trained embedding models behave. For a given anchor (typically a query), one positive example, and a batch of `N-1` negative examples, InfoNCE is:

```
L = -log( exp(sim(a, p) / tau) / sum_over_all_candidates_c( exp(sim(a, c) / tau) ) )
```

where `sim` is typically cosine similarity, `tau` is a temperature parameter, and the denominator sums over the positive plus every negative in the batch. This is exactly the softmax cross-entropy loss applied to a classification problem where the "correct class" is the positive example among a set of candidates — the model is trained to assign high probability mass to the true positive relative to every negative it's shown, in a single batched computation.

```python
import numpy as np

def info_nce_loss(anchor, positive, negatives, temperature=0.05):
    """anchor, positive: single embedding vectors (already L2-normalized)
    negatives: list of embedding vectors (already L2-normalized)"""
    def cos_sim(a, b):
        return np.dot(a, b)  # normalized vectors -> dot product IS cosine sim

    pos_sim = cos_sim(anchor, positive) / temperature
    all_sims = [pos_sim] + [cos_sim(anchor, neg) / temperature for neg in negatives]

    # Numerically stable softmax cross-entropy against index 0 (the positive)
    max_sim = max(all_sims)
    exp_sims = [np.exp(s - max_sim) for s in all_sims]
    loss = -(np.log(exp_sims[0] / sum(exp_sims)))
    return loss
```

Two properties fall directly out of this formula and are worth internalizing. First, **the temperature `tau` controls how hard the model is pushed to sharply separate positives from negatives** — a low temperature makes the softmax much peakier, meaning the loss punishes even small similarity gaps between the positive and the hardest negative very severely, which pushes the model toward sharper, more separated geometry but can make training less stable; a higher temperature is more forgiving and produces smoother, less aggressively separated embeddings. Second, and more practically important, **batch size directly controls how many negatives each training step effectively sees** (in the common in-batch-negatives setup, every other example's positive in the same batch is used as a free negative for the current anchor), so larger batches generally produce better-quality embeddings up to a point, purely because the model gets a harder, more informative contrastive signal per step — this is exactly why serious embedding training runs use large batch sizes (thousands of examples) and why in-batch negative sampling is the standard efficiency trick that makes large-scale contrastive training computationally tractable at all, since it reuses the same forward pass's positives as negatives for every other example in the batch rather than requiring separate negative examples to be encoded.

## Triplet Loss and Its Relationship to InfoNCE

Triplet loss predates InfoNCE as the standard contrastive objective and is conceptually simpler: for an anchor, a positive, and a single negative, it directly penalizes the case where the negative isn't at least a margin `m` farther from the anchor than the positive is:

```
L_triplet = max(0, dist(a, p) - dist(a, n) + margin)
```

```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    """distance-based triplet loss using squared Euclidean distance"""
    d_pos = np.sum((anchor - positive) ** 2)
    d_neg = np.sum((anchor - negative) ** 2)
    return max(0.0, d_pos - d_neg + margin)
```

The practical difference from InfoNCE is that triplet loss only ever compares against one negative at a time, giving a much weaker training signal per step and requiring careful negative selection to make training efficient (a randomly chosen negative is very often already trivially far away, contributing zero gradient since the max(0, ...) term is already satisfied). InfoNCE's batched, many-negatives-at-once formulation generally produces better embeddings faster, which is why it (and its close relatives, such as the multiple-negatives-ranking loss used in the popular `sentence-transformers` training library) has become the more common choice for training modern embedding models, while triplet loss remains conceptually important — it's the clearest way to explain *why* hard negatives matter so much, which is the next topic.

## Hard Negative Mining

The single highest-leverage detail in contrastive embedding training, more consequential in practice than the choice between triplet loss and InfoNCE, is the quality of the negative examples used during training. A "negative" that is trivially unrelated to the anchor (a random unrelated sentence from a completely different topic) provides almost no useful training signal once the model has learned even basic topic-level discrimination early in training — the model already easily assigns it a low similarity score, so the gradient from that example is close to zero and training stalls on the genuinely hard part of the problem: distinguishing subtly different but non-matching content.

**Hard negatives** are examples that are superficially or partially similar to the anchor but are not the correct match — a passage that shares vocabulary and topic with the query but doesn't actually answer it, or, in the classic dense retrieval literature, a passage retrieved by a baseline BM25 or embedding model as highly ranked that turns out, on inspection, to be irrelevant. Training against hard negatives forces the model to learn the finer-grained distinctions that actually matter for retrieval quality, precisely because they're the distinctions a first-pass retrieval system is prone to getting wrong. This is why virtually every serious embedding fine-tuning pipeline includes an explicit hard-negative-mining step rather than relying on random in-batch negatives alone.

A standard hard-negative-mining pipeline looks like: first, train (or use an existing) embedding model to retrieve top-k candidates for each training query; then, for each query, treat highly-ranked-but-incorrect candidates as hard negatives (filtering out any that a labeling process or heuristic identifies as actually being valid alternate answers, since falsely treating a correct answer as a "hard negative" actively corrupts training); then retrain or fine-tune using these mined hard negatives alongside the original positives. This can be iterated — mine harder negatives using the newly fine-tuned model, retrain again — which is essentially how several published state-of-the-art retrieval models (the E5 and BGE model families, for instance) describe their training recipes.

```python
def mine_hard_negatives(query, gold_passage_id, embed_fn, corpus_index, top_k=20):
    """Retrieve top_k candidates with a baseline embedder; anything highly ranked
    that isn't the known-correct passage is a hard negative candidate."""
    query_vec = embed_fn(query)
    candidates = corpus_index.search(query_vec, top_k=top_k)
    hard_negatives = [c for c in candidates if c.id != gold_passage_id]
    return hard_negatives[:10]  # keep the top-ranked wrong answers
```

A specific failure mode worth flagging explicitly, because it's subtle and easy to introduce silently: mined "hard negatives" can turn out to be **false negatives** — passages that are actually valid, correct matches for the query but weren't labeled as such in your ground truth (common in datasets built from click logs or single-answer QA pairs, where a query might legitimately have several correct answers but only one was recorded). Training against false negatives actively teaches the model to push apart things that should be close, which can measurably hurt quality in a way that's hard to diagnose from training loss alone since the loss curve looks perfectly normal. Serious hard-negative-mining pipelines include a denoising step — often using a stronger cross-encoder model to double-check that mined negatives are genuinely irrelevant before they're used in training.

## Generating Training Pairs When You Don't Have Enough

The most common blocker to fine-tuning isn't compute or expertise — it's the absence of enough good-quality positive pairs. A few practical sources are worth knowing because they cover most real situations. Historical query logs paired with the document a user clicked on, or that a support agent ultimately linked in resolving a ticket, are the highest-quality source, since they reflect genuine human relevance judgments rather than a heuristic — but they require an existing product with meaningful usage history, which a new system won't have. Existing structured content — FAQ pairs, documentation with a question-like heading and an answering body, glossary term-to-definition pairs — can be mined directly out of a domain's existing corpus without any new labeling effort, and is often sitting unused in a company's existing documentation.

When neither of those exists in sufficient volume, **LLM-generated synthetic pairs** have become a standard and reasonably effective substitute: prompt a strong LLM to generate a plausible question a user might ask that a given passage would answer, using the passage itself as the grounding context, producing a (synthetic query, real passage) positive pair at scale across an entire corpus. This approach (closely related to the method used to build several open synthetic retrieval training sets) trades some pair quality and diversity (LLM-generated queries tend to be more literal and less idiosyncratically phrased than real user queries) for the ability to bootstrap a training set size that would otherwise require months of real usage data to accumulate.

```python
def generate_synthetic_pair(passage: str, llm_client) -> dict:
    """Bootstrap a (query, passage) positive pair from a corpus passage alone --
    useful when real query logs don't exist yet."""
    prompt = (
        "Write one specific, realistic question that the following passage "
        "directly and completely answers. Return only the question.\n\n"
        f"Passage:\n{passage}"
    )
    synthetic_query = llm_client.complete(prompt)
    return {"query": synthetic_query.strip(), "positive_passage": passage}
```

A quality gate matters here just as much as it did for hard-negative mining: synthetic queries should be spot-checked (or scored by a separate LLM-as-judge pass) for whether they're actually specific and answerable from the passage alone, since a sloppy generation prompt can produce queries so generic ("what is this passage about?") that they teach the model nothing useful, or so vague they'd be equally well "answered" by dozens of unrelated passages, quietly injecting the same false-negative-adjacent noise problem discussed earlier for hard negatives.

## Evaluating a Fine-Tuned Model Honestly

Before trusting a fine-tuned embedding model in production, it needs to clear two separate evaluation bars, and conflating them is a common mistake. The first is **in-domain retrieval quality** — recall@k, MRR, or NDCG on a held-out set of real (or carefully vetted synthetic) query-passage pairs from your actual domain, compared directly against the off-the-shelf baseline on the exact same held-out set. This tells you whether the fine-tuning actually improved the thing you set out to improve. The second, easier to forget, is **regression testing against general capability** — if any fraction of your real traffic includes queries outside the narrow fine-tuning distribution (a compliance-document assistant that occasionally gets a general HR question, for instance), you need to verify the fine-tuned model hasn't gotten meaningfully worse at handling that traffic, since full fine-tuning in particular can cause a model to overfit sharply to its training distribution at the expense of everything else it used to handle reasonably well.

```python
def evaluate_recall_at_k(model, eval_pairs, corpus_embeddings, corpus_ids, k=10):
    """eval_pairs: list of (query, correct_passage_id). Compares against a
    fixed corpus index built with the SAME model being evaluated."""
    hits = 0
    for query, correct_id in eval_pairs:
        query_vec = model.embed(query)
        sims = corpus_embeddings @ query_vec  # assumes normalized embeddings
        top_k_ids = [corpus_ids[i] for i in np.argsort(-sims)[:k]]
        if correct_id in top_k_ids:
            hits += 1
    return hits / len(eval_pairs)

# Compare fairly: rebuild the corpus index separately for each model being tested,
# since a fine-tuned model's embeddings are not interchangeable with the baseline's.
baseline_recall = evaluate_recall_at_k(baseline_model, eval_set, baseline_corpus_emb, ids)
finetuned_recall = evaluate_recall_at_k(finetuned_model, eval_set, finetuned_corpus_emb, ids)
```

A subtlety worth calling out explicitly because it's a common evaluation bug: comparing a fine-tuned model's recall against a baseline's recall is only fair if the entire corpus is *re-embedded separately with each model* before comparison, exactly as the code above does — reusing the baseline's corpus embeddings while querying with the fine-tuned model's query embeddings compares two incompatible coordinate systems and produces meaningless, often catastrophically low, numbers that have nothing to do with either model's actual quality.

## Fine-Tuning Approaches

There are two broad ways to adapt an embedding model to a domain, differing in cost, data requirements, and how much of the model's general-purpose ability you preserve. **Full fine-tuning** continues training all (or most) of the model's weights on domain-specific contrastive pairs, typically starting from a strong pretrained checkpoint rather than from scratch. This can produce the largest quality gains for a well-defined domain but requires meaningfully more training data (typically tens of thousands of good-quality pairs at minimum to avoid overfitting or catastrophic forgetting of general capability) and more compute, and carries real risk of degrading the model's performance on anything outside the fine-tuning distribution if the domain data isn't diverse enough.

**Parameter-efficient fine-tuning** (LoRA-style adapters applied to the embedding model's transformer layers, or training only a lightweight projection head on top of frozen pretrained embeddings) is far cheaper computationally, requires much less domain-specific data to avoid overfitting, and is easier to iterate on, at the cost of a typically smaller (though often still meaningful) quality improvement compared to full fine-tuning. A lightweight linear or shallow-MLP projection head trained on top of frozen base embeddings — sometimes called "embedding adaptation" — is a particularly attractive middle ground for teams with a moderate amount of labeled domain pairs (low thousands) and limited ML infrastructure, since it can be trained quickly even on CPU-scale compute and, crucially, doesn't touch or risk regressing the underlying pretrained model at all.

```python
import numpy as np

class LinearProjectionAdapter:
    """A minimal illustration of the 'adapter on frozen embeddings' pattern:
    learn a linear map from the base model's space into a domain-adapted space,
    trained with a contrastive objective, without touching the base model."""

    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01

    def transform(self, embedding):
        projected = embedding @ self.W
        norm = np.linalg.norm(projected)
        return projected / norm if norm > 0 else projected

    def train_step(self, anchor_emb, positive_emb, negative_embs, lr=0.01, temperature=0.05):
        # In practice this gradient would be computed via autograd (PyTorch);
        # shown here only to illustrate that the *only* trainable parameter is W,
        # while the base embedding model stays completely frozen.
        pass
```

## When Fine-Tuning Is Worth It

The decision framework worth applying, roughly in order: first, establish a solid baseline with a strong off-the-shelf model and proper evaluation (recall@k, MRR, or NDCG against a labeled or carefully human-reviewed validation set specific to your domain and task — not a generic benchmark like MTEB, which won't tell you how the model performs on your actual data). A large fraction of "we need to fine-tune embeddings" conversations start without this baseline evaluation even existing, which makes it impossible to know whether fine-tuning is solving a real problem or a perceived one.

Second, before reaching for embedding fine-tuning, check whether a cross-encoder reranker on top of off-the-shelf retrieval closes the gap. Rerankers are typically far cheaper to deploy than fine-tuning a retrieval model (many strong open-source and hosted cross-encoder rerankers work well zero-shot, with no domain-specific training at all) and directly address the most common failure mode — good candidates being retrieved but ranked suboptimally — without touching your indexing pipeline or requiring you to re-embed your entire corpus. In practice, "off-the-shelf embeddings plus a good reranker" resolves the majority of quality complaints that initially look like they require embedding fine-tuning.

Fine-tuning genuinely earns its cost when the domain vocabulary and semantics diverge sharply enough from general text that even a reranker struggles — highly specialized technical, legal, medical, or internal-jargon-heavy domains where the base model's pretraining data plausibly contained very little relevant material at all, or where retrieval quality on your validation set plateaus well below your target even after adding reranking and improving chunking and hybrid search. It's also more clearly worth it when you have (or can construct) a genuinely solid quantity of representative positive pairs — from historical query logs, existing Q&A pairs, expert-curated examples, or synthetic pairs generated and filtered carefully — since without that data, fine-tuning has a real chance of producing a model that's worse than the off-the-shelf baseline it was meant to improve on, particularly if the fine-tuning set is small, narrow, or noisy.

Finally, remember the earlier point about embedding model incompatibility: choosing to fine-tune is not a small, reversible decision. It means committing to re-embedding your entire corpus, maintaining a custom model in your serving infrastructure indefinitely (including its own versioning, monitoring, and retraining cadence as your domain data evolves), and losing the ability to trivially swap in improved general-purpose models as they're released by model providers — a real ongoing cost that should be weighed explicitly against the quality gain, not treated as a one-time engineering project with no long-term maintenance implication.
