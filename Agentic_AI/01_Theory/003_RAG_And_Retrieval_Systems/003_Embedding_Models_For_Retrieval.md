# Embedding Models for Retrieval

Chapter 2 covered how to split source documents into chunks. Once you have chunks, the next question a RAG pipeline has to answer is: how do you turn a chunk of text (and later, a user's query) into something a machine can compare for similarity? The answer, almost universally in modern systems, is an embedding model — a function that maps text to a fixed-size dense vector such that semantically related text ends up nearby in vector space. Every downstream retrieval decision in a RAG system — which vector database to use, which ANN index type, whether hybrid search is worth the added complexity, how aggressively to rerank — is built on top of whatever this embedding model produces, and a weak or poorly-matched embedding model puts a ceiling on retrieval quality that no amount of cleverness downstream can fully compensate for. This chapter is about that function: how it works, the critical bi-encoder/cross-encoder architectural split that shows up in nearly every senior RAG interview, how to actually pick a model among the dozens available, and the dimensionality trade-offs that determine what that choice costs you in storage and latency. Chapter 4 builds on this to cover dense, sparse, and hybrid retrieval strategies; Chapter 6 goes deep on cross-encoder rerankers, which get introduced here only at the level needed to motivate why they exist.

## From Text to Vector: What an Embedding Model Actually Does

At a mechanical level, a text embedding model is a pipeline with four stages. First, a **tokenizer** breaks the input string into subword tokens using a fixed vocabulary (BPE, WordPiece, or SentencePiece, depending on the model family) and maps each token to an integer ID. Subword tokenization exists precisely so the model never hits a true out-of-vocabulary wall: a rare word like "retrieval-augmented" that never appeared as a whole unit in training data still decomposes into familiar pieces ("retriev", "##al", "-", "augmented") that the vocabulary does contain, at the cost of longer token sequences for unusual or domain-specific vocabulary — one reason legal and medical text, dense with multi-syllable Latinate terms, tends to tokenize into noticeably more tokens per word than everyday English, which quietly inflates both API cost and the chance of hitting a model's max-token ceiling for a given chunk. Second, those token IDs are fed through a **transformer encoder** — architecturally the same self-attention stack that powers language models, except used bidirectionally rather than causally, so every token's representation is informed by every other token in the input, not just the ones before it. This bidirectionality is deliberate and important: a decoder-style causal model, by construction, only lets each token attend to tokens before it, so the first token in a sequence can never incorporate information from the last one, which is a poor fit for building a representation meant to summarize an entire passage regardless of where the important words happen to sit. Classic embedding backbones like BERT and its many derivatives (RoBERTa, MPNet, DeBERTa) are encoder-only for exactly this reason, typically 6-24 transformer layers deep with hidden sizes of 384 to 1024. The output of this stage is not a single vector but a sequence of contextualized token embeddings, one per input token, each of dimensionality `d` (the model's hidden size).

A more recent trend worth knowing about for interviews is the emergence of **decoder-based LLM embedding models** — `e5-mistral-7b-instruct`, NVIDIA's `NV-Embed`, and Alibaba's `gte-Qwen2` family adapt a causal, decoder-only LLM backbone (originally trained for next-token prediction) into an embedding model, usually by removing the causal attention mask so the model can attend bidirectionally during embedding inference, pooling via the last token's hidden state rather than a mean or CLS token, and then contrastively fine-tuning on retrieval pairs exactly as described below. These models trade a much larger parameter count and higher inference cost for embeddings that, on several MTEB tasks, outperform traditional BERT-scale encoders — a reflection of how much general world knowledge a 7B-parameter LLM backbone brings relative to a 100-300M-parameter BERT-style encoder. Whether that quality gain is worth the extra latency and compute is a real production trade-off, not a strictly dominant choice: a 7B-parameter embedding model is meaningfully more expensive to self-host and slower per request than a 100-300M-parameter one, and for many corpora the traditional encoder-based bi-encoders in the model tour below are more than sufficient.

The third stage, **pooling**, is where the sequence of per-token vectors collapses into a single fixed-size vector representing the whole input, regardless of how many tokens it had. This step matters more than it looks like it should, and it's a common interview probe. Two pooling strategies dominate:

- **[CLS]-token pooling**: BERT-style models prepend a special `[CLS]` token to every input, and its final-layer representation is treated as a summary of the whole sequence. This works because during pretraining (next-sentence prediction, or a downstream classification fine-tuning objective), the model is explicitly trained to concentrate whole-sequence information into that one token's representation. If a model was never trained with an objective that rewards the CLS token for summarizing the sequence, using it for pooling produces a mediocre embedding — the vector exists, but nothing forced it to be meaningful as a summary.
- **Mean pooling**: average the contextualized embeddings of every token in the sequence (usually excluding padding tokens, via an attention-mask-weighted average). This is the more common choice for purpose-built sentence-embedding models — Sentence-BERT popularized it, and most of the sentence-transformers ecosystem defaults to it — because it doesn't rely on any single token having learned to be a good summarizer. Every token's representation contributes proportionally, which tends to produce smoother, more robust sentence-level vectors, especially for models that were fine-tuned specifically for semantic similarity/retrieval rather than classification. Empirically, mean pooling tends to outperform CLS pooling for retrieval tasks unless the model was specifically trained with CLS-based supervision (as some newer models are).

```python
import numpy as np

def mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """token_embeddings: (seq_len, hidden_dim) from the encoder's last layer.
    attention_mask: (seq_len,) with 1 for real tokens, 0 for padding."""
    mask = attention_mask[:, None].astype(np.float32)          # (seq_len, 1)
    summed = (token_embeddings * mask).sum(axis=0)              # (hidden_dim,)
    counted = np.clip(mask.sum(axis=0), a_min=1e-9, a_max=None) # avoid div by zero
    return summed / counted

def cls_pool(token_embeddings: np.ndarray) -> np.ndarray:
    """Just take the first token's (the [CLS] token's) final representation."""
    return token_embeddings[0]

# Toy illustration of why mean pooling is more robust when no objective has
# specifically trained a CLS token to summarize the sequence: an untrained or
# generically-pretrained CLS vector is just one token among many, with no
# special claim to representing the whole input, whereas the mean is a
# well-defined summary statistic regardless of what the model was optimized for.
toy_tokens = np.array([
    [0.9, 0.1],   # token 0 -- would be [CLS] in a BERT-style model
    [0.1, 0.9],
    [0.2, 0.8],
    [0.15, 0.85],
])
toy_mask = np.array([1, 1, 1, 1])
print(mean_pool(toy_tokens, toy_mask))  # ~[0.34, 0.66] -- reflects the whole sequence
print(cls_pool(toy_tokens))             # [0.9, 0.1] -- reflects only token 0
```

The fourth and final stage is usually an **L2 normalization** of the pooled vector, and sometimes a small linear projection layer to a target dimensionality. The result is a fixed-size dense vector — 384, 768, 1024, 1536, or 3072 dimensions depending on the model — such that cosine similarity or dot product between two such vectors approximates the semantic similarity of the two original texts. This is the entire contract an embedding model offers a retrieval system: geometric proximity for semantic proximity in meaning.

One operational detail that trips up production pipelines: every embedding model has a hard maximum input length (measured in tokens, not characters), and text beyond that limit is silently truncated by most APIs and libraries rather than raising an error — the tokens past the cutoff are simply dropped before the transformer ever sees them. This is exactly why chunk sizing (Chapter 2) has to be chosen with the target embedding model's token limit in mind, with margin to spare; a chunking strategy tuned for one embedding model's 512-token ceiling will silently lose content if the pipeline is later pointed at a smaller-limit model without re-checking that constraint. There is no universal fix for a chunk that's genuinely too long other than shortening it upstream — some teams average or max-pool multiple sub-chunk embeddings as a workaround, but that reintroduces exactly the pooling-quality trade-offs discussed above at a coarser granularity.

### Why the Space Behaves This Way: Contrastive Training, Briefly

It's worth understanding, at an intuitive level, why this geometric property emerges at all, without turning this chapter into a training deep-dive (that belongs with the fine-tuning material in the Vector Databases & Embeddings folder). Embedding models used for retrieval are trained with a **contrastive objective**: the model is shown a large number of positive pairs — a query and a document that actually answers it, or two paraphrases of the same sentence — along with a batch of unrelated "negative" texts. The loss function, most commonly a variant of **InfoNCE** (also called multiple-negatives-ranking loss in the sentence-transformers ecosystem), pushes the embedding of the positive pair to have high cosine similarity or dot product while pushing the similarity to every negative in the batch down.

```
loss = -log( exp(sim(q, d+) / tau) / sum(exp(sim(q, d_i) / tau) for d_i in {d+} union negatives) )
```

Here `tau` is a temperature parameter controlling how sharply the loss penalizes near-misses: a lower temperature makes the softmax more peaked, so the model is punished harder for giving even slightly too much similarity to a negative, which in practice makes training more sensitive to noisy or mislabeled pairs but yields a more sharply separated embedding space when it works. The negatives are often just the other positive documents belonging to other queries in the same training batch ("in-batch negatives"), which is what makes this loss cheap to compute at scale — a batch of 256 (query, document) pairs yields 256 positives and 255 negatives per query for free, with no separate negative-mining pass required. Better-performing models go further and add explicitly mined **hard negatives** — documents that are lexically or topically similar to the query but not actually a correct match (retrieved, for instance, by running a weaker retriever over the training corpus and keeping its highest-scoring wrong answers) — because random in-batch negatives are usually so obviously unrelated to the query that the model learns very little from them past an early stage of training; hard negatives are what actually sharpen the decision boundary between "related" and "superficially similar but wrong," which is the distinction that matters most at query time in a real retrieval system. Over millions of such pairs, the only way the model can satisfy the loss consistently is to organize its output space so that "semantically related" reliably correlates with "geometrically close." Nothing in the architecture guarantees this outcome directly — it is an emergent property of optimizing this specific objective at scale. This is also why an embedding model trained on, say, question-answering pairs from web forums might not transfer perfectly to legal contract retrieval: the geometry it learned encodes the notion of "relatedness" implicit in its training pairs, not some universal, domain-independent notion of meaning. We return to that gap in the domain adaptation section below.

### Symmetric vs. Asymmetric Retrieval

A subtlety that falls directly out of how the training pairs are constructed, and that's easy to overlook when picking or evaluating a model, is the difference between **symmetric** and **asymmetric** semantic search. Symmetric search is when the query and the target document are the same kind of text and roughly the same length — deduplicating near-identical support tickets, or finding a previously-asked question similar to a new one. Asymmetric search is the far more common RAG situation: a short, often terse query ("how do I reset my password") being matched against long, differently-structured passages from a knowledge base or manual. A model trained primarily on symmetric pairs (sentence-to-sentence paraphrase data, for instance) can underperform on asymmetric retrieval, because nothing in its training ever taught it that a five-word question and a two-hundred-word passage can legitimately be "close" despite looking structurally nothing alike. This is precisely the practical motivation behind E5's `"query: "` / `"passage: "` prefixes and Cohere's `input_type` parameter discussed later in this chapter — they exist because the same underlying model has to handle both roles well, and giving it an explicit signal about which role a given input is playing measurably helps asymmetric retrieval quality. When evaluating a candidate embedding model for a RAG system, it's worth explicitly checking whether its documentation or model card claims strong asymmetric retrieval performance (usually validated against MS MARCO or Natural Questions-style benchmarks) rather than only symmetric similarity performance, since RAG is almost always the asymmetric case.

Some newer models push this instruction idea further than a fixed two-word prefix. `e5-mistral-7b-instruct`, Nomic's instruction-tuned variants, and the `instructor` family of embedding models accept a full natural-language task instruction concatenated in front of the input text — for example, `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}"` — rather than a fixed token like `"query: "`. The instruction can, in principle, describe an arbitrary retrieval intent (question-answering, fact-checking, duplicate detection), letting a single model adapt its embedding behavior per task without any fine-tuning, at the cost of a somewhat longer, more fragile prompt that has to be constructed correctly and consistently at both indexing and query time — the same silent-mismatch failure mode discussed below for the simpler `"query: "`/`"passage: "` prefixes, just with more moving parts to get wrong.

## Bi-Encoders vs. Cross-Encoders

This is one of the most reliably asked architectural questions in RAG interviews, because it explains a design decision that shows up in almost every production retrieval system: why there are two different model architectures doing what looks superficially like the same job of "scoring relevance."

A **bi-encoder** (also called a dual encoder) runs the query and each document through the encoder **independently**, producing two separate fixed-size vectors, and then compares them with a cheap similarity function — cosine similarity or dot product. The critical consequence is that document embeddings never depend on the query. This means every document in a corpus can be embedded once, offline, ahead of any query being asked, and stored in an index. At query time, the system only has to embed the query itself (a single fast forward pass) and then compare that one query vector against millions or billions of pre-computed document vectors using an approximate nearest neighbor (ANN) index — sub-linear time, because the expensive part (encoding) already happened at ingestion time. Every embedding model discussed in this chapter — OpenAI's `text-embedding-3-*`, Cohere's `embed-v3`/`v4`, Voyage, BGE, E5, and the rest — is a bi-encoder used exactly this way.

A **cross-encoder**, in contrast, concatenates the query and a single candidate document into one input sequence (typically `[CLS] query [SEP] document [SEP]`) and passes that combined sequence through a single transformer, letting every query token attend to every document token and vice versa via full self-attention, before producing a single relevance score (often through a classification head on the `[CLS]` output). This joint attention is what makes cross-encoders substantially more accurate at judging fine-grained relevance — the model can directly reason about whether specific query terms are addressed by specific parts of the document, rather than relying on two independently-computed vectors happening to point in a similar direction. The cost is that this scoring cannot be pre-computed at all: a cross-encoder score only exists for a specific (query, document) pair, computed at query time, which means scoring `N` documents against one query requires `N` full transformer forward passes, every single time a new query arrives. This does not scale to searching a corpus of millions of documents directly — it would mean millions of transformer forward passes per query.

It's worth noting that both architectures are typically built from the same family of transformer backbones — a cross-encoder reranker is very often literally the same BERT-style architecture as a bi-encoder embedding model, just fine-tuned with a different objective and a different input format. A bi-encoder is trained contrastively, as described above, to make its two-vector output geometrically meaningful in isolation. A cross-encoder, by contrast, is typically fine-tuned with a pointwise or pairwise ranking loss directly against labeled relevance judgments — datasets like MS MARCO's passage ranking task provide exactly this: real queries paired with passages labeled relevant or not — so the model learns to output a calibrated relevance score for a jointly-encoded pair rather than to organize an embedding space at all. The architectural difference that matters for system design isn't really "different model families," it's "independent scoring that can be precomputed" versus "joint scoring that can't," and that's a direct consequence of what each model's input format and training objective make possible at inference time.

```python
# Bi-encoder: independent encoding, precomputable, scales via ANN search
def biencoder_score(query: str, document: str, encode_fn) -> float:
    q_vec = encode_fn(query)      # can be computed at query time (cheap, one pass)
    d_vec = encode_fn(document)   # precomputed and indexed at ingestion time
    return float(np.dot(q_vec, d_vec))  # assumes both vectors are L2-normalized

# In practice: document vectors are computed once, at ingestion:
doc_vectors = {doc_id: encode_fn(text) for doc_id, text in corpus.items()}  # offline
# ...and at query time you only embed the query and search the precomputed index:
query_vector = encode_fn(user_query)  # online, single forward pass

# Cross-encoder: joint encoding, cannot be precomputed, one full pass per pair
def crossencoder_score(query: str, document: str, model, tokenizer) -> float:
    combined = tokenizer(query, document, return_tensors="pt", truncation=True)
    logits = model(**combined).logits         # full cross-attention between q and d tokens
    return float(logits[0][0])                 # a single relevance score for this exact pair

# This has to be re-run for every candidate document, every time:
scores = [crossencoder_score(user_query, doc_text, model, tokenizer)
          for doc_text in candidate_documents]   # only feasible for a small candidate set
```

It helps to make the scaling gap concrete rather than just qualitative. With a bi-encoder, searching a 10-million-document corpus at query time costs one encoder forward pass (for the query) plus an ANN lookup that touches, in practice, a tiny fraction of the index — typically single-digit milliseconds end to end on modern ANN indexes. Reranking those same 10 million documents with a cross-encoder directly would mean 10 million full transformer forward passes for a single query — utterly infeasible at interactive latencies no matter how much hardware is thrown at it. Even reranking a modest 100-candidate shortlist with a cross-encoder costs 100 forward passes, which is why cross-encoder rerankers are batched and run only after a bi-encoder (or a sparse method like BM25) has already cut the candidate set down from millions to dozens or low hundreds — the two-stage shape isn't a stylistic preference, it's the only combination of these two architectures that is computationally tractable at real corpus sizes.

This asymmetry — bi-encoders are cheap and approximate but scale to entire corpora; cross-encoders are expensive and precise but only scale to small candidate sets — is exactly why production RAG systems almost never use only one of the two. The standard pattern is a **two-stage retrieval pipeline**: a bi-encoder performs broad first-stage retrieval, pulling, say, the top 50-200 candidates out of a corpus of millions using fast ANN search, and then a cross-encoder reranks just those handful of candidates to produce the final, high-precision ordering that actually gets used to build the prompt context. The bi-encoder's job is recall (don't miss the right document out of millions); the cross-encoder's job is precision (put the actually best few documents at the very top). Chapter 6 covers the mechanics, training, and deployment of cross-encoder rerankers in depth — the summary here is only meant to explain why this two-stage shape exists at all, since that reasoning is exactly what interviewers are probing for when they ask "why not just use a cross-encoder for everything?"

A compact summary of the three architectures before moving on:

| Architecture | Document encoding precomputable? | Relevance accuracy | Typical role |
|---|---|---|---|
| Bi-encoder | Yes — fully independent of query | Good | First-stage retrieval over the whole corpus |
| Late-interaction (ColBERT-style) | Yes — per-token, independent of query | Better | First-stage retrieval or lightweight rerank |
| Cross-encoder | No — requires the query at encode time | Best | Reranking a small top-k shortlist (Chapter 6) |

### A Middle Ground: Late-Interaction Models

A third architectural family, worth being aware of even though it's less commonly the default choice, sits between these two extremes: **late-interaction models**, of which **ColBERT** is the canonical example. Instead of pooling a document down to one vector (bi-encoder) or requiring the query at encoding time (cross-encoder), a late-interaction model keeps a separate contextualized embedding for *every token* in the document, precomputes and stores all of them, and at query time computes a fine-grained similarity by comparing every query token's embedding against every stored document token embedding, keeping the best match per query token and summing those best-match scores (an operation usually called "MaxSim"). Because document-side token embeddings are still precomputed independently of any query, this preserves the bi-encoder's key scalability property — no forward pass over the document is needed at query time — while recovering some of the fine-grained, token-level matching precision that makes cross-encoders more accurate than single-vector bi-encoders. The cost is a different one: storing a vector per document token rather than one vector per document multiplies storage substantially (though follow-up work like ColBERTv2 applies aggressive compression to offset this), and the MaxSim comparison, while much cheaper than a full transformer forward pass, is still more expensive per candidate than a single dot product. In practice, late-interaction models occupy a real niche in production retrieval stacks — sometimes as the first-stage retriever itself, sometimes as a cheaper stand-in for a full cross-encoder rerank — but the dominant pattern most teams reach for first is still the plain bi-encoder-plus-cross-encoder-reranker pipeline described above, which is why the remainder of this book treats that as the default and treats late-interaction models as a specialized variant worth knowing exists rather than a primary design pattern.

## A Tour of Embedding Models and How to Choose One

The number of viable embedding models has grown quickly, and the practical question senior engineers get asked is rarely "explain how BERT works" — it's "which embedding model would you pick for this system, and why." The honest answer always starts with the same tension: managed API convenience versus self-hosted control, and general-purpose quality versus domain fit.

**OpenAI's `text-embedding-3` family** is the default many teams reach for first, mainly because it's already available if they're using OpenAI for generation, and because it's genuinely strong on general-domain English retrieval benchmarks. `text-embedding-3-small` produces 1536-dimensional vectors and is priced at roughly **$0.02 per 1 million tokens**; `text-embedding-3-large` produces 3072-dimensional vectors (both support Matryoshka truncation, covered below) and is priced at roughly **$0.13 per 1 million tokens**. It is worth flagging explicitly that pricing is quoted per 1 million tokens, not per 1,000 — a surprising number of blog posts and even some internal docs get this wrong by a factor of 1000 because older embedding models (and most LLM completion pricing tables) were historically quoted per-1K, and the convention shifted. Getting this factor wrong produces wildly incorrect cost projections, so it's worth being able to compute it directly:

```python
def embedding_cost_usd(num_tokens: int, price_per_million: float) -> float:
    return (num_tokens / 1_000_000) * price_per_million

# A corpus of 50,000 chunks, ~300 tokens each, embedded once at ingestion:
total_tokens = 50_000 * 300  # 15,000,000 tokens

print(embedding_cost_usd(total_tokens, price_per_million=0.02))  # text-embedding-3-small: $0.30
print(embedding_cost_usd(total_tokens, price_per_million=0.13))  # text-embedding-3-large: $1.95
```

At this scale the absolute dollar cost of embedding is trivial for either model, which is exactly why, in practice, the dominant cost driver for a managed embedding API is almost never the per-token embedding price itself — it's the downstream storage and ANN search infrastructure sized for whatever dimensionality was chosen, which is precisely the trade-off the dimensionality section below quantifies.

**Cohere's `embed-v3` and newer `embed-v4`** models are a common alternative, with strong multilingual coverage and a notable feature called "compression-aware training," where the model is explicitly optimized to remain accurate even when its output embeddings are quantized to int8 or binary for storage savings — relevant for teams running huge corpora where float32 storage cost dominates the budget.

**Voyage AI's embedding models** (`voyage-3`, `voyage-3-large`, and domain-specific variants like `voyage-code-3` and `voyage-law-2`) have built a reputation specifically around high retrieval accuracy, and are notable because Anthropic's own documentation recommends Voyage as the embedding provider of choice for teams building RAG systems on Claude, since Anthropic does not offer a first-party embedding model. Voyage also publishes domain-specialized models (code, legal, finance), which is a meaningfully different strategy from providers who ship one general-purpose model and hope it generalizes.

On the **open-source** side — relevant whenever data residency, cost at massive scale, or offline/air-gapped deployment rules out a managed API — the standard names to know are `BAAI/bge-large-en-v1.5` and its successors (BGE models are trained with a strong contrastive recipe and have topped MTEB's English retrieval leaderboard repeatedly), `intfloat/e5-large-v2` (notable for requiring `"query: "` and `"passage: "` prefixes prepended to inputs at both training and inference time — forgetting this prefix at inference is a real, common bug that silently degrades retrieval quality), the Alibaba **GTE** family, **Nomic Embed** (open-source, Matryoshka-trained, and notable for being fully reproducible with published training data), and the lightweight `sentence-transformers/all-mpnet-base-v2` and `all-MiniLM-L6-v2` — the latter is a frequent choice for local development, prototyping, or resource-constrained environments because it's small enough (roughly 80MB) to run comfortably on a laptop CPU, at some cost to retrieval quality versus larger models. **Jina Embeddings v3** is another strong open option, notable for supporting long inputs (8192 tokens) and task-specific LoRA adapters (retrieval, classification, clustering) baked into a single base model that gets switched via a task parameter at inference time.

This asymmetric-input pattern used by E5 isn't unique to that model family — Cohere's API exposes an equivalent idea through an explicit `input_type` parameter (`search_query` versus `search_document`) rather than a text prefix the caller has to remember to prepend, which is a slightly safer API design precisely because it can't be silently forgotten the way a manually-typed string prefix can. The underlying reason either mechanism helps is the same: queries and documents are often structurally different kinds of text (a query is short and interrogative; a document is longer and declarative), and telling the model which role a given input is playing lets it apply slightly different internal processing before pooling, rather than treating both as generic, undifferentiated text.

| Model | Dimensions | Max input tokens | Notes |
|---|---|---|---|
| OpenAI `text-embedding-3-small` | 1536 (MRL-truncatable) | 8191 | ~$0.02 / 1M tokens |
| OpenAI `text-embedding-3-large` | 3072 (MRL-truncatable) | 8191 | ~$0.13 / 1M tokens |
| Cohere `embed-v4` | up to 1536 | ~128K | strong multilingual, quantization-aware |
| Voyage `voyage-3-large` | 1024-2048 | 32K | domain variants (code, law, finance) |
| BAAI `bge-large-en-v1.5` | 1024 | 512 | open-source, strong English retrieval |
| intfloat `e5-large-v2` | 1024 | 512 | requires query:/passage: prefixes |
| Nomic Embed v1.5 | 768 (MRL-truncatable) | 8192 | open, reproducible, Matryoshka-trained |
| Jina Embeddings v3 | 1024 | 8192 | task-specific LoRA adapters |
| sentence-transformers `all-mpnet-base-v2` | 768 | 384 | solid general-purpose, self-hosted |
| sentence-transformers `all-MiniLM-L6-v2` | 384 | 256 | lightweight, fast, laptop-friendly |

Self-hosting an open embedding model changes the cost and operational profile of a RAG system in ways worth naming explicitly. Instead of paying per-token API fees with no infrastructure to manage, the team now owns GPU or CPU provisioning, batching logic, model version upgrades, and monitoring for throughput and latency — a `bge-large-en-v1.5` or `e5-large-v2` deployment on a single modern GPU can comfortably embed several thousand short passages per second when requests are batched properly, which for most corpora makes ingestion-time embedding cheap and fast, while query-time embedding (usually one short string, unbatched) is dominated by fixed per-request overhead rather than raw model compute. A common pattern regardless of provider is to **cache embeddings** aggressively: document embeddings almost never need to be recomputed unless the source text or the model itself changes, so a content hash keyed cache (embed once, store the vector alongside a hash of the input text and the model name/version) avoids redundant embedding calls entirely when documents are re-ingested unchanged — a detail that matters more than it sounds, because re-embedding an entire corpus after a pipeline bug fix or a chunking strategy change is one of the most common sources of unplanned embedding API cost.

```python
import hashlib

def cache_key(text: str, model_name: str) -> str:
    # Include the model name/version in the key: the same text embedded by two
    # different models (or two versions of the same model) must never collide.
    digest = hashlib.sha256(f"{model_name}:{text}".encode("utf-8")).hexdigest()
    return digest

def embed_with_cache(text: str, model_name: str, encode_fn, cache: dict) -> np.ndarray:
    key = cache_key(text, model_name)
    if key not in cache:
        cache[key] = encode_fn(text)   # only pay for an actual embedding call on a miss
    return cache[key]
```

This cache is most valuable during the iteration loop most teams actually go through when tuning a chunking strategy (Chapter 2) or comparing embedding models: re-running ingestion after a small change touches only the chunks that actually changed, rather than re-embedding the entire corpus from scratch on every iteration.

### MTEB and Its Limits

The de facto standard for comparing embedding models is the **Massive Text Embedding Benchmark (MTEB)** leaderboard, which aggregates performance across dozens of tasks — retrieval, classification, clustering, semantic textual similarity, reranking — spanning multiple languages and domains, and reports an averaged score alongside per-task breakdowns. The retrieval subset of MTEB is itself built largely on top of **BEIR** (Benchmarking IR), an earlier, retrieval-specific benchmark suite covering datasets like MS MARCO, Natural Questions, HotpotQA, and FiQA; when a model card advertises a specific "retrieval" number rather than the blended overall MTEB average, it's almost always this BEIR-derived subset being reported, which is the more relevant number to look at when the use case is specifically RAG retrieval rather than, say, clustering or classification. It's genuinely useful as a first filter: a model that performs poorly across the board on MTEB's retrieval subset is unlikely to be a good pick regardless of your specific use case. But it has real limitations worth naming in an interview, because uncritically citing "it's #1 on MTEB" as a justification is a tell that someone hasn't actually thought about model selection.

First, **benchmark overfitting**: because MTEB is public and widely used for model comparison, there is competitive pressure for labs to tune training data and hyperparameters specifically to move the needle on MTEB's public tasks, which doesn't necessarily transfer to arbitrary new domains. Second, **domain mismatch**: MTEB's retrieval tasks are drawn from a fixed set of public datasets (MS MARCO, Natural Questions, FiQA, and similar) that look nothing like an internal legal contract corpus, a codebase, or a company's product documentation — a model's MTEB retrieval rank is only a weak prior for how it will perform on your specific corpus, and the only reliable signal is evaluating candidate models against a labeled sample of your own queries and documents. Third, **English-centric bias**: many leaderboard entries are trained and evaluated predominantly on English tasks, and a model's aggregate MTEB score can look strong while its performance on other languages, especially lower-resource ones, is far weaker; MTEB does have multilingual subtasks, but they're easy to overlook if you only glance at the default leaderboard sort. A related, more subtle concern is **eval-set leakage**: because several of MTEB's constituent datasets (MS MARCO in particular) are also popular sources of training data for exactly the kinds of contrastive pairs described earlier in this chapter, a model that was trained on data overlapping with an MTEB test split can post scores that partly reflect memorization rather than genuine generalization — not necessarily through bad faith, but simply because "publicly available, high-quality retrieval pairs" and "MTEB's underlying datasets" draw from overlapping pools. None of this makes MTEB useless — it remains the right starting point for narrowing a shortlist of candidate models — but it's why the shortlist, not the final decision, should come from the leaderboard.

### Practical Selection Criteria

In practice, the decision usually comes down to a short list of concrete constraints rather than leaderboard position. **Managed API versus self-hosted** trades operational simplicity and no infrastructure to maintain against data residency and privacy requirements (sending every document through a third-party API may be a non-starter for regulated data), latency (a network round-trip per embedding call versus local inference), and cost at very large scale, where self-hosting an open model on owned or reserved GPU/CPU capacity can become cheaper than per-token API pricing once volume is high enough. **Multilingual needs** rule out English-only models outright — Cohere's and Jina's multilingual models, or multilingual BGE/E5 variants, become the relevant comparison set instead. **Max input token length** matters more than it first appears: many older models (`all-mpnet-base-v2` at 384 tokens, `bge-large-en-v1.5` and `e5-large-v2` at 512 tokens) will silently truncate longer chunks, quietly discarding content past the limit, whereas newer models (OpenAI, Voyage, Jina, Nomic) support 8K or more — this interacts directly with the chunking strategy from Chapter 2, since a chunking policy tuned for a 512-token-max embedding model produces different (typically smaller) chunks than one tuned for an 8K-token model. Finally, **domain distance from general web text** is often the deciding factor: legal, medical, and source-code retrieval are all domains where general-purpose embedding models systematically underperform, because their contrastive training pairs were drawn overwhelmingly from web, forum, and encyclopedic text, and vocabulary/semantics in specialized domains diverge enough that "close in embedding space" stops reliably tracking "actually relevant" — this is precisely the gap that domain-specialized models (Voyage's law/code variants) or fine-tuning are meant to close.

None of these criteria substitute for actually measuring retrieval quality on a representative sample of your own queries and documents before committing to a model, since MTEB/BEIR rank is only a weak prior for how a model behaves on a specific corpus, as discussed above. The standard way to do this cheaply is with a small hand-labeled evaluation set — a few dozen to a few hundred (query, relevant document ID) pairs is usually enough to reveal a meaningful gap between two candidate models — and a **recall@k** metric: of the top `k` documents an embedding model retrieves for a query, what fraction of the queries had their known-relevant document somewhere in that top `k`.

```python
def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)

def recall_at_k(model_encode_fn, eval_pairs, corpus_texts, corpus_ids, k=10):
    """eval_pairs: list of (query, relevant_doc_id) tuples, hand-labeled.
    corpus_texts / corpus_ids: parallel lists covering the full candidate corpus."""
    corpus_vecs = _l2_normalize_rows(np.array([model_encode_fn(t) for t in corpus_texts]))
    hits = 0
    for query, relevant_id in eval_pairs:
        q_vec = _l2_normalize_rows(np.array(model_encode_fn(query))[None, :])[0]
        scores = corpus_vecs @ q_vec
        top_k_ids = [corpus_ids[i] for i in np.argsort(-scores)[:k]]
        hits += int(relevant_id in top_k_ids)
    return hits / len(eval_pairs)

# Run the same eval_pairs and corpus through two candidate models and compare directly:
# recall_at_k(openai_encode, eval_pairs, corpus_texts, corpus_ids, k=10)   -> e.g. 0.81
# recall_at_k(bge_encode,    eval_pairs, corpus_texts, corpus_ids, k=10)   -> e.g. 0.74
# A gap like this, measured on your own domain, is far more decision-relevant
# than either model's published MTEB average.
```

This kind of lightweight, corpus-specific evaluation is cheap enough that there's rarely a good excuse to skip it before locking in an embedding model choice for a production system, especially given how expensive re-embedding an entire corpus becomes after the fact.

### Versioning: A Production Gotcha Worth Naming

One operational risk that's specific to embedding models, and easy to overlook until it causes an outage, is that **embedding models are not static artifacts you integrate once** — API providers periodically ship new model versions, sometimes under the same model name, and open-source checkpoints get updated on their model hub pages. If an ingestion pipeline calls an API by a bare model name rather than a pinned version, and the provider silently updates what that name points to, every newly-embedded document from that point forward lives in a subtly different vector space than everything embedded before the switch — not different enough to error out, but different enough to degrade relevance in ways that are hard to notice until users complain. The mitigation is unglamorous but important: pin an explicit model version wherever the API supports it, track which model version (including exact name and, for open models, checkpoint commit hash) produced every stored vector as metadata alongside the vector itself, and treat any model version change — deliberate or provider-forced — as an event that requires a full re-embedding pass and a fresh recall@k evaluation against the eval set described above, never a silent drop-in swap.

## Dimensionality Trade-offs and Matryoshka Representation Learning

Embedding dimensionality is not a free quality knob — it's a trade-off along three axes: retrieval quality, storage cost, and search latency (both memory footprint and the compute cost of every similarity comparison scale roughly linearly with dimension, since a dot product or cosine similarity over `d` dimensions is an `O(d)` operation, repeated across every candidate in a search). Higher-dimensional embeddings can, in principle, encode more independent semantic distinctions without those distinctions interfering with each other geometrically, which is part of why the frontier moved from 384/768-dimensional models toward 1536/3072-dimensional ones. But the returns diminish sharply past roughly 1000-1500 dimensions for most retrieval workloads — beyond that point, additional dimensions mostly add redundant capacity and noise rather than meaningfully improving recall or ranking quality, while storage and compute costs keep growing linearly regardless. (The deep mechanics of how this interacts with ANN index structures like HNSW and IVF, and the quantization techniques — like product quantization — used to compress high-dimensional vectors for storage, are covered in the Vector Databases & Embeddings topic folder; the point here is only the quality-versus-cost curve as it bears on choosing an embedding model.)

The storage side of this trade-off is easy to underestimate until it's put in concrete terms:

```python
def storage_estimate_gb(num_vectors: int, dims: int, bytes_per_dim: int = 4) -> float:
    """float32 = 4 bytes/dim; int8 quantization = 1 byte/dim; binary = 1 bit/dim."""
    return (num_vectors * dims * bytes_per_dim) / (1024 ** 3)

for dims in (256, 768, 1536, 3072):
    print(f"dims={dims:5d}  10M vectors, float32: {storage_estimate_gb(10_000_000, dims):.1f} GB")
# dims=  256  10M vectors, float32: 9.5 GB
# dims=  768  10M vectors, float32: 28.6 GB
# dims= 1536  10M vectors, float32: 57.2 GB
# dims= 3072  10M vectors, float32: 114.4 GB
```

At 10 million vectors, the difference between a 256-dimension and a 3072-dimension embedding is roughly 105 GB of raw vector storage before any ANN index overhead is added on top — a difference large enough to change which hardware tier or managed vector database pricing plan is even viable, which is exactly why the "just use the biggest, best-scoring model" instinct needs to be checked against an actual cost projection at the corpus size a system will eventually reach, not just the size it starts at.

The practical response to this trade-off in recent embedding models is **Matryoshka Representation Learning (MRL)**. Models trained with MRL — OpenAI's `text-embedding-3-small`/`large`, Nomic Embed, and a growing number of open models — produce embeddings where a simple **prefix truncation** (keeping just the first `k` of `d` dimensions, discarding the rest, then re-normalizing) still yields a valid, usable embedding whose retrieval quality degrades gracefully rather than collapsing. Truncating `text-embedding-3-large`'s 3072 dimensions down to 256 retains a large majority of its full retrieval quality on most benchmarks, while cutting storage and search compute by roughly 12x.

This works because of how the loss function is constructed during training, not because of any special property of truncation itself. A standard embedding model's contrastive loss only evaluates the *full* output vector — nothing during training ever checks whether a truncated prefix of that vector would still behave sensibly, so truncating an ordinary embedding model's output essentially destroys it (the "important" information is spread arbitrarily across all dimensions with no ordering). MRL training instead computes the contrastive loss redundantly across several nested prefix lengths of the same vector — for example, computing and summing the loss at 64, 128, 256, 512, and the full 1536 dimensions simultaneously for every training example — which forces the model to front-load the most semantically important, most broadly useful information into the earliest dimensions, with each successive block of dimensions adding progressively finer, more marginal refinement. The result is an embedding with a genuine coarse-to-fine internal ordering, which is exactly the property that makes naive truncation safe. This gives a production system a runtime dial that doesn't require re-embedding anything: store or index the full-dimension vector once, and choose a shorter prefix at query time depending on the cost/quality point needed — for instance, doing a fast first pass over the full corpus with 256-dimension truncated vectors, then re-scoring a smaller candidate set with the full 1536 or 3072 dimensions before handing off to a cross-encoder reranker.

## Domain Adaptation: A Pointer, Not a Deep Dive

General-purpose embedding models, trained mostly on web-scale, generic contrastive pairs, routinely underperform on corpora that are lexically or conceptually far from that training distribution — internal product terminology, legal or medical jargon, or source code being the most common examples in practice. When retrieval quality on a specific corpus is unsatisfying, there are three levers worth knowing at a decision-making level, without needing to execute any of them in depth in this chapter:

- **Fine-tune the embedding model** on domain-specific query-document pairs, continuing the same contrastive training recipe described earlier but on in-domain data, so the model's geometry reorganizes around what "related" actually means in that domain. This is the highest-effort, typically highest-payoff option, and its full methodology — data collection, hard-negative mining, evaluation — lives in the Vector Databases & Embeddings topic folder rather than here.
- **Use instruction-style prefixes** the way E5-family models do, prepending `"query: "` or `"passage: "` to inputs (and, for newer instruction-tuned embedding models, more descriptive task prefixes like `"Represent this sentence for searching relevant passages: "`). This doesn't change model weights at all — it's a zero-cost intervention that leans on the fact that some models were trained to condition their pooling behavior on this kind of textual signal — but it only helps if the specific model you're using was actually trained with such prefixes; adding them to a model that wasn't trained this way does nothing useful and can even hurt quality slightly.
- **Fall back to hybrid search**, combining dense embedding similarity with a sparse, exact-match method like BM25. This compensates for a structural blind spot of dense embeddings: rare, exact terms — product SKUs, legal citation numbers, function names, error codes — are often precisely the terms a general-purpose embedding model has the least reliable geometry for, since they're underrepresented in whatever data it was contrastively trained on, while a keyword-based method matches them trivially. Hybrid search is covered in full in Chapter 4; the point to flag here is that it's frequently a cheaper and faster fix for domain mismatch than fine-tuning, and the two approaches are not mutually exclusive.

The decision point to internalize: fine-tuning offers the largest ceiling but the highest engineering cost (data collection, training infrastructure, ongoing maintenance as the domain evolves); prefixes are free but narrow in applicability; hybrid search is a robust, relatively cheap mitigation that specifically targets the exact-match failure mode rather than general domain drift.

The prefix intervention is cheap enough to be worth showing concretely, since it's the kind of one-line detail that's easy to get wrong in an ingestion pipeline:

```python
def embed_document(text: str, encode_fn) -> np.ndarray:
    # E5-family models expect this exact literal prefix at both training-data-generation
    # time and inference time -- mismatching it in either direction degrades retrieval.
    return encode_fn(f"passage: {text}")

def embed_query(text: str, encode_fn) -> np.ndarray:
    return encode_fn(f"query: {text}")

# A common ingestion bug: embedding documents with the "passage: " prefix during
# indexing, then later embedding ad-hoc queries somewhere else in the codebase
# (a debugging script, a batch eval job) without the "query: " prefix -- the two
# resulting vector spaces are no longer aligned the way the model was trained to expect.
```

## Hands-On: Bi-Encoder Retrieval and Matryoshka Truncation

The following demonstrates the two mechanisms covered above end to end: embedding a small corpus and a query with a bi-encoder, ranking by cosine similarity, and then truncating an MRL-trained embedding to a smaller dimensionality and confirming the ranking is largely preserved.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# A bi-encoder: encodes each text independently into a fixed-size vector.
# all-mpnet-base-v2 uses mean pooling over token embeddings, as described above.
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

corpus = [
    "Mean pooling averages token embeddings weighted by the attention mask.",
    "Cross-encoders jointly attend over query and document tokens for scoring.",
    "HNSW builds a navigable small-world graph for approximate nearest neighbor search.",
    "Matryoshka training optimizes nested prefixes of an embedding vector.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
]

# Bi-encoder property: document embeddings are computed once, independent of any query,
# and can be precomputed/indexed ahead of time.
corpus_embeddings = model.encode(corpus, normalize_embeddings=True)  # (5, 768)

query = "How do sentence embedding models turn token vectors into one vector?"
query_embedding = model.encode(query, normalize_embeddings=True)     # (768,)

# Since both are L2-normalized, dot product == cosine similarity.
scores = corpus_embeddings @ query_embedding
ranking = np.argsort(-scores)

for rank, idx in enumerate(ranking, start=1):
    print(f"{rank}. ({scores[idx]:.3f}) {corpus[idx]}")
# Expect the mean-pooling sentence to rank first -- it's the most directly on-topic.
```

Note that `model.encode(corpus, ...)` was called once on the whole list rather than once per string in a loop — batching inputs through the encoder together is a real throughput lever, not a stylistic nicety. A single forward pass over a padded batch of 32 or 64 short passages is dramatically faster per-item on a GPU (and noticeably faster even on CPU) than 32 or 64 separate forward passes, because the fixed overhead of a forward pass — kernel launches, memory allocation — is amortized across the batch. Ingestion pipelines that embed a large corpus should always batch this way; only true query-time encoding, where requests arrive one at a time from users, is naturally unbatched (though some high-traffic systems introduce a small batching window even for queries, trading a few milliseconds of added latency for meaningfully higher throughput under load).

The second snippet demonstrates Matryoshka-style truncation. `all-mpnet-base-v2` above was not trained with an MRL objective, so this part uses a stand-in full-dimension vector to isolate the mechanic; in a real system you would use the actual output of an MRL-trained model such as `text-embedding-3-large` or `nomic-embed-text-v1.5`.

```python
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-9, None)

def matryoshka_truncate(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """Keep the first `target_dim` dimensions and re-normalize.
    Only valid for embeddings from a model explicitly trained with MRL --
    truncating an arbitrary model's output this way discards information
    with no ordering guarantee and produces a degraded, unpredictable vector."""
    truncated = embeddings[..., :target_dim]
    return normalize(truncated)

# Stand-in for real MRL-trained 1536-dim embeddings (e.g., text-embedding-3-small).
rng = np.random.default_rng(seed=7)
full_dim = 1536
mrl_corpus = normalize(rng.standard_normal((5, full_dim)))
mrl_query = normalize(rng.standard_normal((1, full_dim)))[0]

full_scores = mrl_corpus @ mrl_query
full_ranking = np.argsort(-full_scores)

for target_dim in (1536, 512, 256, 64):
    truncated_corpus = matryoshka_truncate(mrl_corpus, target_dim)
    truncated_query = matryoshka_truncate(mrl_query[None, :], target_dim)[0]
    scores = truncated_corpus @ truncated_query
    ranking = np.argsort(-scores)
    agreement = np.mean(ranking == full_ranking)
    print(f"dim={target_dim:5d}  top-1 doc matches full-dim ranking: {ranking[0] == full_ranking[0]}")
    print(f"           storage: {target_dim * 4} bytes/vector (float32)")

# With genuine MRL-trained embeddings, ranking agreement stays high down to
# surprisingly small dimensions (256, even 64); with the random stand-in used
# here, agreement degrades much faster, because there's no trained structure
# concentrating signal into the leading dimensions.
```

The core lesson these two snippets are meant to convey together: bi-encoder retrieval is fundamentally a precompute-then-compare pattern (embed the corpus once, embed each query cheaply, compare via a simple vector operation), and Matryoshka truncation is a way to shrink the "compare" side of that pattern's cost — smaller vectors, less memory, cheaper distance computation — without needing to re-embed the corpus, provided the underlying model was actually trained to support it. Getting both of these right — choosing a bi-encoder appropriate to the domain and traffic pattern, and knowing when a shorter Matryoshka prefix is sufficient versus when full dimensionality or a cross-encoder rerank is warranted — is most of what separates a naively wired-up RAG demo from a retrieval system that has actually been engineered for cost and quality at scale.

## Key Takeaways

A few points from this chapter come up disproportionately often in interviews and are worth being able to state crisply:

- An embedding model's output geometry — the fact that cosine similarity between two vectors tracks semantic similarity between two texts — is not architecturally guaranteed by the transformer encoder itself; it is a learned side effect of contrastive training (InfoNCE / multiple-negatives-ranking loss) that explicitly optimizes related pairs to be close and unrelated pairs to be far apart.
- Mean pooling over token embeddings is the more common and generally more robust pooling strategy for purpose-built sentence/passage embedding models, because it doesn't depend on a single token (`[CLS]`) having been specifically trained to summarize the whole sequence.
- Bi-encoders encode queries and documents independently, which is what makes precomputing and indexing document vectors — and therefore sub-linear ANN search over huge corpora — possible at all; cross-encoders score a concatenated (query, document) pair through joint cross-attention, which is far more accurate but requires one full forward pass per document per query and cannot be precomputed, which is exactly why it's reserved for reranking a small shortlist (Chapter 6) rather than first-stage retrieval over the full corpus.
- No embedding model is universally "best" — MTEB/BEIR leaderboard position is a useful first filter, not a substitute for evaluating candidate models against a labeled sample of your own domain's queries and documents, particularly for legal, medical, code, or other corpora far from general web text.
- Matryoshka Representation Learning lets a single embedding be safely truncated to a shorter prefix with graceful quality degradation, because the training loss explicitly optimizes multiple nested prefix lengths together rather than only the full vector — this is a query-time cost/quality dial, not a technique that can be retrofitted onto a model that wasn't trained with it.
- When retrieval quality lags on a specialized corpus, the decision is between fine-tuning the embedding model (highest ceiling, highest cost — covered in depth in the Vector Databases & Embeddings folder), applying model-specific instruction prefixes (free but narrow), and hybrid dense-plus-keyword search (a robust, comparatively cheap mitigation for the exact-match blind spots dense embeddings tend to have).
- Bi-encoders and cross-encoders aren't the only two points on this spectrum — late-interaction models like ColBERT keep precomputable per-token document embeddings and compare them at query time with a MaxSim operation, trading extra storage for accuracy closer to a cross-encoder while retaining the bi-encoder's key property of not needing the query at document-encoding time.

## Looking Ahead

With an embedding model chosen and its dimensionality and cost trade-offs understood, the next question is what to do with it at retrieval time: how a query vector is actually compared against an indexed corpus, when dense vector search alone is insufficient and a sparse, keyword-based method like BM25 needs to be combined with it, and how to fuse the two into a single ranked result set. That's the subject of Chapter 4.
