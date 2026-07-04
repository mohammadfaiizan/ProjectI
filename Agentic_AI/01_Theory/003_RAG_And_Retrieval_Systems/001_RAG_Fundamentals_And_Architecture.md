# RAG Fundamentals and Architecture

## 1. The problem RAG solves

A large language model is, at its core, a compressed and frozen snapshot of a training corpus, plus a set of learned reasoning and language-generation abilities. Once training finishes, the weights stop changing. Everything the model "knows" as facts was baked in at some cutoff date, and everything it does with those facts — summarizing, arguing, coding, explaining — is a separate, largely orthogonal capability. This split matters enormously in practice, and almost every real limitation of LLM-based products traces back to conflating the two.

Three concrete failure modes fall out of this frozen-knowledge property. First, knowledge cutoffs: a model trained on data through a certain date simply cannot know about anything that happened after that date — a new product release, a regulatory change, last week's incident postmortem. Second, hallucination on facts the model never reliably learned: LLMs are trained to produce plausible continuations, not to say "I don't know," so when asked about something sparsely represented (or absent) in training data, they often generate a fluent, confident, wrong answer rather than abstaining. Third, and most important for enterprise use cases, LLMs have no access to private, proprietary, or dynamic data by construction — your company's internal wiki, your customer's support tickets, today's inventory levels, or a legal contract that was signed an hour ago were never in the training set and never will be, because you cannot (and should not) put confidential data into a foundation model's public training run.

The naive fix — retrain or fine-tune the model every time the world changes — does not scale. Full retraining is enormously expensive in compute and time; even lightweight fine-tuning requires curating a training set, running a training job, evaluating for regressions, and redeploying, for every single knowledge update. If your knowledge base changes daily (ticket resolutions, price changes, new documents), you cannot realistically fine-tune daily. You need a mechanism where updating what the model "knows" is as cheap as writing a row to a database, without touching the model's weights at all.

This is the intuition behind Retrieval-Augmented Generation, formalized by Lewis et al. in their 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": decouple *what the model knows how to do* from *what the model knows*. Keep the model's reasoning, language, and instruction-following ability as the frozen, expensive-to-change part. Move the facts into an external, cheap-to-update store — a document collection indexed for search — and at inference time, before the model generates an answer, retrieve the small number of documents that are actually relevant to the question and hand them to the model as context. The model's job changes from "recall the fact from memory" to "read this evidence and answer based on it," which is a task LLMs are demonstrably much better and much more reliable at.

This reframing buys you several things simultaneously. Knowledge updates become a data-engineering problem (upsert a document into an index) instead of a machine-learning problem (retrain a model). Freshness is trivial — index a new document and it is queryable within seconds. Hallucination on out-of-scope facts drops sharply because the model is grounding its answer in text it can literally see, rather than trying to recall a fuzzy statistical impression of training data. And you get something fine-tuning can never give you for free: traceability. Because you know precisely which chunks of text were retrieved and inserted into the prompt for a given answer, you can show the user "here is the source passage this claim came from," which is not a nice-to-have in legal, medical, or financial domains — it is often a hard compliance requirement.

It is worth knowing the original framing precisely, because it clarifies where the name comes from and what problem the original paper was actually attacking. Lewis et al. described the model as having two kinds of memory: a *parametric* memory (the knowledge implicitly encoded in the trained weights) and a *non-parametric* memory (an explicit, external index of text passages that can be searched at inference time). Their architecture combined a retriever, which pulls the most relevant passages from the non-parametric memory for a given input, with a generator (a sequence-to-sequence model), which conditions its output on those retrieved passages. The key empirical result was that this combination produced more specific, factual, and diverse outputs on knowledge-intensive tasks than a purely parametric model of comparable size, precisely because the non-parametric memory could be swapped or updated without retraining the generator at all. Everything modern RAG systems do — swapping in a different vector index, refreshing documents nightly, pointing the same LLM at a different customer's knowledge base per request — is a direct, practical descendant of that original parametric/non-parametric split.

## 2. RAG vs. fine-tuning vs. long-context stuffing

These three approaches are frequently presented as competitors, but they answer different questions, and a senior engineer's job is to recognize which question is actually being asked before picking a technique.

Fine-tuning changes the model's *behavior*: its tone, its output format, the way it follows a schema, its domain-specific vocabulary, its default reasoning pattern for a narrow task. If you want a model that always responds in a specific JSON schema, that writes in your brand's voice, that has internalized the "shape" of a specialized task (e.g., converting radiology notes into structured findings), fine-tuning is the right tool, because you are teaching a skill, not a fact. Critically, fine-tuning is a poor and expensive tool for teaching a model new, frequently changing *facts*. Every new fact requires curating examples, running a training job, evaluating for regressions against everything the model previously knew, and redeploying — and even after all that effort, there is no guarantee the model reliably recalls the fact rather than blending it with something similar it saw during pretraining. Facts fine-tuned into weights are also opaque: you cannot point to "the sentence that taught the model this," so you lose attribution entirely.

RAG changes what the model has *access to*, not what it *is*. It leaves the base model's weights untouched and instead controls what evidence is placed in front of it at generation time. This makes it the correct tool whenever the requirement is "the model should know about X," where X is factual, voluminous, private, or changes over time. Updating a RAG knowledge base is an upsert into a vector store; updating a fine-tuned model's knowledge is a full training-evaluation-deployment cycle. This asymmetry alone usually decides the question for any system where the underlying data changes weekly or faster.

Long-context stuffing — simply pasting your entire document set into a 200K, 1M, or even larger context window and asking the model to answer directly — looks like it should make retrieval unnecessary once context windows got big enough. It doesn't, for three concrete engineering reasons. First, cost: nearly all commercial APIs charge per input token, and context windows do not change that; stuffing 500K tokens of documents into every single query multiplies your cost by orders of magnitude versus sending the 3-5 relevant chunks a retriever would have selected, even if the huge prompt is heavily cached. Second, latency: time-to-first-token scales with prompt length because the model has to run attention over every token before it can emit the first output token, so a million-token prompt measurably increases response latency compared to a two-thousand-token one, which matters for any interactive product. Third, and most subtly, accuracy: Liu et al. (2023, "Lost in the Middle") showed that LLMs do not attend uniformly across a long context — retrieval accuracy for information placed near the beginning or end of a long prompt is measurably higher than for the same information placed in the middle, even when the model's stated context limit is far larger than the prompt. So even when your entire corpus technically fits in the context window, dumping all of it in front of the model can produce *worse* answers than a retrieval step that pulls out the handful of genuinely relevant passages and places them prominently, because the retrieval step does the attention-focusing work that the model itself does unreliably at scale. Long-context stuffing is a legitimate choice only when the corpus is small and relatively stable (a single contract, a handful of small technical documents), the added per-query cost and latency of resending it is acceptable for your use case, and you don't need document-level filtering across many unrelated sources.

The table below summarizes the trade-off along the axes that matter operationally; the reasoning behind each cell is what was just argued above, not the table itself.

| Dimension | Fine-tuning | RAG | Long-context stuffing |
|---|---|---|---|
| What it changes | Model behavior/style/skill | What evidence the model sees | What evidence the model sees |
| Cost to update knowledge | High (retrain/redeploy cycle) | Low (upsert a document) | Low (add a file) but recurring per-query cost is high |
| Per-query latency impact | None beyond base inference | Small (retrieval + smaller prompt) | Large (attention over huge prompt) |
| Per-query cost | Base inference cost only | Low (small context) | High (tokens billed per request) |
| Source attribution | None | Natural (you know the retrieved chunks) | Weak (model must self-report which part it used) |
| Best for | Tone, format, task specialization | Facts, private data, frequently changing data | Small, stable corpora, one-off deep analysis |

In production systems the three are not mutually exclusive. A common and effective pattern is a fine-tuned model (for instruction-following in your domain's style and for reliably citing retrieved sources in a specific format) sitting on top of a RAG pipeline (for the facts), with long-context techniques reserved for the minority of queries that genuinely require synthesizing an entire large document rather than answering from a few passages.

## 3. The RAG pipeline, end to end

It helps to walk through the pipeline as a story rather than a static diagram, because every stage is a place where a real decision gets made, and the quality of the final answer is bounded by the weakest decision in the chain — a perfect generator cannot fix a retrieval step that fetched the wrong passages, and perfect retrieval cannot fix chunks that were split so badly they lost their meaning.

**Ingest.** Everything starts with pulling raw content out of wherever it actually lives: PDFs, HTML pages, Confluence or Notion wikis, database tables, ticketing systems, APIs. This stage is unglamorous but decides the ceiling on everything downstream — if you ingest a PDF as raw, unstructured text and lose its table structure, no amount of clever retrieval later will recover the table's meaning. Real ingestion pipelines have to handle format-specific parsing (PDF layout extraction, HTML boilerplate stripping, OCR for scanned documents), incremental updates (has this document changed since last time we indexed it), and metadata extraction (author, date, source system, access permissions) that later stages depend on for filtering.

**Chunk.** Once you have clean text, you must split it into retrievable units. This decision is more consequential than it looks: chunks that are too large dilute the embedding signal (a 5-page chunk embeds to a vector that is an average of many unrelated ideas, so it never scores as the top match for any single specific question) and waste context budget when retrieved; chunks that are too small lose surrounding context needed to answer the question at all. Where to draw chunk boundaries — fixed-size windows, sentence/paragraph-aware splitting, semantic chunking, or structure-aware splitting that respects headings and tables — is the entire subject of Chapter 2 (`002_Chunking_Strategies.md`), so here it's enough to recognize that this stage exists and that it is not a mechanical afterthought.

**Embed.** Each chunk is passed through an embedding model that maps it to a dense vector in a fixed-dimensional space, positioned so that semantically similar text ends up close together under a distance metric (usually cosine similarity or dot product). This is what allows retrieval to work on meaning rather than exact keyword overlap — a query about "reducing employee attrition" can retrieve a chunk about "improving staff retention" even though they share almost no words. Which embedding model to use, how dimensionality and training objective affect retrieval quality, and how to handle domain-specific vocabulary are covered in depth in Chapter 3 (`003_Embedding_Models_For_Retrieval.md`).

**Store.** The vectors, together with the original chunk text and its metadata, are written into a vector store or index built for approximate nearest-neighbor (ANN) search — something like FAISS, a managed vector database, or a search engine's vector extension. The indexing structure chosen here (flat index, IVF, HNSW, and so on) trades off recall, latency, and memory, and matters a great deal at scale, but conceptually its job is simple: given a query vector, return the k closest chunk vectors fast, even across millions of chunks.

**Retrieve.** When a user asks a question, the query itself is embedded with the same embedding model, and the store is searched for the nearest chunks. Dense vector similarity is one retrieval signal among several — sparse keyword-based methods like BM25 remain strong for exact-match cases (product codes, names, rare terms an embedding model has never seen), and hybrid approaches combine both. The comparative strengths of dense, sparse, and hybrid retrieval are the focus of Chapter 4 (`004_Retrieval_Strategies_Dense_Sparse_Hybrid.md`); techniques for rewriting or expanding the query before retrieval (to bridge vocabulary mismatch between how users ask and how documents are written) are covered in Chapter 5 (`005_Query_Transformation_And_Expansion.md`); and reordering or fusing multiple retrieved result sets to push the truly best passages to the top is covered in Chapter 6 (`006_Reranking_And_Result_Fusion.md`).

**Augment.** The retrieved chunks are assembled into a prompt alongside the user's original question and system instructions telling the model how to use the evidence — typically something like "answer only using the provided context, and cite which passage supports each claim." How this prompt is structured (ordering chunks to fight the lost-in-the-middle effect, deciding how many chunks to include given context budget, handling the case where retrieval returns nothing relevant) is a real design surface, and several more sophisticated variants of it — multi-hop retrieval, self-correcting RAG, agentic retrieval loops — are the subject of Chapter 7 (`007_Advanced_RAG_Patterns.md`).

**Generate.** Finally the LLM is called with the augmented prompt and produces the answer, ideally grounded in the retrieved evidence and citing it. Whether this grounding actually happened — whether the answer is faithful to the retrieved context, whether retrieval found the right things in the first place, and how to measure both — is the subject of Chapter 8 (`008_RAG_Evaluation_And_Metrics.md`). Everything about turning this pipeline into a system that survives production traffic (caching, indexing pipelines, monitoring for retrieval drift, cost control, latency budgets) is Chapter 9 (`009_Production_RAG_System_Design.md`).

It's worth internalizing that this is a pipeline in the true sense: each stage's output is the next stage's input, errors compound rather than cancel, and there is no stage where "the model is smart enough to figure it out" reliably rescues an upstream mistake. This is why, in interviews and in practice, debugging a bad RAG answer should always start by asking "was the right chunk even retrieved?" before questioning the generation step at all.

## 4. A working basic RAG pipeline

The implementation below is deliberately minimal — a recursive character-based chunker, an in-memory cosine-similarity vector store standing in for FAISS or a managed vector database, and OpenAI-style embedding and chat completion calls — but every piece is structured the way a real system would be, so the shape of the code transfers directly to a production stack. The point is not to build a scalable system here (that's Chapter 9); it's to make every stage from section 3 concrete enough to run.

```python
"""
minimal_rag.py

A minimal, end-to-end Retrieval-Augmented Generation pipeline:
ingest -> chunk -> embed -> store -> retrieve -> augment -> generate.

Dependencies: openai (for embeddings + chat completions), numpy.
    pip install openai numpy
Requires an OPENAI_API_KEY environment variable, or swap `embed_texts`
and `generate_answer` for any other provider's client.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 120
TOP_K = 4


# ---------------------------------------------------------------------------
# 1. INGEST — in a real system this pulls from PDFs, HTML, wikis, DBs, APIs.
#    Here we simulate ingestion with in-memory "documents" carrying metadata,
#    since the parsing step itself is not what this chapter is teaching.
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def ingest_documents() -> List[Document]:
    return [
        Document(
            doc_id="policy-001",
            text=(
                "Our refund policy allows customers to request a full refund "
                "within 30 days of purchase, provided the product is unused "
                "and in its original packaging. After 30 days, only store "
                "credit is issued, and store credit is valid for 12 months "
                "from the date of issue. Digital products are non-refundable "
                "once downloaded, except where required by local consumer law."
            ),
            metadata={"source": "policy-001.pdf", "category": "refunds"},
        ),
        Document(
            doc_id="policy-002",
            text=(
                "Shipping times vary by region. Domestic orders typically "
                "arrive within 3 to 5 business days using standard shipping, "
                "or 1 to 2 business days with expedited shipping. "
                "International orders can take 7 to 21 business days "
                "depending on customs processing in the destination country. "
                "Expedited international shipping is not currently available."
            ),
            metadata={"source": "policy-002.pdf", "category": "shipping"},
        ),
    ]


# ---------------------------------------------------------------------------
# 2. CHUNK — a simple recursive splitter: try to break on paragraph
#    boundaries first, then sentences, then hard character windows, so we
#    avoid slicing a sentence in half whenever the text structure allows it.
#    (See Chapter 2 for a full treatment of chunking strategy trade-offs.)
# ---------------------------------------------------------------------------

def recursive_chunk(text: str, chunk_size: int = CHUNK_SIZE_CHARS,
                     overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    separators = ["\n\n", ". ", " "]

    def split(segment: str, seps: List[str]) -> List[str]:
        if len(segment) <= chunk_size:
            return [segment]
        if not seps:
            # Hard fallback: fixed-size window with overlap.
            chunks = []
            start = 0
            while start < len(segment):
                end = start + chunk_size
                chunks.append(segment[start:end])
                start = end - overlap
            return chunks

        sep, rest_seps = seps[0], seps[1:]
        parts = segment.split(sep)
        chunks, current = [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # The part itself might still be too big; recurse further.
                if len(part) > chunk_size:
                    chunks.extend(split(part, rest_seps))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = split(text.strip(), separators)
    return [c.strip() for c in raw_chunks if c.strip()]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    metadata: dict


def chunk_documents(documents: List[Document]) -> List[Chunk]:
    chunks = []
    for doc in documents:
        pieces = recursive_chunk(doc.text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{i}",
                    text=piece,
                    doc_id=doc.doc_id,
                    metadata=doc.metadata,
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# 3. EMBED — convert chunk text into dense vectors.
#    (See Chapter 3 for how to choose an embedding model.)
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. STORE — a minimal in-memory vector store using cosine similarity.
#    A real system would use FAISS, a managed vector DB, or a search
#    engine's vector index for ANN search at scale; the interface below
#    is intentionally shaped the same way those clients are.
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    def __init__(self):
        self._chunks: List[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        # Normalize so a dot product equals cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.clip(norms, 1e-10, None)
        self._chunks.extend(chunks)
        self._vectors = (
            normalized if self._vectors is None
            else np.vstack([self._vectors, normalized])
        )

    def search(self, query_vector: np.ndarray, top_k: int = TOP_K) -> List[tuple]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        q = query_vector / np.clip(np.linalg.norm(query_vector), 1e-10, None)
        scores = self._vectors @ q  # cosine similarity, since both are unit norm
        top_indices = np.argsort(-scores)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# 5. RETRIEVE — embed the query and search the store.
#    (Chapter 4 covers dense/sparse/hybrid retrieval; Chapter 5 covers
#    query rewriting/expansion; Chapter 6 covers reranking result fusion.)
# ---------------------------------------------------------------------------

def retrieve(query: str, store: InMemoryVectorStore, top_k: int = TOP_K) -> List[tuple]:
    query_vector = embed_texts([query])[0]
    return store.search(query_vector, top_k=top_k)


# ---------------------------------------------------------------------------
# 6. AUGMENT — build a prompt from the retrieved chunks and the question.
#    (Chapter 7 covers more advanced augmentation and multi-hop patterns.)
# ---------------------------------------------------------------------------

def build_prompt(query: str, retrieved: List[tuple]) -> str:
    context_blocks = []
    for chunk, score in retrieved:
        source = chunk.metadata.get("source", chunk.doc_id)
        context_blocks.append(f"[Source: {source}]\n{chunk.text}")
    context_text = "\n\n---\n\n".join(context_blocks)

    return (
        "You are a helpful assistant answering questions using only the "
        "provided context. If the context does not contain the answer, say "
        "you don't know rather than guessing. Cite the source for each claim "
        "using the [Source: ...] tags shown.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# 7. GENERATE — call the LLM with the augmented prompt.
#    (Chapter 8 covers evaluating whether the answer is actually faithful
#    to the retrieved context.)
# ---------------------------------------------------------------------------

def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Orchestration: wire the stages into one pipeline.
# ---------------------------------------------------------------------------

def build_index() -> InMemoryVectorStore:
    documents = ingest_documents()
    chunks = chunk_documents(documents)
    vectors = embed_texts([c.text for c in chunks])
    store = InMemoryVectorStore()
    store.add(chunks, vectors)
    return store


def answer_question(query: str, store: InMemoryVectorStore) -> str:
    retrieved = retrieve(query, store, top_k=TOP_K)
    prompt = build_prompt(query, retrieved)
    return generate_answer(prompt)


if __name__ == "__main__":
    index = build_index()

    questions = [
        "How long do I have to return a product for a full refund?",
        "How long does international shipping take?",
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {answer_question(q, index)}")
        print()
```

A few implementation notes worth calling out because they matter beyond this toy example. The vector store normalizes vectors once, at insertion and query time, so that a plain dot product gives cosine similarity — this is a standard trick to avoid recomputing norms on every comparison. The prompt explicitly instructs the model to say it doesn't know rather than guess, which is a small but important guardrail: RAG reduces hallucination but does not eliminate it, and an unconstrained prompt will still happily generate a fluent answer even when the retrieved context is irrelevant. And the `InMemoryVectorStore.search` method has the exact same shape (`add`, `search`) that FAISS or a managed vector database client would have, so swapping it out for a production-grade ANN index later is a drop-in replacement, not a redesign.

## 5. When to use RAG, and when not to

RAG is not a default you reach for on every LLM project; it adds real infrastructure (an ingestion pipeline, an index to keep fresh, a retrieval step to tune and monitor) and it is only worth that cost when the problem it solves is actually present.

RAG is the right choice when the model needs access to facts that are too large, too private, or too dynamic to live in a prompt or in the model's weights: an internal knowledge base, a customer's account history, a document corpus that updates daily, or any domain where you need to show your work by citing sources. It is also the right choice whenever hallucination on specific facts is unacceptable and grounding in retrievable evidence is the practical way to constrain the model's output.

RAG is the wrong choice, or at least unnecessary overhead, in a few recognizable situations. If the entire relevant knowledge fits comfortably in a single prompt and rarely changes — a short FAQ, one policy document, a small fixed dataset — you gain nothing from building a retrieval pipeline over what you could just paste directly into the system prompt; this is exactly the small-and-stable-corpus case where prompt stuffing is legitimately simpler and cheaper to operate than a vector store, ingestion jobs, and retrieval tuning. If your latency budget is extremely tight — sub-100-millisecond response requirements, as in some real-time inference paths — the added round trip of embedding the query and searching an index (on top of the LLM call itself) may simply not fit the budget, and you need either a cached/precomputed answer or a fundamentally different, non-generative approach for that path. And if what you actually need is a change in the model's behavior rather than its knowledge — a different tone, stricter adherence to an output schema, a specialized reasoning pattern for a narrow task — RAG will not help at all, because the problem isn't that the model lacks facts, it's that the model's default behavior isn't what you want, which is precisely the problem fine-tuning (or, for lighter cases, prompt engineering) is built to solve. Recognizing which of these three situations you're actually in, before writing any retrieval code, is the single highest-leverage decision in designing a RAG system, and it is exactly the kind of judgment call senior engineering interviews are designed to probe.

A useful way to internalize the decision, and one that tends to hold up well under interview follow-up questions, is to ask three things about the problem in front of you before touching any code. Does the answer depend on information that is large, private, or changes faster than you're willing to retrain a model — if yes, that pulls toward RAG. Does the task actually require the model to behave differently rather than know more — a stricter format, a different voice, a specialized skill — if yes, that pulls toward fine-tuning, and no amount of retrieved context will fix a formatting or tone problem. And is the relevant knowledge small and stable enough to just live in the prompt, with a cost and latency profile you're comfortable paying on every single request — if yes, retrieval may be solving a problem you don't have yet. Most production systems that mature past a prototype end up combining more than one of these — a fine-tuned model for consistent behavior, sitting on top of a RAG pipeline for facts, occasionally falling back to long-context handling for the rare query that spans an entire document — and the remaining eight chapters of this section go deep on each moving part of that RAG half of the system, starting with chunking in Chapter 2.
