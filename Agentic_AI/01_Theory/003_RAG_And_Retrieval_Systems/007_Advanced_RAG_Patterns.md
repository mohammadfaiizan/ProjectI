# Advanced RAG Patterns

## 1. Why go beyond retrieve-then-generate

Every pipeline in the preceding six chapters, no matter how sophisticated the chunking, embedding, hybrid retrieval, query rewriting, or reranking got, still ultimately executes a single pass: retrieve once, stuff the results into a prompt, generate once, return the answer. That single-pass shape has a structural ceiling, and it is worth being precise about where the ceiling actually is, because each pattern in this chapter exists to break through exactly one piece of it.

The first crack is that retrieval is *always invoked, unconditionally*, whether or not the query segment actually needs it. A user who says "Thanks, that's helpful, one more question — what's the cancellation policy?" doesn't need external context to parse "Thanks, that's helpful," but a naive pipeline still retrieves for the whole turn and stuffs irrelevant chunks into the prompt, which at best wastes context budget and at worst introduces a distractor passage the model tries to awkwardly incorporate. The second crack is that single-pass RAG has no mechanism to notice when retrieval failed. If the top-k chunks are all irrelevant — wrong document, stale content, a query that doesn't match anything in the corpus — the pipeline generates from bad context anyway, because nothing in the architecture asks "was this retrieval actually any good?" before generation happens. The third crack is that a single vector search over chunk embeddings is structurally incapable of answering questions that require synthesizing information spread across the *entire* corpus — "what are the recurring themes across all of last quarter's incident reports?" has no single chunk, no single embedding, that is "about" the answer, because the answer only exists at the level of the whole collection. And the fourth crack is that single-pass RAG treats retrieval as a fixed pipeline stage rather than a decision: it cannot decide to look again with a better query after seeing disappointing results, and it cannot naturally interleave retrieval with other operations (a calculation, a database lookup, a second, differently-scoped search) the way a genuine multi-step research process does.

The five patterns in this chapter — Self-RAG, Corrective RAG (CRAG), GraphRAG, RAPTOR, and Agentic RAG — are five different answers to those four cracks. Self-RAG attacks the first and second by teaching the model to decide, at a fine grain, whether to retrieve and whether what it retrieved and generated actually holds up. CRAG attacks the second directly, adding an explicit quality gate on retrieval with a corrective fallback. GraphRAG and RAPTOR both attack the third, via two different mechanisms — a knowledge graph with community summaries versus a recursive summarization tree — for building structures that *do* have a representation of "the whole corpus," not just of individual chunks. Agentic RAG attacks the fourth by promoting retrieval from a pipeline stage to a tool the model calls at will, arbitrarily many times, interleaved with anything else it needs. None of these are free — every one of them spends extra LLM calls and extra latency to buy the improvement — and the closing section of this chapter is about deciding when that trade is actually worth making.

## 2. Self-RAG: retrieval and generation under explicit self-critique

### 2.1 The problem it targets

Naive RAG has two independent failure points that Self-RAG (Asai, Wu, Wang, Sil, and Hajishirzi, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," 2023) was designed to catch. The first is over-retrieval: real responses are usually a mix of segments that need grounding ("the deductible on the gold plan is $500") and segments that don't ("Happy to help with that — here's what I found"). A fixed pipeline retrieves for the entire response regardless, which both wastes tokens and risks injecting an irrelevant passage that pulls the generation off course. The second is silent unfaithfulness: even when retrieval succeeds and finds a genuinely relevant passage, nothing forces the generator to actually stay faithful to it — the model can still produce a claim that isn't supported by the retrieved text, and a standard pipeline has no internal check that would catch this before the answer reaches the user.

Self-RAG's fix is to train (in the original paper, via a critic model that generates training labels and then a target model fine-tuned to imitate them) a language model that emits explicit *reflection tokens* interleaved with its own generation: a decision on whether retrieval is needed at all for the segment it's about to produce, and, once passages have been retrieved, per-passage and per-segment critique tokens. Concretely, the paper defines four reflection categories worth knowing by name because they show up constantly in interview discussions of the technique:

- **Retrieve**: a binary/ternary decision — does generating the next segment benefit from retrieved evidence, or can the model just continue generating from its own knowledge (or is retrieval unnecessary because the segment is purely conversational)?
- **ISREL** (is relevant): for each retrieved passage, is it actually relevant to the query/segment being generated, or is it noise that should be ignored?
- **ISSUP** (is supported): for a generated segment produced using a given passage, is the segment actually supported by that passage's content, partially supported, or not supported at all (i.e., is this a hallucination riding on top of retrieved context)?
- **ISUSE** (is useful): holistically, how useful is the overall generated response to the original request, on a graded scale — this is the signal used to pick the best among several candidate generations.

The mechanism that makes this powerful is that these are not post-hoc metrics computed after the fact for a dashboard (that's the subject of Chapter 8) — they are decisions the model itself makes *during* generation, which lets the system act on them in real time: skip retrieval when it isn't warranted, discard an irrelevant passage before it pollutes the prompt, and — crucially — reject or regenerate a segment when ISSUP indicates the model is about to make an unsupported claim, before that claim ever reaches the user. That last property is what separates Self-RAG from a rerank-then-generate pipeline: reranking only judges retrieval quality, whereas Self-RAG also judges whether the *generation* it is about to emit is actually faithful to what was retrieved.

### 2.2 A simplified reproduction

Training a genuine Self-RAG model requires the critic/target model fine-tuning pipeline from the paper, which is out of scope for an engineer building on top of an existing LLM API. What's practical, and what interviewers actually want to see you can reason through, is a *prompted* reproduction: use a capable instruction-following LLM as its own critic by asking it to emit the same categories of judgment as structured output, then have your orchestration code branch on those judgments exactly the way the trained reflection tokens would.

```python
"""
self_rag_lite.py

A simplified, prompted reproduction of the Self-RAG loop: for each segment
of a response, decide whether retrieval is needed; if so, retrieve, score
each passage's relevance (ISREL), generate conditioned on the passages
judged relevant, score the generation's faithfulness (ISSUP), and revise
or regenerate if the segment is unsupported. A final ISUSE pass picks
between the settled answer and a fallback that abstains.

This is a reproduction with LLM-judge calls standing in for trained
reflection-token classifiers, not the trained model from Asai et al. 2023 --
useful for building the *architecture* on top of any off-the-shelf LLM.
"""

import json
from dataclasses import dataclass
from typing import List
from openai import OpenAI

client = OpenAI()
CHAT_MODEL = "gpt-4o-mini"


@dataclass
class Passage:
    text: str
    source: str


def llm_json(system: str, user: str) -> dict:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Reflection decision 1: Retrieve -- does this query/segment need external
# evidence at all, or is it answerable (or non-substantive) without it?
# ---------------------------------------------------------------------------

def needs_retrieval(query: str) -> bool:
    verdict = llm_json(
        "You decide whether answering a user message requires retrieving "
        "external documents. Conversational pleasantries, requests you can "
        "safely answer from general reasoning without specific facts, and "
        "simple acknowledgements do NOT need retrieval. Questions asking "
        "for specific facts, policies, numbers, or proprietary information "
        "DO need retrieval. Respond as JSON: {\"retrieve\": true|false, "
        "\"reason\": \"...\"}",
        query,
    )
    return bool(verdict.get("retrieve", True))


# ---------------------------------------------------------------------------
# Reflection decision 2: ISREL -- score each retrieved passage's relevance.
# ---------------------------------------------------------------------------

def score_relevance(query: str, passage: Passage) -> bool:
    verdict = llm_json(
        "You judge whether a retrieved passage is relevant enough to help "
        "answer the query. Respond as JSON: {\"isrel\": true|false}.",
        f"Query: {query}\n\nPassage:\n{passage.text}",
    )
    return bool(verdict.get("isrel", False))


# ---------------------------------------------------------------------------
# Generate a segment conditioned only on passages that passed ISREL.
# ---------------------------------------------------------------------------

def generate_segment(query: str, passages: List[Passage]) -> str:
    if not passages:
        return client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": query}],
            temperature=0.2,
        ).choices[0].message.content

    context = "\n\n".join(f"[{p.source}] {p.text}" for p in passages)
    prompt = (
        "Answer the query using only the passages below. Every factual "
        "claim must be traceable to a passage.\n\n"
        f"Passages:\n{context}\n\nQuery: {query}\nAnswer:"
    )
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    ).choices[0].message.content


# ---------------------------------------------------------------------------
# Reflection decision 3: ISSUP -- is the generated segment actually
# supported by the passages it was conditioned on?
# ---------------------------------------------------------------------------

def score_support(segment: str, passages: List[Passage]) -> str:
    context = "\n\n".join(p.text for p in passages) if passages else "(no passages used)"
    verdict = llm_json(
        "You judge whether a generated answer is supported by the given "
        "passages. Respond as JSON: {\"issup\": \"fully\"|\"partially\"|"
        "\"none\", \"unsupported_claims\": [\"...\"]}.",
        f"Passages:\n{context}\n\nGenerated answer:\n{segment}",
    )
    return verdict.get("issup", "none")


# ---------------------------------------------------------------------------
# Reflection decision 4: ISUSE -- holistic usefulness, used to pick between
# a grounded answer and an honest abstention.
# ---------------------------------------------------------------------------

def score_usefulness(query: str, segment: str) -> int:
    verdict = llm_json(
        "Rate how useful this answer is for the query on a 1-5 scale. "
        "Respond as JSON: {\"isuse\": 1-5}.",
        f"Query: {query}\n\nAnswer:\n{segment}",
    )
    return int(verdict.get("isuse", 3))


def self_rag_answer(query: str, retrieve_fn, max_regenerations: int = 2) -> str:
    if not needs_retrieval(query):
        return generate_segment(query, passages=[])

    candidate_passages = retrieve_fn(query)
    relevant = [p for p in candidate_passages if score_relevance(query, p)]

    if not relevant:
        # Retrieval fired but found nothing usable -- fall back to an
        # explicit abstention rather than generating from noise.
        return ("I looked for relevant information but couldn't find "
                "anything that reliably answers this. Could you clarify "
                "or point me to the right source?")

    segment = generate_segment(query, relevant)
    for _ in range(max_regenerations):
        support = score_support(segment, relevant)
        if support == "fully":
            break
        # ISSUP caught an unsupported or partially supported claim --
        # regenerate with an explicit instruction to tighten grounding.
        tighten_prompt = (
            "Your previous answer made claims not fully backed by the "
            "passages. Rewrite it, keeping only claims directly supported "
            f"by the passages.\n\nPassages:\n"
            + "\n\n".join(p.text for p in relevant)
            + f"\n\nPrevious answer:\n{segment}\n\nRevised answer:"
        )
        segment = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": tighten_prompt}],
            temperature=0.0,
        ).choices[0].message.content

    if score_usefulness(query, segment) < 2:
        return ("The available sources don't fully answer this question; "
                "here is the most reliable partial answer I can support:\n\n" + segment)
    return segment
```

The orchestration logic above is the whole point: `needs_retrieval` prevents wasted, potentially harmful retrieval on segments that don't need it; the ISREL filter (`score_relevance`) keeps noisy passages out of the generation prompt entirely rather than relying on the generator to ignore them; and the ISSUP loop (`score_support`) is what catches a hallucinated claim *before* it reaches the user and gives the system a concrete, actionable instruction ("rewrite it, keeping only claims directly supported") rather than just flagging the problem in a log line after the fact. ISUSE is the final gate deciding whether to present the answer confidently or hedge it. None of the individual LLM-judge calls here are as reliable as a model actually trained end-to-end on reflection tokens the way the paper describes, but the control flow — decide to retrieve, filter by relevance, generate, verify support, loop, gate on usefulness — is exactly the architecture, and it composes with any base LLM you already have API access to.

## 3. Corrective RAG (CRAG): repairing bad retrieval, not critiquing every token

### 3.1 The problem it targets

Self-RAG's critique loop is general-purpose and operates at the level of generated segments. Corrective RAG (Yan, Gu, Zhu, and Ling, "Corrective Retrieval Augmented Generation," 2024) is narrower and more surgical: it assumes the dominant failure mode in most production RAG systems is not "the generator hallucinated despite good context" but "the retriever handed the generator bad context in the first place," and it builds a dedicated repair mechanism specifically for that failure. This distinction is worth stating plainly, because it's the crux of how CRAG differs from Self-RAG: CRAG does not touch the generation step's internal faithfulness at all — it never asks "is this generated sentence supported by this passage." Instead it inserts a quality gate *before* generation that asks a single, cheaper question — "is what we retrieved actually good?" — and if the answer is no, it goes and gets better material rather than trying to generate carefully around bad material.

### 3.2 The mechanism

CRAG introduces a lightweight retrieval evaluator — in the paper, a small fine-tuned model (T5-based), though in a practical LLM-based reproduction this is just another LLM judge call — that scores the first-stage retrieved set into one of three buckets:

- **Correct**: at least one retrieved document is clearly relevant and reliable enough to answer the query. The system proceeds to a *knowledge refinement* step rather than dumping the raw documents into the prompt: it decomposes each retrieved document into small "strips" (roughly sentence- or clause-level segments), scores each strip for relevance, and recomposes only the relevant strips back into a compact, denoised context. This decompose-then-recompose step matters because even a genuinely relevant retrieved document is often mostly irrelevant filler around the one or two sentences that actually answer the question, and feeding the whole document in anyway reintroduces the distractor-token problem that Chapter 6 discussed in the context of reranking.
- **Incorrect**: none of the retrieved documents are usable. The system discards the entire retrieved set — refinement can't save documents that aren't relevant to begin with — and falls back to an external knowledge source, in the paper's implementation a web search, to fetch different candidate evidence, which is then run through the same refinement step before generation.
- **Ambiguous**: the evaluator isn't confident either way. CRAG hedges by combining both signals — refined strips from the original internal retrieval *and* freshly retrieved external results — and lets the generator work from the union, which is a reasonable middle ground when the evaluator itself is uncertain rather than confidently wrong.

The reason this differs meaningfully from just adding a reranker (Chapter 6) is that a reranker only re-orders or filters within the retrieved set — it has no fallback when the entire set is bad, and no mechanism to reach outside the original corpus for better evidence. CRAG's Incorrect branch is explicitly an escape hatch out of the original retrieval source entirely, which is what makes it "corrective" rather than merely "selective."

### 3.3 A working implementation

```python
"""
corrective_rag.py

CRAG: evaluate first-stage retrieval quality, then branch:
  Correct    -> refine (decompose into strips, keep relevant strips, recompose)
  Incorrect  -> discard retrieval entirely, fall back to external web search
  Ambiguous  -> combine refined internal strips with external search results

The retrieval evaluator and strip-relevance scorer are LLM-judge calls
standing in for the paper's lightweight fine-tuned evaluator -- swappable
for a distilled classifier in production for latency/cost reasons.
"""

import json
import re
from dataclasses import dataclass
from typing import List
from openai import OpenAI

client = OpenAI()
CHAT_MODEL = "gpt-4o-mini"

CORRECT_THRESHOLD = 0.7
INCORRECT_THRESHOLD = 0.3


@dataclass
class Document:
    text: str
    source: str


def llm_json(system: str, user: str) -> dict:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Step 1: retrieval evaluator -- a single confidence score for the whole
# retrieved set, standing in for CRAG's fine-tuned Correct/Incorrect/
# Ambiguous classifier.
# ---------------------------------------------------------------------------

def evaluate_retrieval(query: str, documents: List[Document]) -> float:
    context = "\n\n".join(f"[{d.source}] {d.text}" for d in documents) or "(nothing retrieved)"
    verdict = llm_json(
        "Score, from 0.0 to 1.0, how well the retrieved documents as a "
        "whole are able to correctly and completely answer the query. "
        "1.0 = clearly sufficient and reliable. 0.0 = irrelevant or "
        "empty. Respond as JSON: {\"confidence\": 0.0-1.0}.",
        f"Query: {query}\n\nRetrieved documents:\n{context}",
    )
    return float(verdict.get("confidence", 0.0))


def classify(confidence: float) -> str:
    if confidence >= CORRECT_THRESHOLD:
        return "correct"
    if confidence <= INCORRECT_THRESHOLD:
        return "incorrect"
    return "ambiguous"


# ---------------------------------------------------------------------------
# Step 2: knowledge refinement -- decompose each document into strips,
# keep only relevant strips, recompose into a compact context.
# ---------------------------------------------------------------------------

def decompose_into_strips(document: Document) -> List[str]:
    # Sentence-level split stands in for the paper's finer-grained strips;
    # in production this could be clause-level or fixed-token windows.
    raw = re.split(r"(?<=[.!?])\s+", document.text.strip())
    return [s.strip() for s in raw if s.strip()]


def refine(query: str, documents: List[Document]) -> str:
    all_strips = []
    for doc in documents:
        all_strips.extend(decompose_into_strips(doc))
    if not all_strips:
        return ""

    verdict = llm_json(
        "Given a query and a numbered list of candidate text strips, "
        "return the indices of strips that directly help answer the "
        "query, discarding filler/off-topic strips. Respond as JSON: "
        "{\"keep\": [indices]}.",
        f"Query: {query}\n\nStrips:\n"
        + "\n".join(f"{i}: {s}" for i, s in enumerate(all_strips)),
    )
    kept = [all_strips[i] for i in verdict.get("keep", []) if i < len(all_strips)]
    return " ".join(kept)


# ---------------------------------------------------------------------------
# Step 3: external fallback -- a broader/different source when internal
# retrieval is judged Incorrect. Here it's a stubbed web-search call;
# swap in a real search API in production.
# ---------------------------------------------------------------------------

def web_search_fallback(query: str) -> List[Document]:
    # Placeholder: in production this calls a real web/search API and
    # wraps results as Document objects the same way internal retrieval does.
    return [Document(text=f"(stubbed external search result for: {query})", source="web")]


# ---------------------------------------------------------------------------
# Orchestration: evaluate -> branch -> generate.
# ---------------------------------------------------------------------------

def crag_answer(query: str, internal_retrieve_fn, top_k: int = 5) -> str:
    documents = internal_retrieve_fn(query, top_k)
    confidence = evaluate_retrieval(query, documents)
    verdict = classify(confidence)

    if verdict == "correct":
        context = refine(query, documents)
    elif verdict == "incorrect":
        external_docs = web_search_fallback(query)
        context = refine(query, external_docs)
    else:  # ambiguous -- combine both refined internal and external context
        internal_context = refine(query, documents)
        external_docs = web_search_fallback(query)
        external_context = refine(query, external_docs)
        context = f"{internal_context}\n\n{external_context}".strip()

    prompt = (
        "Answer the query using only the context below. If the context "
        "is insufficient, say so explicitly.\n\n"
        f"Context:\n{context}\n\nQuery: {query}\nAnswer:"
    )
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    ).choices[0].message.content
```

Notice how narrow and cheap the branch condition is relative to Self-RAG's segment-by-segment critique loop: one evaluator call decides the fate of the entire retrieved set, and the three branches all funnel back into the same refine-then-generate shape. That's the practical appeal of CRAG in production — it is a bounded, one-shot correction (retrieve, maybe re-retrieve once from a fallback source, generate) rather than an open-ended critique-and-regenerate loop, which makes its latency and cost overhead far more predictable than Self-RAG's.

## 4. GraphRAG: answering questions no single chunk can answer

### 4.1 The problem it targets

Every retrieval method covered so far — dense, sparse, hybrid, reranked, even query-rewritten — shares one structural assumption: the answer to the user's question lives inside some (possibly small) subset of chunks, and the retriever's job is to find that subset. That assumption is simply false for a large and important category of real questions: "What are the major themes across this entire set of customer interviews?" "How do these five departments' incident reports relate to each other?" "Summarize the overall narrative arc of this book." No single chunk's embedding is "about" the answer to these questions, because the answer is a property of the *whole collection*, synthesized across dozens or hundreds of chunks that individually look unrelated. Increasing top-k does not fix this — you can retrieve 50 chunks and still be missing the connective tissue between them, and stuffing 50 chunks into a prompt reintroduces the lost-in-the-middle problem discussed in Chapter 1 without actually giving the model a *structure* to reason over.

Microsoft's GraphRAG (Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," 2024) attacks this by building an intermediate representation that sits above individual chunks: a knowledge graph of entities and relationships extracted from the corpus, plus a hierarchy of pre-computed summaries over clusters of that graph. The graph captures connections between facts that live in different chunks (the retriever's traditional blind spot), and the community summaries give the system something that individual chunk embeddings never could — a compact, pre-synthesized description of "what this whole region of the corpus is about," at multiple levels of granularity.

### 4.2 The mechanism

Building the graph happens once, offline, as an indexing-time process, not per query. First, an LLM sweeps the corpus and extracts entities (people, organizations, concepts, products — whatever's salient for the domain) and the relationships between them, typically chunk by chunk, producing a set of (entity, relationship, entity) triples along with short descriptions. Second, these extractions are merged into a single graph, with duplicate entity mentions resolved and multiple relationship observations between the same pair of entities consolidated. Third, a community-detection algorithm — the paper uses the Leiden algorithm, chosen because it produces good-quality hierarchical partitions efficiently at scale — groups the graph into communities: sets of entities that are much more densely connected to each other than to the rest of the graph, at multiple levels of a hierarchy (fine-grained sub-communities nested inside broader ones). Fourth, for every community at every level, an LLM generates a summary describing what that community is "about" — its dominant entities, their relationships, and the themes tying them together. This produces a layered set of pre-computed summaries, from tightly-scoped local clusters up to broad, corpus-spanning ones.

At query time this structure supports two genuinely different retrieval modes, and picking the right one is itself a meaningful architectural decision:

- **Local search** starts by identifying the specific entities named or implied in the query, then traverses the graph outward from those entities — pulling in directly connected entities, their relationships, and the source text chunks that mentioned them. This is the right mode for specific, entity-centric questions ("What products does Acme Corp's partnership with Globex cover?") where the answer is genuinely localized to a neighborhood of the graph, and it plays a role similar to traditional chunk retrieval but with the graph providing connective structure that pure vector similarity would miss (two chunks that never use overlapping vocabulary but describe the same entity's two different relationships are only linkable via the graph).
- **Global search** ignores individual entities and instead operates over the pre-computed community summaries, typically via a map-reduce pattern: the query is run against every relevant community summary (or a sample of them) in a "map" step, each producing a partial answer with an importance/relevance score, and then a "reduce" step synthesizes the highest-scoring partial answers into one final response. This is the mode that actually answers corpus-wide questions, because the community summaries — not any single chunk — are the artifact in the system that already represents "what large swaths of this corpus are about."

### 4.3 A working implementation

The following is deliberately toy-scale (a graph with a handful of entities is enough to demonstrate the mechanism), but every step — LLM-based extraction, graph construction, community detection, per-community summarization, and both search modes — mirrors the real architecture.

```python
"""
graphrag_lite.py

A simplified GraphRAG pipeline:
  1. LLM-based entity/relationship extraction per chunk.
  2. Graph construction with networkx.
  3. Community detection (greedy modularity as a dependency-light stand-in
     for Leiden -- swap for python-igraph + leidenalg in production).
  4. LLM-generated summaries per community.
  5. Local search (entity-neighborhood traversal) and global search
     (map-reduce over community summaries).

Dependencies: networkx, openai.
    pip install networkx openai
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from openai import OpenAI

client = OpenAI()
CHAT_MODEL = "gpt-4o-mini"


def llm_json(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Step 1: extract entities and relationships from each chunk.
# ---------------------------------------------------------------------------

def extract_entities_relationships(chunk_text: str) -> dict:
    prompt = (
        "Extract entities and relationships from the text below. "
        "Respond as JSON: {\"entities\": [{\"name\": \"...\", "
        "\"type\": \"...\", \"description\": \"...\"}], "
        "\"relationships\": [{\"source\": \"...\", \"target\": \"...\", "
        "\"description\": \"...\"}]}\n\n"
        f"Text:\n{chunk_text}"
    )
    return llm_json(prompt)


# ---------------------------------------------------------------------------
# Step 2: build the graph from extractions across all chunks.
# ---------------------------------------------------------------------------

def build_graph(chunks: List[str]) -> nx.Graph:
    graph = nx.Graph()
    for chunk in chunks:
        extraction = extract_entities_relationships(chunk)
        for entity in extraction.get("entities", []):
            name = entity["name"]
            if graph.has_node(name):
                graph.nodes[name]["description"] += " " + entity.get("description", "")
            else:
                graph.add_node(name, type=entity.get("type", "unknown"),
                                description=entity.get("description", ""),
                                source_chunks=[chunk])
            graph.nodes[name].setdefault("source_chunks", []).append(chunk)
        for rel in extraction.get("relationships", []):
            source, target = rel.get("source"), rel.get("target")
            if source and target and graph.has_node(source) and graph.has_node(target):
                graph.add_edge(source, target, description=rel.get("description", ""))
    return graph


# ---------------------------------------------------------------------------
# Step 3: community detection. greedy_modularity_communities is a
# dependency-light stand-in for Leiden; swap for leidenalg at scale, since
# Leiden gives better-quality, more stable hierarchical partitions.
# ---------------------------------------------------------------------------

def detect_communities(graph: nx.Graph) -> List[set]:
    if graph.number_of_edges() == 0:
        return [{node} for node in graph.nodes]
    return list(greedy_modularity_communities(graph))


# ---------------------------------------------------------------------------
# Step 4: summarize each community with an LLM.
# ---------------------------------------------------------------------------

def summarize_community(graph: nx.Graph, community: set) -> str:
    lines = []
    for node in community:
        data = graph.nodes[node]
        lines.append(f"Entity: {node} ({data.get('type')}) - {data.get('description')}")
    for u, v, data in graph.edges(data=True):
        if u in community and v in community:
            lines.append(f"Relationship: {u} -> {v}: {data.get('description')}")

    prompt = (
        "Summarize the following cluster of entities and relationships "
        "into a short paragraph describing the overall theme this "
        "cluster represents.\n\n" + "\n".join(lines)
    )
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    ).choices[0].message.content


def build_community_summaries(graph: nx.Graph) -> Dict[int, str]:
    communities = detect_communities(graph)
    return {i: summarize_community(graph, community) for i, community in enumerate(communities)}


# ---------------------------------------------------------------------------
# Local search: traverse outward from entities named in the query.
# ---------------------------------------------------------------------------

def find_query_entities(graph: nx.Graph, query: str) -> List[str]:
    query_lower = query.lower()
    return [node for node in graph.nodes if node.lower() in query_lower]


def local_search(graph: nx.Graph, query: str, hops: int = 1) -> str:
    seed_entities = find_query_entities(graph, query)
    if not seed_entities:
        return "No specific entities from the query were found in the graph."

    neighborhood = set(seed_entities)
    frontier = set(seed_entities)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(graph.neighbors(node))
        neighborhood.update(next_frontier)
        frontier = next_frontier

    context_lines = [f"{n}: {graph.nodes[n].get('description')}" for n in neighborhood]
    for u, v, data in graph.edges(data=True):
        if u in neighborhood and v in neighborhood:
            context_lines.append(f"{u} -> {v}: {data.get('description')}")

    prompt = (
        "Answer the query using the entity/relationship context below, "
        f"which is the local graph neighborhood around {seed_entities}.\n\n"
        f"Context:\n" + "\n".join(context_lines) + f"\n\nQuery: {query}\nAnswer:"
    )
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    ).choices[0].message.content


# ---------------------------------------------------------------------------
# Global search: map-reduce over pre-computed community summaries. This is
# the mode that answers broad, corpus-wide questions no single chunk or
# entity neighborhood could answer on its own.
# ---------------------------------------------------------------------------

def global_search(community_summaries: Dict[int, str], query: str) -> str:
    # Map step: score each community summary's relevance + partial answer.
    partial_answers = []
    for community_id, summary in community_summaries.items():
        result = llm_json(
            "Given a broad query and one community summary from a larger "
            "corpus, produce a partial answer to the query using only "
            "this summary, and rate its relevance 0-10. Respond as JSON: "
            "{\"partial_answer\": \"...\", \"relevance\": 0-10}.\n\n"
            f"Community summary:\n{summary}\n\nQuery: {query}"
        )
        if result.get("relevance", 0) > 0:
            partial_answers.append(result)

    # Reduce step: synthesize the highest-relevance partial answers.
    partial_answers.sort(key=lambda r: r["relevance"], reverse=True)
    top_partials = "\n\n".join(
        f"(relevance {r['relevance']}) {r['partial_answer']}" for r in partial_answers[:10]
    )
    prompt = (
        "Synthesize a single, coherent answer to the query from these "
        f"partial answers, each drawn from a different part of the corpus.\n\n"
        f"Partial answers:\n{top_partials}\n\nQuery: {query}\nFinal answer:"
    )
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    ).choices[0].message.content
```

The key architectural fact to hold onto is that `build_graph` and `build_community_summaries` run once at indexing time (they're expensive — an LLM call per chunk for extraction, plus one per community for summarization — but amortized across every future query), while `local_search` and `global_search` run per query against structures that are already built. Choosing between the two modes in production is usually done either by a router (an LLM classifies the query as specific/entity-centric versus broad/thematic) or by exposing both as separate tools in an agentic setup, which is exactly the pattern in Section 6.

## 5. RAPTOR: a summarization tree instead of a knowledge graph

### 5.1 The problem it targets

RAPTOR (Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval," 2024) is aimed at the same structural gap as GraphRAG's global search — a flat store of leaf-level chunks has no representation of the corpus at higher levels of abstraction, so broad questions that require reasoning across many chunks or whole documents are unanswerable by any single retrieval hit — but it gets there through a completely different mechanism: recursive clustering and summarization instead of entity/relationship graph extraction. Where GraphRAG's structure is built from explicit entities and their connections, RAPTOR's structure is built purely from the embedding space of the chunks themselves, which makes it considerably cheaper to build (no per-chunk entity extraction step) and domain-agnostic (it needs no notion of "entity type" at all — it works on anything you can embed and summarize).

### 5.2 The mechanism

Start with the leaf level: the normal chunks produced by whatever chunking strategy Chapter 2 covers, each embedded as usual. RAPTOR then clusters these chunk embeddings — the paper uses a Gaussian Mixture Model with soft clustering (a chunk can belong to more than one cluster, which matters because real text chunks often legitimately span more than one topic) — into groups of semantically related chunks. Each cluster is then summarized by an LLM into a single new "node" — a synthetic higher-level text unit that condenses what the whole cluster is about. Crucially, the process does not stop there: those cluster summaries are themselves embedded and re-clustered, and re-summarized, recursively, forming successive layers — level 1 summaries of leaf chunks, level 2 summaries of level-1 summaries, and so on, up until clustering stops producing meaningfully separable clusters (typically when everything collapses into one cluster spanning the whole document set, which becomes the root). The result is a tree: leaves are raw chunks, intermediate nodes are progressively broader summaries, and the root is a summary of the entire corpus.

At query time, this tree gives you a choice of granularity that a flat store never could: a narrow factual question ("What was the exact refund window mentioned in policy-001?") is best answered by a leaf-level chunk, because summarizing away the raw text would lose the precise number, while a broad question ("What's the overall tone and philosophy behind our customer-facing policies?") is best answered by a high-level node, because no leaf chunk individually captures a corpus-wide pattern but a level-3 summary might. Two retrieval strategies exploit this tree, and the choice between them is a real design decision worth naming precisely, since it comes up in interviews: **collapsed-tree retrieval** flattens every node at every level (leaves and all intermediate/root summaries) into a single pool and ranks them all together by similarity to the query, letting the most relevant node win regardless of which level it's on — simple and, per RAPTOR's own ablations, usually the stronger default; **tree-traversal retrieval** instead starts at the root and greedily descends only into the child branches whose summaries look most relevant to the query, narrowing level by level until it reaches the chunks it will actually use — cheaper at query time (it never scores the whole tree) but riskier, because a bad early branching decision near the root prunes away subtrees it can never recover from later.

### 5.3 A working implementation

```python
"""
raptor_lite.py

Recursive clustering + summarization tree, RAPTOR-style:
  1. Embed leaf chunks.
  2. Cluster embeddings with Gaussian Mixture Models (soft clustering).
  3. Summarize each cluster with an LLM into a new node.
  4. Recurse on the new nodes' embeddings until a single cluster remains.
  5. Collapsed-tree retrieval: pool every node at every level, rank by
     similarity to the query.

Dependencies: numpy, scikit-learn, openai.
    pip install numpy scikit-learn openai
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from sklearn.mixture import GaussianMixture
from openai import OpenAI

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

MAX_CLUSTER_SIZE_FOR_STOP = 1   # stop recursing once one cluster covers everything
SOFT_ASSIGNMENT_THRESHOLD = 0.15  # a node can belong to multiple clusters


@dataclass
class TreeNode:
    text: str
    level: int                       # 0 = leaf chunk, increases toward the root
    embedding: np.ndarray
    children: List["TreeNode"] = field(default_factory=list)


def embed_texts(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def summarize_cluster(texts: List[str]) -> str:
    joined = "\n\n".join(texts)
    prompt = (
        "Write a concise summary that captures the shared content and "
        f"themes of the following related passages:\n\n{joined}"
    )
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    ).choices[0].message.content


def cluster_embeddings(embeddings: np.ndarray) -> List[List[int]]:
    """Soft-cluster embeddings with a GMM, choosing cluster count via BIC.
    Returns a list of clusters, each a list of member indices; an item can
    appear in more than one cluster if its membership probability clears
    the threshold for multiple components."""
    n_samples = len(embeddings)
    if n_samples <= 2:
        return [list(range(n_samples))]

    max_components = min(n_samples - 1, 10)
    best_gmm, best_bic = None, float("inf")
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=0, n_init=1)
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_bic, best_gmm = bic, gmm

    probabilities = best_gmm.predict_proba(embeddings)
    clusters = [[] for _ in range(best_gmm.n_components)]
    for idx, probs in enumerate(probabilities):
        for cluster_idx, prob in enumerate(probs):
            if prob >= SOFT_ASSIGNMENT_THRESHOLD:
                clusters[cluster_idx].append(idx)
    return [c for c in clusters if c]


def build_raptor_tree(chunk_texts: List[str]) -> List[TreeNode]:
    """Builds the tree bottom-up and returns ALL nodes across all levels
    (this flat list of every node, leaves through root, is exactly what
    collapsed-tree retrieval searches over)."""
    leaf_embeddings = embed_texts(chunk_texts)
    current_level_nodes = [
        TreeNode(text=text, level=0, embedding=emb)
        for text, emb in zip(chunk_texts, leaf_embeddings)
    ]
    all_nodes: List[TreeNode] = list(current_level_nodes)
    level = 0

    while len(current_level_nodes) > MAX_CLUSTER_SIZE_FOR_STOP:
        embeddings = np.stack([node.embedding for node in current_level_nodes])
        clusters = cluster_embeddings(embeddings)

        # If clustering can't reduce the node count at all, stop -- we've
        # collapsed as far as the corpus structure allows.
        if len(clusters) >= len(current_level_nodes):
            break

        next_level_nodes = []
        for member_indices in clusters:
            members = [current_level_nodes[i] for i in member_indices]
            summary_text = summarize_cluster([m.text for m in members])
            summary_embedding = embed_texts([summary_text])[0]
            new_node = TreeNode(
                text=summary_text, level=level + 1,
                embedding=summary_embedding, children=members,
            )
            next_level_nodes.append(new_node)

        all_nodes.extend(next_level_nodes)
        current_level_nodes = next_level_nodes
        level += 1

    return all_nodes


# ---------------------------------------------------------------------------
# Collapsed-tree retrieval: pool every node from every level and rank by
# cosine similarity to the query, letting the best-matching granularity
# win regardless of whether it's a leaf or a high-level summary.
# ---------------------------------------------------------------------------

def collapsed_tree_retrieve(query: str, all_nodes: List[TreeNode], top_k: int = 5) -> List[TreeNode]:
    query_embedding = embed_texts([query])[0]
    q_norm = query_embedding / np.linalg.norm(query_embedding)

    scored = []
    for node in all_nodes:
        n_norm = node.embedding / np.linalg.norm(node.embedding)
        similarity = float(np.dot(q_norm, n_norm))
        scored.append((similarity, node))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [node for _, node in scored[:top_k]]
```

Two implementation details are worth calling out because they're easy to get wrong when reproducing RAPTOR. First, the recursion terminates on cluster count, not tree depth — if a corpus is small or homogeneous enough that GMM clustering collapses to a single cluster after the very first pass, the tree will only have two or three levels, and that's correct behavior, not a bug. Second, `build_raptor_tree` deliberately returns *all* nodes from every level in one flat list rather than just the root or just the leaves, because that flat pool across levels is precisely what collapsed-tree retrieval needs to search over — the tree structure (via `children`) is preserved on each node in case you also want to implement tree-traversal retrieval, which would instead start from the top-level nodes and only expand into `node.children` when a node scores above some relevance bar, stopping the search on branches that don't.

## 6. Agentic RAG: retrieval as a tool, not a stage

### 6.1 The problem it targets

Self-RAG, CRAG, GraphRAG, and RAPTOR all improve on naive RAG, but every one of them is still, structurally, a *pipeline* — a fixed sequence of stages (even CRAG's branching, or Self-RAG's regeneration loop, is a bounded, pre-determined control flow that the engineer wrote in advance). None of them let the model itself decide, at reasoning time, that it needs a second, differently-scoped retrieval after seeing the first result, or that it needs to check a fact against a SQL table between two retrieval calls, or that a completely different tool entirely (a calculator, a code interpreter) is what the next step actually requires. Real research tasks — the kind a competent human analyst does — routinely need exactly this: look something up, notice the result raises a new question, look up the new question, maybe compute something with the combined findings, then answer. Hardcoding that shape as a fixed pipeline (even a branching one) doesn't scale to open-ended tasks, because you can't anticipate every branch in advance.

Agentic RAG's shift is to stop treating retrieval as a pipeline stage the orchestration code invokes on the model's behalf, and instead expose it as a *tool* the model itself chooses to call, the same way it might call a calculator or a code execution sandbox — inside a ReAct-style ("Reasoning + Acting," Yao et al. 2022) loop where the model alternates between emitting a thought, choosing (or not choosing) to invoke a tool, observing the tool's result, and deciding what to do next, including whether to call a tool again. This is a strict generalization of everything earlier in this chapter: an agentic system *can* reproduce CRAG's evaluate-and-fallback behavior (by having the model notice weak results and choose to call a differently-scoped search tool) or GraphRAG's mode-switching (by exposing local-search and global-search as two distinct tools and letting the model pick), but it can also do things none of those fixed pipelines can, like interleaving three retrieval calls against different indexes with a calculator call in between, in whatever order the specific question actually demands, with no engineer having pre-wired that particular sequence.

### 6.2 A working implementation

```python
"""
agentic_rag.py

A minimal ReAct-style loop where retrieval is exposed as one callable tool
among several, and the LLM decides for itself whether to call it, what
query to issue, whether results are sufficient, and when to stop.

Dependencies: openai.
    pip install openai
"""

import json
from typing import Callable, List
from openai import OpenAI

client = OpenAI()
CHAT_MODEL = "gpt-4o-mini"
MAX_STEPS = 6


# ---------------------------------------------------------------------------
# Tool definitions. `search_knowledge_base` wraps whatever retrieval stack
# earlier chapters built (dense/sparse/hybrid + reranking); it is exposed
# here as just another function the model can choose to call, with its own
# argument schema, exactly like the calculator tool beside it.
# ---------------------------------------------------------------------------

def search_knowledge_base(query: str, retrieve_fn: Callable) -> str:
    results = retrieve_fn(query, top_k=3)
    if not results:
        return "No results found."
    return "\n\n".join(f"[{r.source}] {r.text}" for r in results)


def calculator(expression: str) -> str:
    try:
        # A real system would use a safe expression evaluator, not eval();
        # shown minimally here since the point is the tool-calling loop.
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the internal knowledge base for information relevant "
                "to a query. Call this whenever you need a specific fact you "
                "don't already have. You may call it multiple times with "
                "refined queries if earlier results were insufficient."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a numeric arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def run_agentic_rag(user_query: str, retrieve_fn: Callable) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. You have tools available: "
                "search_knowledge_base and calculator. Decide for yourself "
                "whether you need to call a tool, what to search for (it "
                "may differ from the user's exact wording), and whether "
                "results are sufficient or you need another, refined call. "
                "Only answer directly, without calling any tool, once you "
                "have enough grounding -- do not guess at facts you could "
                "look up."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.0,
        )
        message = response.choices[0].message

        # Stopping condition: the model chose to answer directly instead
        # of calling a tool -- it has judged its own grounding sufficient.
        if not message.tool_calls:
            return message.content

        messages.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "search_knowledge_base":
                result = search_knowledge_base(args["query"], retrieve_fn)
            elif tool_call.function.name == "calculator":
                result = calculator(args["expression"])
            else:
                result = f"Unknown tool: {tool_call.function.name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # Safety valve: force a final answer if the loop ran to MAX_STEPS
    # without the model naturally stopping, so the system never hangs
    # or returns nothing on a pathological query.
    messages.append({
        "role": "user",
        "content": "Please provide your best answer now based on everything gathered so far.",
    })
    final = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.0)
    return final.choices[0].message.content
```

The stopping condition — `if not message.tool_calls: return message.content` — is the entire architectural difference from every previous pattern in this chapter distilled into one line: the model itself decides when it has enough grounding to answer, rather than the orchestration code deciding after a fixed number of stages. The `MAX_STEPS` cap and the forced-final-answer fallback are the necessary production guardrail against a model that loops indefinitely or never converges; without a bound like this, an agentic loop is a genuine reliability and cost risk, not just a performance one, since each additional step is a full LLM call. Note also that `search_knowledge_base` here is a thin wrapper around whatever `retrieve_fn` you already built in Chapters 4-6 — hybrid retrieval, reranking, query transformation — none of that infrastructure is replaced by going agentic; it simply gets called zero, one, or several times, on queries the model composes itself, instead of being invoked exactly once on the user's literal question.

## 7. Choosing among them in production

These five patterns are not mutually exclusive tiers of a ladder where you always pick the fanciest one — they solve different problems, and a senior engineer's job is matching the pattern to the failure mode actually observed in evaluation, not defaulting to the most sophisticated architecture available.

Reach for Self-RAG or CRAG when the corpus is a fairly standard document collection (support docs, policies, internal wikis) and the problem you're actually seeing is trust and correctness on largely factoid, single-hop questions: users get answers, but too many of those answers are subtly unsupported, or retrieval occasionally returns garbage that the generator dutifully answers from anyway. Self-RAG is the better fit when the failure is diffuse across generation — unsupported claims creeping into otherwise fine answers — because its segment-level ISSUP critique is built to catch exactly that. CRAG is the better fit when the failure is concentrated in retrieval itself — queries that miss the corpus, stale or sparse indexes, cases where "go check somewhere else" is a real, available option — because its evaluate-then-fallback design is cheaper and more predictable than a general critique loop when that's the specific problem.

Reach for GraphRAG or RAPTOR when evaluation shows the corpus is being asked genuinely broad, cross-document, thematic questions in addition to narrow factoid ones — "summarize the key risks across all vendor contracts," "what's changed in our incident patterns over the year" — because these are exactly the questions where Chapters 1 through 6's entire toolkit structurally cannot help: no amount of better chunking, embedding, hybrid search, or reranking fixes the fact that no single chunk (or top-k set of chunks) represents "the whole corpus." Between the two, GraphRAG tends to win when the domain has rich, nameable entities and relationships worth making explicit (organizations, people, products, regulations, and how they interconnect) and when you specifically need entity-centric local search as well as thematic global search; RAPTOR tends to win when the content doesn't decompose naturally into a clean entity graph (long-form narrative, freeform research notes, mixed unstructured content) but does benefit from progressively broader summarization, and it is meaningfully cheaper and simpler to build since it skips entity/relationship extraction and community detection entirely in favor of embedding-space clustering the corpus already supports.

Reach for Agentic RAG when the task itself is genuinely open-ended and multi-step, rather than "answer one question well" — when the system needs to combine retrieval with other tools (a calculator, a SQL query, a code execution sandbox, a second and differently-scoped search index) in an order that can't be anticipated and hardcoded in advance, or when the user's requests are more like research assignments than single questions ("compare these three vendors' SLAs and flag any clause that conflicts with our compliance policy"). Agentic RAG subsumes the others architecturally — an agent can be given a CRAG-style evaluate-and-refine tool, or GraphRAG's local/global search as two separate tools, or a Self-RAG-style critique step as another tool call — but that generality is exactly why it's the most expensive and least predictable option to run: an unbounded reasoning loop over a general-purpose tool set is harder to test, harder to put a firm latency SLA on, and harder to debug when it goes wrong than a fixed pipeline with two or three explicit branches.

The trade every single one of these five patterns makes is the same, and it's worth stating explicitly because it's the point evaluation (Chapter 8) exists to settle: each pattern spends additional LLM calls — a critique pass, an evaluator call, a map-reduce over dozens of community summaries, a multi-step agent loop — to buy higher answer quality, better grounding, or the ability to answer a class of question plain retrieval structurally cannot touch. That is real latency and real inference cost, multiplied per user query, not a one-time engineering cost you pay off and forget. None of these patterns should be adopted by default or because they are the current state of the art; they should be adopted when your evaluation numbers on your actual corpus and your actual query distribution show that plain single-pass RAG's answer quality, faithfulness, or coverage of broad questions is genuinely insufficient for the product — and even then, adopt the narrowest pattern that fixes the specific failure mode you measured, not the most architecturally impressive one available.
