# Query Transformation and Expansion

## The Query Is Not the Search Key

The previous chapter treated retrieval quality as a function of the index: dense embeddings, sparse lexical signals, and hybrid fusion between the two. All of that machinery shares a silent assumption — that the string being embedded or tokenized for search is a good representation of what the user actually needs. That assumption is frequently false, and it fails in a specific, structural way that has nothing to do with how good the retriever is.

A user asks a RAG system, "What caused the 2008 crash?" That is a perfectly good question for a human to receive — it's short, unambiguous in intent, and exactly the right length for a conversation turn. But it is a poor *search key*. The documents in the corpus that actually answer it are not phrased as questions; they are phrased as answers: paragraphs about subprime mortgage securitization, credit default swaps, Lehman Brothers' collateral exposure, and Federal Reserve rate policy in the mid-2000s. A dense retriever embeds the query "What caused the 2008 crash?" and the passage "Excessive leverage in mortgage-backed securities, combined with inadequate risk disclosure by ratings agencies, left major institutions exposed when housing prices reversed in 2007..." into the same vector space, but these two pieces of text differ in more than topic — they differ in register, length, grammatical mood (interrogative versus declarative), and vocabulary density. Embedding models are trained to place semantically related text nearby, but "related" is doing a lot of work when the two texts are structurally this different. This is often called the **asymmetry gap** between questions and answers, and it is one of the most consistent, well-documented failure modes in production RAG systems — not because the embedding model is bad, but because it was never given a fair comparison to make.

A second, independent failure mode is **underspecification**. A single short query is a single point in embedding space. If the truly relevant passage happens to use different terminology, sit at a different level of abstraction, or address a sub-aspect of the question that the literal wording didn't emphasize, that one point can simply miss it — not because the passage is irrelevant, but because nearest-neighbor search from one query vector has a limited "reach," and the geometry didn't cooperate.

A third failure mode is **compositionality**. Some questions cannot be answered by any single passage in the corpus at all, no matter how well phrased the query is, because the answer requires combining facts that live in different documents. No amount of clever embedding of the original question closes that gap, because the problem isn't retrieval quality — it's that the question, as posed, doesn't correspond to a single retrieval target.

Query transformation is the family of techniques that address all three failure modes *before* the retriever is ever invoked: rewrite, expand, or decompose the query into something (or several somethings) that make better search keys than the user's literal words. This chapter covers four techniques that solve for each failure mode in a different way — HyDE for the asymmetry gap, multi-query generation for underspecification, decomposition for compositionality, and step-back prompting for a related but distinct problem: retrieving background knowledge the narrow question doesn't ask for but the correct answer depends on. All four share a common shape: spend one or more extra LLM calls up front to produce a better retrieval query (or set of queries) than the raw user input, then hand the *transformed* text to the same dense/sparse/hybrid retrieval stack from Chapter 4 unchanged. Nothing here replaces the retriever — it only changes what gets fed into it. Chapter 6 covers what happens to the results *after* they come back (reranking), and Chapter 7 covers architectures that loop retrieval and generation together adaptively (Self-RAG, CRAG, agentic multi-step retrieval); this chapter is deliberately scoped to the single step that happens before the vector store or search index is ever touched.

## HyDE: Hypothetical Document Embeddings

### The Core Trick

HyDE, introduced by Gao et al. in "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022), attacks the asymmetry gap head-on with a trick that sounds counterintuitive the first time you hear it: instead of embedding the user's real question, ask an LLM to *hallucinate an answer* to it, and embed the hallucination instead.

Concretely, the pipeline is: take the query, prompt an LLM to write a plausible-sounding passage that would answer it — in the style of the kind of document the corpus actually contains (a paragraph of documentation, an encyclopedia entry, a paper abstract, whatever fits the domain) — then embed *that generated passage*, not the original query, and use the resulting vector to search the index. The generated passage is never shown to the user and never treated as a source of truth; it exists purely as a better-shaped probe into embedding space. It is completely fine, even expected, that the hypothetical document contains factual errors, invented numbers, or a wrong causal story — the LLM is very likely to get *entity-level details* wrong (dates, names, specific figures) precisely because it's being asked to answer from parametric memory without grounding. What it reliably gets right is *style*: an LLM asked to write "a passage explaining what caused the 2008 financial crash" produces declarative, answer-shaped prose using the same vocabulary and structural register that genuine finance-domain documents use — because that's the distribution the LLM learned to imitate. That answer-shaped text sits much closer, in embedding space, to the real relevant passage than the short interrogative original query does, even though the hypothetical document's specific claims may be wrong. The retriever isn't trusting the hypothetical document's *facts* — it's exploiting its *shape*.

This is why HyDE is described as doing "relevance modeling via generation" rather than via labeled data: it needs no query-passage relevance pairs, no fine-tuning of the embedding model, and no training signal specific to the target corpus. It works zero-shot on any domain the base LLM has enough general knowledge to plausibly imitate the writing style of, which is a surprisingly low bar — the LLM doesn't need to know the *right answer*, only what a right-shaped answer *looks like*.

### Implementation

A minimal HyDE pipeline is three steps: generate the hypothetical document, embed it, and search with that embedding instead of the raw query's embedding.

```python
class HyDERetriever:
    def __init__(self, llm, embed_fn, vector_store, style_hint: str = "a detailed encyclopedia passage"):
        self.llm = llm
        self.embed_fn = embed_fn          # text -> vector
        self.vector_store = vector_store  # exposes .search(vector, top_k)
        self.style_hint = style_hint

    def generate_hypothetical_document(self, query: str) -> str:
        prompt = f"""Write {self.style_hint} that directly answers the following
question. Write as if you are certain of the answer, even if you are not.
Do not mention that this is hypothetical or express uncertainty. Aim for the
length and tone of a real reference passage, not a short reply.

Question: {query}

Passage:"""
        return self.llm.generate(prompt, max_tokens=256, temperature=0.7)

    def retrieve(self, query: str, top_k: int = 5):
        hypothetical_doc = self.generate_hypothetical_document(query)
        hyde_vector = self.embed_fn(hypothetical_doc)
        return self.vector_store.search(hyde_vector, top_k=top_k)
```

The `temperature=0.7` is a deliberate choice, not a default left unconsidered: HyDE benefits from a moderately creative, natural-sounding generation rather than the terse, hedged output a low-temperature, safety-tuned model tends to produce for questions it isn't fully sure about. A hedgy hypothetical document ("it is difficult to say precisely, but generally speaking...") is a worse search key than a confident, wrong one, because hedging language doesn't resemble the assertive register of real reference documents either.

### Guarding Against Drift: The Ensemble Fallback

HyDE's failure mode is the mirror image of its strength: if the hypothetical document hallucinates in a *wrong direction* — confidently answering with a plausible-sounding but topically off-base story — the embedding of that document can pull retrieval toward an entirely wrong region of the corpus, one that resembles the hallucination's shape but not the query's actual intent. This is more likely on obscure or highly specific factual questions where the LLM has little real signal to draw on and effectively invents an answer from a different-but-similar-sounding domain.

The standard mitigation is to never rely on HyDE alone: retrieve with both the hypothetical document's embedding and the original query's embedding, then merge the two result sets. If HyDE's guess is on-topic, its results reinforce or extend the original query's results. If HyDE drifts, the original query's results are still present in the merged set as a safety net.

```python
class EnsembleHyDERetriever(HyDERetriever):
    def retrieve(self, query: str, top_k: int = 5, hyde_weight: float = 0.6):
        hypothetical_doc = self.generate_hypothetical_document(query)
        hyde_vector = self.embed_fn(hypothetical_doc)
        original_vector = self.embed_fn(query)

        hyde_hits = self.vector_store.search(hyde_vector, top_k=top_k * 2)
        original_hits = self.vector_store.search(original_vector, top_k=top_k * 2)

        # Merge by combining scores for chunks retrieved by either query,
        # weighting HyDE's contribution against the literal query's.
        combined_scores = {}
        for hit in hyde_hits:
            combined_scores[hit.chunk_id] = combined_scores.get(hit.chunk_id, 0.0) + hyde_weight * hit.score
        for hit in original_hits:
            combined_scores[hit.chunk_id] = combined_scores.get(hit.chunk_id, 0.0) + (1 - hyde_weight) * hit.score

        ranked_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        chunk_lookup = {hit.chunk_id: hit for hit in hyde_hits + original_hits}
        return [chunk_lookup[cid] for cid in ranked_ids]
```

`hyde_weight` above is tunable per domain: corpora with a strong asymmetry gap (short conversational queries against long technical documentation) usually benefit from weighting HyDE higher, while corpora where queries are already fairly close in style to the documents (e.g., searching a FAQ database with FAQ-style questions) get less benefit from HyDE and should weight the original query higher, or skip HyDE for that traffic altogether — a point the closing section returns to.

## Multi-Query Generation

### Why One Query Vector Isn't Enough

HyDE changes *how* a single query is phrased before embedding it. Multi-query generation instead accepts that a single embedding, however well-shaped, is still one point in a high-dimensional space, and a relevant passage can simply be sitting in a nearby-but-distinct region that a single point doesn't reach. The fix is to stop relying on one point: ask an LLM to generate several different phrasings of the same underlying information need — different vocabulary, different framing of the question, different angles on sub-aspects the user's phrasing didn't emphasize, different levels of specificity — then retrieve independently for each variant and take the union (or a fused ranking) of the results.

This works because relevance in embedding space is not perfectly smooth with respect to paraphrase: two questions that a human would consider identical in meaning can still embed measurably differently, and if the corpus's relevant passage happens to sit closer to phrasing B than to phrasing A, a system that only ever tries phrasing A will simply never find it. Sampling several plausible phrasings of the same intent is a way of hedging against exactly that geometric bad luck, and it costs nothing in the underlying retriever's quality — it's purely a way of giving the same retriever more chances to succeed.

### Implementation

```python
class MultiQueryRetriever:
    def __init__(self, llm, embed_fn, vector_store, n_variants: int = 4):
        self.llm = llm
        self.embed_fn = embed_fn
        self.vector_store = vector_store
        self.n_variants = n_variants

    def generate_query_variants(self, query: str) -> list[str]:
        prompt = f"""Generate {self.n_variants} alternative versions of the following
search query. Each version should preserve the original intent but vary in
wording, vocabulary, specificity, or the particular sub-aspect emphasized.
Return one query per line, with no numbering or extra commentary.

Original query: {query}"""
        response = self.llm.generate(prompt, max_tokens=200, temperature=0.8)
        variants = [line.strip() for line in response.splitlines() if line.strip()]
        return [query] + variants[: self.n_variants]  # always include the original

    def retrieve(self, query: str, top_k_per_query: int = 5, final_top_k: int = 8):
        variants = self.generate_query_variants(query)

        # Retrieve independently per variant, tracking which rank each chunk
        # achieved under each query -- needed for reciprocal rank fusion below.
        per_query_rankings = []
        for variant in variants:
            hits = self.vector_store.search(self.embed_fn(variant), top_k=top_k_per_query)
            per_query_rankings.append(hits)

        fused = self._reciprocal_rank_fusion(per_query_rankings)
        return fused[:final_top_k]

    def _reciprocal_rank_fusion(self, rankings: list[list], k: int = 60):
        """Combine multiple ranked lists into one, rewarding chunks that
        rank well across several query variants rather than just the single
        best variant. RRF is a standard, hyperparameter-light fusion choice
        because it only needs rank position, not comparable raw scores --
        important here since each variant's similarity scores aren't
        necessarily on the same scale."""
        rrf_scores: dict[str, float] = {}
        chunk_lookup = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking):
                rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
                chunk_lookup[hit.chunk_id] = hit
        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [chunk_lookup[cid] for cid in ranked_ids]
```

Two details in this implementation matter in practice. First, deduplication is not a separate pass bolted on afterward — it falls out naturally from keying `rrf_scores` and `chunk_lookup` by `chunk_id`: a chunk retrieved by three different query variants simply accumulates score contributions from all three appearances rather than showing up three times in the final list, and it is exactly this accumulation that lets RRF reward *consensus* across variants, not just a single strong hit. Second, reciprocal rank fusion is used here rather than a naive max-of-scores combination because raw similarity scores from independently embedded queries are not guaranteed to be comparable in scale or calibration (variant A's cosine similarities might cluster in a narrower range than variant B's), whereas rank position is comparable by construction — "this chunk was the retriever's #1 pick" means the same thing regardless of which query produced it. A simpler alternative, take-the-max-score-per-chunk-across-variants, is easier to implement and works reasonably when all variants are embedded with the same model and the score distributions are known to be well-behaved, but RRF is the more robust default when that assumption is shaky.

## Query Decomposition for Multi-Hop Questions

### Single-Hop vs. Multi-Hop

A single-hop question is answerable from a single retrieved passage: "What is the boiling point of water at sea level?" has one passage somewhere that states the answer directly. A multi-hop question requires combining facts that live in separate, often disjoint passages or documents, and no single retrieval call — however well-phrased — can return a chunk that simply *contains* the compound answer, because no such chunk exists in the corpus. Consider: "Which of the two companies that merged in 2019 had a higher IPO valuation, and who was its CEO at the time?" Answering this correctly requires, in order: identifying which two companies merged in 2019, retrieving each company's IPO valuation, comparing the two valuations to determine which was higher, and then retrieving the identity of that specific company's CEO at the relevant time. Every one of those is a separate fact, plausibly living in a separate document, and the comparison step (which valuation was higher) is not even a retrieval operation at all — it's arithmetic that has to happen *between* two retrieval steps.

Embedding the compound question as a single query and hoping the retriever surfaces something useful for all of that in one shot is asking too much of the retriever, no matter how good it is — this is the compositionality failure mode named earlier, and neither HyDE nor multi-query generation solves it, because both still treat the question as a single, atomic information need to be phrased better or sampled more broadly. Decomposition treats it correctly: as a sequence of atomic sub-questions, each of which *is* a single-hop question with its own well-shaped retrieval target.

### Sequential vs. Parallel Decomposition

Once an LLM has broken a compound question into sub-questions, there are two ways to resolve them. If the sub-questions are genuinely independent (e.g., "compare feature X of product A and feature X of product B" decomposes into two lookups that don't depend on each other), they can be retrieved and answered in parallel, then synthesized together at the end — this is cheaper in wall-clock time since the sub-question resolutions can run concurrently.

If a later sub-question depends on an entity or fact that only becomes known after an earlier sub-question is resolved — as in the merger example, where you cannot even ask "who was the CEO of the higher-valued company" until you know *which company* that is — the sub-questions must be resolved sequentially, feeding the answer to each step into the retrieval query (and reasoning context) for the next. This interleaving of retrieval and reasoning, where each step's *result* determines the next step's *query*, is the same underlying idea behind IRCoT ("Interleaving Retrieval with Chain-of-Thought Reasoning," Trivedi et al. 2022): rather than deciding all retrieval queries upfront, let the reasoning chain generate the next retrieval query only once the information needed to phrase it correctly is actually in hand.

### Implementation: Decomposition with Sequential Resolution

```python
class DecomposingRetriever:
    def __init__(self, llm, retriever, max_sub_questions: int = 4):
        self.llm = llm
        self.retriever = retriever  # any Chapter 4 dense/sparse/hybrid retriever
        self.max_sub_questions = max_sub_questions

    def decompose(self, question: str) -> list[str]:
        prompt = f"""Break the following question into an ordered list of atomic
sub-questions that must be answered in sequence to reach the final answer.
Each sub-question should be answerable independently, but later
sub-questions may depend on facts resolved by earlier ones -- in that case,
phrase the later sub-question generically (e.g., "What was the IPO
valuation of [company]?") rather than trying to guess the entity now.
Return at most {self.max_sub_questions} sub-questions, one per line.

Question: {question}"""
        response = self.llm.generate(prompt, max_tokens=200, temperature=0.2)
        return [line.strip("- ").strip() for line in response.splitlines() if line.strip()]

    def resolve_sequentially(self, question: str) -> dict:
        sub_questions = self.decompose(question)
        resolved_facts = []  # running log of (sub_question, answer) pairs

        for raw_sub_question in sub_questions:
            # Substitute any placeholders (e.g. "[company]") using facts
            # resolved so far, so later retrieval queries are concrete.
            grounded_sub_question = self._ground_with_prior_facts(raw_sub_question, resolved_facts)

            hits = self.retriever.search(grounded_sub_question, top_k=5)
            context = "\n\n".join(hit.text for hit in hits)

            answer_prompt = f"""Using only the context below, answer the sub-question
concisely. If the context is insufficient, say so explicitly.

Context:
{context}

Sub-question: {grounded_sub_question}

Answer:"""
            sub_answer = self.llm.generate(answer_prompt, max_tokens=150, temperature=0.0)
            resolved_facts.append((grounded_sub_question, sub_answer))

        synthesis_prompt = self._build_synthesis_prompt(question, resolved_facts)
        final_answer = self.llm.generate(synthesis_prompt, max_tokens=250, temperature=0.0)
        return {"final_answer": final_answer, "trace": resolved_facts}

    def _ground_with_prior_facts(self, sub_question: str, resolved_facts: list[tuple]) -> str:
        if not resolved_facts:
            return sub_question
        facts_summary = "\n".join(f"- {q} -> {a}" for q, a in resolved_facts)
        prompt = f"""Given these previously resolved facts:
{facts_summary}

Rewrite this sub-question, replacing any placeholder or vague reference
with the specific entity or value it refers to, so it can be used as a
standalone search query:

{sub_question}

Rewritten:"""
        return self.llm.generate(prompt, max_tokens=60, temperature=0.0).strip()

    def _build_synthesis_prompt(self, original_question: str, resolved_facts: list[tuple]) -> str:
        facts_block = "\n".join(f"- {q}\n  Answer: {a}" for q, a in resolved_facts)
        return f"""Original question: {original_question}

Resolved sub-questions and their answers:
{facts_block}

Using only the facts above, give a final, direct answer to the original
question."""
```

The `_ground_with_prior_facts` step is the load-bearing piece of this loop and the direct analog of IRCoT's interleaving: without it, the second sub-question in the merger example would be retrieved as the literal string "What was the IPO valuation of [company]?" — a query containing an unresolved placeholder, which no retriever can search against meaningfully. By rewriting it into a concrete entity-bearing query ("What was the IPO valuation of Company X?") using the fact resolved in the previous step, each retrieval call in the sequence gets a query that is as well-formed and specific as if a human had known the whole answer chain in advance and asked each question one at a time. This is also exactly where the sequential design pays for itself over a naive parallel decomposition: a parallel version would have tried to retrieve for the ungrounded placeholder query up front and failed to find anything useful, because that sub-question's real search key doesn't exist until an earlier step supplies it.

For genuinely independent sub-questions, the same `decompose` method's output can instead be fanned out concurrently — retrieve and answer each sub-question without the grounding step, then feed all resolved facts into a single synthesis prompt at the end, saving the sequential latency the dependent case requires.

## Step-Back Prompting

### Retrieving the Principle Behind the Specific Question

Step-back prompting, from Zheng et al.'s "Take a Step Back: Evoking Reasoning via Abstraction" (2023), addresses a failure mode distinct from all three above: sometimes the literal question is perfectly answerable by a single passage, phrased in a perfectly retrievable way, and still the retrieved context is insufficient — not because retrieval missed the relevant fact, but because the fact alone isn't enough to reason correctly, and the *background principle* needed to interpret it correctly was never asked for.

The paper's own example is illustrative: "What was the temperature at which water boiled in Denver during the 1962 experiment?" is a narrow, detail-heavy question. Even a retriever that perfectly finds a passage describing the 1962 Denver experiment may return a passage that states the observed temperature without explaining *why* it differs from the familiar 100°C figure — and an LLM reasoning only from that narrow passage might flag the number as an error, or fail to explain it, because it lacks the general fact that boiling point depends on atmospheric pressure, which itself depends on altitude. The narrow question, however well retrieved, doesn't surface that background principle, because the passage answering it isn't obligated to restate high-school chemistry.

Step-back prompting's fix: before retrieving for the specific question, ask the LLM to first generate a more abstract, general "step-back" question about the underlying principle or broader topic — in this case, "What factors affect the boiling point of a liquid?" Retrieve for *both* the step-back question and the original specific question, and combine the retrieved context from both searches before generating the final answer. The step-back retrieval surfaces foundational passages (atmospheric pressure and altitude effects on boiling point) that the narrow query would never have surfaced on its own, because those passages don't mention Denver, 1962, or any of the specific entities in the original question at all — they'd never rank highly against the original query's embedding, but they rank very highly against the abstracted one.

### Implementation

```python
class StepBackRetriever:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def generate_step_back_question(self, query: str) -> str:
        # Few-shot examples anchor the abstraction level the model should
        # aim for -- too abstract loses connection to the question, too
        # narrow just restates it.
        prompt = f"""You are an expert at rephrasing specific questions into more
general, higher-level questions that would help retrieve foundational
background knowledge relevant to answering the specific question.

Example:
Specific: What was the temperature at which water boiled in Denver during
the 1962 experiment?
Step-back: What factors affect the boiling point of a liquid?

Example:
Specific: Which Formula 1 driver won the most races in the 2016 season under
the V6 turbo-hybrid regulations introduced in 2014?
Step-back: What changed about Formula 1 competition after the 2014
regulation changes?

Now generate a step-back question for this one:
Specific: {query}
Step-back:"""
        return self.llm.generate(prompt, max_tokens=60, temperature=0.0).strip()

    def retrieve(self, query: str, top_k: int = 5):
        step_back_query = self.generate_step_back_question(query)

        specific_hits = self.retriever.search(query, top_k=top_k)
        general_hits = self.retriever.search(step_back_query, top_k=top_k)

        combined = self._dedupe_preserving_order(specific_hits + general_hits)
        return {
            "step_back_question": step_back_query,
            "context_chunks": combined,
        }

    def _dedupe_preserving_order(self, hits: list) -> list:
        seen = set()
        deduped = []
        for hit in hits:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                deduped.append(hit)
        return deduped

    def answer(self, query: str, top_k: int = 5) -> str:
        result = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(chunk.text for chunk in result["context_chunks"])
        prompt = f"""Use the context below -- which includes both specific details and
general background principles -- to answer the question. Use the
background principles to correctly interpret or explain the specific
details where relevant.

Context:
{context}

Question: {query}

Answer:"""
        return self.llm.generate(prompt, max_tokens=250, temperature=0.0)
```

Notice that the few-shot examples in `generate_step_back_question` are doing most of the calibration work here: without them, an LLM asked to "generalize" a question will sometimes overshoot into something so abstract it loses any useful connection to the original ("What is science?"), or undershoot into a trivial restatement that retrieves nothing new. Good step-back examples show the model the specific altitude of abstraction that's useful — one level up, not several — which is a domain-dependent calibration worth tuning with real examples from the target corpus rather than relying on the paper's original examples verbatim.

It's worth being precise about how step-back prompting differs from HyDE and multi-query generation, since all three "ask the LLM to produce alternative text before retrieving." HyDE changes the *register* of the query (question to answer-shaped text) while preserving its scope. Multi-query generation preserves both scope and register while sampling different *phrasings*. Step-back prompting deliberately *widens the scope* to a different, more general question, precisely because the original scope is too narrow to retrieve the background knowledge the answer depends on. They are not mutually exclusive — a production system can combine step-back retrieval with HyDE by generating a hypothetical document for the step-back question rather than embedding it directly — but each targets a different, specific gap between what the user asked and what the retriever needs.

## Comparing the Four Techniques

Each technique targets a distinct failure mode, adds a different multiple of LLM calls, and fails in its own characteristic way when misapplied — a useful summary to have on hand when deciding which one a given query actually needs.

| Technique | Failure mode addressed | Extra LLM calls | Characteristic failure if misapplied |
|---|---|---|---|
| HyDE | Asymmetry gap between question phrasing and answer phrasing | 1 (generation) | Hypothetical document hallucinates off-topic, dragging retrieval with it |
| Multi-query generation | Underspecification — one query vector missing a nearby relevant passage | 1 (fans out to N retrievals) | Variants converge on the same phrasing, adding cost without adding recall |
| Decomposition | Compositionality — multi-hop questions with no single matching passage | 1 planning call + 1 per sub-question | Sub-questions are wrongly assumed independent, so a dependent one retrieves against an ungrounded placeholder |
| Step-back prompting | Missing background principle needed to interpret a narrow fact correctly | 1 (generation) | Step-back question is pitched too abstractly, retrieving generic content disconnected from the original question |

None of these techniques are mutually exclusive, and production systems frequently combine them — most commonly HyDE applied to a step-back question, or multi-query generation applied independently to each sub-question produced by decomposition. The combinatorics get expensive quickly, though, which is exactly why the routing decision below matters as much as the techniques themselves.

## Cost, Latency, and When to Skip All of This

Every technique in this chapter buys retrieval quality by spending an LLM call before retrieval even starts — HyDE and step-back prompting each add one, multi-query generation adds one call that fans out into several retrieval requests, and decomposition adds one planning call plus one additional LLM call per sub-question for grounding and answering. None of this is free, and none of it is close to free at scale: a naive RAG pipeline that always runs the full transformation stack on every incoming query has turned what used to be "one retrieval call plus one generation call" into anywhere from three to a dozen LLM calls per user turn, multiplying both latency (every extra LLM call is another sequential round trip, unless carefully parallelized) and cost (every extra call is billed tokens, and generation calls for hypothetical documents or sub-answers are not small).

For a simple, well-formed factoid query — "What is the capital of Australia?", "What does the `--force` flag do in `git push`?" — none of this machinery helps, because there is no asymmetry gap to close (the query is already about the right length and register to match a direct-answer passage), no meaningful ambiguity for multi-query sampling to hedge against, no compositional structure for decomposition to unpack, and no missing background principle for step-back prompting to surface. Running the full transformation stack on such a query adds pure latency and cost with no corresponding accuracy gain, and in the case of HyDE specifically, even carries a small risk of the hypothetical document *introducing* drift that a plain, already-good query embedding wouldn't have had.

The production pattern that has emerged in response is selective routing: a fast, cheap classifier or a small set of heuristics inspects the incoming query before deciding whether to invoke a transformation pipeline at all. Signals that push toward transformation include query length and complexity (long, multi-clause questions are more likely to be multi-hop), the presence of comparison or superlative language ("which of... had more," "the highest," "compared to") that hints at compositional structure, low confidence or low top-score spread in a first-pass retrieval attempt (if the top results all score similarly and mediocre, the query embedding likely isn't finding a strong match, which is a signal worth reacting to before generation rather than after), or simply a domain-specific heuristic library built from observed failure cases. Simple, high-confidence factoid queries skip straight to retrieval and generation; queries flagged as ambiguous, comparison-heavy, or multi-hop get routed into the appropriate transformation — multi-query for suspected ambiguity, decomposition for suspected multi-hop structure, step-back for suspected missing-context needs, HyDE as a general-purpose asymmetry fix that's cheap enough to apply somewhat more liberally than the others. This routing decision itself can be as simple as a small fine-tuned classifier or even a single fast LLM call with a cheap model, since the cost of *deciding whether to transform* is far lower than the cost of the transformation itself.

It's also worth being clear about what query transformation does not solve. All four techniques in this chapter improve the *query* that reaches the retriever, but none of them evaluate whether the retrieval that comes back is actually any good — they are open-loop with respect to their own outcome. A HyDE-generated hypothetical document can still drift, a decomposition can still produce sub-questions that don't retrieve well, and a step-back question can still be pitched at the wrong level of abstraction, and none of the code in this chapter would notice or correct for it. That closed-loop correction — retrieving, grading the retrieved context's relevance, and deciding whether to retry, reformulate, or fall back to a different strategy based on that grade — is exactly the problem Self-RAG and CRAG are built to solve, and is covered in the next chapter's discussion of advanced RAG architectures. Query transformation and self-correction are complementary, not competing: transformation improves the odds of a good retrieval on the first attempt, and self-correction catches the cases where, despite a well-transformed query, the first attempt still wasn't good enough.
