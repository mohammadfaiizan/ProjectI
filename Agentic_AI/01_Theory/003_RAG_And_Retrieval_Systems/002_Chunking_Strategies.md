# Chunking Strategies

## Why Chunking Exists

Every RAG system is built on top of two hard physical constraints. First, embedding models have a maximum input length — typically somewhere between 256 and 8192 tokens depending on the model — beyond which text is either rejected or silently truncated. Second, LLM context windows, while much larger than they used to be, are still a finite and expensive resource: every token of retrieved context is a token you pay for, a token that adds latency, and a token that competes for the model's attention with every other token in the prompt. You cannot simply embed an entire 200-page PDF as a single vector and expect that vector to usefully represent "what this document is about" for the purpose of matching it against a narrow user question. Chunking is the answer to both constraints at once: it breaks a large corpus into retrieval-sized units that fit inside an embedding model's input limit and that can be selectively assembled into a prompt without blowing the context budget.

But chunking is not merely an engineering workaround for size limits — it is arguably the single highest-leverage design decision in the entire RAG pipeline, more consequential in practice than the choice of embedding model or vector database. This is because chunking determines the *granularity of retrieval*, and granularity sits directly on the precision/recall trade-off that governs whether RAG actually works. Get chunk size right, and the system retrieves passages that contain exactly the information needed to answer a question, with just enough surrounding context for the LLM to interpret it correctly. Get it wrong in either direction, and no amount of embedding model quality or reranking sophistication can fully compensate.

The core tension runs in two directions, and it is worth internalizing both because interview questions on chunking almost always probe for an understanding of the trade-off rather than a single "right answer."

When chunks are too large, two distinct problems compound. The first is an embedding quality problem: dense embedding models produce a single fixed-length vector that is, in effect, a weighted average of the semantic content of everything in the input. If a chunk covers five different subtopics — say, a document section that discusses pricing, then shipping policy, then a customer support contact, then a legal disclaimer — the resulting embedding is a blend that doesn't strongly represent any one of those topics. A query about shipping policy will produce a query vector that is reasonably close to the shipping portion of that blended embedding, but the cosine similarity will be diluted by the other four unrelated topics baked into the same vector. This is often called the "diluted embedding" or "semantic averaging" problem, and it is the single most common root cause of "the right document is in my corpus but the retriever didn't find it." The second problem with oversized chunks is purely economic: even when retrieval works, you now pay to inject a large block of text into the LLM's context window, most of which is irrelevant filler surrounding the two sentences that actually answer the question. This wastes tokens, raises latency and cost, and — per the well-documented "lost in the middle" phenomenon — can actually make the LLM worse at using the relevant portion of a long retrieved passage, because attention degrades for information buried in the middle of a long context block.

When chunks are too small, the failure mode flips. A chunk that is a single clause or a fragment of a sentence may embed with high precision for a narrow keyword match, but it strips away the surrounding context the LLM needs to reason about what that fragment means. A chunk containing only the sentence "It must be renewed annually" is useless to an LLM if it doesn't also know that "it" refers to a specific insurance policy discussed two sentences earlier. Overly small chunking also multiplies the number of vectors you must store and search, increases the chance that a coherent idea gets split across a chunk boundary (so that neither half, in isolation, is a strong match for the query that's actually about the whole idea), and forces the retriever to pull back many small pieces to reconstruct one coherent thought — which reintroduces some of the same "lots of tokens, low information density" problem that oversized chunks caused, just via a different mechanism.

The rest of this chapter works through the major chunking strategies in increasing order of sophistication — fixed-size, recursive/overlapping, semantic, sentence-window, and document-structure-aware — and closes with a concrete framework for reasoning about chunk size as a cost/quality dial rather than a single default value to memorize.

## Fixed-Size Chunking

Fixed-size chunking is the simplest possible strategy: pick a chunk length (in characters or tokens), slide a window across the document at that length, optionally overlapping consecutive windows by some fixed amount, and stop. It makes no attempt to respect sentence boundaries, paragraph structure, or semantic coherence — it is a purely mechanical split.

```python
from dataclasses import dataclass
from typing import List
import tiktoken


@dataclass
class Chunk:
    text: str
    start_offset: int
    end_offset: int
    chunk_index: int


def fixed_size_chunk_by_chars(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """Character-based sliding-window chunking with overlap."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    index = 0
    text_length = len(text)
    stride = chunk_size - chunk_overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]
        chunks.append(Chunk(text=chunk_text, start_offset=start, end_offset=end, chunk_index=index))
        index += 1
        if end == text_length:
            break
        start += stride

    return chunks


def fixed_size_chunk_by_tokens(
    text: str,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    encoding_name: str = "cl100k_base",
) -> List[Chunk]:
    """Token-based sliding-window chunking, aligned to how the embedding
    model actually counts length rather than approximating with characters."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    stride = chunk_size - chunk_overlap

    chunks = []
    index = 0
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        token_slice = tokens[start:end]
        chunk_text = encoding.decode(token_slice)
        chunks.append(Chunk(text=chunk_text, start_offset=start, end_offset=end, chunk_index=index))
        index += 1
        if end == len(tokens):
            break
        start += stride

    return chunks
```

Two implementation details matter more than they look. First, token-based chunking is almost always preferable to character-based chunking in production, because the thing you're actually bounded by — the embedding model's input limit and, downstream, the LLM's context window — is measured in tokens, not characters, and the character-to-token ratio varies meaningfully across languages, punctuation-heavy text, and code. Chunking by a fixed character count can silently overshoot the embedding model's real token limit on token-dense text (dense code, non-English text, heavy markdown) even though it looked safe by character count. Second, the `stride = chunk_size - chunk_overlap` computation is the part people get wrong when hand-rolling this: overlap is not an extra chunk tacked onto the end, it's a reduction in how far the window advances, so that the tail of chunk *N* reappears as the head of chunk *N+1*.

The advantages of fixed-size chunking are exactly what you'd expect from doing the simplest possible thing: it's fast (no NLP tooling, no embedding calls at chunking time), deterministic, trivially parallelizable across documents, and requires no understanding of the document's structure or language. Its weaknesses are equally direct: it has zero awareness of sentence or idea boundaries, so it will frequently cut a sentence, a table row, or a code statement in half, and semantically unrelated content that happens to be textually adjacent gets forced into the same chunk purely because of where the character count landed.

Fixed-size chunking is genuinely the right choice, not just the lazy choice, in a specific set of situations: large corpora of relatively homogeneous, low-structure prose (transcripts, chat logs, OCR output with unreliable formatting) where the cost of building or maintaining a more structure-aware pipeline exceeds the retrieval-quality benefit; early-stage prototypes where you need a RAG pipeline working end-to-end before optimizing any one stage; and extremely high-volume ingestion pipelines where the computational overhead of NLP-based sentence segmentation or embedding-based boundary detection across millions of documents becomes a real cost and latency concern. The rule of thumb: reach for fixed-size chunking when you need something working today and your content doesn't have strong internal structure to lose in the first place.

## Recursive / Overlapping Chunking

Recursive chunking (popularized by LangChain's `RecursiveCharacterTextSplitter`) fixes the most obvious flaw of fixed-size chunking — indifference to structure — while staying almost as simple to implement. The idea is a separator hierarchy: try to split on the "biggest" structural boundary first (typically double newlines, i.e., paragraph breaks), and only fall back to a smaller-granularity separator (single newline, then sentence-ending punctuation, then whitespace, then raw characters) for any piece that is still larger than the target chunk size after splitting on the coarser separator. This means the splitter naturally respects paragraph and sentence boundaries wherever the document's paragraphs happen to already be close to your target chunk size, and only resorts to a mid-sentence cut in the worst case, where a "paragraph" is itself a huge wall of text with no smaller natural boundary.

```python
from typing import List, Optional


class RecursiveCharacterSplitter:
    """A from-scratch implementation of separator-hierarchy chunking,
    equivalent in spirit to LangChain's RecursiveCharacterTextSplitter."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        separators: Optional[List[str]] = None,
        length_fn=len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_fn = length_fn

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []

        # Pick the first separator that actually appears in the text;
        # fall back to the final (empty-string / char-level) separator.
        separator = separators[-1]
        remaining_separators = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                remaining_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        final_chunks: List[str] = []

        for piece in splits:
            if self.length_fn(piece) < self.chunk_size:
                good_splits.append(piece)
            else:
                # Flush what we've accumulated, then recurse into the
                # oversized piece with the next, finer-grained separator.
                if good_splits:
                    final_chunks.extend(self._merge(good_splits, separator))
                    good_splits = []
                if remaining_separators:
                    final_chunks.extend(self._split(piece, remaining_separators))
                else:
                    final_chunks.append(piece)

        if good_splits:
            final_chunks.extend(self._merge(good_splits, separator))

        return final_chunks

    def _merge(self, splits: List[str], separator: str) -> List[str]:
        """Greedily pack small splits back together up to chunk_size,
        carrying `chunk_overlap` worth of trailing text into the next chunk."""
        chunks = []
        current: List[str] = []
        current_len = 0

        for split in splits:
            split_len = self.length_fn(split)
            added_len = split_len + (len(separator) if current else 0)

            if current_len + added_len > self.chunk_size and current:
                chunks.append(separator.join(current))
                # Build overlap: keep trailing splits whose combined length
                # is <= chunk_overlap, then continue accumulating.
                overlap_splits: List[str] = []
                overlap_len = 0
                for s in reversed(current):
                    s_len = self.length_fn(s)
                    if overlap_len + s_len > self.chunk_overlap:
                        break
                    overlap_splits.insert(0, s)
                    overlap_len += s_len
                current = overlap_splits
                current_len = overlap_len

            current.append(split)
            current_len += added_len

        if current:
            chunks.append(separator.join(current))

        return chunks
```

The overlap serves a specific purpose that is easy to state but worth being precise about: it insures against the case where a self-contained idea straddles a chunk boundary. Without overlap, if a key sentence spans the exact cut point, neither the tail of chunk *N* nor the head of chunk *N+1* contains the whole idea, and a query matching that idea might not score highly against either fragment. With overlap, the last portion of chunk *N* reappears intact at the start of chunk *N+1*, so at least one of the two chunks contains the complete thought even in the worst-case boundary placement.

Choosing the overlap percentage is a genuine design decision, not a copy-pasted default, and the right way to reason about it is in terms of what overlap is actually insuring against: the fixed "risk zone" near a boundary where an idea might get split. That risk zone is roughly constant in absolute size — a sentence or two, maybe 50-150 characters/tokens — regardless of how big your chunks are, because it's a property of natural language sentence length, not of your chosen chunk size. That's why the *relative* overlap percentage should shrink as chunk size grows:

- **Small chunks (roughly 100-300 tokens)**: use 10-20% overlap. At this size, the fixed boundary risk zone is a large fraction of the whole chunk — a 30-token overlap on a 150-token chunk is 20% of the content, but that 30 tokens might be the entire sentence that matters. Skimping on overlap here disproportionately increases the chance of a badly split idea.
- **Medium chunks (roughly 300-800 tokens)**: use 15-25% overlap. This is the most commonly used range in production RAG (chunk sizes around 500-512 tokens with 50-100 token overlap are a very common default), because it balances boundary safety against the redundant-storage cost of overlap without either extreme.
- **Large chunks (roughly 800-2000+ tokens)**: use 10-15% overlap. Here the same fixed-size risk zone is a small fraction of a much bigger chunk, so a smaller overlap percentage still fully covers it. Using a large overlap percentage at this scale would mostly duplicate large amounts of already-well-contained text for little additional boundary protection, while directly inflating embedding and storage cost.

In other words: overlap size in absolute tokens should stay roughly constant (large enough to contain a sentence or two), while overlap size as a *percentage* of chunk size should shrink as chunk size grows, because the denominator is growing while the thing you're protecting against stays the same size.

## Semantic Chunking

Fixed-size and recursive chunking both decide boundaries using surface-level signals — character counts and punctuation — with no reference to what the text actually *means*. Semantic chunking replaces that with a content-aware decision rule: split the document into small units (usually sentences), embed each one, and cut a new chunk boundary wherever the semantic similarity between consecutive units drops sharply, on the theory that a big drop in similarity marks a topic shift.

```python
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import re


def split_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter; swap for spaCy/nltk for production
    robustness against abbreviations, decimals, etc."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        breakpoint_percentile: float = 95.0,
        buffer_size: int = 1,
        min_chunk_sentences: int = 1,
    ):
        """
        breakpoint_percentile: how extreme a similarity drop must be
            (relative to the distribution of drops in this document) to
            count as a chunk boundary. Higher = fewer, larger chunks.
        buffer_size: number of sentences on each side to combine into a
            'window' before embedding, which smooths out noise from very
            short sentences.
        """
        self.model = SentenceTransformer(model_name)
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.min_chunk_sentences = min_chunk_sentences

    def _combine_with_buffer(self, sentences: List[str]) -> List[str]:
        combined = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            combined.append(" ".join(sentences[start:end]))
        return combined

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return 1.0 - cos_sim

    def chunk(self, text: str) -> List[str]:
        sentences = split_into_sentences(text)
        if len(sentences) <= self.min_chunk_sentences:
            return [text]

        # Embed a windowed version of each sentence rather than the raw
        # sentence alone -- single sentences are often too short and noisy
        # to embed reliably on their own.
        windows = self._combine_with_buffer(sentences)
        embeddings = self.model.encode(windows, normalize_embeddings=True)

        # Distance between each pair of *adjacent* sentence windows.
        distances = [
            self._cosine_distance(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        if not distances:
            return [text]

        # A boundary is a distance that is an outlier relative to this
        # document's own distribution of adjacent-sentence distances --
        # this makes the threshold adaptive per document rather than a
        # single global magic number.
        threshold = np.percentile(distances, self.breakpoint_percentile)
        breakpoints = {i for i, d in enumerate(distances) if d > threshold}

        chunks = []
        current: List[str] = []
        for i, sentence in enumerate(sentences):
            current.append(sentence)
            if i in breakpoints and len(current) >= self.min_chunk_sentences:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))

        return chunks
```

The mechanism is worth walking through concretely: rather than embedding raw individual sentences (which are often too short to embed reliably — a five-word sentence doesn't give an encoder much to work with), the implementation above embeds a small sliding window of sentences around each position, then measures the cosine distance between adjacent windows. Wherever that distance spikes above a percentile-based threshold computed from the document's own distribution of distances, that's flagged as a topic boundary and a new chunk starts. Using a percentile of the document's own distance distribution, rather than a single global cutoff like "0.3," makes the method adaptive to different writing styles and topic densities — a rambling blog post and a tightly structured technical spec have very different baseline sentence-to-sentence similarity, and a fixed global threshold would either over-split one or under-split the other.

The benefit is real: chunks produced this way tend to correspond much more closely to actual topical units in the source document than any fixed-length window could, which directly improves embedding quality (no more diluted, multi-topic vectors) and improves the coherence of what gets handed to the LLM. But the cost is equally real and worth stating plainly in an interview setting, because it's the kind of trade-off that shows you understand production trade-offs rather than just algorithms: semantic chunking requires running an embedding model over every sentence (or sentence window) at *ingestion* time, purely to make the chunking decision, which is a meaningfully more expensive and slower pipeline stage than a string split. It also produces variable-size chunks by construction — one chunk might be two sentences, the next might be twenty — which makes it much harder to reason about and budget for downstream costs (how many chunks fit in a context window, how much you'll pay per retrieval call) compared to the predictable, uniform sizing of fixed or recursive chunking. In practice, semantic chunking is usually reserved for corpora where retrieval precision is the dominant concern and ingestion is a one-time or infrequent batch cost — knowledge bases of long-form, topically dense articles rather than high-velocity streams of short documents.

## Sentence-Window Retrieval

Sentence-window retrieval addresses the tension between chunk size and retrieval precision by refusing to accept that "the unit you search over" and "the unit you hand to the LLM" have to be the same thing. Instead, it indexes very small units — typically individual sentences — for the similarity search itself, which maximizes retrieval precision, because a single-sentence embedding is not diluted by any surrounding unrelated content. Then, at retrieval time, instead of returning the matched sentence in isolation, the system expands it to include a window of surrounding context (a fixed number of neighboring sentences, or the entire parent paragraph/section) before that expanded text is handed to the LLM.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class IndexedSentence:
    sentence_id: str
    text: str
    embedding: np.ndarray
    doc_id: str
    sentence_index: int          # position of this sentence within its document
    document_sentences: List[str] = field(repr=False)  # full sentence list, for window expansion


class SentenceWindowIndex:
    def __init__(self, embed_fn, window_size: int = 2):
        """
        embed_fn: callable text -> np.ndarray, e.g. a SentenceTransformer.encode
        window_size: number of sentences on EACH side to include when expanding
        """
        self.embed_fn = embed_fn
        self.window_size = window_size
        self.entries: List[IndexedSentence] = []

    def index_document(self, doc_id: str, text: str, sentence_splitter):
        sentences = sentence_splitter(text)
        embeddings = self.embed_fn(sentences)  # batch-encode for efficiency
        for i, (sentence, emb) in enumerate(zip(sentences, embeddings)):
            self.entries.append(
                IndexedSentence(
                    sentence_id=f"{doc_id}::{i}",
                    text=sentence,
                    embedding=np.asarray(emb),
                    doc_id=doc_id,
                    sentence_index=i,
                    document_sentences=sentences,
                )
            )

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_emb = np.asarray(self.embed_fn([query])[0])
        scored = [
            (self._cosine_sim(query_emb, entry.embedding), entry)
            for entry in self.entries
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        results = []
        for score, entry in scored[:top_k]:
            results.append({
                "score": score,
                "matched_sentence": entry.text,
                "expanded_context": self._expand_window(entry),
                "doc_id": entry.doc_id,
                "sentence_index": entry.sentence_index,
            })
        return results

    def _expand_window(self, entry: IndexedSentence) -> str:
        sentences = entry.document_sentences
        start = max(0, entry.sentence_index - self.window_size)
        end = min(len(sentences), entry.sentence_index + self.window_size + 1)
        return " ".join(sentences[start:end])
```

The reason this often improves precision *and* recall simultaneously, rather than trading one for the other, is worth spelling out. Precision improves because the vector actually being matched against the query is a clean, single-topic sentence embedding rather than a blended, multi-sentence chunk embedding — there's no dilution to blur the similarity score, so a genuinely relevant sentence surfaces with a higher, more distinguishable score, and irrelevant sentences don't get an artificial boost from happening to share a chunk with something relevant. Recall improves because the search space is now much finer-grained: a document that mentions the exact fact you need only once, in one sentence buried inside an otherwise off-topic section, would have been diluted into obscurity as part of a large chunk embedding, but as its own sentence-level vector it stands a real chance of surfacing. The window expansion at retrieval time is what prevents this fine-grained indexing from producing the classic "fragment without context" failure mode described earlier — the LLM never actually sees an isolated sentence, it sees that sentence plus enough surrounding text to resolve pronouns, understand qualifiers, and follow the argument. This decoupling of retrieval unit from context unit is one of the more elegant ideas in modern RAG design, and it's implemented natively in frameworks like LlamaIndex as the `SentenceWindowNodeParser` plus `MetadataReplacementPostProcessor` pairing, which is exactly the pattern this code reproduces from scratch.

The cost of this approach is indexing overhead: you now have one vector per sentence rather than one per multi-sentence chunk, which multiplies the number of embeddings you compute and store, and multiplies the number of vector comparisons a naive search has to make (though this is a solved problem for any real ANN index). It's a reasonable trade for corpora where individual facts are what users query for — FAQs, reference documentation, structured knowledge bases — and less obviously worth it for corpora dominated by narrative or argumentative text where the "unit of meaning" doesn't cleanly compress into a single sentence.

## Document-Structure-Aware Chunking

None of the strategies above know anything about the *document's own structure* — that a Markdown file has headers marking section boundaries, that a table is a single atomic unit whose rows must never be separated from its header row, or that a code block is a syntactic unit that becomes meaningless (and often literally un-parseable) if split mid-function. Structure-aware chunking treats these as first-class boundaries: it always cuts at section breaks before considering any length-based split, and it treats tables and code fences as atomic — never split internally, even if that means a single chunk exceeds the "target" size — because a broken table or truncated function body is actively worse than a slightly oversized chunk.

```python
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StructuredChunk:
    text: str
    header_path: List[str] = field(default_factory=list)
    chunk_type: str = "text"   # "text", "code", "table"

    @property
    def section_label(self) -> str:
        return " > ".join(self.header_path) if self.header_path else "(root)"


HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
CODE_FENCE_RE = re.compile(r"^```")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def parse_markdown_blocks(text: str) -> List[Dict]:
    """Split raw markdown into an ordered list of typed blocks: headers,
    code fences (atomic), tables (atomic), and plain paragraphs."""
    lines = text.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]

        header_match = HEADER_RE.match(line)
        if header_match:
            blocks.append({
                "type": "header",
                "level": len(header_match.group(1)),
                "text": header_match.group(2).strip(),
            })
            i += 1
            continue

        if CODE_FENCE_RE.match(line):
            code_lines = [line]
            i += 1
            while i < len(lines) and not CODE_FENCE_RE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])  # closing fence
                i += 1
            blocks.append({"type": "code", "text": "\n".join(code_lines)})
            continue

        if TABLE_ROW_RE.match(line):
            table_lines = [line]
            i += 1
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append({"type": "table", "text": "\n".join(table_lines)})
            continue

        if line.strip() == "":
            i += 1
            continue

        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() != "" and not HEADER_RE.match(lines[i]) \
                and not CODE_FENCE_RE.match(lines[i]) and not TABLE_ROW_RE.match(lines[i]):
            para_lines.append(lines[i])
            i += 1
        blocks.append({"type": "paragraph", "text": "\n".join(para_lines)})

    return blocks


def structure_aware_chunk(
    text: str,
    max_chunk_size: int = 800,
    length_fn=len,
) -> List[StructuredChunk]:
    """Chunk markdown by section, keeping code and tables atomic and
    stamping every chunk with the header path it belongs to."""
    blocks = parse_markdown_blocks(text)

    chunks: List[StructuredChunk] = []
    header_stack: List[str] = []          # e.g. ["Refund Policy", "3.2 Exceptions"]
    current_text_parts: List[str] = []
    current_len = 0

    def flush():
        nonlocal current_text_parts, current_len
        if current_text_parts:
            chunks.append(StructuredChunk(
                text="\n\n".join(current_text_parts),
                header_path=list(header_stack),
                chunk_type="text",
            ))
            current_text_parts = []
            current_len = 0

    for block in blocks:
        if block["type"] == "header":
            # A new header always starts a fresh chunk -- section
            # boundaries take priority over length-based packing.
            flush()
            level = block["level"]
            # Truncate the header stack back to this level, then push.
            header_stack = header_stack[: level - 1]
            header_stack.append(block["text"])
            continue

        if block["type"] in ("code", "table"):
            # Atomic units: never split internally. If it doesn't fit in
            # the current chunk, flush first, then emit it as its own
            # chunk even if that exceeds max_chunk_size.
            flush()
            chunks.append(StructuredChunk(
                text=block["text"],
                header_path=list(header_stack),
                chunk_type=block["type"],
            ))
            continue

        # Plain paragraph: pack greedily, respecting max_chunk_size.
        block_len = length_fn(block["text"])
        if current_len + block_len > max_chunk_size and current_text_parts:
            flush()
        current_text_parts.append(block["text"])
        current_len += block_len

    flush()
    return chunks


def render_chunk_with_context(chunk: StructuredChunk) -> str:
    """Prefix the chunk with its section path so the LLM sees where the
    content came from -- this is what gets embedded and/or injected."""
    prefix = f"[Section: {chunk.section_label}]\n" if chunk.header_path else ""
    return prefix + chunk.text
```

Two design choices in this implementation are the ones interviewers most often probe on. First, headers always force a chunk boundary, even if the current accumulated text is well under `max_chunk_size` — this is a deliberate choice to never let a chunk silently straddle a section boundary, because doing so would mean a single chunk represents two different topics under two different headers, reintroducing the semantic-dilution problem discussed earlier, just triggered by document structure instead of raw length. Second, code blocks and tables are treated as atomic regardless of size: the function explicitly allows them to exceed `max_chunk_size` rather than ever splitting them, because a table missing its header row is misleading (the LLM can't tell which column is which) and a function body cut in half is not just lower quality, it's often actively wrong (a partial function definition can look syntactically plausible while being semantically nonsensical, which is worse than obviously-broken output because it doesn't visibly signal that something is missing).

The `header_path` carried on every chunk is the other important piece. Storing "Section: Billing > 3.2 Refund Policy" as chunk metadata, and prepending it to the chunk text before embedding (or before injecting the chunk into the LLM prompt, or both), does two things: it gives the LLM disambiguating context it wouldn't otherwise have (a chunk that just says "the fee is waived in this case" is far more useful when the model also knows it's from the "Refund Policy" section rather than, say, "Shipping Policy"), and it gives the embedding model extra signal that can help distinguish superficially similar chunks that live in different parts of the document (two chunks that both mention "processing time" might belong to entirely different sections, and the header path is what disambiguates them at the vector level too, not just at the prompt level).

## Chunk Size, Cost, and Quality: How to Actually Decide

Every chunking decision is simultaneously a retrieval-quality decision and a cost decision, and the two pull in opposite directions, which is exactly why there is no single universally correct chunk size — the "right" answer depends on where your system sits on that trade-off curve.

Smaller chunks mean more chunks per document, which means more embedding API calls (or more compute if self-hosting an embedding model) at ingestion time, and more vectors to store and index — vector database storage and index-build cost scale roughly linearly with vector count, so halving your chunk size roughly doubles both. But smaller chunks typically improve retrieval precision for the reasons already covered (less dilution), and — this is the part that's easy to miss — they often *reduce* the per-query cost of generation, because each retrieved chunk carries less irrelevant filler alongside the relevant sentence, so the same "true" information fits into fewer total tokens injected into the LLM prompt. You're trading a larger, one-time indexing cost for a smaller, recurring per-query cost.

Larger chunks invert this: fewer chunks means cheaper indexing (fewer embedding calls, fewer vectors to store and search), but each retrieved chunk drags along more irrelevant surrounding text, which means every single query pays a larger token cost in the LLM call, retrieval precision degrades due to embedding dilution, and you're more exposed to "lost in the middle" effects if the model needs to find one sentence buried inside a much longer retrieved block. Since queries typically vastly outnumber ingestion events over a corpus's lifetime, the "cheaper indexing, more expensive per-query" trade of large chunks is usually the wrong economic trade for any system with meaningful query volume — the indexing cost is paid once, the per-query token tax is paid forever.

Content type should drive the starting point far more than any generic default, because different content types have very different natural "unit of meaning" sizes:

| Content type | Typical chunk size | Reasoning |
|---|---|---|
| Source code | 50-200 lines (or one function/class, whichever is smaller) | The natural semantic unit is a function or class; splitting mid-function destroys correctness, and a whole file is usually too large and multi-topic to embed well. Structure-aware, AST-based chunking (split on function/class boundaries) outperforms any fixed size here. |
| Technical documentation / API docs | 300-600 tokens | Docs are already organized into short, single-purpose sections (one concept, one endpoint, one parameter); chunk size should track that natural section size rather than impose an arbitrary one, which is why structure-aware chunking by header is especially effective for this content type. |
| General articles / blog posts / wiki pages | 500-800 tokens | Prose paragraphs typically develop one idea across several sentences; this range usually captures a full idea (or two related ones) without pulling in the next unrelated section, and it's the range where recursive chunking on paragraph boundaries tends to land naturally. |
| Books / long-form narrative | 800-1500 tokens | Narrative and argumentative text often needs more surrounding context for a passage to make sense (character references, ongoing arguments spanning several paragraphs), and the query patterns against this content type ("what happens when," "what does the author argue about X") tend to need broader context than a single paragraph provides. |
| Legal / contract text | 200-500 tokens, section/clause-aligned | Legal text is dense, and small drafting differences between clauses matter enormously, so diluting multiple clauses into one embedding is especially costly for retrieval precision. Chunking must align to the document's own clause/section numbering (never split a numbered clause across chunks) rather than any generic length target, because a retrieved fragment of a clause without its qualifying conditions can be actively misleading. |

Treat every number in that table as a starting hypothesis to validate against your own corpus and query patterns, not a constant to hardcode. The only reliable way to actually tune chunk size and overlap for a production system is to build a small labeled evaluation set — realistic queries paired with the passages that should be retrieved to answer them — and measure retrieval metrics (recall@k, mean reciprocal rank) across a few candidate chunking configurations before committing to one. Chunking strategy is not a one-time decision made at project kickoff; it's a parameter that should be revisited whenever the corpus's content type shifts, whenever the embedding model changes (different models have different optimal input lengths and different sensitivity to dilution), or whenever evaluation reveals a systematic retrieval failure pattern that traces back to chunk boundaries.
