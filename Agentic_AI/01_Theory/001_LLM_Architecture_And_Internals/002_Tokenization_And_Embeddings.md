# Tokenization and Embeddings

## Why Tokenization Has To Exist At All

A transformer's mathematical machinery — matrix multiplications, softmax, dot products — only operates on vectors of numbers. Raw text is a sequence of characters, so before any of the architecture described elsewhere in this folder can run, text has to be converted into a sequence of discrete integer IDs, each of which indexes into a lookup table of learned vectors. The question tokenization answers is: what should the "units" of that sequence be? This sounds like a minor preprocessing detail, but the choice has first-order effects on sequence length, training efficiency, vocabulary size, model quality on rare words and non-English text, and even the model's ability to do arithmetic — which is why it deserves the same depth of understanding as the architecture itself.

There are three natural extremes to consider, and understanding why all three fail in different ways motivates why every modern LLM uses something in between.

**Character-level tokenization** treats each character as a token. Its vocabulary is tiny (a few hundred symbols at most, covering an alphabet, punctuation, and digits), so there's never an out-of-vocabulary problem, and it handles novel words, typos, and rare strings gracefully because everything can be spelled out. Its fatal flaw is sequence length: a paragraph that might be a few hundred word-level tokens becomes many times longer in characters, and since a transformer's self-attention cost scales quadratically with sequence length, and every additional token also costs a proportional share of the fixed context window, character-level tokenization is extremely wasteful of both compute and the model's limited context budget. It also forces the model to spend a great deal of its representational effort re-deriving word- and morpheme-level structure that a smarter tokenizer could have handed to it for free.

**Word-level tokenization** treats each whitespace/punctuation-delimited word as a token. This produces short, semantically meaningful sequences, but it has the opposite problem: the vocabulary needed to cover a language's full set of words (including inflections, proper nouns, technical terms, and typos) is enormous and, worse, open-ended — you will always encounter words at inference time that never appeared in training. This forces an explicit out-of-vocabulary (`<UNK>`) mechanism, which is destructive: any unseen word collapses to the same generic symbol, throwing away all of its information. Word-level tokenization also handles morphology poorly — "run," "runs," "running," and "runner" are treated as four entirely unrelated symbols with no shared substructure, so the model must learn their relationship purely from co-occurrence statistics rather than getting it for free from a shared subword.

**Subword tokenization** is the compromise every production LLM actually uses. The vocabulary is built from frequently-occurring character sequences of variable length — common whole words remain single tokens, but rarer or morphologically complex words get split into a small number of meaningful subword pieces (e.g., "tokenization" might become "token" + "ization"), and in the worst case, an entirely novel string can still always be represented by falling back to individual characters or bytes, so there is no true out-of-vocabulary problem. This gives a tunable middle ground: vocabulary size is bounded and fixed in advance (unlike open-ended word-level vocabularies), sequences stay much shorter than character-level ones for common text, and the model can still represent arbitrary rare or unseen strings by decomposing them into smaller known pieces. Every scheme discussed below — BPE, WordPiece, and Unigram/SentencePiece — is a different algorithm for deciding which subword units belong in that fixed vocabulary.

## Byte-Pair Encoding (BPE)

### The core algorithm

BPE was originally a data-compression algorithm, repurposed for tokenization by Sennrich et al. for neural machine translation, and it is the basis for GPT-2, GPT-3, GPT-4 (via OpenAI's `tiktoken` byte-level variant), and, with different pre-tokenization details, Llama's tokenizer. BPE builds its vocabulary bottom-up: start with a base vocabulary of individual characters (or bytes — more on that distinction below), then repeatedly find the single most frequent adjacent pair of symbols across the entire training corpus and merge that pair into one new symbol, adding it to the vocabulary. This process repeats for a fixed number of merges, chosen in advance to hit a target vocabulary size, and the sequence of merges learned during training is exactly what gets replayed, in the same order, to tokenize new text at inference time.

### A worked example

Suppose the training corpus, after splitting into words and appending an end-of-word marker `_`, gives us the word "low" appearing 5 times, "lower" 2 times, "newest" 6 times, and "widest" 3 times. We start by representing every word as a sequence of individual characters:

```
l o w _        (freq 5)
l o w e r _    (freq 2)
n e w e s t _  (freq 6)
w i d e s t _  (freq 3)
```

The base vocabulary is the set of unique characters seen: `{l, o, w, _, e, r, n, s, t, i, d}`. Now we count all adjacent symbol pairs across the corpus, weighted by word frequency. The pair `(e, s)` appears in "newest" (freq 6) and "widest" (freq 3), for a total count of 9 — the most frequent pair. We merge `e` and `s` into a new symbol `es`, and add `es` to the vocabulary:

```
l o w _            (freq 5)
l o w e r _        (freq 2)
n e w es t _       (freq 6)
w i d es t _       (freq 3)
```

Next, the most frequent remaining pair is `(es, t)`, appearing 9 times total (6 + 3), so we merge it into `est`:

```
l o w _
l o w e r _
n e w est _
w i d est _
```

We keep repeating this: the next most frequent pair might be `(l, o)` from "low"/"lower" (freq 5 + 2 = 7), merging into `lo`, then `(lo, w)` into `low`, and so on, until we've performed however many merges the target vocabulary size calls for (real tokenizers run tens of thousands of merges). Each merge is recorded, in order, as a rule; tokenizing a brand-new word at inference time means starting from its raw characters and greedily applying the learned merge rules in the same order they were learned, until no more merges apply.

```python
from collections import Counter, defaultdict

def get_pair_counts(corpus):
    """corpus: dict mapping tuple-of-symbols -> frequency"""
    pairs = Counter()
    for symbols, freq in corpus.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_pair(pair, corpus):
    a, b = pair
    merged = a + b
    new_corpus = {}
    for symbols, freq in corpus.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_corpus[tuple(new_symbols)] = new_corpus.get(tuple(new_symbols), 0) + freq
    return new_corpus

# words -> frequency, pre-split into characters with an end-of-word marker
corpus = {
    tuple("low_"):    5,
    tuple("lower_"):  2,
    tuple("newest_"): 6,
    tuple("widest_"): 3,
}

num_merges = 10
merges = []
for _ in range(num_merges):
    pair_counts = get_pair_counts(corpus)
    if not pair_counts:
        break
    best_pair = max(pair_counts, key=pair_counts.get)
    merges.append(best_pair)
    corpus = merge_pair(best_pair, corpus)

print(merges)
print(corpus)
```

This toy loop is the entire conceptual core of BPE training. Production tokenizer trainers (Hugging Face `tokenizers`, `sentencepiece`, `tiktoken`'s training utilities) do the same thing with far more efficient data structures (a priority queue over pair counts with incremental updates rather than a full rescan per merge), operating over billions of tokens of raw text.

## WordPiece: Likelihood-Based Merging

WordPiece, introduced by Google for their neural machine translation system and used by BERT, is structurally very similar to BPE — it also starts from characters and greedily merges pairs — but it changes the *criterion* used to pick which pair to merge. Plain BPE always merges whichever adjacent pair is most frequent in raw counts. WordPiece instead merges the pair that maximizes the likelihood of the training corpus under a unigram language model built from the current vocabulary, which in practice reduces to picking the pair `(a, b)` that maximizes:

```
score(a, b) = count(a, b) / (count(a) * count(b))
```

This is a pointwise-mutual-information-like criterion: it doesn't just ask "how often do `a` and `b` co-occur," it asks "how often do they co-occur *relative to how often we'd expect by chance*, given how common `a` and `b` individually are." This means WordPiece can choose to merge a pair that is not the single most frequent pair in absolute terms, if that pair's co-occurrence is unusually strong relative to the individual frequencies of its parts. Intuitively, this discourages merging two extremely common, largely independent symbols just because their sheer individual frequency makes their raw co-occurrence count high, and instead favors merges that indicate a genuinely strong statistical association — the same instinct behind using PMI instead of raw co-occurrence counts in classical NLP collocation extraction.

## SentencePiece and the Unigram Language Model Tokenizer

### SentencePiece as an implementation wrapper

SentencePiece, from Google, is worth distinguishing from BPE/WordPiece/Unigram because it operates at a different layer: it is a tokenizer *training and inference framework* that can implement either BPE or the Unigram algorithm underneath, and its defining practical feature is that it treats the input as a raw, un-pre-tokenized stream of Unicode characters (or bytes), including whitespace, rather than assuming the text has already been split into words by, say, a whitespace-based pre-tokenizer. It typically represents a space as an explicit meta-symbol (commonly rendered as `▁`, U+2581) so that the original text, whitespace included, can always be losslessly reconstructed from the token sequence. This language-agnostic, pre-tokenization-free property matters a great deal for languages like Japanese, Chinese, or Thai, which don't use whitespace to separate words at all, and it's why Llama's tokenizer is described as "a SentencePiece-based BPE tokenizer" — SentencePiece is the framework and byte-level handling machinery, and BPE is the specific merge algorithm it's configured to run.

### The Unigram Language Model tokenizer

The Unigram algorithm (also implementable inside SentencePiece) is fundamentally different in direction from both BPE and WordPiece, and this is one of the more conceptually interesting points to be able to explain clearly. BPE and WordPiece are both *constructive*: they start from the smallest possible vocabulary (individual characters) and grow it one merge at a time until reaching the target size. Unigram is *destructive*: it starts from a very large candidate vocabulary — for instance, every substring that appears often enough in the corpus, or the output of a preliminary BPE-like pass — and then iteratively *prunes* it down to the target size.

Concretely, Unigram assumes a probabilistic model in which the corpus is generated by independently sampling subword tokens according to a unigram (single-token, no-context) probability distribution over the current vocabulary. Given that model, any input string generally has multiple possible segmentations into vocabulary tokens, and the algorithm uses the Viterbi algorithm (or an EM-style expectation-maximization procedure) to find and score the most probable segmentation of the training corpus under the current vocabulary and its estimated token probabilities. It then estimates, for each token currently in the vocabulary, how much the overall corpus likelihood would drop if that token were removed (forcing all occurrences to be re-segmented using the remaining vocabulary). Tokens whose removal barely hurts the likelihood are pruned first, and this shrink-and-re-estimate cycle repeats until the vocabulary reaches its target size.

The practical benefit of this top-down approach is that the final vocabulary and the model's probability estimates naturally support tokenizing any given input in *multiple valid ways*, each with an associated probability, rather than the single deterministic greedy path that BPE's merge-order or WordPiece's greedy longest-match produces. This makes Unigram naturally suited to a regularization technique called subword regularization, where during training, a word is randomly tokenized using a different plausible segmentation each time (sampled proportional to the model's segmentation probabilities) rather than always the single best one, which acts like a data-augmentation scheme over tokenization itself and has been shown to improve robustness, particularly for lower-resource languages and translation tasks. This is precisely why SentencePiece's Unigram mode is popular for multilingual models beyond just the whitespace-agnostic property discussed above.

## How Production Tokenizers Actually Differ

It's worth being concrete about what specific well-known models actually use, since interviewers often probe for these exact facts.

OpenAI's models (GPT-2 through GPT-4 and beyond) use a **byte-level BPE** tokenizer, implemented efficiently in the open-source `tiktoken` library. "Byte-level" means the base vocabulary, before any merges, is the 256 possible byte values rather than Unicode characters. This is a clever trick: because every possible sequence of bytes (and therefore every possible Unicode string, once UTF-8 encoded) can be represented starting from just 256 base symbols, byte-level BPE has zero out-of-vocabulary strings by mathematical construction — there's no need for an explicit `<UNK>` token or a separate fallback mechanism, because the base vocabulary already spans every possible byte value, and BPE merges on top of that just make common byte sequences (which usually align with common characters and words in practice) more compact. GPT-2's tokenizer also has a specific pre-tokenization detail worth knowing: it uses a regex-based pre-splitting step (to keep, for instance, contractions, punctuation, and numbers from merging across category boundaries in certain ways) before running byte-level BPE merges on each resulting piece, and different GPT tokenizer versions (`r50k_base`, `p50k_base`, `cl100k_base`, `o200k_base`) differ mainly in vocabulary size and details of this pre-tokenization regex.

Llama's tokenizer (and Mistral's, which is nearly identical) is a **SentencePiece-trained BPE** tokenizer operating over raw bytes with an added **byte-fallback** mechanism: if a character or subword sequence is encountered that isn't in the trained vocabulary, instead of mapping it to a generic `<UNK>` symbol, the tokenizer falls back to representing it as a sequence of raw UTF-8 byte tokens (Llama's vocabulary reserves 256 explicit byte tokens for exactly this purpose). This gives Llama the same "no true out-of-vocabulary input" guarantee as GPT's byte-level BPE, but arrived at through a different mechanism — GPT bakes byte-level coverage into the base vocabulary from the start, while Llama trains its BPE merges primarily over more conventional Unicode-character/subword statistics and keeps an explicit byte-level escape hatch only for the residual cases the trained vocabulary doesn't cover well (this matters especially for rarer scripts and multilingual text that were underrepresented in the training-corpus statistics used to pick merges).

## Vocabulary Size Trade-offs

Vocabulary size is a real hyperparameter with consequences that show up throughout the whole model, not just in preprocessing. GPT-2 used roughly 50,000 tokens; GPT-4's `cl100k_base` uses roughly 100,000; Llama 1/2 used 32,000; Llama 3 jumped to 128,000; some multilingual models go higher still. The trade-off runs in both directions.

A **larger vocabulary** means more distinct strings can be represented as a single token, so the same piece of text tokenizes into a shorter sequence. Shorter sequences are valuable for two direct reasons: attention's compute cost is quadratic in sequence length, so fewer tokens per document means cheaper training and inference, and a fixed context window (measured in tokens) covers more actual text when each token carries more information. But a larger vocabulary also directly enlarges the input embedding matrix and the output (LM head) projection matrix, both of shape roughly `(vocab_size, d_model)`, which is a real, non-trivial chunk of total parameter count, especially for smaller models where the embedding matrices can be a significant fraction of total parameters. There is also a training-signal concern: with a larger vocabulary, individual rare tokens are seen far fewer times over the course of pretraining (the same total token budget gets spread across more distinct symbols), so rare-token embeddings tend to be relatively undertrained, receiving noisier and sparser gradient updates than common-token embeddings, which can show up as worse behavior specifically on rare words, unusual formatting, or rare-language text.

A **smaller vocabulary** produces the opposite profile: sequences get longer (more tokens per unit of text), raising compute and context-budget cost, but every token in the vocabulary is seen far more frequently during training, giving denser, better-trained embeddings even for what would have been "rare" tokens under a larger vocabulary, since they're now decomposed into smaller, more frequently-recurring subword pieces. Choosing vocabulary size is therefore a genuine trade-off between sequence-length efficiency and per-token training density, and the trend across model generations (32K to 100K+ to 128K+) reflects that as pretraining corpora and compute budgets have grown enormously, the "rare token undertraining" cost has become relatively less important than the sequence-length efficiency gain, especially as models are increasingly used with very long contexts where the quadratic attention cost of longer sequences dominates.

## Well-Known Tokenization Pathologies

Several specific, well-documented failure modes are worth knowing by name because they come up constantly in both interviews and real production debugging.

**Inconsistent number splitting** is probably the most cited cause of LLMs' historically weak arithmetic. Because BPE/WordPiece merges are learned from corpus frequency statistics, a number like "380" might end up as a single token in one context, while "381" splits into two tokens ("38" + "1"), and "1000" might split completely differently than "999" or "1001" — there is no guarantee of any consistent, place-value-aligned segmentation of digit sequences, since the merge rules only reflect which digit substrings happened to co-occur frequently in training text. This means the model isn't reliably seeing numbers as a consistent positional (units, tens, hundreds) representation the way a human or a hand-designed arithmetic system would — it's seeing an arbitrary, frequency-driven chunking that can differ between two numbers that are numerically very close to each other. This is a direct, well-understood contributor to LLMs' well-documented struggles with multi-digit arithmetic, and it's part of why some newer tokenizers (Llama 3's, for instance) deliberately special-case digit tokenization, e.g., always splitting numbers into individual digit tokens or fixed-size digit groups, specifically to give the model a consistent positional structure to learn arithmetic over.

**The "strawberry" letter-counting failure** is the most viral illustration of this class of problem, and it is worth being able to explain precisely rather than just cite. When a model is asked "how many letters 'r' are in the word strawberry," it is not perceiving the word the way a human reader does, as ten individual characters s-t-r-a-w-b-e-r-r-y. A BPE tokenizer typically encodes "strawberry" as a small number of opaque subword chunks (something like `st` + `raw` + `berry`, or `straw` + `berry`, depending on the exact tokenizer and its merge rules) — the model's entire perceptual access to the word is those one-to-three integer token IDs, each pointing to a learned embedding vector that represents the chunk holistically. There is no direct, explicit channel exposing the individual characters inside a token to the transformer's computation; whatever letter-level information exists has to have been indirectly absorbed into the embedding during training, purely as a side effect of that token's usage patterns, not because the architecture provides any character-counting mechanism. Asking the model to count a specific letter inside a token is therefore roughly analogous to asking a person to count how many times a particular pen-stroke curve appears inside a word rendered in an unfamiliar font — the requested information lives at a level of granularity below the one the perceiving system actually operates on. Models can sometimes still get this right, since large-scale training does teach some implicit spelling knowledge (character-level tasks and data do appear in training corpora), but performance is inconsistent and word-dependent precisely because the model is fighting its own input representation rather than being supported by it — which is exactly why the failure is a tokenization artifact and not evidence of some deeper reasoning deficiency.

**Leading-whitespace sensitivity** is a well-known GPT-tokenizer quirk: because GPT's byte-level BPE treats a leading space as part of the token itself (so `"the"` and `" the"` — with a leading space — are two entirely different token IDs, and likewise for whether a token is at the start of a line versus mid-sentence), the exact same word can tokenize completely differently depending on what precedes it in the raw text. This is why, historically, prompting guidance for GPT models cared about details like whether or not to include a trailing space after a prompt, and why token-level manipulations (like constrained decoding or logit biasing on a specific "word") have to be done carefully, keeping in mind that the "word" you want might correspond to multiple different token IDs depending on surrounding whitespace.

**Poor compression ratio for non-English/non-Latin-script languages** is a significant, well-measured fairness and cost issue. Because these tokenizers' merge vocabularies are learned from training corpora that are overwhelmingly English/Latin-script-dominant, the resulting merges compress common English words and phrases into very few tokens, while text in scripts like Chinese, Japanese, Korean, Thai, or Arabic — and even non-English Latin-script languages with different morphology, such as Finnish or Turkish — ends up requiring substantially more tokens to represent the same amount of semantic content, sometimes several times more per "word" of meaning. This has two compounding real-world consequences: since API pricing and context-window limits are measured in tokens, the same amount of meaningful content costs more and fits less of it in context for these languages, and since the model's effective training exposure per unit of real-world text is also lower for these languages, quality often lags for exactly the same underlying reason. This is one of the concrete, well-documented reasons multilingual model development includes deliberately expanding and rebalancing the tokenizer's training corpus (and vocabulary size) to better cover non-English scripts, rather than only scaling English-centric vocabularies.

**Glitch tokens** — the "SolidGoldMagikarp" phenomenon, named after a specific Reddit username token discovered in GPT-2/GPT-3's vocabulary — occur when a token ends up in the trained vocabulary (often because it appeared with reasonably high frequency in some scraped corpus, such as forum usernames or repeated boilerplate strings) but then appears extremely rarely or not at all in the actual downstream fine-tuning/RLHF data used to shape the model's behavior. The result is a token whose embedding receives essentially no meaningful gradient signal during the phases of training that teach the model sensible behavior, leaving it as a barely-trained, near-random vector sitting in the embedding table. When such a token is later fed into the model (deliberately or by accident), the model's behavior can become highly erratic — refusing to repeat the word, generating unrelated or bizarre completions, or exhibiting other undefined behavior — because the model is operating on an embedding region it essentially never learned to handle. This is a good concrete illustration of a broader point: tokenizer vocabulary and model behavior are trained somewhat separately (the tokenizer is typically fixed before or independently of the main behavioral training/fine-tuning data mix), and mismatches between what the tokenizer's training corpus contained versus what the model's later training phases actually reinforced can leave "holes" in the model's learned representation space.

## Embedding Lookup and Weight Tying

### From token ID to vector

Once text is tokenized into a sequence of integer IDs, the first thing that happens inside the model is an embedding lookup: a learned matrix `E` of shape `(vocab_size, d_model)` is indexed by each token ID to retrieve that token's dense vector representation. This is simply a differentiable table lookup — gradients flow back into exactly the rows of `E` corresponding to tokens that appeared in the training batch, which is itself part of why rare tokens get comparatively little training signal, as discussed above; a token that appears once in a training run updates its embedding row exactly as many times as it happens to occur, while an extremely common token's row is updated on nearly every batch.

At the very end of the network, after the final transformer block and final normalization, the model needs to convert its final hidden state vector back into a probability distribution over the vocabulary — this is the LM head, a linear projection of shape `(d_model, vocab_size)` followed by a softmax, producing the logits used for next-token prediction.

### Weight tying

Notice that the input embedding matrix `E` has shape `(vocab_size, d_model)` and the output projection matrix needs shape `(d_model, vocab_size)` — these are transposes of each other, and **weight tying** is the practice of literally using the same parameter matrix for both, i.e., the output projection is computed as `hidden_state @ E^T` rather than learning an entirely separate matrix `W_out`.

The argument for why this works, beyond simply saving parameters, is a genuine conceptual one, originating from the paper "Using the Output Embedding to Improve Language Models" (Press & Wolf) and independently motivated in the original GPT/GPT-2 line: the input embedding for a token and the output projection "direction" for that same token are conceptually doing related jobs. The input embedding needs to place semantically or syntactically similar tokens near each other in vector space so the model can process them similarly; the output projection row for a token effectively defines the direction in hidden-state space that makes the model assign high probability to predicting that token next. If two tokens are used in very similar contexts (making their input embeddings naturally similar), it's also reasonable that the hidden states that should lead the model to *predict* those two tokens would be similar directions — so sharing the matrix lets training signal that improves one usage (as input) also directly improve the other (as output target), rather than learning what is effectively the same semantic geometry twice with two independent, only indirectly-related sets of parameters.

Tying also has a clear practical benefit: for models with large vocabularies, the embedding and output matrices are each `vocab_size * d_model` parameters — for GPT-2 small (`vocab_size` ~50K, `d_model` 768), that's about 38 million parameters per matrix, and tying halves this specific cost, which was proportionally significant for smaller models. As models have grown much larger relative to their vocabulary size, this parameter-count argument has become proportionally less important (38M is a rounding error against a 70-billion-parameter model), which is part of why weight tying is not universal in modern large-scale models — several large models (for instance, larger members of the PaLM and GPT-3-scale family, and in general models where the authors found empirically that untying and letting the two matrices specialize independently gave a small quality improvement) disable weight tying once the model is large enough that the parameter savings no longer matter much, and the extra flexibility of two independently-learned matrices sometimes measurably helps, since strictly identical geometry for the "reading" and "writing" roles of a token is a real constraint on the model's expressiveness, not a free lunch.

```python
import numpy as np

vocab_size, d_model = 1000, 64
E = np.random.randn(vocab_size, d_model) * 0.02   # shared embedding / LM head weight

def embed(token_ids):
    return E[token_ids]                            # (seq_len, d_model)

def lm_head_logits(hidden_states):
    return hidden_states @ E.T                      # (seq_len, vocab_size), tied weights

# example
token_ids = np.array([5, 42, 999, 5])
x = embed(token_ids)
logits = lm_head_logits(x)   # shape (4, 1000)
```

In small-to-medium open models trained on limited compute budgets, weight tying remains common and genuinely useful; in the largest frontier models, whether it's used is more of a per-model empirical decision, and both choices appear across current production systems.
