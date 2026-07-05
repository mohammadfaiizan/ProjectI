# Pretraining Objectives: Overview

## 0. What a pretraining objective actually is

A pretraining objective is a self-supervised task computed entirely from raw text (or, more generally, raw tokens) with no human labels: you take a corpus, define a corruption or ordering scheme over it, and train a model to predict the missing/future/hidden piece. "Self-supervised" here means the label is derived mechanically from the input itself — the label for "predict the next token" is just the next token, already present in the raw text. This is the property that let language model pretraining scale to trillions of tokens: there is no labeling bottleneck, no annotator pipeline, no cost that grows with dataset size beyond compute and data acquisition/cleaning.

Every objective in this file answers the same three design questions differently, and it is worth holding these three axes explicit because they are exactly what differs across causal LM, masked LM, span corruption, prefix-LM, and GLM-style blank infilling:

1. **What gets hidden from the model, and how much of the input?** (a single next token; 15% of tokens scattered as masks; contiguous spans; an entire suffix)
2. **What attention pattern does the surviving, visible context get?** (strictly causal; fully bidirectional; bidirectional-over-a-prefix-then-causal)
3. **How is the loss computed — over every position, or only over the hidden positions?**

Everything downstream — training efficiency, whether the model can generate autoregressively at inference time, whether it can be used as a bidirectional encoder for embeddings, how naturally in-context learning falls out of it — is a consequence of how a given objective answers these three questions, not an independent design choice. Keep these three axes in mind; the rest of this file is organized around them.

## 1. Causal (next-token) language modeling

### 1.1 Mechanics

Given a token sequence \(x_1, \dots, x_T\), the causal LM objective factorizes the joint probability of the sequence autoregressively, left to right:

\[
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})
\]

and trains a single network to model \(P(x_t \mid x_{<t})\) at every position simultaneously. The loss is the negative log-likelihood, averaged (or summed) over positions:

\[
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
\]

Mechanically, a decoder-only transformer computes this in a single forward pass over the whole sequence using a **causal (lower-triangular) attention mask**: position \(t\) may attend to positions \(1, \dots, t\) but never to \(t+1, \dots, T\). Because the mask is static and doesn't depend on which position you're "currently predicting," every position's hidden state can be computed in parallel, and the model produces a full vector of next-token logits at every position in one pass — this is **teacher forcing**: during training the ground-truth tokens \(x_{<t}\) (not the model's own possibly-wrong predictions) are fed in as context at every position, so the loss at position \(t\) is well-defined and doesn't depend on earlier prediction errors compounding within the same batch. In code, the whole objective for one sequence is nothing more than:

```python
import torch
import torch.nn.functional as F

def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # logits: (batch, seq_len, vocab_size) -- model's output at every position
    # input_ids: (batch, seq_len)
    # The label for position t is the token that actually occupies position t+1.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
```

The "shift by one" is the entire mechanical content of causal LM training: the model at position \(t\) has already seen \(x_1 \dots x_t\) (via the causal mask), and its target is simply \(x_{t+1}\), which sits one slot to the right in the same tensor. No separate corruption step, no sentinel tokens, no masking schedule — the raw text *is* the training signal, shifted by one position. This is a major part of why the recipe is operationally simple (Section 6).

### 1.2 Loss coverage: every token is supervised

A critical, easily underweighted property: **every single position in the sequence contributes a training signal.** A sequence of length \(T\) yields \(T\) prediction problems (ignoring the first token, which has no context, or handling it as an unconditional prior). Contrast this immediately with BERT-style masking (Section 2), where roughly 15% of positions get a gradient signal per forward pass and the other 85% are consumed purely as context with zero direct supervision. This difference in "supervision density per token processed" is one of the underappreciated reasons causal LM pretraining scales token-efficiently — see Section 6.4.

### 1.3 Inference-time symmetry

Because training uses exactly the causal factorization that generation uses — sample \(x_{t+1} \sim P_\theta(\cdot \mid x_{<t})\), append it, repeat — there is no train/inference mismatch in the *structure* of the computation. The only mismatch is that at inference the model conditions on its own previously *sampled* tokens rather than ground truth (a classic exposure-bias concern in principle, though in practice at LLM scale this is not treated as a first-order problem the field spends much effort correcting, unlike in earlier seq2seq NMT literature). This structural symmetry — train the way you'll be asked to generate — is the single most important practical property distinguishing causal LM from every other objective in this file, and is revisited directly in Section 6.

## 2. Masked language modeling (BERT-style)

### 2.1 Mechanics

BERT's objective (Devlin et al., 2019) is **not** autoregressive. Given a sequence, sample ~15% of token positions; for each selected position, apply one of three corruptions (in BERT's original recipe: 80% of the time replace with a literal `[MASK]` token, 10% of the time replace with a random other token, 10% of the time leave unchanged) — this 80/10/10 split exists specifically to reduce train/inference mismatch, since `[MASK]` never appears at inference/fine-tuning time, and always corrupting to `[MASK]` would teach the model a representation that's only well-calibrated in the presence of a token that will never occur downstream. The model then predicts the *original* identity of every masked position, conditioning on **the entire sequence, bidirectionally** — no causal mask at all; position \(i\)'s hidden state is computed using full self-attention over positions \(1, \dots, T\) including positions after \(i\).

```python
import random

def bert_style_mask(tokens: list[int], mask_token_id: int, vocab_size: int,
                      mask_prob: float = 0.15) -> tuple[list[int], list[int]]:
    """Returns (corrupted_tokens, labels) where labels[i] = -100 (ignored)
    for unmasked positions and the original token id for masked positions."""
    corrupted = list(tokens)
    labels = [-100] * len(tokens)
    for i, tok in enumerate(tokens):
        if random.random() < mask_prob:
            labels[i] = tok  # supervise only here
            r = random.random()
            if r < 0.8:
                corrupted[i] = mask_token_id
            elif r < 0.9:
                corrupted[i] = random.randrange(vocab_size)
            # else: leave unchanged (10%), still supervised at this position
    return corrupted, labels
```

The loss is cross-entropy over only the masked positions (`-100` is the standard "ignore this position" sentinel in PyTorch's `F.cross_entropy`). The attention is bidirectional because there is no notion of "future" to hide — the whole point is that the model can look in both directions to infer a plausible filler, which is exactly what makes BERT-style pretraining good for building a general-purpose *encoder* representation (used for classification, retrieval embeddings, token tagging) rather than a generator.

### 2.2 Why masked LM cannot serve as a general-purpose generator

Two independent problems compound here. First, at generation time you'd need every future token to already exist in order to attend to it bidirectionally — obviously self-defeating for sequential generation. Non-autoregressive generation schemes exist (iteratively refine an initially-all-masked sequence, e.g., the mechanism behind some diffusion-style text models and BERT-adjacent generation research) but they are a fundamentally different, more complex inference procedure than "sample one token, append, repeat," and none of them displaced autoregressive decoding as the dominant generation paradigm for general-purpose LLMs. Second, and more subtly: even if you engineered a workable iterative-unmasking generation procedure, masked LM's per-position independence assumption at *training* time is a real modeling weakness for generation — BERT predicts every masked position's distribution independently and simultaneously, conditioned on the *corrupted* input, so it never learns \(P(x_i \mid x_{<i})\) in the sense of conditioning token \(i\)'s prediction on token \(i-1\)'s actual sampled value within the same masked span. If two adjacent positions are both masked, BERT's training objective has no mechanism to make their joint prediction *coherent* (e.g., producing "New York" rather than independently plausible-but-mismatched "New" and "Francisco") — each is scored purely against the ground truth at that position in isolation. This is precisely the joint-coherence gap that GLM's span-autoregressive design (Section 4) was built to close while retaining bidirectional context.

### 2.3 What masked LM is genuinely good for

None of the above is a claim that masked LM is a bad objective in an absolute sense — it produces excellent general-purpose bidirectional representations for exactly the tasks that don't require autoregressive generation: sentence/document embeddings for retrieval, classification, span extraction (QA-as-extraction), token-level tagging (NER). Encoder-only models trained this way (BERT, RoBERTa, and the current generation of embedding-model backbones) remain the standard choice when the downstream product is "produce a representation," not "generate free-form text," precisely because bidirectional attention over the *entire, uncorrupted-at-inference* input is a better inductive bias for those tasks than a causal model that can only look backward.

## 3. Span corruption (T5-style)

### 3.1 Mechanics

T5 (Raffel et al., 2020, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer") generalizes BERT's single-token masking to **contiguous span corruption**, and moves to an **encoder-decoder** architecture rather than a single shared stack. Given an input sequence, sample several non-overlapping spans of tokens (span lengths determined by a mean-length parameter, T5's default corrupts ~15% of tokens using a mean span length of 3); replace each entire span with a single unique **sentinel token** (`<extra_id_0>`, `<extra_id_1>`, ...). The corrupted sequence — now shorter than the original, since a whole span collapses to one sentinel — is fed to the **encoder** with full bidirectional attention (same as BERT: no causal masking on the input side). The **decoder** is then trained, autoregressively and causally, to emit each sentinel token followed by the actual span content it replaced, concatenated across all spans in original left-to-right order, terminated by a final sentinel:

```
Input:  "Thank you <X> me to your party <Y> week."
Target: "<X> for inviting <Y> last <Z>"
```//`<X>`, `<Y>`, `<Z>` are sentinels; the decoder must reproduce which text belonged to which slot, autoregressively, conditioning on the full bidirectional encoding of the corrupted input via cross-attention.

```python
def t5_span_corrupt(tokens: list[int], sentinel_ids: list[int],
                     corrupt_rate: float = 0.15, mean_span: float = 3.0):
    """Simplified sketch: returns (encoder_input, decoder_target)."""
    n = len(tokens)
    target_corrupted = int(n * corrupt_rate)
    spans, covered, i, sentinel_idx = [], 0, 0, 0
    while covered < target_corrupted and i < n:
        span_len = max(1, round(random.expovariate(1 / mean_span)))
        span_len = min(span_len, n - i)
        spans.append((i, i + span_len))
        covered += span_len
        i += span_len + random.randint(1, 3)  # gap before next span
    encoder_input, decoder_target = [], []
    prev_end = 0
    for (s, e) in spans:
        encoder_input += tokens[prev_end:s] + [sentinel_ids[sentinel_idx]]
        decoder_target += [sentinel_ids[sentinel_idx]] + tokens[s:e]
        sentinel_idx += 1
        prev_end = e
    encoder_input += tokens[prev_end:]
    decoder_target.append(sentinel_ids[sentinel_idx])  # final sentinel
    return encoder_input, decoder_target
```

### 3.2 Why span corruption over single-token masking

Two motivations, both empirical findings from the T5 paper's own ablations. First, **compute efficiency**: collapsing a whole span to one sentinel token shortens the target sequence the decoder must produce relative to predicting every corrupted position individually (as BERT effectively does), which reduces the number of decode steps needed per unit of corrupted content — a meaningful wall-clock/FLOPs saving at scale. Second, **span-level coherence**: because the decoder generates each span's content autoregressively (token \(i\) of the span conditions on token \(i-1\) of the same span, having already been emitted), span corruption inherits the same joint-coherence advantage over independent-per-position masking that GLM's design (Section 4) also exploits — this is a shared insight across T5 and GLM, arrived at independently, both correcting the same BERT weakness described in Section 2.2.

### 3.3 Encoder-decoder vs. shared-stack cost structure

T5's encoder-decoder split means there are, in effect, two stacks of parameters and two forward computations per training example (encode once bidirectionally, then decode autoregressively with cross-attention back into the encoder's output) — for a fixed total parameter budget, this splits capacity between an encoder and a decoder rather than putting all of it into a single stack. Whether that split is worth it depends on the task distribution: for tasks with an even, clean input/output separation (translation, summarization) an encoder-decoder is a very natural fit; for open-ended generation where there isn't a crisp "input" versus "output" boundary at all (open dialogue, free-text continuation), the split doesn't correspond to anything in the task structure, and the extra architectural machinery (a second stack, cross-attention layers, a fixed input/output boundary) buys comparatively little. This is one strand of the argument in Section 6 for why decoder-only became dominant for general-purpose LLMs specifically, even though T5-style objectives remain a defensible, still-used choice for well-bounded sequence-to-sequence tasks.

## 4. GLM-style autoregressive blank infilling

The GLM objective (Du et al., 2021, scaled up in GLM-130B, carried into the ChatGLM/GLM-4 lineage) is the most direct attempt in this list to have one objective subsume both BERT's bidirectional-understanding strength and GPT's autoregressive-generation strength within a **single shared transformer stack** (not an encoder-decoder split). The full mechanical derivation — the Part A / Part B split, the resulting prefix-LM attention pattern, and the two-dimensional positional encoding needed once spans are randomly permuted — is worked through in detail in `..\..\OpenSource\010_GLM4.md` (Section 2); it is not re-derived here. The essential shape, for this file's comparative purposes:

- Sample spans to corrupt (as in span corruption), but keep the **surviving, uncorrupted context** (GLM's "Part A") visible to itself **bidirectionally** — full BERT-style attention among the non-masked tokens.
- Generate the **content of the masked spans** ("Part B") **autoregressively**, causally, with each Part-B token attending back over the *entire* bidirectionally-encoded Part A plus every earlier Part-B token — this closes exactly the joint-coherence gap identified in Section 2.2 and Section 3.2, in a single stack rather than T5's separate encoder/decoder.
- The attention pattern this produces — full bidirectional attention within a "prefix" (Part A) and causal attention within the "continuation" (Part B), with Part B free to look back at all of Part A but not vice versa — is exactly the **prefix-LM** pattern discussed generally in Section 5. GLM is best understood as: *span corruption's masking scheme, T5-style span-to-single-placeholder collapsing, executed via a prefix-LM attention pattern inside one shared decoder stack instead of a separate encoder and decoder.*

The practical payoff GLM's authors argued for was a single pretrained checkpoint that is simultaneously strong on discriminative/understanding benchmarks (via the bidirectional Part-A mechanism, closer to what BERT-style pretraining buys you) and on open-ended generation (via the autoregressive Part-B mechanism and, specifically, the `[gMASK]`-single-trailing-span regime, which degenerates to ordinary left-to-right generation conditioned on a prefix). As documented in `010_GLM4.md`, by the ChatGLM2/3/GLM-4 generations, usage converged heavily toward exactly that `[gMASK]`-dominant regime — in practice looking close to conventional causal LM pretraining with an infilling objective mixed in, rather than the original paper's full multi-span-with-permutation recipe. That convergence is itself informative: it's a real-world data point for the argument in Section 6 that pure causal, next-token training is a strong attractor for a general-purpose LLM recipe even for labs that started from a deliberately more general unified objective.

## 5. Prefix-LM

Prefix-LM is best thought of as the *general pattern* that GLM's Part-A/Part-B split (Section 4) is one specific instance of, and it is worth stating independently because it recurs elsewhere (e.g., it is the attention pattern underlying UL2's mixture-of-denoisers work, and shows up whenever a model needs to condition generation on a block of context that itself deserves bidirectional treatment — for instance, a system prompt, a retrieved document, or a multi-turn conversation history that should be freely cross-referenced by every part of itself before generation begins).

Mechanically: split a sequence into a **prefix** \(x_1, \dots, x_k\) and a **continuation** \(x_{k+1}, \dots, x_T\). Attention mask:

- Within the prefix: full bidirectional attention (every prefix position attends to every other prefix position, forward and backward).
- Within the continuation: causal attention (position \(i > k\) attends to \(1, \dots, i\), never beyond).
- Continuation positions may attend into the prefix; prefix positions may never attend into the continuation.

```python
def prefix_lm_mask(seq_len: int, prefix_len: int) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)  # True = allowed
    mask[:prefix_len, :prefix_len] = True                    # prefix: bidirectional
    for i in range(prefix_len, seq_len):
        mask[i, :i + 1] = True                                # continuation: causal
    return mask
```

The loss is computed only over the continuation (the prefix is pure context, analogous to how BERT's unmasked tokens are pure context with no direct loss). Prefix-LM sits at a genuine midpoint on the causal-vs-bidirectional spectrum: strictly more expressive context-encoding than pure causal LM (the prefix isn't artificially prevented from looking ahead within itself), while still preserving a clean autoregressive generation story for everything after the prefix boundary — you can sample the continuation left-to-right exactly as with causal LM, you've just allowed the "given" part of the input richer self-attention. The cost is a training/inference asymmetry that causal LM doesn't have: the prefix/continuation boundary has to be decided (either fixed by task structure, as in GLM's masked-span design, or randomized per example, as in some prefix-LM pretraining variants explored in the UL2 line of work), and unlike causal LM's single static mask, the mask itself is data-dependent.

## 5a. A worked numeric illustration of supervision density

Section 1.2 asserts that causal LM supervises every position while BERT-style masking supervises only ~15%; it's worth putting a number on what that difference means for effective training-signal throughput per token processed.

```python
def effective_supervised_tokens(corpus_tokens: int, objective: str, mask_prob: float = 0.15) -> int:
    """Rough count of positions that receive a direct loss term per pass
    over a corpus of the given size, under each objective's masking scheme."""
    if objective == "causal_lm":
        return corpus_tokens - 1          # every position except the last has a target
    if objective == "masked_lm":
        return int(corpus_tokens * mask_prob)
    if objective in ("span_corruption", "glm_blank_infilling"):
        return int(corpus_tokens * mask_prob)   # same nominal corruption rate, contiguous spans
    raise ValueError(objective)

for obj in ["causal_lm", "masked_lm", "span_corruption"]:
    print(obj, effective_supervised_tokens(1_000_000, obj))
```

At the same nominal 15% corruption rate, causal LM supervises roughly 6.7x as many positions per forward pass over the same raw token count as BERT-style masking. This does not, by itself, prove causal LM is 6.7x more sample-efficient in terms of final model quality per token (masked positions may carry a denser, harder prediction problem per position, which could partially offset the raw count difference) — but it is a concrete, checkable number behind the qualitative claim in Section 6.4, and a staff-level answer should be able to produce this kind of estimate rather than only gesture at "causal LM sees more supervision."

## 5b. UL2's mixture-of-denoisers as an explicit attempt to unify these objectives

It is worth naming one more data point in this space directly, because it demonstrates that the field has explicitly tried to resolve the "which objective" question by combining several of the objectives in this file rather than picking one. UL2 (Tay et al., 2022, "Unifying Language Learning Paradigms") trains a single model against a **mixture of denoising objectives** spanning the spectrum this file lays out: an "R-denoiser" (short-span corruption, closer to T5-style span corruption at a low corruption rate), an "S-denoiser" (a prefix-LM-style objective, predicting a long suffix from a prefix — structurally identical to Section 5's prefix-LM pattern), and an "X-denoiser" (aggressive, long-span or high-corruption-rate corruption, pushing toward something closer to open-ended generation from very little given context). Each training example is randomly assigned one of these regimes (signaled to the model via a distinct sentinel/mode token), and the paper's claim is that mixing all three during pretraining produces a single checkpoint that transfers better, across a broader range of downstream task types, than a model pretrained on any single regime alone.

UL2 is useful here as a second, independent data point alongside GLM (Section 4) for the same underlying question this file keeps returning to: is there a single best pretraining objective, or does unifying multiple regimes into one model buy real, general-purpose flexibility? Both UL2 and GLM answer "unify them" at the *research* level, and both are genuine, citable contributions — but neither displaced causal decoder-only pretraining as the dominant recipe for the field's largest, most capability-defining general-purpose models, which is exactly the empirical outcome Section 6 tries to explain rather than merely assert. The honest reading: mixture-of-denoiser and blank-infilling-style unified objectives are real, validated research directions with genuine benefits for certain transfer-learning settings, and the field's overwhelming practical convergence on causal-only pretraining for frontier general-purpose LLMs specifically is a statement about what wins under the scale, infrastructure, and in-context-learning priorities Section 6 describes — not a statement that the unified-objective research program was mistaken or uninteresting.

## 6. Why causal decoder-only became the dominant recipe for general-purpose LLMs

This is the section a staff interview will actually probe hardest, and it is important to make the argument precisely rather than reciting "it's simpler." The honest claim is narrower and more interesting than "causal LM is the best objective": **it is not obviously the best objective for any single task in isolation** — bidirectional encoders are typically better for embeddings/retrieval/classification (Section 2.3); encoder-decoder span corruption may be a better fit for a well-bounded translation or summarization product (Section 3.3) — **but it is the best objective for building one general-purpose system that must generate free-form text, follow arbitrary instructions, and support in-context learning, under a scaling regime where architectural and infrastructure simplicity compounds.** Several independent arguments stack on top of each other:

**6.1 Structural train/inference symmetry (Section 1.3).** Causal LM is the only objective in this file where the *computation performed during training* and the *computation performed during autoregressive generation* are literally the same operation, just with ground-truth vs. sampled tokens fed in as context. There is no separate "now switch to iterative denoising" or "now run the decoder in a different mode" step at deployment time. Every other objective here (masked LM, span corruption, even GLM's `[MASK]`-regime) requires either a non-trivial inference-time procedure to actually generate free text, or — as GLM's own usage evolution shows — a drift back toward the causal, single-trailing-span regime specifically *because* that's what makes open-ended generation easy.

**6.2 Natural fit with in-context learning.** In-context learning (a model performing a task from a natural-language instruction plus a few demonstrations placed directly in the prompt, with zero gradient updates — see `..\..\GPT\003_GPT3.md` Section 7 for the full empirical case) is, mechanically, nothing more than causal next-token prediction conditioned on an unusually structured prompt. There is no separate mechanism to build; the capability emerges directly from the pretraining objective applied at sufficient scale, because "predict what comes next" *is* "given this instruction-and-examples pattern, produce the continuation that pattern-completes it." Masked LM and span corruption don't have this property nearly as cleanly: their training-time task shape (fill in scattered/contiguous corruptions, in a fixed, pre-determined location, of a document you're also allowed to see the rest of) does not naturally resemble "here is a novel instruction, followed by a blank you must fill with free-form text of unknown length at an unknown location, namely the end."

**6.3 Simplicity of implementation, training, and infrastructure investment.** A causal decoder-only model needs exactly one attention mask (static, data-independent, lower-triangular), one stack of parameters, one loss (cross-entropy over every position, no `-100`-style position filtering, no sentinel-token bookkeeping), and one inference procedure (KV-cache-based autoregressive decoding, discussed at length across the per-model docs in `..\..\OpenSource\` and `..\..\GPT\`). Every distributed-training optimization the field has built over the last several years — FlashAttention-style fused causal-masked kernels, KV-cache reuse across decode steps, speculative decoding, continuous batching in serving systems — is built against *this specific, simple contract*. Committing to causal LM as the pretraining objective means every subsequent infrastructure investment compounds cleanly; committing to, say, a randomized-prefix-boundary prefix-LM objective means every one of those downstream systems has to handle a more variable masking pattern. At the scale frontier labs operate at (clusters of thousands of accelerators, training runs costing tens of millions of dollars — see `002_Scaling_Laws_And_Compute_Optimal_Training.md`), this kind of infrastructure simplicity is not a minor convenience; it is a first-order cost and risk factor in itself.

**6.4 Supervision density and observed scaling behavior.** As noted in Section 1.2, causal LM extracts a gradient signal from *every* token in the corpus, whereas BERT-style masking directly supervises only ~15% of tokens per forward pass (the other 85% are context-only). Empirically, autoregressive decoder-only models have been the vehicle for essentially all of the large, well-documented power-law scaling results (Kaplan et al. 2020; Hoffmann et al. 2022 — see `002_Scaling_Laws_And_Compute_Optimal_Training.md`), and it is this same causal-decoder-only family that every frontier lab has bet its largest, most expensive training runs on. This is partly a self-fulfilling pattern (the objective that got the most investment produced the most scaling evidence, which justified more investment), but the supervision-density argument gives an independent, mechanistic reason to expect causal LM to be comparatively token-efficient even before accounting for the field's investment bias.

**6.5 What is actually given up.** None of this means the tradeoffs in Sections 2–5 vanish — they are simply judged, for a *general-purpose* system, to matter less than the properties above. A causal-only model is a systematically worse building block for a pure embedding/retrieval product than a bidirectional encoder (hence encoder-only and bidirectional-friendly architectures remain the default for that specific product category, often now built by extracting representations from a causal model's hidden states or via lightweight bidirectional fine-tuning on top of a causal base, rather than from a from-scratch causal pretraining run). And a from-scratch encoder-decoder can still be the more sample-efficient, better-architected choice for a narrowly-scoped, well-bounded seq2seq product where "input" and "output" are genuinely different distributions (classic MT is the textbook case). The staff-level framing is: causal decoder-only won the race for *general-purpose* pretraining not because it dominates every axis, but because (a) it is the objective whose native computational shape matches free-form generation and in-context learning with zero adaptation, and (b) at the compute and infrastructure scale frontier labs operate at, its simplicity compounds into a decisive operational advantage that offsets its task-specific disadvantages elsewhere.

## 7. Summary table

| Objective | Attention over context | What's hidden | Loss coverage | Native inference mode | Best fit |
|---|---|---|---|---|---|
| Causal LM | Strictly causal | Next token, everywhere | Every position | Autoregressive sampling (= training op) | General-purpose generation, ICL |
| Masked LM (BERT) | Full bidirectional | ~15% scattered tokens | Masked positions only | Not a generator (needs iterative/non-autoregressive scheme) | Embeddings, classification, retrieval |
| Span corruption (T5) | Bidirectional (encoder) / causal (decoder) | Contiguous spans, collapsed to sentinels | Decoder target sequence | Encoder-decoder autoregressive decode | Bounded seq2seq (translation, summarization) |
| GLM blank infilling | Bidirectional (Part A) / causal (Part B) | Permuted spans | Part B only | `[gMASK]`-regime ≈ causal generation | Unified understanding + generation in one stack |
| Prefix-LM | Bidirectional (prefix) / causal (continuation) | Continuation | Continuation only | Causal decode after a bidirectional prefix | Context-block-conditioned generation |

The remaining five files in this module assume the causal decoder-only objective as the baseline recipe (consistent with essentially every frontier general-purpose LLM covered elsewhere in this collection) and go deep on how that recipe is actually scaled, trained, staged, and validated in practice.

## 8. A survey of which objective each model in this collection actually uses

Holding this file's taxonomy against the specific models documented elsewhere in this collection is a useful sanity check on Section 6's central claim:

| Model | Pretraining objective | Source |
|---|---|---|
| GPT-1/2/3, GPT-4-era models | Causal decoder-only | `..\..\GPT\` |
| Llama 1/2/3, Mistral, Mixtral, DeepSeek-V2/V3, Qwen2.5 | Causal decoder-only | `..\..\OpenSource\` |
| BERT, RoBERTa (not individually documented in this collection, referenced for contrast) | Masked LM | N/A — background reference only |
| T5 | Span corruption, encoder-decoder | Background reference; not individually documented in this collection |
| Original GLM / GLM-130B | Autoregressive blank infilling (prefix-LM pattern) | `..\..\OpenSource\010_GLM4.md` |
| ChatGLM2/3, GLM-4 | Converged toward `[gMASK]`-dominant, causal-generation-like regime | `..\..\OpenSource\010_GLM4.md`, Section 2 |
| Claude family, Gemini/PaLM family | Causal decoder-only (architecture details vary; objective is consistently causal LM across the disclosed record) | `..\..\Claude\`, `..\..\Gemini\` |

The pattern is stark: of every frontier, general-purpose model covered in this entire collection, the *only* one that started from a genuinely non-causal-only pretraining objective (GLM) drifted toward the causal-generation-dominant regime by its later, more product-oriented generations. This is not proof by itself that causal-only is the only viable choice — masked LM and span corruption remain the right defaults for their respective non-generative use cases (Sections 2.3, 3.3) — but it is strong, consistent empirical evidence for Section 6's specific claim about what wins for *general-purpose* frontier LLM pretraining specifically.

## 9. A closing checklist

1. Can you state, for any objective in this file, all three of its defining axes (what's hidden, what attention pattern the context gets, what the loss covers) without conflating them?
2. Can you explain precisely why BERT's independent per-position masked prediction lacks joint span-level coherence, and why GLM's and T5's autoregressive-span-generation designs each independently fix this?
3. Can you state the mechanistic reason in-context learning falls out of causal LM pretraining specifically, rather than merely asserting that it does?
4. Can you name, honestly, what a causal-only pretraining recipe gives up relative to a bidirectional encoder or an encoder-decoder for their respective specialized use cases (Section 6.5), rather than treating causal LM as unconditionally superior?
5. Can you connect GLM's real-world usage drift (Section 4, Section 8) and UL2's mixture-of-denoisers (Section 5b) as two independent data points bearing on the same underlying question, rather than only being aware of one of them?

## 10. Quick-reference glossary

- **Teacher forcing** — feeding the ground-truth prefix, not the model's own sampled output, as context during causal-LM training, which is what allows all positions in a sequence to be trained in parallel (Section 1.1).
- **Supervision density** — the fraction of positions in a training sequence that receive a direct loss term per forward pass; causal LM supervises every position, BERT-style masking only the masked subset (Section 1.2, Section 5a).
- **Sentinel token** — a placeholder token (e.g., T5's `<extra_id_0>`) that collapses an entire corrupted span to a single position in the corrupted input, with the decoder trained to reproduce the span's actual content after it (Section 3.1).
- **Prefix-LM mask** — full bidirectional attention within a leading "prefix" block, causal attention within the following "continuation," with the continuation free to attend back into the prefix but not vice versa (Section 5).
- **Mixture-of-denoisers** — UL2's approach of training a single model against several distinct corruption/generation regimes (short-span, prefix-LM-style, aggressive long-span) simultaneously, signaled via a mode token (Section 5b).

## 11. See also

The scaling and compute-allocation consequences of committing to causal decoder-only pretraining are developed in `002_Scaling_Laws_And_Compute_Optimal_Training.md`. The architecture decisions (attention variant, dense-vs-MoE) that are typically made jointly with, but are distinct from, the objective choice covered here are developed in `003_Model_Architecture_Decisions_At_Pretraining_Time.md`. Worked, staff-level applications of this file's specific framework are in `007_Interview_Questions_Part1.md`, Q1-Q4 and Q14 and Q19.
