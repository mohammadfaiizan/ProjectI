## LLaMA (2023)

### 1. Overview & Strategic Context

Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (Meta AI, Feb 2023). The paper's stated goal was to show that state-of-the-art results were achievable by training smaller models longer on more tokens, rather than chasing parameter count -- a direct rebuttal of the GPT-3-era assumption that quality scales primarily with model size. LLaMA-13B was shown to outperform GPT-3 (175B) on most benchmarks despite being roughly 13x smaller, and LLaMA-65B was competitive with Chinchilla-70B and PaLM-540B.

The strategically consequential decision, from an open-source-ecosystem standpoint, was training exclusively on data that is "publicly available" and reproducible without proprietary access -- CommonCrawl, C4, GitHub, Wikipedia, two book corpora (Gutenberg and Books3), ArXiv, and StackExchange. Contemporary large models (GPT-3, PaLM, Chinchilla, OPT) either used undisclosed proprietary data mixtures or data with licensing ambiguity. LLaMA's constraint was deliberate: it made the training recipe auditable and (mostly) replicable by third parties, which is precisely why LLaMA became the substrate for the open-model ecosystem (Alpaca, Vicuna, and eventually the entire "Llama-style block" architecture lineage) despite being released under a noncommercial research license rather than a permissive one. The weights leaked within days of release and that leak, arguably more than the paper itself, catalyzed the 2023 open-weight LLM boom.

Architecturally, LLaMA is not novel in any single component -- RMSNorm, RoPE, and SwiGLU each predate the paper by 1-3 years -- but it is the paper that canonicalized their combination as the default recipe for decoder-only causal LMs. Nearly every open model released afterward (Llama 2/3, Mistral, Qwen, Yi, Falcon 2, Gemma) starts from this exact block and modifies it incrementally (adding GQA, sliding windows, MoE) rather than replacing it.

### 2. Architecture Deep Dive

LLaMA is a standard decoder-only, causal, dense transformer with three deviations from the original "Attention Is All You Need" / GPT-2 block, applied uniformly across all four sizes:

**Pre-normalization with RMSNorm.** Instead of post-LN (original Transformer) or the by-then-standard pre-LN with LayerNorm, LLaMA normalizes the input to each sub-layer (attention and FFN) using RMSNorm (Zhang & Sennrich, 2019):

RMSNorm(x) = (x / RMS(x)) * g, where RMS(x) = sqrt((1/d) * sum_i x_i^2 + eps)

There is no mean-centering (unlike LayerNorm) and no bias term -- only a learned per-channel gain g. Dropping mean subtraction removes one reduction and is empirically just as stable for transformer hidden states, at lower compute cost. Pre-normalization (normalizing the sub-layer *input*, then adding the residual) rather than post-normalization is used specifically because it was shown (by GPT-2 and follow-ups) to improve training stability at scale, at some cost to final performance versus post-LN, a tradeoff the paper accepts.

**Rotary Position Embeddings (RoPE), replacing absolute/learned positional embeddings.** Su et al. (2021). Rather than adding a position vector to the token embedding, RoPE rotates each 2D subspace of the query and key vectors by an angle proportional to absolute position. For a query/key vector split into pairs (x_{2i}, x_{2i+1}), position m, and frequency theta_i = base^(-2i/d) with base = 10000:

q'_m = R(m*theta) * q_m, k'_n = R(n*theta) * k_n

where R is the block-diagonal rotation matrix. The key property is that the dot product q'_m . k'_n depends only on the *relative* position (m-n), not on absolute position, which is injected directly into the attention score rather than into the token representation. This gives LLaMA a form of relative position awareness "for free," with no learned position parameters and (empirically) better extrapolation behavior than learned absolute embeddings, though LLaMA 1 was still trained and evaluated at a fixed 2048-token context and no explicit long-context extrapolation claims are made in the paper.

**SwiGLU activation in the FFN, replacing ReLU.** Shazeer (2020), "GLU Variants Improve Transformer." The FFN becomes a gated linear unit using SiLU/Swish as the gate nonlinearity:

FFN(x) = (Swish(x W1) ⊙ (x W3)) W2, Swish(z) = z * sigmoid(z)

This uses three weight matrices instead of two (an extra gating projection W3), so for a fixed parameter budget the hidden dimension is scaled down: LLaMA uses hidden dim = (2/3) * 4d (rounded to a multiple of 256, or 64 in some released configs) rather than the conventional 4d, keeping the FFN's total parameter count roughly comparable to a ReLU-MLP FFN of hidden size 4d while gaining the empirical quality benefit of the gating mechanism.

**Attention** is standard multi-head causal self-attention -- no GQA, no MQA, no sliding window. Every attention head has its own full-size K and V projection at every layer; n_heads = n_kv_heads for all four sizes. This is the most significant point of divergence from Llama 2/3: LLaMA 1's inference cost and KV-cache memory scale with the full head count, a cost that later models start amortizing via GQA.

**Tokenizer:** byte-pair encoding (BPE) via SentencePiece, 32,000-token vocabulary. Numbers are split into individual digits, and unknown UTF-8 byte sequences fall back to byte-level tokens, which guarantees the tokenizer can represent arbitrary text without an `<unk>` token dead-end.

**Context length:** 2048 tokens, fixed, for all four sizes.

Per-size configuration (dim = hidden size, n_heads = n_layers = number of transformer blocks):

| Size | dim | n_heads | n_layers | Learning rate | Batch size (tokens) |
|---|---|---|---|---|---|
| 6.7B (7B) | 4096 | 32 | 32 | 3.0e-4 | 4M |
| 13.0B (13B) | 5120 | 40 | 40 | 3.0e-4 | 4M |
| 32.5B (33B) | 6656 | 52 | 60 | 1.5e-4 | 4M |
| 65.2B (65B) | 8192 | 64 | 80 | 1.5e-4 | 4M |

Optimizer: AdamW (beta1=0.9, beta2=0.95), cosine learning-rate schedule decaying to 10% of peak, weight decay 0.1, gradient clipping at 1.0, and 2000 warmup steps.

### 3. Scale -- Parameters, Data, Compute

Four dense sizes: 6.7B, 13.0B, 32.5B, 65.2B parameters (commonly rounded to 7B/13B/33B/65B).

Training tokens: the 7B and 13B models were trained on ~1.0T tokens; the 33B and 65B models were trained on ~1.4T tokens. All models see each source-dataset mixture roughly the same number of epochs except Wikipedia and Books, which are up-weighted to about two epochs given their comparatively higher per-token quality.

Pretraining corpus mixture (by sampling proportion): CommonCrawl 67%, C4 15%, GitHub 4.5%, Wikipedia (20 languages) 4.5%, Books (Gutenberg + Books3) 4.5%, ArXiv 2.5%, StackExchange 2%. Total corpus size after tokenization: roughly 1.4T tokens available, matching the largest models' full pass count -- i.e., the 33B/65B models are trained for approximately one epoch over the entire deduplicated corpus.

Disclosed compute: training used 2048 A100-80GB GPUs. For the 65B model, the paper reports a processing throughput of about 380 tokens/second/GPU, implying roughly 21 days of wall-clock training for 1.4T tokens on the full cluster. This places LLaMA training compute in the same order of magnitude as Chinchilla-70B despite LLaMA-65B being trained on the same 1.4T-token count -- the paper's central point is that this token budget, applied to a much smaller model with efficient attention/checkpointing, buys competitive quality at a fraction of both training and (especially) inference FLOPs versus GPT-3/PaLM-scale models.

### 4. Training Infrastructure & Distributed Training

The paper discloses two specific systems-level optimizations, both aimed at reducing the memory and compute overhead of attention and backpropagation at this scale, rather than a novel distributed-training framework:

**Efficient causal attention implementation.** LLaMA uses the memory-efficient attention formulation from Rabe & Staats (2021) as implemented in the `xformers` library, which avoids materializing the full attention weight matrix and avoids computing attention weights for masked-out (future) positions in a causal mask, reducing both memory footprint and FLOPs relative to a naive attention implementation.

**Manual backward pass with activation checkpointing.** Rather than relying on generic autograd checkpointing, the authors manually implement the backward function for the transformer block so that the more expensive-to-recompute activations (notably the outputs of the linear layers, which have the highest computational cost to reconstruct) are saved rather than recomputed, while cheaper activations are recomputed on the backward pass. This is a hand-tuned tradeoff between memory and recompute FLOPs rather than the blanket "checkpoint everything" strategy, and it materially improved achieved throughput on the 2048-A100 cluster.

**Parallelism strategy:** model parallelism (what would later be called tensor parallelism) is used to shard the largest models (33B, 65B) across GPUs, following the approach of Korthikanti et al. (2022) ("Reducing Activation Recomputation in Large Transformer Models"), combined with standard data parallelism across GPU groups. The paper does not disclose pipeline-parallel configuration details or a named end-to-end framework (no "we built X" claim analogous to Megatron-LM or DeepSpeed branding) -- the systems contribution is presented as an engineering optimization on top of existing primitives rather than a new framework.

### 5. Pretraining Data & Objective

Standard autoregressive (causal) language-modeling objective: cross-entropy loss predicting the next token given all previous tokens, no auxiliary objectives.

The data is exclusively from sources the authors characterize as "publicly available and compatible with open sourcing" -- a constraint absent from GPT-3 (which used a large undisclosed web + books mixture), PaLM (undisclosed proprietary mixture), and even Chinchilla/Gopher (DeepMind's MassiveText, not publicly released). The seven constituent sources:

- **CommonCrawl** (67% of tokens): filtered via a linear classifier trained to distinguish Wikipedia-referenced pages from random CommonCrawl pages, plus n-gram-based quality/language filtering, deduplicated at the line level.
- **C4** (15%): the standard cleaned CommonCrawl derivative from Raffel et al. (2020), included alongside raw CommonCrawl because its distinct heuristic-based cleaning pipeline provides diversity in filtering criteria.
- **GitHub** (4.5%): public GitHub repositories under permissive licenses (Apache, BSD, MIT), filtered by heuristics on line length and alphanumeric character fraction, with boilerplate/headers removed via regex, deduplicated at file level.
- **Wikipedia** (4.5%): dumps covering 20 languages, with markup/citations removed.
- **Books** (4.5%): the Gutenberg project and the Books3 section of ThePile, deduplicated at the book level with high content-overlap threshold.
- **ArXiv** (2.5%): scientific papers, with LaTeX macros/bibliography stripped to keep signal-dense text.
- **StackExchange** (2%): Q&A pairs from the 28 largest sites, sorted by answer score, HTML stripped.

No instruction data, no RLHF data, and no synthetic data are part of this mixture -- LLaMA 1 is a pure base/foundation model trained end-to-end on this raw-text objective.

### 6. Post-Training / Alignment Approach

None. LLaMA 1 was released purely as a set of base (pretrained) language models with no instruction-tuning or RLHF stage from Meta. The paper itself only reports few-shot and zero-shot benchmark evaluation of the base models. Any instruction-following or chat behavior seen from "LLaMA" in practice (Alpaca, Vicuna, Koala, etc.) came from third parties who applied their own SFT (and in some cases RLHF-adjacent) recipes on top of the leaked base weights -- these are downstream community artifacts, not part of Meta's release. This absence of an official alignment stage is itself a notable contrast with Llama 2, where Meta shipped fully documented RLHF'd chat variants alongside the base models.

### 7. Key Research Contributions & Novel Techniques

Nothing in LLaMA's component list is individually new -- RMSNorm (2019), RoPE (2021), SwiGLU (2020), and memory-efficient attention (2021) all predate the paper. The genuine contributions are:

1. **Empirical demonstration that the "smaller model, more tokens" regime beats the "bigger model, fewer tokens" regime at fixed inference budget** -- reframing the Chinchilla compute-optimal-training result (which optimizes training-time compute for a fixed loss) around a different, arguably more practically relevant, axis: given that a model will be served at inference for a long deployment lifetime, a smaller model trained on more tokens than compute-optimal training would suggest can match a larger model's quality at drastically lower serving cost. This logic is the direct conceptual ancestor of Llama 3's much more extreme "overtraining" strategy (see `003_Llama3.md`).
2. **Canonicalizing the "Llama block"** -- the specific combination of pre-norm RMSNorm + RoPE + SwiGLU-FFN + standard MHA, trained purely causally, became the reference architecture that essentially every subsequent open decoder-only LLM (Mistral, Qwen, Yi, Falcon2, Gemma, DeepSeek's dense components) starts from verbatim, only modifying attention (GQA, sliding window) or adding MoE.
3. **Training solely on publicly available/reproducible data as a first-class design constraint at this scale and quality bar** -- proving it does not sacrifice competitiveness against contemporaries trained on proprietary data mixtures.

### 8. Training Challenges & Engineering Solutions

The main disclosed engineering challenge is compute efficiency at fixed hardware (2048 A100s, no larger cluster available for this project) rather than any exotic instability. The two systems optimizations described in Section 4 (xformers-based memory-efficient causal attention, and the hand-written backward pass that selectively checkpoints only the expensive-to-recompute activations) exist specifically to maximize achieved tokens/sec/GPU on that fixed hardware budget, since the token count (1-1.4T) was itself a deliberate design choice constrained by available compute and desired training wall-clock time, not by data availability (much more CommonCrawl exists than was used).

The paper reports training-loss curves that are smooth and monotonically decreasing for all four sizes with no discussion of loss spikes, divergence, or restart events -- unlike later, larger-scale efforts (e.g., OPT-175B, BLOOM) whose logbooks documented significant instability. At LLaMA's scale (max 65B, max 1.4T tokens, max 2048 GPUs) this kind of instability was less of a binding constraint than it became for subsequent 100B+ parameter or 10K+ GPU efforts.

### 9. Inference & Serving Considerations

Because attention is standard MHA with no KV-head reduction, LLaMA's KV cache scales as 2 * n_layers * n_heads * head_dim * seq_len * batch (for K and V combined) with no discount -- e.g., 65B's cache at full 2048-token context requires materially more memory per token than the GQA-equipped Llama 2 70B, let alone Llama 3's universally-GQA'd models. This is precisely the inference-cost problem that GQA (introduced in Llama 2, and made universal in Llama 3) was designed to address; LLaMA 1 predates that fix.

At 2048-token context, absolute memory pressure from the KV cache is modest by today's standards, so this was not yet a first-order serving concern for LLaMA 1 specifically -- it becomes one once later models push context to 4K, 8K, 32K, and 128K while keeping the same head-count-scaling attention. LLaMA 1's fixed 2048-token training context is itself a serving constraint: extrapolating beyond it degrades quality (RoPE gives some graceful extrapolation behavior, but the model was never trained or evaluated beyond 2048 tokens).

### 10. Evaluation, Benchmarks & Known Limitations

Reported zero-/few-shot results (selected, from the paper):

- **Common-sense reasoning** (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e/c, OpenBookQA, zero-shot): LLaMA-65B matches or exceeds PaLM-540B on most of these tasks despite being ~8x smaller in parameter count.
- **Closed-book QA** (Natural Questions, TriviaQA): LLaMA-65B is competitive with or exceeds Chinchilla-70B and PaLM-540B.
- **Reading comprehension** (RACE): competitive with PaLM-540B.
- **MMLU** (5-shot): LLaMA-65B scores below Chinchilla-70B and PaLM-540B, attributed by the authors to the relatively small amount of books/academic data (Books + ArXiv is only 7% of the corpus) relative to those models' training sets.
- **Headline result:** LLaMA-13B outperforms GPT-3 (175B) on most benchmarks tested, while being runnable on a single high-end GPU -- the paper's central efficiency claim.
- **Code generation** (HumanEval, MBPP): competitive given GitHub is only 4.5% of the corpus.

Known limitations, as disclosed or evident from the paper: 2048-token fixed context is short by later standards; no instruction-tuning/RLHF means the base models are poor at following instructions or open-ended chat out of the box (few-shot prompting is required to elicit good task performance); MMLU underperformance versus contemporaries reflects the data-mixture choice; and toxicity/bias evaluations (RealToxicityPrompts, CrowS-Pairs, WinoGender) are reported but show LLaMA inherits the biases typical of web-scale training data, with no mitigation applied since there is no alignment stage.

### 11. Confirmed Facts vs. Speculation

This is an open-weight model with a published paper disclosing architecture, data mixture percentages, token counts, hardware, and benchmark numbers in detail, so nearly everything above is directly confirmed by the paper. This section is accordingly short. The few areas of genuine ambiguity:

- The paper does not disclose the exact per-source epoch counts beyond stating Wikipedia and Books are seen "around twice," and does not give an exact FLOPs or dollar-cost figure for the full training run (only GPU-count and days-of-wall-clock for the 65B model can be derived/estimated from the throughput figure given).
- The precise deduplication and filtering thresholds (e.g., exact n-gram overlap cutoffs for CommonCrawl filtering) are described qualitatively, not as exact reproducible parameters.
- No official instruction-tuned or chat variant exists from Meta for LLaMA 1; anything claiming to be "instruction-tuned LLaMA 1" is a third-party derivative (Alpaca/Vicuna/etc.), not part of the confirmed Meta release.

### 12. Staff/Research Interview Talking Points

- Be able to state precisely *why* RMSNorm is cheaper and equally effective versus LayerNorm (no mean-centering, no bias) and why pre-norm is chosen over post-norm (training stability at depth, at some cost to peak quality) -- this is a frequently probed "do you actually understand normalization" check.
- Be able to derive, on a whiteboard, why RoPE's rotation makes q'_m . k'_n a function only of (m - n), and why that is a better inductive bias for a causal LM than adding a learned absolute-position vector to the embedding.
- Explain the SwiGLU hidden-dimension bookkeeping: three matrices instead of two means hidden_dim is scaled to ~(2/3)*4d specifically to keep total FFN parameter count roughly constant against a ReLU-MLP baseline of hidden_dim 4d -- interviewers use this to check whether you understand that "adding a gate" is not architecturally free.
- Be ready to contrast the "compute-optimal" (Chinchilla) framing against the "inference-optimal" framing this paper implicitly introduces: Chinchilla asks "what model size and token count minimizes training loss for fixed training FLOPs," while LLaMA's framing asks "what is the cheapest model to serve at inference that hits a target quality," which is the direct conceptual seed of Llama 3's much larger overtraining ratio.
- Know the concrete number: LLaMA-13B beating GPT-3-175B on most benchmarks is the paper's headline result and a common context point for "why did the industry pivot toward smaller, longer-trained models."
- Understand that LLaMA 1 has zero alignment step -- if asked to compare "LLaMA vs. Llama 2 chat," the correct answer starts with "these aren't comparable without noting one has no RLHF/SFT stage at all."
