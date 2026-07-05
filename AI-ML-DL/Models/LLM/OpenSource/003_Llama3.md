## Llama 3 (2024)

### 1. Overview & Strategic Context

Meta released Llama 3 in two stages. The initial April 2024 release shipped 8B and 70B dense models trained on over 15T tokens with an 8K context window. The July 2024 "Llama 3.1" release ("The Llama 3 Herd of Models," Dubey et al.) added a 405B dense model and extended all three sizes (8B, 70B, 405B) to a 128K context window, alongside improved multilingual and tool-use capability. This document treats the two releases as one continuous model family, as the underlying paper does.

Three changes define this generation relative to Llama 2. First, GQA is applied universally, including the 8B model -- Llama 2 treated GQA as a cost-saving measure justified only at 70B; Llama 3 treats it as a default at every scale, reflecting that KV-cache-bandwidth cost matters even at 8B once context routinely extends to 8K-128K tokens. Second, the tokenizer vocabulary quadruples to 128K tokens (from Llama 2's 32K), directly reducing tokens-per-character for a given text and materially improving compression on non-English text and code. Third, and most consequential for how the field thinks about pretraining-scale decisions: the 8B and 70B models are trained on 15T+ tokens, a token count that is deliberately, by a wide margin, past what Chinchilla-style compute-optimal scaling laws would prescribe for their parameter counts. This is not an oversight -- it is the paper's explicit strategy, and the reasoning behind it (training compute-optimality minimizes training FLOPs for a target loss, but says nothing about inference cost, which for a widely-deployed model dominates total cost of ownership) is one of the most load-bearing ideas in modern LLM engineering strategy and a frequent staff-level interview topic.

### 2. Architecture Deep Dive

The block is architecturally the same lineage as LLaMA 1/2: pre-norm RMSNorm, RoPE, SwiGLU FFN. What changes:

**GQA everywhere, including 8B.** Every released size -- 8B, 70B, 405B -- uses grouped-query attention with 8 KV heads. For 8B (32 query heads, 8 KV heads: 4:1 grouping) and 70B (64 query heads, 8 KV heads: 8:1 grouping, unchanged ratio from Llama 2's 70B) this is a meaningful new design decision at the smaller size; for 405B (128 query heads, 8 KV heads: 16:1 grouping) it is a necessity -- without GQA, 405B's KV cache at long context would be prohibitively expensive to serve even before considering the model's own weight-memory footprint.

**Vocabulary expanded to 128,256 tokens** using a tokenizer built on OpenAI's `tiktoken`-style byte-level BPE (specifically extending the `cl100k_base` vocabulary Llama 3's tokenizer descends from, per the released tokenizer implementation), plus roughly 28K additional tokens targeted at improving non-English-language compression. A larger vocabulary means, for the same text, fewer tokens are needed to represent it -- this reduces both the sequence length the attention mechanism must process for a fixed amount of "real" content and the number of autoregressive decode steps needed to generate a fixed amount of output text, at the cost of a larger embedding matrix and LM head (128K x dim parameters each way, versus 32K x dim in Llama 2 -- a non-trivial parameter-count and compute increase specifically in the embedding/unembedding layers, especially proportionally at 8B).

**RoPE base frequency increased to 500,000** (from 10,000 in Llama 1/2) for the 128K-context Llama 3.1 models. Increasing the RoPE base frequency theta stretches the wavelengths of the lowest-frequency rotation components, which is the standard technique (matching the family of "RoPE scaling" approaches explored across the field for context extension) for adapting a RoPE-based model originally trained at a shorter context to a much longer one without the highest-frequency components aliasing at long relative distances.

**Context length:** 8K at the initial Llama 3 (8B/70B) release; extended to 128K for all three sizes (8B/70B/405B) in the Llama 3.1 release, achieved via continued pretraining on longer sequences with the adjusted RoPE base, in stages that progressively increase the maximum sequence length rather than jumping directly to 128K.

Per-size configuration (Llama 3.1):

| Size | dim | n_heads | n_kv_heads | n_layers | Vocab | Context |
|---|---|---|---|---|---|---|
| 8B | 4096 | 32 | 8 | 32 | 128,256 | 128K |
| 70B | 8192 | 64 | 8 | 80 | 128,256 | 128K |
| 405B | 16,384 | 128 | 8 | 126 | 128,256 | 128K |

405B's feed-forward hidden dimension and exact SwiGLU sizing follow the same (2/3)*4*dim-with-rounding convention as earlier Llama generations, scaled to the larger dim.

### 3. Scale -- Parameters, Data, Compute

Dense (non-MoE) models at 8B, 70B, and 405B parameters. Pretraining token count: over 15T tokens for the initial 8B/70B release; the 405B model (and the continued-pretraining/context-extension work for all three sizes in Llama 3.1) is trained on a similarly large multilingual- and code-heavy corpus, with the paper stating the total training corpus is roughly 50% general knowledge tokens, with the remainder split across code, multilingual, reasoning/math, and long-context-specific data added during later training stages.

Disclosed compute for 405B: on the order of 3.8 x 10^25 FLOPs of training compute (the paper's own headline compute figure), trained on up to 16,000 H100 GPUs at once -- the largest disclosed single-model training run in the Llama lineage by a wide margin. This makes 405B, at release, one of the largest publicly disclosed open-weight training runs by both parameter count and compute.

Context-length-driven token accounting: extending context to 128K required a dedicated continued-pretraining phase on long-document data (and synthetically constructed long-context data) after the main pretraining run, rather than being trained end-to-end at 128K from step zero -- consistent with how most long-context extensions are done across the field (short-context pretraining is cheaper per token; long-context adaptation is a comparatively short, targeted continued-training phase).

### 4. Training Infrastructure & Distributed Training

405B training is disclosed as using a combination of data parallelism, tensor parallelism, pipeline parallelism, and context parallelism (4D parallelism) across the up-to-16K-H100 cluster -- the explicit mention of *context parallelism* (splitting a single long sequence's computation across GPUs along the sequence dimension) is notable and specific to supporting the eventual 128K-context training/inference regime; standard 3D parallelism (data/tensor/pipeline) alone becomes insufficient once per-sequence activation memory at 128K tokens is itself too large to fit even after tensor+pipeline sharding.

The paper discloses substantial infrastructure engineering investment: a custom training stack built around PyTorch, with the storage, networking (RDMA over Converged Ethernet at very large scale, moving away from InfiniBand-only assumptions for a cluster this size), and fault-tolerance/checkpointing systems described as first-class engineering deliverables of the paper, not afterthoughts -- Meta explicitly frames the 405B run's infrastructure engineering (job scheduling, failure recovery, network topology) as comparably important to the algorithmic recipe, and reports non-trivial GPU failure/interruption rates over the training run requiring automated detection and restart tooling at that scale.

Data parallelism uses FSDP-style sharding for optimizer/gradient state; the exact parallelism-degree combination (e.g., specific TP x PP x DP x CP grid dimensions) is disclosed at a high level in the paper without being a full reproducible configuration table for every stage of training.

### 5. Pretraining Data & Objective

Standard causal LM objective. The pretraining corpus is described as substantially larger and more heavily curated for quality than Llama 2's -- more code, more multilingual data, and a much larger token budget overall (15T+ vs. Llama 2's 2T). The paper discloses extensive data-quality-classification pipelines (using earlier Llama models themselves as quality classifiers/filters over web data, a form of self-generated-signal data curation), deduplication at multiple granularities, and a deliberate curriculum -- later stages of pretraining up-weight higher-quality and longer-context data relative to earlier stages ("annealing" on a smaller, higher-quality data mixture at the end of pretraining, a technique the paper reports measurably improves benchmark performance relative to ending training on the raw, unweighted mixture).

**The overtraining/inference-cost argument, in depth.** Chinchilla-style compute-optimal scaling (Hoffmann et al., 2022) answers the question: for a fixed training-compute budget C, what parameter count N and token count D minimize training loss? The empirical answer is roughly N and D should scale together, with D/N landing near 20 tokens-per-parameter at compute-optimal allocation for the loss-vs-compute frontier Chinchilla fit. Under that heuristic, an 8B model's compute-optimal token count would be on the order of 150-200B tokens -- two orders of magnitude below the 15T+ tokens Llama 3's 8B model is actually trained on.

The reason this is not a mistake is that Chinchilla optimality is a statement about *training* compute only. It says nothing about what it costs to *serve* the resulting model. A model that will be queried billions of times over a multi-year deployment lifetime incurs an inference-compute bill that, in aggregate, can dwarf the one-time training-compute bill -- and inference cost per query scales with parameter count (roughly linearly, for a dense model, ignoring cache effects), not with how many tokens the model happened to be trained on. Given a fixed target quality bar, the total-cost-of-ownership-minimizing strategy is therefore not "train the Chinchilla-optimal model for that quality level" but "find the *smallest* model that reaches that quality level, even if that means training it on far more tokens than compute-optimal, because extra pretraining compute is a one-time cost while extra parameters are a recurring cost paid on every future inference call." This is precisely why Llama 3's 8B model continues to improve measurably even after 15T tokens -- Meta's own reported loss curves show log-linear improvement continuing well past the point a Chinchilla-optimal recipe would have stopped training an 8B model, and the paper explicitly frames the decision to keep training as informed by observing that the smaller models had not saturated. The 405B model, deliberately sized close to what the available compute and desired training duration could support, is the one member of the family trained closest to compute-optimal for its own size -- reinforcing that the "overtraining" strategy is specifically a smaller-model-for-cheaper-inference decision, not a blanket policy applied at every size.

### 6. Post-Training / Alignment Approach

Llama 3's post-training pipeline evolves from Llama 2's PPO-centric RLHF toward a pipeline built around **SFT, rejection sampling, and Direct Preference Optimization (DPO)**, run over multiple rounds, rather than PPO. The paper states this shift was made because DPO-based post-training was found to be simpler to tune and scale reliably across model sizes than online PPO, while still achieving comparable or better alignment quality -- consistent with the broader field's move away from PPO-based RLHF toward direct preference-optimization methods over 2023-2024.

The pipeline: collect human and model-generated preference data; train reward model(s); use rejection sampling (sample multiple completions per prompt from the current SFT/DPO model, filter/rank with the reward model, keep the best as new SFT targets); apply DPO directly on preference pairs; iterate this loop across several rounds, progressively improving both the policy and the quality of the data used to train it. Synthetic data generation plays a larger explicit role than in Llama 2 -- using the model itself (and specialized fine-tuned variants of it) to generate and self-filter training data for specific skills (coding, reasoning, tool use, long-context instruction following), then validating with execution feedback where applicable (e.g., code that is checked by actually running it) rather than relying purely on human or reward-model judgment.

Long-context instruction-following and multilinguality receive dedicated post-training attention distinct from Llama 2's process, reflecting the new 128K context and expanded language coverage introduced in Llama 3.1.

### 7. Key Research Contributions & Novel Techniques

1. **Universal GQA as a default rather than a largest-model-only optimization** -- validating and generalizing the pattern Llama 2 only applied at 70B.
2. **The overtraining/inference-cost-dominance argument made explicit and executed at extreme scale (15T+ tokens for an 8B model)** -- the clearest, most quantitatively extreme public demonstration of "compute-optimal training and inference-optimal deployment are different optimization problems," and arguably the paper's single most cited strategic insight in industry discussion.
3. **4D parallelism (data + tensor + pipeline + context parallelism) disclosed for training at 128K context on a 405B model** -- context parallelism specifically as a response to per-sequence activation memory becoming the binding constraint at very long context, rather than parameter count alone.
4. **Data annealing** -- deliberately up-weighting a small, high-quality data mixture in the final phase of pretraining, reported to measurably improve downstream benchmark scores relative to a flat data mixture through to the end of training.
5. **Shift from PPO-centric to DPO-centric post-training** at frontier open-model scale, alongside heavy reliance on model-generated, execution-validated synthetic training data for skills like code and tool use.
6. **128K context extension via staged continued pretraining with RoPE base-frequency rescaling**, applied uniformly across all three sizes after the initial shorter-context pretraining run.

### 8. Training Challenges & Engineering Solutions

Training on 15T+ tokens creates engineering pressure at every layer of the stack that a 1-2T-token run does not:

- **Data pipeline throughput becomes a first-class systems problem.** Feeding a cluster of up to 16,000 H100s a continuous stream of well-shuffled, deduplicated, quality-filtered tokens at the throughput those GPUs can consume requires the data pipeline itself (storage I/O, deduplication, filtering, tokenization) to be engineered as carefully as the model-parallel training code -- becoming I/O-bound on the data pipeline at this token count is a real risk the paper's infrastructure discussion addresses.
- **Failure rate over a run this long becomes a dominant scheduling concern.** A training run spanning enough wall-clock time to process 15T+ tokens (even at high hardware utilization) will, at a 16K-GPU scale, encounter a non-trivial number of individual GPU/node failures purely due to hardware MTBF statistics; the paper discloses building automated failure-detection and fast-restart tooling as a necessity, not a nice-to-have, and reports the observed failure/interruption rate over the 405B run.
- **Checkpointing cost and cadence** at 405B-parameter scale (optimizer state alone, in mixed precision with Adam's two moment buffers, is several times the raw parameter memory) must be balanced against the wall-clock cost of checkpointing itself; frequent checkpointing protects against failures but adds overhead, a tradeoff the paper's infrastructure section addresses via its checkpointing/fault-tolerance system design.
- **Long-context training's quadratic-in-sequence-length activation memory** at 128K tokens is the specific reason context parallelism (splitting a single sequence's attention computation across GPUs) was needed in addition to the data/tensor/pipeline parallelism that sufficed for 8K-context Llama 2 training.

Rather than being solved by any single trick, the 15T-token/128K-context/405B-parameter regime is disclosed as requiring simultaneous engineering investment across the data pipeline, the parallelism strategy, and the fault-tolerance/checkpointing system -- the paper's own framing is that infrastructure engineering at this scale is now inseparable from the "modeling" contribution.

### 9. Inference & Serving Considerations

The 128K-token expanded vocabulary directly reduces inference cost per unit of "real" text processed or generated -- fewer tokens for the same content means fewer forward passes for prefill and fewer decode steps for generation of a fixed amount of output, at the cost of a larger (but still comparatively small relative to total model size, except at 8B) embedding/unembedding matrix.

Universal GQA (8 KV heads at every size) keeps the KV-cache-bandwidth cost bounded even as context is pushed to 128K -- without it, 405B's KV cache at 128K context with full MHA (128 heads) would be enormous; with 8 KV heads it remains the same absolute per-token cost regardless of query-head count, which is precisely the point of applying GQA universally rather than only at the top size.

405B specifically raises weight-memory serving cost as the dominant constraint (over 800GB of parameters at fp16, before KV cache), which typically requires multi-GPU tensor-parallel serving regardless of context length -- a genuinely different serving regime from 8B/70B, which can be served on a single high-memory GPU or a small number of GPUs respectively. This is exactly the cost 8B's "overtrained" strategy is designed to let most deployments avoid: a much cheaper-to-serve model reaching a meaningfully higher quality bar than its parameter count would suggest under compute-optimal training.

### 10. Evaluation, Benchmarks & Known Limitations

Reported results (Llama 3.1, from the paper, selective): Llama-3.1-405B is reported as competitive with GPT-4-class closed models on a broad benchmark suite (including MMLU, GSM8K/MATH-style reasoning benchmarks, code benchmarks like HumanEval, and multilingual benchmarks), marking it as the first Llama generation explicitly positioned by Meta as competitive with the best closed frontier models rather than only with other open models. Llama-3.1-70B and 8B show substantial gains over their Llama 2 counterparts of the same size, consistent with the combination of far more pretraining tokens, the larger tokenizer, and the improved post-training pipeline.

Long-context evaluation (needle-in-a-haystack-style retrieval tests and long-document benchmarks) is reported specifically to validate the 128K context extension, since simply supporting a long input length architecturally does not guarantee the model can actually use information placed anywhere within that window.

Known limitations disclosed or evident: despite the overtraining strategy, 405B still trails the very best closed frontier models on some benchmarks at release; multilingual performance, while improved, is not uniform across all covered languages; and the paper is candid that further scaling of both data and parameters was still yielding gains at the point training was stopped for 405B, implying the model was not trained to a point of clear diminishing returns, i.e., a larger/longer-trained model would likely have scored higher still (a standard caveat for any frontier-scale release).

### 11. Confirmed Facts vs. Speculation

This is an open-weight model family with a detailed accompanying paper ("The Llama 3 Herd of Models"), so most figures above (token counts, GPU counts, FLOPs estimate, GQA configuration, tokenizer size, context lengths) are directly disclosed. This section is accordingly short. Remaining ambiguity:

- The exact per-stage token/data-mixture breakdown (how many of the 15T+ tokens are code vs. multilingual vs. general web text, stage by stage) is described qualitatively rather than as a fully itemized table comparable to LLaMA 1's per-source percentage table.
- The precise parallelism-degree grid (exact TP/PP/DP/CP dimensions used at each stage of the 405B run) is discussed at a high level rather than published as a complete reproducible configuration.
- The exact GPU-failure/interruption rate and specific fault-tolerance implementation details are disclosed at a summary level (the paper discusses the phenomenon and the general mitigation) rather than as an exhaustive incident log.
- "15T+ tokens" is the disclosed figure for the initial Llama 3 release; the exact total token count used across the full Llama 3.1 training + continued-pretraining + long-context-extension pipeline for 405B specifically is not broken out as a single clean number distinct from the 15T+ headline figure.

### 12. Staff/Research Interview Talking Points

- Be able to state the Chinchilla compute-optimal heuristic (roughly 20 tokens/parameter at compute-optimal allocation) and then immediately explain why Llama 3's 8B model trained on 15T+ tokens (75-100x past that ratio) is not a contradiction of Chinchilla but an optimization of a *different* objective -- inference-amortized total cost of ownership, not training FLOPs for a target loss. This is one of the single most common "do you actually understand scaling laws" staff-interview checks in 2024-2025.
- Be ready to reason quantitatively: if a model will be queried N times over its deployment lifetime, the total cost is (training FLOPs cost) + N * (inference FLOPs cost per query), and inference cost per query scales with parameters, not training tokens -- so for large N, minimizing parameters at a fixed quality bar dominates the total-cost calculation even at the price of a much larger one-time training-token bill.
- Know why GQA became universal here rather than staying 70B-only: as context length grows (Llama 3 pushes to 128K), KV-cache bandwidth cost grows with it at every model size, not just the largest, so the case for GQA strengthens even at 8B once long context is a design goal.
- Understand why context parallelism specifically (as opposed to just more tensor/pipeline parallelism) becomes necessary at 128K-token training -- per-sequence activation memory, not just parameter memory, is the new binding constraint at very long context.
- Be able to explain, at a mechanism level, why a 4x larger vocabulary (128K vs. 32K) reduces inference cost independent of any other model change -- fewer tokens per unit of text means fewer decode steps and shorter attention sequences for the same content, a "free" efficiency lever paid for once in a slightly larger embedding/unembedding matrix.
