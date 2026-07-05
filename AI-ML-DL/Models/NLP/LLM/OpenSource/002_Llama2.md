## Llama 2 (2023)

### 1. Overview & Strategic Context

Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Meta AI, July 2023). Llama 2 is the direct successor to LLaMA 1, with three headline changes: a larger and cleaner pretraining corpus (2T tokens, ~40% more than LLaMA 1's ~1.4T), a doubled context length (4096 vs. 2048), and -- for the first time from Meta -- fully documented, RLHF'd chat variants (Llama-2-Chat) released alongside the base models under a license permissive enough for commercial use (subject to a monthly-active-user threshold clause). This last point is the strategic pivot: LLaMA 1 was a research-license base-model-only release that the community had to instruction-tune itself; Llama 2 is an end-to-end product release competing directly with closed chat APIs, with a 70-page paper documenting the RLHF recipe in a level of detail that was, at the time, unprecedented for an open release.

Architecturally Llama 2 is conservative relative to LLaMA 1 -- same RMSNorm/RoPE/SwiGLU block -- with exactly one structural change: grouped-query attention (GQA), and only in the 70B model. The 7B and 13B models keep standard multi-head attention. This is the paper's own explicit engineering tradeoff, and it is worth understanding precisely why: GQA trades a small amount of quality (fewer distinct KV projections) for a large reduction in KV-cache memory and memory-bandwidth pressure during autoregressive decoding. At 7B/13B, the per-token KV cache is already small enough that MHA's cost is tolerable; at 70B, with far more heads and far larger KV cache per token, the memory-bandwidth savings from GQA become large enough to matter for real-world serving economics, so Meta applied it only there. This is the first data point in what becomes, by Llama 3, a universal design decision -- Llama 2 treats GQA as a cost-saving measure justified only at the most inference-expensive scale, not yet as a default.

### 2. Architecture Deep Dive

Base block is unchanged from LLaMA 1: pre-normalization with RMSNorm, RoPE for position information, SwiGLU-gated FFN with hidden dimension ~(2/3)*4*dim rounded to a configured multiple. Context length is doubled to 4096 tokens (all sizes), which also required recomputing the RoPE frequency table over the longer range (base theta = 10000, unchanged from LLaMA 1).

**Grouped-Query Attention (GQA), 70B only.** GQA (Ainslie et al., 2023, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints") sits between standard multi-head attention (MHA, one K/V head per Q head) and multi-query attention (MQA, a single shared K/V head for all Q heads, Shazeer 2019). GQA partitions the n_heads query heads into n_kv_heads groups, and every query head within a group shares one K and one V projection:

- Query heads: 64, per-head dim 128, hidden dim 8192 (unchanged 70B config from LLaMA 1's 65B analog, scaled)
- KV heads: 8 (i.e., groups of 64/8 = 8 query heads share each KV head)
- MHA (n_kv_heads = n_heads = 64) is the n_kv_heads=64 special case; MQA (n_kv_heads = 1) is the other extreme; Llama-2-70B's 8 is a deliberate middle point.

The KV cache for a given sequence scales as 2 * n_layers * n_kv_heads * head_dim * seq_len (K and V), so reducing n_kv_heads from 64 to 8 is an 8x reduction in per-token KV-cache memory and, more importantly for decode-time throughput, an 8x reduction in the bytes that must be streamed from HBM per decoding step -- since autoregressive decoding is memory-bandwidth-bound on the KV cache, not compute-bound, this directly increases achievable batch size and decode throughput at serving time. The 7B and 13B models retain n_kv_heads = n_heads (standard MHA), since at those sizes the KV-cache-bandwidth bottleneck is much less severe relative to the compute cost of the rest of the forward pass.

**Tokenizer:** unchanged from LLaMA 1 -- same SentencePiece BPE, 32,000-token vocabulary, byte-level fallback, digit-splitting.

**Context length:** 4096 tokens for all sizes (2x LLaMA 1's 2048).

Per-size configuration:

| Size | dim | n_heads | n_kv_heads | n_layers | Context |
|---|---|---|---|---|---|
| 7B | 4096 | 32 | 32 (MHA) | 32 | 4096 |
| 13B | 5120 | 40 | 40 (MHA) | 40 | 4096 |
| 34B* | 6656 | 52 | 8 (GQA) | 60 | 4096 |
| 70B | 8192 | 64 | 8 (GQA) | 80 | 4096 |

*A 34B model was trained and is described in the paper's tables but was not publicly released at the time -- Meta stated it was withheld pending further safety work, though it later informed CodeLlama-34B. It is included here because the paper discusses it and because it shows GQA was applied to 34B as well as 70B, i.e., the cutoff for "GQA vs. MHA" in Llama 2 is size-based, not "70B only" in the strictest sense -- but among *released* Llama 2 models, only 70B uses GQA.

### 3. Scale -- Parameters, Data, Compute

Released sizes: 7B, 13B, 70B (34B trained but not released). Pretraining tokens: 2.0T for all released sizes -- about 40% more than LLaMA 1's largest-model token count (1.4T), from a new pretraining mixture that up-weights the most factual sources and adds new sources not detailed at the same granularity as LLaMA 1's mixture (Meta disclosed proportions less precisely for Llama 2, citing safety/privacy review; a small amount of additional data was excluded).

Disclosed compute: 3,311,616 total GPU-hours across all Llama 2 pretraining (all sizes combined), on A100-80GB hardware, all training done on Meta's Research Super Cluster (RSC) and internal production clusters. The 70B model alone accounts for the majority of that -- the paper's per-model GPU-hours table reports roughly 1.7M GPU-hours for 70B, versus roughly 184K for 7B and 368K for 13B, with 34B (unreleased) at roughly 1.04M. Estimated carbon emissions are also disclosed per model in the paper (a first for a Meta open release at this level of granularity).

RLHF-specific scale: over 1 million human preference comparisons were collected across the iterative fine-tuning rounds (helpfulness + safety combined), a substantially larger human-annotation investment than typical academic RLHF work at the time and the largest disclosed for an open-weight chat model up to that point.

### 4. Training Infrastructure & Distributed Training

Pretraining infrastructure is essentially the same class of setup as LLaMA 1 -- A100-80GB GPUs, standard data + tensor (model) parallelism for the larger sizes -- run on Meta's Research Super Cluster and internal production GPU clusters. The paper does not introduce a new named distributed-training framework; the systems contribution of the Llama 2 paper is concentrated in the *post-training* pipeline rather than pretraining infrastructure.

The RLHF pipeline itself is a multi-stage, iterative process requiring its own infrastructure:

1. SFT on a curated set of ~27,540 high-quality instruction-response pairs (Meta found that a smaller set of very high quality annotations outperformed larger, noisier third-party instruction datasets -- a specific, disclosed finding).
2. Two separate reward models trained from human preference comparisons: a **helpfulness reward model** and a **safety reward model**, each initialized from the pretrained base checkpoint and trained as a binary/margin-ranking classifier over pairs of model outputs. Keeping the two objectives as separate reward models (rather than one blended reward) was a deliberate choice to avoid one objective (typically safety) being drowned out by the other during optimization.
3. Iterative RLHF over five successive versions (RLHF-V1 through RLHF-V5), alternating between **rejection sampling fine-tuning** (sample K outputs per prompt from the current policy, score with the reward model, fine-tune on the best-scoring one -- used for the largest models, 70B, and only in later rounds for smaller models) and **PPO** (standard on-policy RL update against the reward model signal, applied on top of the rejection-sampling-tuned checkpoint in later iterations).
4. **Ghost Attention (GAtt)**, a synthetic-data-based fine-tuning technique to fix multi-turn instruction "forgetting" -- system-prompt instructions given at turn 1 are concatenated to all user turns during a synthetic-data construction pass, then the loss on that synthetic concatenation is zeroed out for all but the final turn, teaching the model to keep obeying an early instruction across a multi-turn conversation without needing it repeated. This is a data-construction / loss-masking trick, not an architectural change.

Human annotation pipeline: comparisons were collected continuously throughout the iterative RLHF loop specifically to keep the reward model's training distribution matched to the current policy's output distribution -- a "reward model / policy co-evolution" concern the paper discusses explicitly as important for RLHF stability.

### 5. Pretraining Data & Objective

Standard causal LM objective, unchanged from LLaMA 1. Data mixture: 2T tokens from "publicly available sources," with the paper stating that no Meta user data was included and that sources known to contain a high volume of personal information about private individuals were up-weighted down or removed. Unlike LLaMA 1, the paper does not give the same fine-grained per-source percentage table (CommonCrawl/C4/GitHub/etc. breakdown) -- this is a disclosed reduction in mixture transparency relative to LLaMA 1, attributed to expanded safety/privacy review of the sources. The most factual sources were up-sampled in the new mixture based on findings from LLaMA 1's evaluation (e.g., under-representation of high-quality knowledge sources was one hypothesis for LLaMA 1's relative MMLU weakness).

### 6. Post-Training / Alignment Approach

This is the section where Llama 2 differs most sharply from LLaMA 1: a fully documented, multi-stage alignment pipeline shipped as first-class released artifacts (Llama-2-Chat-7B/13B/70B).

**Stage 1 -- Supervised fine-tuning (SFT).** ~27,540 curated prompt-response pairs, mixing a small amount of vendor-sourced annotations with Meta-internal high-quality examples; the paper's specific finding is that annotation *quality* mattered far more than *quantity* at this stage, and that continuing to add third-party SFT data beyond a point stopped improving (and sometimes hurt) output quality.

**Stage 2 -- Reward modeling.** Two reward models, helpfulness and safety, each trained on separate human-preference comparison datasets (over 1M binary comparisons total across the whole RLHF process). Each RM outputs a scalar score; the training loss is a margin-ranking loss over the human-labeled preference pairs, with an explicit margin term scaled by how large the human-labeled preference gap is ("significantly better," "better," "slightly better," "negligibly better," reflecting graded rather than purely binary preference labels).

**Stage 3 -- Iterative RLHF.** Five rounds (RLHF-V1 to RLHF-V5). Each round retrains the reward models on freshly collected comparisons from the current policy's outputs, then updates the policy via rejection-sampling fine-tuning and/or PPO against the (updated) reward models. For the largest model (70B), rejection sampling with a large number of samples (the paper explores K up to 30) is used before PPO is layered on top, motivated by empirical evidence that PPO with too few candidate samples is less sample-efficient at improving output quality than picking the best of many rejection samples first, then polishing with PPO.

**Ghost Attention (GAtt).** Addresses multi-turn instruction adherence without any architectural change -- a synthetic-data trick applied during the fine-tuning data construction, described in Section 4 above.

**Safety-specific measures:** context distillation for safety (generating safety-conscious responses using a safety-focused system prompt, then fine-tuning without that prompt so the safety behavior becomes "baked in" rather than dependent on the prompt), extensive red-teaming (over 350 person-hours of red-teaming disclosed), and a separate safety-focused RLHF track using the safety reward model, with an explicit "safety tax" analysis quantifying how much helpfulness score is traded off for safety gains at each iteration.

### 7. Key Research Contributions & Novel Techniques

1. **The most detailed publicly documented RLHF recipe for an open-weight model at the time** -- the paper's process disclosure (dual reward models, iterative rejection-sampling + PPO, margin-ranking loss with graded preference strength, reward-model/policy co-evolution) is itself the primary research contribution, more so than any single novel algorithm.
2. **Ghost Attention (GAtt)** -- a lightweight, purely data/loss-masking-based fix for multi-turn system-prompt adherence, notable for solving an architectural-feeling problem (the model "forgets" instructions across turns) without touching the architecture.
3. **GQA adopted for the first time in the Llama lineage, but scoped only to the largest released size** -- establishing the empirical pattern (validated further in Llama 3) that GQA's quality cost is small and its serving-cost benefit is large, setting up the case for making it universal one generation later.
4. **Explicit "helpfulness vs. safety" reward-model separation with quantified safety-tax tradeoff curves** -- rather than a single blended reward, which the paper argues would let one objective dominate silently.

### 8. Training Challenges & Engineering Solutions

The paper documents **reward hacking / distributional drift** as the central RLHF engineering challenge: as the policy improves, the outputs it produces drift away from the distribution the reward model was originally trained on, causing the RM's scores to become miscalibrated (reward over-optimization). The disclosed solution is the iterative loop itself -- continuously collecting fresh human preference data on the *current* policy's outputs and periodically retraining the reward models, rather than training a single static reward model and optimizing against it indefinitely. This is explicitly framed in the paper as necessary for RLHF stability at this scale of iteration (five rounds).

A second documented challenge is the **safety/helpfulness tension**: naive RLHF against a single blended objective tends to push the model toward one extreme (typically over-refusal for safety-tuned single-objective setups). Maintaining two separate reward models and separately tunable weighting between them is the disclosed mitigation, along with targeted safety-specific data collection (adversarial/red-team prompts) rather than relying on the general helpfulness preference data to also carry safety signal.

At the pretraining level, no major instability is reported beyond what LLaMA 1 already handled at smaller scale -- the 40% token-count increase and longer 4096 context did not require new stabilization techniques beyond the existing recipe.

### 9. Inference & Serving Considerations

The 70B GQA configuration is the direct, disclosed serving-cost motivation: an 8x reduction in KV-cache size and KV-cache memory-bandwidth at decode time relative to full MHA, letting 70B be served with meaningfully larger batch sizes and higher throughput per GPU than a hypothetical MHA-70B would allow. The 7B/13B models, still using MHA, do not get this benefit but also do not need it as urgently -- their absolute KV-cache size is much smaller in the first place (fewer heads, smaller hidden dim), so the memory-bandwidth bottleneck is less binding.

Doubling context length to 4096 also doubles the KV cache's linear-in-sequence-length term for a given model, which is precisely the kind of pressure that makes GQA's constant-factor savings matter more as context grows -- a dynamic that becomes even more pronounced once Llama 3 pushes context to 128K and, for that reason, makes GQA universal rather than 70B-only.

### 10. Evaluation, Benchmarks & Known Limitations

Selected reported results (base models, from the paper's benchmark tables): Llama-2-70B outperforms LLaMA 1-65B and all open models the authors compared against at the time (PaLM-540B is close on some benchmarks; Llama-2-70B trails GPT-3.5/GPT-4 and PaLM-2-L on most academic benchmarks, which the paper states explicitly). On MMLU, Llama-2-70B improves substantially over LLaMA-65B, closing much of the gap the LLaMA 1 paper flagged.

Llama-2-Chat evaluation: human evaluation (not just academic benchmarks) comparing Llama-2-Chat-70B against ChatGPT (GPT-3.5) on helpfulness -- the paper reports Llama-2-Chat-70B winning or tying against ChatGPT in a majority of human-judged single-turn and multi-turn prompts in their internal evaluation set, a headline claim of the paper (with the usual caveats about human-eval prompt-set composition and annotator variance that the paper itself flags).

Safety evaluation: lower violation rate on the paper's safety benchmark set relative to Llama-2-Chat's predecessor comparisons and relative to some other open chat models evaluated at the time, attributed to the dedicated safety RLHF track and context distillation.

Known limitations disclosed in the paper: knowledge cutoff limits factual recency; 4096-token context is still short relative to some contemporaneous closed models; the model can still produce unsafe or biased content despite safety tuning (the paper explicitly does not claim safety tuning eliminates the issue, only reduces its rate); and the paper flags that human-evaluation results are sensitive to the specific prompt set and annotator guidelines used, urging caution in over-generalizing the ChatGPT-comparison headline number.

### 11. Confirmed Facts vs. Speculation

This is an open-weight model with an unusually detailed accompanying paper (70 pages, including the RLHF process, reward model training details, GQA ablations, and safety evaluation), so the great majority of this document is directly confirmed. Remaining points of genuine ambiguity:

- The exact per-source percentage breakdown of the 2T-token pretraining mixture is *not* disclosed at the same granularity as LLaMA 1's table -- the paper describes the mixture qualitatively (more factual sources up-weighted, privacy-sensitive sources down-weighted/removed) without exact percentages. Any specific percentage breakdown claim for Llama 2's pretraining mixture beyond "2T tokens, ~40% more than LLaMA 1" should be treated as unconfirmed.
- 34B was trained (GPU-hours and benchmark numbers are in the paper's tables) but never publicly released; Meta's stated reason was insufficient time to complete their safety review before the paper's release window, not a claim about the model's quality.
- Exact reward-model architecture details (e.g., whether the RM head is a single scalar linear layer on top of the base transformer, precise margin-loss hyperparameters per round) are given but not with full training-hyperparameter tables for every one of the five RLHF iterations -- some iteration-specific hyperparameters are described qualitatively/in aggregate rather than exhaustively tabulated.

### 12. Staff/Research Interview Talking Points

- Be able to explain GQA's mechanism precisely (query heads grouped, each group sharing one K/V head) and its KV-cache-size and memory-bandwidth math, and be ready for the follow-up "why only 70B, not 7B/13B" -- the answer is decode-time memory bandwidth on the KV cache is the bottleneck being solved, and that bottleneck's absolute severity scales with n_heads * head_dim, which is much larger at 70B.
- Know that GQA (Ainslie et al., 2023) is positioned as an interpolation between MHA (n_kv_heads = n_heads) and MQA (n_kv_heads = 1, Shazeer 2019) -- and be able to state why MQA alone was seen as too aggressive a quality tradeoff for a flagship-scale model, motivating the middle-ground choice of 8 KV heads.
- Understand the significance of *dual* reward models (helpfulness, safety) versus a single blended reward -- this is a recurring theme in real production RLHF systems and a strong interview signal if you can explain why blending them risks one objective dominating silently during optimization.
