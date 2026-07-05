## DeepSeek-V2 (2024)

### 1. Overview & Strategic Context

DeepSeek-V2 (released May 2024, technical report "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model") is a 236B-total / 21B-activated-per-token Mixture-of-Experts (MoE) language model. It sits between DeepSeek's earlier dense/MoE experiments (DeepSeek-LLM, DeepSeekMoE) and the much larger DeepSeek-V3, and it is the paper that introduces **Multi-head Latent Attention (MLA)** — the single most consequential architectural idea DeepSeek contributed to the open-model ecosystem, later reused unchanged in V3 and R1.

The strategic framing in the paper is explicit: V2 is positioned as "economical" not just in training FLOPs (a fine-grained MoE keeps activated parameters at ~21B against a 236B pool) but in **inference serving cost**, and specifically in KV-cache footprint at long context. By 2024 it was clear that for many production LLM deployments the binding constraint on throughput is not FLOPs but HBM capacity consumed by the KV cache — every concurrent request at long context holds a cache proportional to (num_layers × seq_len × num_heads × head_dim × 2) bytes, and once that exceeds available memory, batch size (and therefore throughput and $/token) collapses. GQA (Grouped-Query Attention) attacks this by sharing K/V heads across groups of Q heads, trading quality for cache size. MLA takes a structurally different approach: compress the *entire* per-token K/V information into one small shared low-rank latent vector, cache only that, and reconstruct full per-head K/V from it on the fly at attention time. This gets a cache footprint smaller than aggressive GQA while empirically matching or exceeding standard MHA quality — the paper reports V2's KV cache per token is comparable to GQA with only 2.25 groups, at MHA-level or better benchmark performance.

V2 also carries forward **DeepSeekMoE** (fine-grained expert segmentation + shared-expert isolation, from DeepSeek's earlier MoE paper) and introduces **device-limited routing** to bound the all-to-all communication cost of expert parallelism — a precursor to the communication engineering that becomes central in V3.

### 2. Architecture Deep Dive

**Backbone.** 60 transformer layers, hidden dimension d = 5120, SwiGLU FFN in dense layers, RMSNorm, RoPE for positional encoding, context length extended to 128K via YaRN. The first layer uses a dense FFN; all subsequent layers use the DeepSeekMoE block. Attention in every layer is MLA.

**Multi-head Latent Attention (MLA) — the core mechanism.**

Standard MHA computes, per token t, per-head queries/keys/values of dimension d_h from the residual stream h_t ∈ R^d, and caches every head's K and V — cache size per token = 2 × n_h × d_h. GQA reduces this by sharing K/V across groups of heads, cache size = 2 × n_g × d_h for n_g groups (n_g ≪ n_h).

MLA instead **compresses the key/value information into a single shared low-rank latent vector before it is ever split into heads**, and defers the "split into heads" step to attention time via learned up-projections:

- Down-projection (KV path): `c_t^{KV} = W^{DKV} h_t`, where `c_t^{KV} ∈ R^{d_c}` is the compressed latent, with d_c ≪ n_h·d_h (V2 uses d_c = 512 against an effective n_h·d_h = 128×128 = 16384 — a ~32× compression of the naive per-head K/V dimensionality).
- Up-projection to reconstruct content keys: `k_t^{C} = W^{UK} c_t^{KV}`, `k_t^{C} ∈ R^{n_h·d_h}`, reshaped into n_h per-head keys.
- Up-projection to reconstruct values: `v_t^{C} = W^{UV} c_t^{KV}`, reshaped into n_h per-head values.
- **Only `c_t^{KV}` is cached** — the up-projection matrices W^{UK}, W^{UV} are model weights (not per-token state), so reconstruction at every future attention step is a matmul against cached activations, not a memory-bound cache read of full per-head tensors.

**The RoPE complication and the decoupled-RoPE fix.** RoPE is a position-dependent rotation applied per-head to queries/keys before the dot product. If you rotate the compressed latent directly, the rotation does not commute cleanly with the up-projection (rotating before vs. after up-projecting are not equivalent), which would break the ability to fold W^{UK} into the query-side projection at inference time for a matmul-absorption trick. DeepSeek-V2's fix is to **decouple a small extra RoPE-carrying key/query from the compressed content path**:

- A separate, small shared "rotary key" is produced directly from h_t (not through the latent bottleneck): `k_t^{R} = RoPE(W^{KR} h_t)`, of dimension d_h^R = 64, shared across all heads (only one rotary key per token, not one per head).
- The query side is compressed too (for activation-memory savings during training, not caching, since queries are never cached): `c_t^{Q} = W^{DQ} h_t` (d_c' = 1536), then `q_t^{C} = W^{UQ} c_t^{Q}` (content part, per head) and `q_t^{R} = RoPE(W^{QR} c_t^{Q})` (rotary part, per head, dimension d_h^R).
- Final per-head key and query are concatenations of content and rotary parts: `k_t = [k_t^{C}; k_t^{R}]`, `q_t = [q_t^{C}; q_t^{R}]`. Attention scores are the usual scaled dot product `q_t^T k_t / sqrt(d_h + d_h^R)`.
- Values use only the content path: `v_t = v_t^{C}`.

**What actually gets cached, per token:** `c_t^{KV}` (d_c = 512 values) plus `k_t^{R}` (d_h^R = 64 values, shared across heads) = 576 scalars per token per layer — versus 2 × 128 × 128 = 32768 scalars for standard MHA at V2's head count/width. This ~57× reduction is what the paper compares to "GQA with 2.25 groups" in cache-size terms, while V2's benchmark quality tracks full MHA rather than degraded GQA.

**DeepSeekMoE FFN.** Each MoE layer has N_s = 2 always-active **shared experts** (capturing common/general knowledge so routed experts don't have to redundantly relearn it) plus N_r = 160 **routed experts**, of which top-k = 6 are activated per token via a softmax-gated router. Experts are deliberately fine-grained (smaller intermediate size per expert, e.g. 1536, versus a coarse handful of large experts) so that the combinatorial space of {shared experts} ∪ {6-of-160 routed experts} gives much finer per-token specialization than an 8-way coarse MoE would. **Device-limited routing** additionally caps each token's routed experts to reside on at most M distinct devices, bounding the all-to-all communication fan-out that expert parallelism would otherwise incur.

### 3. Scale — Parameters, Data, Compute

- Total parameters: 236B. Activated parameters per token: 21B (~8.9% of total) — the combination of 2 shared experts + 6-of-160 routed experts + attention/norm parameters.
- Training data: 8.1T tokens, a multilingual (English/Chinese-heavy) corpus with an emphasis on math and code, built by DeepSeek's own data pipeline (details of exact composition are not as fully disclosed as V3's).
- Context length: pretrained at 4K, then extended to 128K via YaRN-based RoPE scaling in a continued-training stage.
- MLA dimensions: hidden d = 5120, n_h = 128 heads, d_h = 128 per-head dim, d_c = 512 (KV compression dim), d_c' = 1536 (query compression dim), d_h^R = 64 (decoupled RoPE dim).
- MoE dimensions: N_r = 160 routed experts, N_s = 2 shared experts, top-k = 6 routed experts/token, expert FFN intermediate size 1536, 59 MoE layers + 1 dense first layer.

### 4. Training Infrastructure & Distributed Training

V2 trains with a combination of pipeline parallelism, expert parallelism (necessary once expert count reaches 160 — no single device holds all experts), and ZeRO-style data-parallel optimizer sharding. Expert parallelism's core cost is the all-to-all dispatch/combine communication needed to route each token's activation to the (possibly remote) devices holding its selected experts and route the outputs back; **device-limited routing** (each token's 6 selected experts constrained to at most M devices) is specifically an infrastructure-driven architectural choice — it trades a small amount of routing freedom for a bounded, predictable communication volume, which matters enormously when a training step's wall-clock time is communication-bound rather than compute-bound at this expert count. DeepSeek reports V2 achieves higher training efficiency (in tokens/GPU-day terms) than a dense model of comparable capability, and lower KV-cache memory pressure translates directly into inference throughput gains (larger feasible batch size per GPU at a given context length).

### 5. Pretraining Data & Objective

Standard autoregressive next-token prediction over the 8.1T-token corpus. The tokenizer is a BBPE (byte-level BPE) vocabulary sized for strong English/Chinese/code coverage. The report emphasizes data quality filtering and deduplication pipelines carried over/extended from DeepSeek's earlier LLM work, with increased representation of math and code relative to typical web-text-dominated corpora of the period. Exact data-mixture percentages are not disclosed at the level of detail V3 later provides.

### 6. Post-Training / Alignment Approach

DeepSeek-V2-Chat is produced via a conventional two-stage pipeline: supervised fine-tuning (SFT) on instruction data, followed by reinforcement learning — DeepSeek's Group Relative Policy Optimization (GRPO) is introduced in this line of work (used for V2's RL stage) as a critic-free alternative to PPO, predating and setting up the much larger-scale RLVR application in R1. At the V2 stage, the RL reward is a more conventional learned/preference-based reward model (RLHF-style) rather than the verifiable-reward regime R1 later exploits; GRPO's group-baseline mechanic (see the R1 document for the full derivation) is applied here mainly for RL training efficiency, not yet for reasoning-specific verifiable rewards.

### 7. Key Research Contributions & Novel Techniques

1. **Multi-head Latent Attention (MLA).** The headline contribution: compress K/V into a shared low-rank latent cached once per token instead of per-head, reconstruct per-head K/V via learned up-projections at attention time, and decouple a small shared RoPE component so the compression is compatible with rotary position encoding. This is the first attention variant to beat GQA on cache size *and* MHA on quality simultaneously (per the paper's own ablations), rather than trading one for the other.
2. **DeepSeekMoE fine-grained expert design carried to production scale** — many small experts plus isolated always-on shared experts, at 160 routed experts, is finer-grained than the 8-or-16-expert MoEs common at the time (e.g., Mixtral's 8 experts).
3. **Device-limited routing** as a communication-aware architectural constraint baked into the router itself, rather than purely a systems-level mitigation.
4. **YaRN-based long-context extension to 128K** applied post-hoc to a model pretrained at much shorter context, validating that MLA's cache savings compound with long-context serving (the whole point of the cache-efficiency argument only matters once context is actually long).

### 8. Training Challenges & Engineering Solutions

- **Load imbalance across 160 routed experts** is addressed with auxiliary load-balancing losses at this stage (V2 still uses explicit auxiliary losses — expert-level, device-level, and communication-balance losses — unlike V3's later bias-based, loss-free approach). Multiple loss terms are needed because imbalance can occur at the expert level and independently at the device level under device-limited routing.
- **RoPE incompatibility with low-rank KV compression** is the central technical obstacle MLA had to solve; the decoupled shared-RoPE-key design is the engineering answer, and it is what allows the large up-projection matrices to be algebraically absorbed at inference time (see Section 9) rather than materialized as an extra step.
- **Numerical/training stability** at 236B total parameters with a fine-grained 160-expert router is managed with careful initialization and the standard DeepSeekMoE balancing losses.

### 9. Inference & Serving Considerations

This is MLA's raison d'être. Per-token, per-layer KV cache footprint:

- Standard MHA (V2's head config, n_h=128, d_h=128): 2 × 128 × 128 = 32,768 elements.
- GQA at a moderate group count (say 8 groups): 2 × 8 × 128 = 2,048 elements.
- MLA (V2): d_c + d_h^R = 512 + 64 = 576 elements — smaller than even an aggressive GQA configuration, and the paper's own comparison frames it as equivalent-cache to **GQA with 2.25 groups**, a degree of sharing that would normally cost significant quality; MLA pays no such quality tax because the compression is followed by a learned, full-rank-effective up-projection rather than literal head-sharing.

At inference time, the up-projection for keys, W^{UK}, can be algebraically folded into the query projection: since attention score computation is `q_t^T k_t = q_t^T (W^{UK} c_t^{KV})= (W^{UK,T} q_t)^T c_t^{KV}`, the model can precompute `W^{UK,T} W^{UQ}` (or absorb W^{UK} into the effective query weight) so that scores are computed directly against the cached latent `c_t^{KV}` without ever materializing per-head keys — this is the standard "matrix absorption" trick that makes MLA not just cache-efficient but also avoids extra FLOPs for reconstructing K at every decode step. V is reconstructed post-softmax (the output projection can similarly absorb W^{UV}). Net effect: smaller cache → larger feasible batch size at fixed HBM budget → higher decode throughput, particularly pronounced at long context lengths where KV cache (not weights) dominates memory.

### 10. Evaluation, Benchmarks & Known Limitations

DeepSeek-V2 reports competitive or superior results against comparable open MoE and dense models of the period (e.g., LLaMA 3 70B, Mixtral 8x22B) on MMLU, C-Eval, GSM8K, MATH, HumanEval, and Chinese-language benchmarks, while activating only 21B parameters. The paper's efficiency claims are specifically about **training cost vs. a dense model of similar capability** and **inference cost vs. MHA/GQA at similar quality** — both are cache/FLOP arguments rather than claims of absolute SOTA. Known limitations at the time: English-benchmark performance trails the very best dense frontier models of similar or larger active-parameter budgets in some categories; the 128K context extension via YaRN is a post-hoc adaptation rather than native long-context pretraining, which can leave some long-context degradation relative to models trained natively at that length.

### 11. Confirmed Facts vs. Speculation

**Confirmed (from the technical report):** total/activated parameter counts (236B/21B), MLA's down/up-projection structure and the decoupled-RoPE design, N_r=160/N_s=2/top-k=6 MoE configuration, 8.1T training tokens, 128K context via YaRN, device-limited routing, the KV-cache-vs-GQA-2.25-groups comparison, use of auxiliary balancing losses (not yet the bias-based scheme).

**Speculative / not fully disclosed:** exact data mixture ratios (math/code/web/multilingual percentages), full RLHF/GRPO reward-model details for DeepSeek-V2-Chat, exact GPU cluster configuration and total training cost for V2 specifically (V3's report is far more forthcoming on this than V2's).

### 12. Staff/Research Interview Talking Points

- Be able to derive, from first principles, why RoPE and low-rank KV compression are in tension, and reproduce the decoupled-RoPE fix — this is the question most likely to separate "read the abstract" from "understood the mechanism."
- Be able to state precisely *why* KV cache (not parameter count) is the binding constraint at long-context serving: cache scales with batch × seq_len × layers × heads × head_dim, independent of how many parameters are "cold" in HBM; MLA is a direct attack on this specific scaling term.
- Be able to explain the inference-time matrix-absorption trick (folding W^{UK} into the query path) — this is what makes MLA cheap in FLOPs as well as cheap in memory, not merely a compression that shifts cost from memory to compute.
- Contrast MLA with GQA/MQA precisely: GQA reduces cache by *literally sharing* K/V tensors across head groups (a lossy, structural sharing decided at architecture time); MLA reduces cache via a *learned low-rank bottleneck* with full up-rank reconstruction, which is why it does not pay GQA's typical quality tax at comparable cache size.
- Know that MLA long predates its "aha" reception in V3/R1 — V2 is the origin, and the mechanism is unchanged across the DeepSeek-V2/V3/R1 line.
