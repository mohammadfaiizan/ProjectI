## Quantization for Inference

### 1. Why quantize, and what it costs you

A model's weights are trained in fp32 or, almost universally by 2024-2025, bf16/fp16 (2 bytes per parameter). Inference-time quantization takes a *trained* model and re-represents its weights (and optionally activations and the KV cache) in a lower-precision numeric format — int8, int4, fp8 — after the fact, without retraining from scratch. This is a distinct discipline from quantization-aware training; this file is entirely about **post-training quantization (PTQ)**, the dominant regime for serving already-trained frontier-scale models where a full retrain is not on the table.

The motivation is threefold, and the three benefits don't always arrive together:

- **Memory**: an int4 weight is a quarter the size of an fp16 weight. A 70B-parameter model at fp16 is 140 GiB; at int4, roughly 35 GiB — the difference between needing a multi-GPU node and fitting on a single high-end GPU.
- **Bandwidth / speed**: decode-time inference is memory-bandwidth-bound (file 001 / file 005), meaning the bottleneck is *moving weights from HBM into the compute units*, not the arithmetic itself. Smaller weights move faster, directly reducing per-token latency, independent of any change in raw FLOPs.
- **Cost**: fewer/cheaper GPUs needed per unit of served throughput.

The cost is quality degradation: every quantization scheme introduces *some* approximation error relative to the original weights, and the entire discipline is about minimizing that error's effect on model output quality for a given bit-width budget, and about being honest with yourself about where the error actually lands (uniformly small everywhere, or concentrated in a few catastrophic outlier weights/activations that wreck specific behaviors).

**PTQ versus quantization-aware training (QAT), briefly.** QAT simulates quantization *during* training or fine-tuning (inserting fake-quantize operations into the forward pass so the model's own gradients learn to compensate for quantization error), which generally achieves better quality-per-bit than any post-hoc PTQ scheme, for the obvious reason that the model itself was optimized with the eventual quantization already accounted for rather than having it imposed afterward on weights that were never trained with that constraint in mind. The reason PTQ dominates the inference-serving conversation anyway — and the reason this file is entirely about PTQ rather than QAT — is practical: QAT requires a training-infrastructure investment (data, compute, and engineering effort comparable to a fine-tuning run) for every quantization configuration you want to support, while PTQ is a comparatively cheap, offline, single-GPU-hours-scale post-processing step applicable to any already-trained checkpoint, including third-party open-weight models whose original training pipeline you have no access to at all. A staff engineer choosing between the two is really asking "do I control training compute and is quality-per-bit important enough to justify it, or am I taking an already-trained checkpoint as a given" — the latter is the overwhelmingly more common situation for a serving team working with either open-weight models or a frontier lab's already-released checkpoints.

### 2. The mechanics of uniform quantization: scale and zero-point

The basic operation converts a real-valued tensor `W` (fp16) into an integer tensor `W_q` (e.g. int8, range `[-128, 127]`) plus a small amount of floating-point metadata that lets you approximately reconstruct the original values:

```
W_q = round(W / scale) + zero_point         (quantize)
W_hat = (W_q - zero_point) * scale          (dequantize, W_hat ≈ W)
```

- **scale** is a single fp16/fp32 number (or a small vector, see "granularity" below) chosen so the integer range `[-128, 127]` (or `[0, 255]` for unsigned) spans the actual range of values in `W`. `scale = (max(W) - min(W)) / (q_max - q_min)`.
- **zero_point** is an integer offset that lets an asymmetric range (e.g. weights that are mostly positive, or activations after a ReLU that are strictly non-negative) map efficiently onto the integer grid without wasting representable codes on values that never occur. **Symmetric** quantization fixes `zero_point = 0` and requires `W`'s range to be centered on zero (common for weights, which are typically near-zero-mean); **asymmetric** quantization allows a nonzero `zero_point` and is common for activations, whose distributions are often skewed.

The core engineering problem is choosing `scale` (and `zero_point`) well. This is what **calibration** means: you cannot know the right scale without looking at real data, because the right scale depends on the actual distribution of values, especially the tails.

```python
import numpy as np

def compute_scale_zp(x: np.ndarray, qmin=-128, qmax=127, symmetric=True):
    if symmetric:
        max_abs = np.max(np.abs(x))
        scale = max_abs / qmax if max_abs > 0 else 1.0
        zero_point = 0
    else:
        x_min, x_max = np.min(x), np.max(x)
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = round(qmin - x_min / scale)
    return scale, zero_point

def quantize(x, scale, zero_point, qmin=-128, qmax=127):
    q = np.round(x / scale) + zero_point
    return np.clip(q, qmin, qmax).astype(np.int8)

def dequantize(q, scale, zero_point):
    return (q.astype(np.float32) - zero_point) * scale
```

**The calibration process** for a real model: run a small, representative sample of inputs (a "calibration set" — a few hundred to a few thousand sequences resembling production traffic) through the *original* fp16 model, and record the actual distribution of values seen — either weights directly (which don't need data, since weights are static — you can compute their scale directly from the weight tensor itself) or, crucially, **activations**, which are input-dependent and therefore genuinely require running data through the model to observe. The scale is then set from these observed statistics, commonly via one of:

- **min/max**: scale spans the full observed range. Simple, but a single extreme outlier value stretches `scale` for the entire tensor, wasting precision on the vast bulk of "normal" values (this is the central weakness min/max calibration has, and the reason more sophisticated schemes exist).
- **percentile clipping**: clip to, say, the 99.9th percentile of observed magnitudes and saturate anything beyond it, trading a small amount of clipping error on rare outliers for much better resolution on the typical case.
- **MSE-minimizing / entropy-based calibration**: search over candidate scales and pick the one that minimizes a reconstruction-error metric (e.g. KL-divergence between the original and quantized activation histograms) rather than using a fixed percentile rule.

**Granularity** is the other key calibration decision: quantizing with **one scale per entire tensor** is simplest but coarsest; **per-channel** (one scale per output channel of a weight matrix) or **per-group** (one scale per contiguous block of, say, 128 weights along a channel) quantization uses many small scales instead of one big one, letting each slice of the tensor use a range tuned to its own local distribution. This matters enormously in practice: weight matrices routinely have some output channels with much larger typical magnitude than others, and a single tensor-wide scale would force the small-magnitude channels to be represented with almost no effective precision (their whole range might collapse to a handful of integer codes) just to accommodate the large-magnitude channels' range. Per-group quantization is the standard choice for aggressive (4-bit and below) weight quantization for exactly this reason, and is the granularity GPTQ, AWQ, and the GGUF k-quant formats below all use.

### 3. GPTQ: Hessian-based per-layer error compensation

Naive round-to-nearest quantization (Section 2) treats every weight independently: quantize each one, done. GPTQ's key insight (building on the earlier Optimal Brain Quantization / Optimal Brain Compression line of work) is that weights in a matrix are **not independent** with respect to their effect on the layer's output — a linear layer computes `y = Wx`, and quantizing one weight introduces an error that, propagated through the same input `x`, can be *partially cancelled* by deliberately adjusting the still-unquantized weights in a compensating direction. Quantizing weights **greedily and independently** throws away this opportunity; GPTQ exploits it.

Conceptually, GPTQ processes one layer's weight matrix column by column (or in small groups):

1. Quantize one column (round to the nearest representable value on the target grid).
2. Measure the error this introduces in the layer's output for the calibration data.
3. **Redistribute** that error into the remaining, not-yet-quantized columns by nudging them in the direction that best cancels the error — computed using the inverse of the **Hessian** of the layer's local quadratic loss surface with respect to the weights (the second-order curvature information tells you exactly how much each remaining weight should move to compensate for a given perturbation, in a least-squares-optimal sense).
4. Move to the next column and repeat, now working with already-compensated weights.

This is exactly the logic of the classical Optimal Brain Surgeon pruning/compression framework, adapted to quantization: instead of asking "which weight can I zero out with least damage" (pruning), GPTQ asks "which weight can I round to the nearest quantization level with least damage, given that I can compensate by adjusting everything not yet touched." The Hessian here is computed cheaply per layer from a small calibration set (it does not require the full training-loss Hessian over the whole model — it's a local, per-layer, least-squares proxy: minimizing `||Wx - W_q x||^2` over the calibration activations `x`), which is what makes GPTQ tractable to run on a single GPU in a few hours for a 70B-parameter model, rather than requiring anything resembling a training run.

The practical result: GPTQ reliably gets 4-bit weight-only quantization (`W4A16` — 4-bit weights, fp16 activations) to within a small perplexity/accuracy gap of the original fp16 model for most models, which naive round-to-nearest at 4 bits typically cannot match (naive RTN degrades much more sharply below 8 bits because it has no mechanism to compensate for the compounding rounding error across a whole matrix).

### 4. AWQ: activation-aware weight scaling

AWQ starts from a different empirical observation: not all weights matter equally, but the reason isn't intrinsic to the weight — it's about which weights multiply against **large-magnitude activations**. In a matmul `y = Wx`, a weight `w_ij` that happens to always multiply against an activation `x_j` with large typical magnitude has an outsized effect on `y` relative to a weight multiplying a small-magnitude activation, *even if the two weights have identical magnitude themselves*. Empirically, in trained transformers, a small number of activation channels carry systematically larger magnitudes than the rest (a well-documented "outlier channel" phenomenon, especially pronounced after certain normalization/activation choices) — and it is precisely the weights feeding into those channels that are most "salient" (most damaging to quantize coarsely), independent of the weight values' own distribution.

AWQ's fix does **not** keep those salient weights in higher precision (which would require mixed-precision kernels — different weights at different bit-widths within the same matrix — an implementation headache and a poor fit for dense, uniform low-bit GPU kernels). Instead, it exploits a **mathematical identity**: for a given input channel `j`, you can scale up the corresponding column of `W` by a per-channel factor `s_j > 1` and scale *down* the corresponding activation channel `x_j` by `1/s_j`, and the product `Wx` is exactly unchanged:

```
y = sum_j w_j * x_j = sum_j (w_j * s_j) * (x_j / s_j)
```

Scaling up a salient channel's weights before quantizing gives that channel's weights a *larger effective range relative to the fixed quantization grid*, meaning the same integer grid spacing now represents that channel's values with proportionally less relative rounding error — you've effectively borrowed some of the "headroom" that less-salient channels didn't need. The `1/s_j` activation-side scaling is then folded into an adjacent operation (e.g. absorbed into a preceding layer's output scale, or, since activations are typically kept in fp16 rather than quantized at all in AWQ's usual configuration, simply applied as a cheap elementwise pre-scaling at inference time) so the correction is mathematically exact, not approximate — the model computes the identical function, just with the weight tensor pre-conditioned to be more quantization-friendly.

The channel saliency itself is determined during calibration: run representative data through the model, measure per-channel activation magnitude statistics, and pick the scaling factors `s_j` (per input channel) that minimize quantization error for the *most salient* channels — typically via a small search over candidate scaling strengths per layer, since scaling too aggressively also has costs (it can push previously well-behaved weights into a poorly-represented range).

**AWQ vs GPTQ, conceptually.** GPTQ works purely at the weight level, using second-order (Hessian) information to compensate quantization error across weights within a layer, and needs no notion of "salient channel" — its calibration data is used only to estimate the local loss curvature. AWQ works by pre-conditioning weights based on which *activation channels* they interact with, using only first-order magnitude statistics (no Hessian), and is comparatively cheaper to compute (no per-layer matrix inversion). In practice both are widely used 4-bit weight-only PTQ methods with broadly comparable quality outcomes on most models; GPTQ is generally considered to squeeze out marginally better accuracy at the cost of a more expensive calibration procedure, while AWQ is faster to calibrate and reportedly more robust across a wider range of model families and less prone to catastrophic failure on any single outlier layer — but exact relative rankings are model- and task-dependent and shift with new evaluations, so treat any specific numeric superiority claim between the two as a moving target rather than a fixed fact.

### 5. GGUF and llama.cpp-style k-quants

GGUF (the file format used by llama.cpp and its ecosystem) targets a different deployment regime than GPTQ/AWQ: **CPU and consumer-GPU edge inference**, where the constraints are total file size (fitting a model in limited RAM/VRAM on a laptop or phone), and where the serving stack cares about single-request, low-batch-size latency rather than high-throughput multi-tenant serving.

The "k-quant" formats (named `Q4_K_M`, `Q5_K_S`, `Q2_K`, etc.) extend simple uniform quantization along two axes:

- **Superblock structuring**: weights are grouped into small blocks (e.g. 32 weights), and blocks are further grouped into "superblocks" (e.g. 8 blocks = 256 weights). Each small block gets its own scale (fine-grained, like the per-group quantization in Section 2), and the superblock stores a shared, more coarsely quantized scale-of-scales, so you get most of the accuracy benefit of very fine-grained per-block scales without paying the full fp16-scale-per-block metadata overhead — the scale metadata itself is quantized.
- **Mixed bit-width across layers/tensors within one model file**, chosen empirically per weight-matrix type: the `_S`/`_M`/`_L` suffixes (small/medium/large) and the letter K variants encode different recipes for *which tensors get more bits*. A common pattern (llama.cpp's empirically-tuned defaults) allocates slightly higher precision to certain layers observed to be more quantization-sensitive (e.g., attention output projections, or the final layers) and lower precision to less sensitive ones, all while keeping the nominal "4-bit" or "5-bit" label as an average across the whole file rather than a uniform per-weight guarantee. This is a lighter-weight, heuristic cousin of AWQ/GPTQ's more principled per-channel or Hessian-driven precision allocation — tuned by extensive empirical sweeps across many models rather than derived per-model from calibration data, which makes it fast to apply to any new model checkpoint (often no calibration run at all is required, only the static weight statistics) at some cost in the guaranteed-optimality that a calibration-driven method offers.

The practical trade-off GGUF occupies: it is optimized for **portability and ease of use across heterogeneous, often CPU-only or unified-memory hardware** (Apple Silicon, consumer GPUs, CPU-only servers) with a simple single-file format and no requirement for a GPU-only quantization/calibration pipeline, at some cost in absolute quality-per-bit relative to a carefully calibrated GPTQ/AWQ run targeted at one specific model and one specific target hardware. It is essentially never the format of choice for high-throughput multi-tenant cloud serving of a frontier-scale model — that regime is GPTQ/AWQ/FP8-on-GPU territory — but it is close to the default choice for "run this open-weight model on my laptop."

### 6. Activation quantization is the harder half of the problem

Everything above focused on **weight-only** quantization (`WxA16`: weights at x bits, activations kept in fp16). This is by far the most common regime for serving because weights are static (no data-dependence, easy to calibrate once) and quantizing them alone already captures most of the memory-and-bandwidth win (weights dominate static memory; activations are comparatively small, transient, per-forward-pass tensors).

**Why activation quantization (`W8A8`, `W4A8`, etc.) is harder**: activations are the *output* of a nonlinear function (attention, GELU/SwiGLU, layernorm) applied to a data-dependent input, and empirically exhibit far more extreme, structured outliers than weights do — specific channels in specific layers routinely produce activation magnitudes 10-100x larger than the typical value in that same tensor, a phenomenon documented extensively in the LLM quantization literature (this is precisely the observation AWQ's channel-scaling trick and, separately, the SmoothQuant technique for `W8A8` are built around — SmoothQuant applies a similar migrate-the-scale-from-activations-to-weights identity to make activation quantization tractable at int8). A single such outlier, if it sets the scale for the whole tensor via naive min/max calibration, wastes almost all of the integer range's resolution on the rest of the (small-magnitude, common-case) tensor. This is why activation quantization schemes at aggressive bit-widths (below int8) remain comparatively less mature and less universally reliable than weight-only quantization, and why most production serving stacks default to `W4A16`/`W8A16`-style weight-only regimes unless specifically chasing the additional compute-throughput gains that come from also running the matmul itself in low-precision (which requires *both* operands quantized — hence needing activation quantization, not just weight quantization, to actually speed up the matmul rather than only shrinking memory).

**FP8**, supported natively by recent GPU tensor cores (H100 and later), is worth calling out separately: it is a floating-point format (typically `E4M3` or `E5M2` — 4 or 5 exponent bits, 3 or 2 mantissa bits), not an integer format, and floating-point's non-uniform, larger dynamic range per bit makes it considerably more forgiving of exactly the outlier problem that plagues int8 activation quantization — an fp8 value can represent a large-magnitude outlier and a small typical value with proportionally similar *relative* precision, whereas int8's uniform grid spacing cannot. This is a major reason fp8 (for both weights and activations, and increasingly for the KV cache itself, connecting back to file 001 Section 7) has become a preferred choice for high-throughput GPU serving where hardware support exists, capturing much of int8's speed/memory benefit with meaningfully less of int8's outlier-driven accuracy risk.

### 7. The practical decision framework

A staff engineer picking a quantization scheme is really answering three coupled questions: what hardware will this run on, what's the acceptable quality bar for this specific product surface, and what's the actual bottleneck (memory capacity, memory bandwidth, or compute throughput)?

```
Decision sketch:

Target = high-throughput multi-tenant GPU serving (H100/similar), quality-sensitive product
  -> weight-only INT4/INT8 via GPTQ or AWQ (W4A16/W8A16), or native FP8 if hardware supports it
     and the serving stack (vLLM, TensorRT-LLM, etc.) has a mature FP8 kernel path.
  -> Keep KV cache at fp16 or fp8 (file 001 Section 7) but rarely below that for a
     quality-sensitive deployment.

Target = maximize GPU throughput/cost, quality bar is looser (e.g. cheap tier of a
routing system, file 006), compute-bound workload
  -> Push toward W8A8 or FP8 end-to-end (weights AND activations), since this is the
     regime that actually speeds up the matmul (not just memory), at the cost of the
     extra activation-outlier risk discussed in Section 6 -- validate hard on eval suites.

Target = edge / CPU / consumer GPU / on-device, single-user latency matters more than
multi-tenant throughput, no calibration infra available
  -> GGUF k-quants (Q4_K_M as a reasonable default starting point), accept a modest
     quality gap relative to a calibrated GPU-targeted scheme in exchange for
     portability and zero-calibration-pipeline simplicity.

Any target, before shipping
  -> Validate on task-representative evals, not just perplexity. Perplexity is a coarse,
     aggregate signal and can look fine while specific capabilities (e.g. long-context
     retrieval, precise arithmetic, structured-output adherence) degrade
     disproportionately -- these are exactly the behaviors most sensitive to the small
     number of "important" weights/channels a coarse quantization scheme is most likely
     to damage.
```

The **quality-memory-speed trade-off curve**, stated qualitatively rather than with fabricated numbers (exact percentages are model- and method-specific and shift with each new quantization technique, so treat any specific number you haven't personally benchmarked as provisional): going from fp16 to int8 weight-only is close to a free lunch for most models — memory halves, speed improves, and measured quality loss is typically negligible to unmeasurable on standard evals. Going from int8 to int4 is where the real trade-off curve bends: memory halves again and speed improves further, but quality loss becomes clearly measurable and method-dependent (naive RTN degrades noticeably; GPTQ/AWQ claw back most, but rarely literally all, of the gap). Below int4 (int3, int2, extreme k-quants) quality loss becomes severe and inconsistent across model families for all but the most careful, often model-specific, quantization recipes — this is the region where "does this actually work for *my* model and *my* eval suite" stops being a question you can answer from general PTQ literature and requires direct empirical validation.

### 8. SmoothQuant, in a bit more depth

Section 6 mentioned SmoothQuant only in passing as "a related technique" to AWQ; it's worth understanding on its own terms because it targets a distinct point in the design space — genuine `W8A8` (both weights and activations quantized to int8), rather than the weight-only regime AWQ and GPTQ primarily target.

The observation SmoothQuant starts from is the same outlier-channel phenomenon AWQ exploits, but applied in the opposite direction: rather than scaling weights *up* on salient channels to give them more effective range (AWQ's move), SmoothQuant scales the *activations* down on the channels with the largest outliers, and pushes the compensating inverse scale onto the weights, using the identical mathematical identity from Section 4 (`Wx = (W \cdot s) (x / s)` for a per-channel scale `s`). The direction of the trade matters: activations are the harder tensor to quantize well (Section 6's discussion of why), so SmoothQuant's design goal is specifically to *flatten* the activation distribution — shrink its outliers — even at the cost of making the weight distribution slightly less flat than it was, since weights are the easier tensor to absorb that cost into. This is a smoothing operation performed once, analytically, from calibration statistics, prior to applying whatever int8 quantization scheme (symmetric or asymmetric, per-tensor or per-channel) is used on both tensors afterward — it is a *pre-conditioning* step, not a quantization scheme in its own right, in the same sense AWQ's channel scaling is a pre-conditioning step rather than the quantization arithmetic itself.

The practical payoff: once activations are smoothed this way, standard int8 quantization of both weights and activations becomes tractable at accuracy levels close to the original fp16 model for many architectures, unlocking int8 matmul kernels (faster in raw FLOP/s terms than fp16 on hardware with native int8 tensor-core support, not just smaller in memory) rather than settling for the memory-only benefit of weight-only quantization. This is the `W8A8` end of the spectrum sketched in Section 7's decision framework — appropriate when the workload is genuinely compute-bound enough (large batch decode, or prefill, file 005) that a faster matmul actually matters, as opposed to a small-batch decode workload where the bottleneck is memory bandwidth regardless of how fast the matmul arithmetic itself is (file 001, file 005 Section 1) and where weight-only quantization already captures most of the achievable win.

### 9. A worked calibration walkthrough

To make Section 2's calibration process fully concrete rather than purely conceptual, here is what a calibration pass for a single linear layer's weight-only int4 quantization actually involves end to end, stated as a sequence of discrete engineering steps rather than an abstract description:

```python
import numpy as np

def calibrate_and_quantize_layer(W: np.ndarray, calibration_inputs: np.ndarray,
                                  bits: int = 4, group_size: int = 128):
    """W: [out_features, in_features]. calibration_inputs: [n_samples, in_features],
    a small representative batch of real activations feeding this layer, collected by
    running the ORIGINAL fp16 model forward over a calibration dataset and hooking this
    layer's input. Returns per-group quantized weights, scales, and a reconstruction-
    error diagnostic to decide whether this layer needs special handling (Section 7)."""
    out_features, in_features = W.shape
    qmax = 2 ** (bits - 1) - 1
    n_groups = (in_features + group_size - 1) // group_size

    W_q = np.zeros_like(W)
    scales = np.zeros((out_features, n_groups))

    for g in range(n_groups):
        lo, hi = g * group_size, min((g + 1) * group_size, in_features)
        w_slice = W[:, lo:hi]
        # Per-group, per-output-channel scale (a common granularity choice, Section 2).
        max_abs = np.max(np.abs(w_slice), axis=1, keepdims=True)
        scale = np.where(max_abs > 0, max_abs / qmax, 1.0)
        W_q[:, lo:hi] = np.clip(np.round(w_slice / scale), -qmax, qmax) * scale
        scales[:, g] = scale.squeeze(axis=1)

    # Reconstruction-error diagnostic on REAL calibration activations, not weights alone --
    # this is the step that actually validates whether this scheme is safe for this layer.
    y_true = calibration_inputs @ W.T
    y_quant = calibration_inputs @ W_q.T
    relative_error = np.linalg.norm(y_quant - y_true) / np.linalg.norm(y_true)
    return W_q, scales, relative_error


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, size=(16, 512)).astype(np.float64)
    calib = rng.normal(0, 1, size=(256, 512)).astype(np.float64)
    W_q, scales, rel_err = calibrate_and_quantize_layer(W, calib, bits=4, group_size=128)
    print(f"relative output reconstruction error: {rel_err:.4f}")
```

The diagnostic in the last line is the practically important habit: quantizing a layer without measuring its *output*-level reconstruction error against real calibration data (rather than trusting a bit-width choice blindly) is exactly how a staff engineer catches, before shipping, that a specific layer (commonly an attention output projection, or a layer feeding directly into the unembedding/output head) needs a higher bit-width allocation or a different granularity than the rest of the model — precisely the kind of per-layer sensitivity GGUF's k-quant recipes (Section 5) encode as static, empirically-derived heuristics, and that GPTQ/AWQ's calibration process is discovering dynamically, per model, rather than assuming from a fixed prior recipe.

### 10. Evaluating a quantized model before shipping it

Section 7 already argued that perplexity alone is an insufficient quality gate; it's worth being explicit about what a more complete quantization-validation protocol looks like, since "run more evals" is correct but underspecified.

- **Perplexity / cross-entropy loss on a held-out corpus**, as a coarse, cheap, always-run first check — a large perplexity delta versus the unquantized model is disqualifying on its own, but a *small* delta is necessary, not sufficient, evidence of safety (Section 7, and Part 1 Q9's worked scenario).
- **KL-divergence between the quantized and unquantized model's next-token distributions**, measured token-by-token on representative sequences — a more sensitive signal than perplexity alone, because it can reveal cases where the quantized model's *ranking* of plausible next tokens shifted meaningfully even though the loss on the single ground-truth token barely moved (perplexity only scores the probability assigned to the one token that actually occurred; KL-divergence scores the whole distribution's shift).
- **Task-representative capability evals**, chosen specifically to cover whatever the deployment actually depends on — code-generation pass rates for a coding assistant, exact-match/structured-output-validity rates for a workflow-automation product, multi-step reasoning benchmarks for an agentic product — rather than a generic, deployment-agnostic benchmark suite. The point of this module's Q9 (Part 1) scenario is precisely that perplexity and generic benchmarks can look fine while a specific, deployment-critical capability collapses.
- **Long-context-specific evaluation**, if KV-cache quantization (file 001 Section 7) is also being applied — a model can pass every short-context eval while exhibiting degraded long-context retrieval or reasoning specifically because cache quantization error compounds across many attended positions, a failure mode that simply doesn't show up in a short-context test set.
- **A/B or canary-style comparison against the unquantized model on live or held-out production-representative traffic** (file 006 Section 3's canary discussion, applied here to a quantization rollout specifically) before a full cutover, rather than trusting an offline eval suite as the sole gate — offline evals are, at best, a proxy for the traffic distribution that actually matters.

### 11. Interaction with the rest of the serving stack

Quantization is not independent of the other techniques in this module. Smaller weights mean more HBM headroom for KV cache (file 001), which directly increases the achievable batch size in a continuous-batching server (file 003) — this is often the *dominant* practical benefit of weight quantization in a production serving context, larger than the raw per-token latency improvement: quantizing a 70B model from fp16 to int4 doesn't just make each forward pass faster, it frees roughly 105 GiB of HBM (140 GiB - 35 GiB) that can now hold KV cache for dozens of additional concurrent long-context requests. Any serving-cost model (file 005) that only accounts for "quantization speeds up the matmul" and ignores "quantization frees memory for more concurrent batching" is missing the larger of the two effects at realistic production concurrency levels.

Quantization also interacts with speculative decoding (file 004): a quantized draft model is an obvious way to make the draft step even cheaper (lower `c`, the draft/target cost ratio in file 004's speedup formula), but quantizing the draft model too aggressively risks widening the gap between its distribution and the target's, lowering the acceptance rate `alpha` — the same speed/quality coupling seen everywhere else in this module, now expressed as a genuine tuning trade-off between two knobs (draft cost, draft fidelity) that both affect the same speedup formula in opposite directions. This is worth having ready as a concrete example of how the six files in this module compose rather than operate in isolation: a change made purely to satisfy file 002's concerns (draft-model memory footprint) has a direct, quantifiable effect on file 004's speedup math, and a staff-level analysis of "should we quantize the draft model" has to trace through both.

### 12. Hardware support is not uniform, and it constrains the decision as much as quality does

None of the schemes above are useful in production unless the target hardware's kernels actually execute them efficiently — a quantization format with no fast matmul kernel for the deployment GPU is a memory-savings-only technique at best (still useful for fitting a model at all, per file 001's crossover argument, but not a speed win) and a genuinely unusable one at worst. A rough, non-exhaustive orientation, offered as a snapshot rather than a permanent hardware fact given how quickly kernel support evolves:

| Format | Typical benefit realized | Hardware dependency |
|---|---|---|
| INT8 weight-only | Memory + some speedup | Broadly supported across recent GPU generations |
| INT4 weight-only (GPTQ/AWQ) | Memory + speedup via dedicated low-bit kernels | Requires a serving stack with matching kernel support (e.g. vLLM/TensorRT-LLM's specific INT4 kernels) |
| FP8 (weights and/or activations) | Memory + genuine matmul speedup | Native tensor-core support required (H100-class and later); no benefit, or a slow emulated path, on older hardware |
| GGUF k-quants | Memory, portability | CPU and broad consumer-GPU support by design; not the target for high-throughput datacenter GPU serving |

The practical consequence: a quantization decision is never made purely on a quality-vs-memory curve in isolation — it is made jointly with "what does my actual serving stack, on my actual target hardware, have a fast kernel for," and a scheme that looks best on a quality/memory chart but has no efficient kernel path on the deployment GPU is, for that deployment, the wrong choice regardless of its paper results.

### 13. A note on what remains genuinely unsettled

Two things are worth flagging honestly rather than presenting as settled science, since a staff interview will sometimes probe exactly this boundary. First, the relative ranking between GPTQ and AWQ (and the growing set of newer PTQ methods published after both) shifts with each new evaluation and is genuinely model-family-dependent — treat any specific claim of "method X beats method Y by Z%" as provisional to the paper or benchmark that produced it, not as an established, permanent ordering. Second, extreme low-bit quantization (sub-4-bit, and especially sub-2-bit) remains an active research area where no single technique has emerged as a clearly dominant, universally reliable default the way GPTQ/AWQ have for 4-bit weight-only quantization — deploying at that end of the spectrum for a production, quality-sensitive use case should be treated as requiring direct empirical validation on the specific target model and workload, not as something safely generalizable from published results on a different model family.

A staff-level answer on this topic should therefore always carry an implicit "as of the evidence I've actually validated" qualifier for any specific numeric claim, while still being able to state the *structural* facts in this file — what calibration is, why per-group granularity matters, what GPTQ's and AWQ's mechanisms actually do, why activation quantization is harder than weight quantization, and how quantization's memory savings interact with KV-cache-driven batch size — with full confidence, since those are mechanism-level facts rather than moving benchmark numbers.

As a final, practical habit: whenever a quantization decision is being made for a real deployment, write down the three things this file has argued are jointly load-bearing — the target hardware's actual kernel support (Section 12), the specific capabilities the deployment depends on and how they'll be validated (Section 10), and the expected memory-freed-for-batching effect on achievable concurrency (Section 11) — before committing to a specific bit-width and method, rather than defaulting to whichever scheme is most commonly cited in the current literature.

A quick self-test worth being able to pass without notes: (1) write the symmetric int8 quantize/dequantize formulas from memory; (2) explain, in one sentence each, what GPTQ's Hessian-based compensation and AWQ's channel scaling are each actually doing differently; (3) state why activation quantization is harder than weight quantization, citing the specific outlier phenomenon responsible; (4) name the two distinct benefits weight quantization delivers to a serving stack (raw speed, and freed HBM for more KV-cache-driven batching) and explain which is usually larger in practice; and (5) state what perplexity does and does not validate, and name one capability that can fail while perplexity looks fine. These five map directly onto Sections 2, 3/4, 6, 11, and 10 respectively.
