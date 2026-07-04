# Quantization and Inference Optimization

## Why Inference Is a Memory-Bandwidth Problem, Not a Compute Problem

It's tempting to think of running a large language model as primarily a matter of raw arithmetic throughput — a 70B-parameter model does a lot of matrix multiplications, so surely the bottleneck is how many FLOPs your GPU can execute per second. For autoregressive decoding, that intuition is almost exactly backwards, and understanding why is the single most important prerequisite for understanding both quantization and every serving-optimization technique in the second half of this chapter.

Consider what actually happens when you generate one new token from a model that already has a populated KV cache. The forward pass processes exactly one new token position through every layer. At each linear projection, the GPU multiplies a tiny `(1, hidden_size)` activation vector against a huge weight matrix — say `(hidden_size, hidden_size)` or `(hidden_size, 4 * hidden_size)` for the MLP block. The number of floating-point operations involved is proportional to the number of weight elements touched, but so is the number of *bytes that have to be read* from the GPU's high-bandwidth memory (HBM) to get those weights into the compute units in the first place. The crucial asymmetry is this: for a batch size of 1 (or any batch size small relative to the weight matrix's dimensions), you read the *entire* weight matrix from HBM just to perform *one* multiply-accumulate per weight — you get essentially no reuse of the data you paid to move. This ratio of "compute performed per byte moved" is called arithmetic intensity, and single-token decoding has catastrophically low arithmetic intensity compared to what modern GPUs are built to exploit. A100/H100-class GPUs have compute throughput (FLOPs/sec) that vastly outpaces their memory bandwidth (bytes/sec) when measured in the units that matter — meaning that for low-arithmetic-intensity workloads, the compute units sit idle, stalled waiting for weights to arrive from HBM, and the wall-clock time per generated token is determined almost entirely by how many bytes of weights had to be streamed off HBM, not by how many multiply-accumulates were performed on them. This is precisely the same "memory-bandwidth-bound, not compute-bound" observation that motivates FlashAttention (covered in file 004) — it shows up again here because it is a structural property of autoregressive decoding itself, not something specific to attention.

This single fact has an enormous, very concrete consequence: if you can shrink the number of bytes needed to represent each weight, you directly and almost linearly speed up token generation, because you have shrunk the dominant cost. This is the entire economic case for quantization. Halving the bytes per weight (say, from 16-bit to 8-bit) doesn't just halve your GPU memory footprint — under the memory-bandwidth-bound regime decoding operates in, it can very nearly halve your per-token latency too, and it lets you fit models on smaller/cheaper GPUs or fit more concurrent requests into the memory you do have. Batching multiple requests together helps because the same loaded weights get reused across every sequence in the batch during that step (which is part of why continuous batching, discussed later, matters so much for throughput), but for a given batch size, decoding latency is still dominated by bytes moved, not FLOPs performed. Training, by contrast, generally runs in a much more compute-bound regime (large batches supply plenty of arithmetic intensity per byte of weights loaded), which is one reason quantization is overwhelmingly an *inference-time* technique — you train in fp32/bf16 for numerical stability and gradient quality, then quantize the resulting weights afterward for serving.

## Precision Formats: What the Bits Actually Buy You

Before getting into quantization algorithms, it's worth being crisp about what "precision" means numerically. Standard training and much of inference historically used FP32 (32 bits: 1 sign, 8 exponent, 23 mantissa) or, more commonly for LLMs, FP16 (1 sign, 5 exponent, 10 mantissa) or BF16 (1 sign, 8 exponent, 7 mantissa — the same exponent range as FP32, so more robust to overflow, at the cost of less mantissa precision than FP16). Quantization pushes further down to INT8 (a single byte, 256 distinct integer levels) and, increasingly aggressively, INT4 (a nibble, only 16 distinct levels).

The step from FP16 to INT8 is a 2x reduction in bytes per weight; the step to INT4 is 4x. But the *representational* cost of that reduction is not linear in an obvious way — going from 256 possible values (INT8) down to 16 possible values (INT4) is a drastic drop in resolution, and it's why INT4 quantization requires meaningfully more careful algorithms (GPTQ, AWQ, and friends, discussed below) to avoid a large quality hit, whereas INT8 quantization is comparatively forgiving and can often be done with fairly naive, round-to-nearest schemes and still preserve near-original model quality. A useful mental model: INT8 quantization is mostly a "free lunch" (2x smaller, roughly 2x faster decoding, negligible quality loss with reasonable care) while INT4 is a real trade-off that has to be actively managed, which is exactly why so much research effort has gone specifically into making 4-bit quantization viable.

## The Core Quantization Math: Scale and Zero-Point

At its heart, quantization is just an affine mapping between a continuous (or high-resolution) range of real numbers and a small, fixed set of integers. Every quantization scheme, no matter how sophisticated the surrounding algorithm, boils down to choosing two numbers — a **scale** and a **zero-point** — that define this mapping, and then rounding.

**Symmetric quantization** assumes the values being quantized are roughly centered around zero, a reasonable assumption for most neural network weights, which are typically initialized and trained to have a roughly zero-centered distribution. It maps the range `[-max_abs, +max_abs]` onto an integer range symmetric about zero, with no separate offset needed:

```
scale = max(abs(W)) / (2^(bits-1) - 1)
q = round(W / scale)              # q is an integer in [-(2^(bits-1)-1), 2^(bits-1)-1]
W_dequantized = q * scale
```

**Asymmetric quantization** instead maps the actual observed range `[min(W), max(W)]` onto the full unsigned integer range `[0, 2^bits - 1]`, which requires an extra **zero-point** offset so real value `0.0` still maps to *some* exact integer — important, because operations like padding or a ReLU nonlinearity frequently produce literal zeros you don't want corrupted by rounding error:

```
scale = (max(W) - min(W)) / (2^bits - 1)
zero_point = round(-min(W) / scale)
q = round(W / scale) + zero_point
W_dequantized = (q - zero_point) * scale
```

Asymmetric quantization is strictly more expressive — it can represent skewed distributions (very common for *activations*, especially post-ReLU/GELU distributions that are heavily one-sided) more accurately than symmetric quantization can, at the cost of needing to store and apply the extra zero-point term and slightly more complex integer arithmetic during dequantization. In practice, weights (roughly symmetric, zero-centered) are very often quantized symmetrically, while activations (often skewed) more frequently use asymmetric quantization — though many production quantization schemes for LLMs (GPTQ, AWQ) are weight-only and leave activations in fp16 entirely, sidestepping the activation-quantization problem altogether, a deliberate design choice discussed further below.

A minimal, concrete illustration of symmetric INT8 quantization and dequantization of a weight tensor makes the arithmetic tangible:

```python
import numpy as np

def quantize_symmetric_int8(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric INT8 quantization. Returns the quantized
    integer tensor and the single scale needed to dequantize it."""
    max_abs = np.max(np.abs(weights))
    scale = max_abs / 127.0  # symmetric INT8 range is [-127, 127], not -128, to stay symmetric
    q = np.round(weights / scale).astype(np.int8)
    return q, scale

def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale

# Example
rng = np.random.default_rng(0)
W = rng.normal(loc=0.0, scale=0.02, size=(4, 4)).astype(np.float32)
q, scale = quantize_symmetric_int8(W)
W_hat = dequantize(q, scale)

print("original   :", W[0])
print("quantized  :", q[0])
print("dequantized:", W_hat[0])
print("max abs error:", np.max(np.abs(W - W_hat)))
```

Running this reveals the fundamental tension in all quantization: the dequantized weights are never bit-identical to the originals — there is always some quantization error, `W - W_hat`, introduced by the round-to-nearest-integer step. Every algorithm discussed in this chapter is, at bottom, a strategy for minimizing the downstream *impact* of that unavoidable rounding error on the model's actual output, rather than a strategy for eliminating the error itself, which is impossible once you have committed to fewer bits.

## Granularity: Per-Tensor, Per-Channel, and Per-Group

The scale (and zero-point) don't have to be a single number for an entire weight matrix — you can compute a separate scale for different subsets of the weights, and how finely you subdivide is a direct quality-versus-overhead trade-off called quantization granularity.

**Per-tensor** quantization uses one scale for an entire weight matrix (or even an entire layer). It's the cheapest in terms of storage overhead — a single float per tensor — but the least accurate, because a single outlier weight anywhere in that tensor forces `max_abs` to be large, which stretches the scale for *every other weight in the tensor*, wasting most of the available integer levels on values that never occur and crushing the resolution available to the bulk of "normal-sized" weights.

**Per-channel** (also called per-output-channel or per-row) quantization computes a separate scale for each output channel of a weight matrix — for example, a separate scale for each row of a `(out_features, in_features)` linear layer's weight. This confines the damage from any single outlier to just the one channel it lives in, rather than degrading the whole matrix, and it costs only `out_features` extra floats — negligible relative to the size of the weight matrix itself. This is why per-channel quantization is close to a free improvement over per-tensor and is used almost universally as the minimum granularity in any serious quantization scheme.

**Per-group** (also called per-block) quantization goes one step further, splitting each row into fixed-size contiguous groups (commonly 32, 64, or 128 weights per group) and giving *each group its own scale*. This is the granularity used by GPTQ, AWQ, and the GGUF k-quant formats discussed below, because it localizes the effect of an outlier down to a tiny neighborhood of weights rather than an entire row, which matters enormously at INT4, where every bit of preserved resolution counts. The cost scales with how many groups you create — a group size of 128 on a row of 4096 weights needs 32 separate scale values instead of 1, a meaningfully larger (though still small relative to the weights themselves, since scales are often stored compactly in fp16) storage overhead. Choosing group size is therefore a direct dial: smaller groups give better quality — finer-grained adaptation to local weight distributions — at the cost of more overhead and, in some kernel implementations, added computational complexity during dequantization at inference time; a group size around 128 is a very common practical sweet spot in modern 4-bit quantization tooling. The general pattern across all three levels is the same trade-off curve you'll see again and again in this chapter: finer granularity buys quality by localizing error, at a storage and complexity cost that only becomes worth paying once you push aggressively enough on bit-width that naive coarse-grained quantization starts visibly hurting the model.

## The Calibration Problem

Everything above assumes you already know the range — `min`/`max`, or some robust proxy for it — of the values you are quantizing. For weights this is trivial: the weights are a fixed, known tensor you can inspect directly, with no dependence on any particular input. But the more sophisticated algorithms below (GPTQ, AWQ) don't just look at weight magnitudes in isolation; they need to know how weights interact with *actual activations* the model produces on real inputs, in order to determine which weights matter most for preserving the model's actual output distribution. That requires **calibration data**: a small (typically a few hundred to a couple thousand samples) set of text sequences that gets run through the model in a forward-pass-only mode, purely to observe the statistics of intermediate activations and inform the quantization decisions, without any gradient computation or weight update at all.

The calibration set doesn't need to be enormous, but it does need to be reasonably representative of the domains and input distributions the model will actually see in production. Calibrating a code-generation model on generic web prose, or calibrating a multilingual model exclusively on English text, can leave the quantization poorly tuned for the true deployment distribution, because the "important" weight channels — the ones that see high-magnitude activations, which is exactly what AWQ cares about below — can genuinely differ across domains: a channel that is quiet on English prose might be exactly the channel that lights up on source code or on a different language's tokens. Getting calibration data selection wrong is a subtle and easy-to-overlook failure mode: the quantized model can look fine on whatever benchmark resembles the calibration distribution while quietly degrading on distributions the calibration set didn't cover, which is a good thing to flag if asked about failure modes of quantization in an interview.

This calibration step is also exactly what distinguishes **static** quantization — activation ranges are pre-computed once from calibration data and then fixed for all future inference, which is fast at serving time but relies on a fixed range that might not perfectly match every future input — from **dynamic** quantization, where activation ranges (particularly `min`/`max`) are recomputed on the fly for every actual inference batch, which is more accurate to the specific input at hand but adds a small amount of runtime overhead for that min/max reduction. Because GPTQ and AWQ are weight-only quantization schemes that leave activations in fp16, their use of calibration data is specifically about observing activation statistics to decide how best to quantize the *weights*, rather than about quantizing the activations themselves — but the same representativeness concern applies just as sharply: a poorly chosen calibration set gives both algorithms a distorted picture of which weights actually matter.

## GPTQ: One-Shot Weight Quantization via Approximate Second-Order Information

GPTQ (Frantar et al., 2022, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers") solves a very specific optimization problem, one layer at a time: given a weight matrix `W` and a set of calibration activations `X` that would normally be multiplied against it (`Y = WX`), find a quantized version `Ŵ` that minimizes the squared reconstruction error `||WX - ŴX||²` — in other words, quantize the weights so that the layer's *output*, on realistic inputs, changes as little as possible, rather than simply minimizing the raw distance between `W` and `Ŵ` in isolation (which is what naive round-to-nearest does, and which has no way to know that some weights matter far more than others depending on the activations they get multiplied against). This objective is exactly the right one, because a weight multiplied by activations that are almost always near zero can tolerate huge quantization error with almost no effect on the output, while a weight multiplied by consistently large activations needs to be quantized far more carefully.

GPTQ's approach traces its lineage directly to the Optimal Brain Surgeon (OBS) framework originally developed for neural network *pruning* in the 1990s, adapted here for quantization instead of outright weight removal. The key mathematical tool is the layer's Hessian with respect to this reconstruction loss, which for a squared-error objective of this form works out to be proportional to `H = 2 X Xᵀ`, computable directly from the calibration activations without ever needing to run a backward pass through the whole network. The Hessian captures, in second-order terms, how sensitive the output error is to perturbing each weight, and how those sensitivities interact across different weights — a weight with high curvature in this Hessian is one whose value matters a lot for output fidelity, and a weight with low curvature can absorb more rounding error more cheaply.

The algorithm then proceeds column by column: quantize one column of the weight matrix to the nearest representable value at the target bit-width, measure the error this quantization step introduced, and then — using the inverse of the Hessian — distribute that error as a compensating adjustment across all of the *not-yet-quantized* remaining columns, nudging them slightly to make up for the error just introduced. This is repeated column by column across the entire matrix, so that by the time you reach the last column, every previous quantization decision has already had its downstream damage partially absorbed and corrected for by adjustments made to later columns. Because this compensation makes the whole procedure numerically sensitive to the order and precision of Hessian inversion, the original GPTQ paper introduces some practical stability tricks — a Cholesky decomposition of the Hessian inverse computed once up front, plus a lazy-batch-update scheme — to make this tractable and numerically stable even for matrices with tens of thousands of rows and columns.

The practical upshot of all this machinery is that GPTQ is a genuinely **one-shot, post-training** method: no gradient descent through the model, no retraining loop, no backpropagation at all — just a single calibration forward pass to collect activation statistics, followed by this layer-by-layer reconstruction procedure. This is why GPTQ became so popular so quickly for the open-weight LLM community: quantizing a 175B-parameter-class model to 4-bit can be done in a few GPU-hours on a single high-end GPU, dramatically cheaper than any approach requiring even a light fine-tuning pass, let alone full retraining.

```python
import numpy as np

def gptq_quantize_layer(W: np.ndarray, X: np.ndarray, bits: int = 4, group_size: int = 128):
    """Heavily simplified illustration of the GPTQ column-by-column
    reconstruction idea (omits Cholesky-based numerical stabilization
    and lazy batching used in the real algorithm, to keep the core
    logic -- quantize a column, propagate the error via the inverse
    Hessian -- visible).

    W: (out_features, in_features) weight matrix
    X: (in_features, num_calibration_samples) calibration activations
    """
    out_features, in_features = W.shape
    H = 2 * (X @ X.T)                       # approximate Hessian, (in_features, in_features)
    H += np.eye(in_features) * 1e-4         # damping for numerical stability
    H_inv = np.linalg.inv(H)

    W_hat = W.copy()
    for col in range(in_features):
        group_start = (col // group_size) * group_size
        group = W_hat[:, group_start: group_start + group_size]
        scale = np.max(np.abs(group)) / (2 ** (bits - 1) - 1)

        w_col = W_hat[:, col]
        q_col = np.clip(np.round(w_col / scale), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
        w_col_hat = q_col * scale

        error = (w_col - w_col_hat) / H_inv[col, col]
        # propagate the quantization error of this column into all
        # not-yet-quantized columns, weighted by the inverse Hessian
        remaining = slice(col + 1, in_features)
        W_hat[:, remaining] -= np.outer(error, H_inv[col, remaining])
        W_hat[:, col] = w_col_hat

    return W_hat
```

## AWQ: Activation-Aware Weight Quantization

AWQ (Lin et al., 2023, "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration") starts from a different, arguably simpler, empirical observation: not all weights matter equally for preserving model quality, and — crucially — the weights that matter most are *not* the ones with the largest magnitude, but the ones that get multiplied by activations with the largest magnitude. The paper found that a very small fraction of weight channels, often well under 1%, correspond to consistently large-magnitude activations, and that protecting *just those* channels from aggressive quantization error preserves the vast majority of the model's downstream quality, while the remaining 99%+ of weights can be quantized fairly aggressively with comparatively little impact.

The naive way to exploit this observation would be to keep that salient 1% of weight channels in full fp16 precision and quantize the rest to INT4 — a mixed-precision scheme. AWQ deliberately avoids this, because mixed-precision storage at the individual-channel level is a poor fit for GPU kernel efficiency: GPU tensor cores are built for fast, uniform-precision matrix multiplication, and having to special-case a scattered 1% of channels at a different bit-width during the actual matmul either requires slow gather/scatter operations or an entirely separate, awkward compute path, largely defeating the speed benefit quantization was supposed to deliver in the first place.

AWQ's actual solution is more elegant: instead of physically storing the salient channels at higher precision, it **rescales** them before quantization. For each salient channel, the weight is multiplied by some factor `s > 1` before quantization, which spreads that channel's values across a wider range and improves the relative precision it retains after rounding (quantization error is roughly proportional to a value's magnitude relative to the fixed quantization step size, so a larger-magnitude value suffers proportionally less from the same absolute rounding error). To keep the actual layer output mathematically unchanged, the corresponding *activation* channel is divided by that same factor `s` before it hits the now-quantized weight — a purely mathematical identity, `(s * W) * (X / s) = W * X`, that requires no extra storage format and no mixed-precision kernel at all, since every weight is still quantized uniformly to the same bit-width; only the pre-quantization scaling factor differs per channel, and the compensating activation scale can typically be fused directly into a preceding layer's output (for instance, folded into a preceding LayerNorm's scale parameter) at essentially zero runtime cost. AWQ determines which channels are salient, and what scaling factor `s` to apply to each, by running calibration data through the model, observing which weight channels correspond to the largest activation magnitudes, and then doing a small grid search over candidate scaling factors that empirically minimizes output reconstruction error on that calibration data — a much cheaper search than GPTQ's full Hessian-based reconstruction, since it never needs to invert anything.

Because AWQ never reconstructs the weights based on a specific calibration set's *fine-grained* second-order statistics the way GPTQ does, it tends to be somewhat less prone to overfitting narrowly to the calibration distribution, and is reported in the AWQ paper to generalize better across evaluation domains than GPTQ at very low bit-widths (particularly INT3/INT4), while also being substantially cheaper and faster to run, since it needs no Hessian computation or matrix inversion. This is part of why AWQ has become a very popular default choice for 4-bit quantization of open-weight models in serving frameworks like vLLM and TGI, often used alongside or instead of GPTQ.

```python
def awq_apply_channel_scaling(W: np.ndarray, activation_magnitudes: np.ndarray,
                               salience_threshold_percentile: float = 99.0):
    """Illustrative sketch of AWQ's core trick: identify salient
    input channels (by activation magnitude, not weight magnitude),
    scale those weight channels UP before quantization, and record
    the compensating activation scale-down that must be applied
    at inference time to keep the math equivalent.
    """
    threshold = np.percentile(activation_magnitudes, salience_threshold_percentile)
    salient_channels = activation_magnitudes >= threshold   # boolean mask over in_features

    scales = np.ones(W.shape[1])
    scales[salient_channels] = 2.0   # in practice: found via a small grid search, not fixed

    W_scaled = W * scales[np.newaxis, :]        # scale up salient input channels
    activation_correction = 1.0 / scales        # divide the matching activations by the same factor
    return W_scaled, activation_correction
```

## GGUF and llama.cpp: Quantization for CPUs and Consumer Hardware

GPTQ and AWQ were both designed with GPU serving in mind — their quantized weights are packed and dequantized inside custom CUDA kernels (ExLlama, AutoAWQ, Marlin, and others) tuned specifically for NVIDIA tensor cores. A parallel ecosystem grew up around `llama.cpp` and its GGUF file format, aimed at a different deployment target entirely: efficient CPU inference, Apple Silicon (via Metal), and consumer-grade GPUs that don't necessarily have the specialized low-bit tensor-core support server GPUs do. This difference in target hardware is the real reason the two ecosystems diverged rather than converging on one shared format — CPU SIMD instructions and Apple's unified-memory architecture reward a different data layout and dequantization strategy than CUDA tensor cores do, and portability to hardware with no CUDA at all (a laptop, a phone, an old desktop) was the whole point of the llama.cpp project from its inception.

GGUF's quantization schemes, commonly called **k-quants** (named `Q2_K` through `Q8_0`, with popular middle-ground options like `Q4_K_M` and `Q5_K_M`), use a block-based mixed-precision structure: weights are split into small blocks (commonly 32 weights), each block gets its own scale (and sometimes its own minimum, for asymmetric variants), and blocks are further grouped into "superblocks" that share a higher-level scale, giving a two-level hierarchy of quantization granularity that trades off storage overhead against fidelity more finely than a flat per-group scheme. The "K" and "S"/"M"/"L" suffixes in names like `Q4_K_M` denote different mixes of which layers or tensors within the model get slightly higher versus lower effective bit-widths — it's common practice, for instance, to keep more sensitive tensors (the final output projection, the embedding table) at a higher effective precision than the bulk of the transformer's linear layers, since empirically those tensors are more quality-sensitive per bit spent on them. `llama.cpp` also supports an **importance matrix** (`imatrix`) calibration step, conceptually similar in spirit to GPTQ/AWQ's use of calibration data: it runs representative text through the model to weight which weights matter most for output fidelity, then biases the quantization/rounding decisions accordingly, which meaningfully improves quality at very low bit-widths (`Q2_K`, `Q3_K`) versus quantizing blind.

The practical reason this ecosystem matters for an interview answer, beyond the algorithmic detail, is deployment target: GGUF models are the default choice for running LLMs locally on a laptop, a Mac with unified memory, or a machine with no discrete GPU at all, whereas GPTQ/AWQ-quantized models are the default choice for GPU-based serving stacks like vLLM or TGI where CUDA kernel support for those specific quantization formats already exists. Knowing which ecosystem to reach for is as much a hardware-target question as it is a quality question, and it is a very common practical distinction that separates "I've read about quantization" from "I've actually deployed a quantized model."

## Serving-Side Optimization, Beyond Quantizing the Weights

Quantization shrinks what has to move through memory per token. The remaining large chunk of production LLM serving optimization is about how you schedule and batch requests, and how you manage the other big memory consumer discussed in file 004 — the KV cache — across many concurrent users. As a quick recap: every sequence being generated needs its own KV cache, that cache grows linearly with how many tokens have been generated so far, and it has to persist in GPU memory for the entire lifetime of the request, frequently exceeding the memory used by the model's weights themselves for long-context, high-concurrency workloads. The techniques below are all about serving many such requests efficiently and concurrently rather than about the per-layer computation itself.

### Continuous Batching

The naive way to batch LLM inference requests is exactly like batching in traditional deep learning: collect a fixed set of requests, pad them to a common length, run the whole batch through the model together, and don't start a new batch until every sequence in the current one has finished generating. This is called **static batching**, and it has an obvious, serious flaw for autoregressive generation specifically: request lengths vary enormously — one user's prompt might need 20 output tokens, another's 2000 — and a static batch can only finish, and free up its GPU slot for new requests, once its *longest-running* sequence completes. Every sequence that finished early sits idle, its slot in the batch wasted, computed on with padding tokens that contribute nothing, while the GPU keeps grinding through the one straggling sequence that needs many more tokens. At production request-volume and length variance, this wastes a very large fraction of total GPU-time.

**Continuous batching** (sometimes called iteration-level scheduling, a term introduced in the influential Orca paper) fixes this by scheduling at the granularity of individual decoding steps — token-generation iterations — rather than entire requests. After every single token-generation step, the scheduler checks which sequences in the current batch have just finished (hit an EOS token or their length limit) and immediately evicts them, freeing their slot; it then immediately admits new, freshly-arrived requests into those now-open slots, without waiting for every other sequence in the batch to also finish. The GPU's batch composition is therefore constantly churning, token-by-token, always kept as full as the available memory — largely bounded by KV-cache capacity — allows, rather than being locked to a fixed roster of requests for an entire generation.

```python
class ContinuousBatchScheduler:
    """Simplified illustration of iteration-level (continuous) batching.
    A real implementation (vLLM, TGI) also has to handle KV-cache
    allocation per admitted request, prefill vs. decode scheduling,
    and fairness/priority policies -- omitted here for clarity."""

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.active_requests: list = []
        self.pending_queue: list = []

    def step(self, model_step_fn):
        # 1. Run one decode step for every currently active request, together.
        finished = []
        for req in self.active_requests:
            next_token = model_step_fn(req)
            req.append(next_token)
            if req.is_finished():
                finished.append(req)

        # 2. Evict anything that just finished -- frees a slot immediately,
        #    rather than waiting for the whole batch to complete.
        for req in finished:
            self.active_requests.remove(req)

        # 3. Backfill freed slots with newly arrived requests right away.
        while len(self.active_requests) < self.max_batch_size and self.pending_queue:
            self.active_requests.append(self.pending_queue.pop(0))

        return finished
```

The result is dramatically higher GPU utilization and throughput under realistic, length-heterogeneous production traffic, without changing the model or the mathematical output of any individual request at all — it is a pure scheduling optimization. This is the scheduling strategy underlying essentially every modern production LLM serving stack (vLLM, TGI, TensorRT-LLM), and citing it correctly is a strong signal in an interview that you understand LLM serving as a systems problem, not just a modeling problem.

### PagedAttention and vLLM

Continuous batching solves *when* to admit and evict requests, but it doesn't solve a separate, equally serious problem: how to actually *lay out* each request's KV cache in GPU memory when you don't know in advance how long that request's generation will run. The naive approach pre-allocates a large, contiguous block of memory per sequence sized for the maximum possible sequence length "just in case" — but this wastes enormous amounts of memory on sequences that end up much shorter than the worst case (**internal fragmentation**), and because different requests need different, unpredictable amounts of contiguous memory, the GPU's memory pool also accumulates unusable gaps between allocations over time as requests come and go (**external fragmentation**). Empirically, naive contiguous KV-cache allocation was found, in the original vLLM/PagedAttention paper (Kwon et al., 2023), to waste 60-80% of allocated KV-cache memory to these two forms of fragmentation — a staggering amount of GPU memory doing nothing useful.

PagedAttention's solution is a direct, deliberate borrowing from how operating systems manage virtual memory. Instead of requiring each sequence's KV cache to live in one contiguous chunk of GPU memory, PagedAttention divides the KV cache into small, fixed-size **blocks** (analogous to OS memory pages — for instance, holding the K/V vectors for 16 tokens each), maintains a pool of physical blocks that can be allocated to any sequence as needed, and keeps a per-sequence **block table** that maps that sequence's *logical* token positions onto whatever *physical* blocks happen to be holding them, which need not be contiguous in memory at all. When a sequence needs one more block of KV-cache space, the system just grabs any free physical block from the pool and appends it to that sequence's block table; there's no need to have pre-reserved a large contiguous region up front, which eliminates internal fragmentation (blocks are only ever allocated as actually needed, in small fixed increments) and external fragmentation (every physical block is the same fixed size, so any freed block can satisfy any future request's next allocation, unlike variable-sized contiguous chunks, which can leave awkward unusable gaps).

The paging analogy pays an extra dividend beyond just fixing fragmentation: it makes memory **sharing** between sequences straightforward via the same copy-on-write mechanism operating systems use for shared memory pages. If multiple sequences share an identical prefix — the classic examples being several parallel samples in beam search that all branch from the same partial sequence, or many concurrent requests that all start with the same long system prompt — their block tables can simply point to the *same* physical blocks for that shared prefix, with a reference-counted block only actually being copied (or a private write triggered) if and when one of the sharing sequences needs to diverge and write something different into that region. This turns what would otherwise be fully duplicated KV cache across every sequence in a batch into a single shared physical copy for as long as the sequences remain identical, a substantial additional memory savings on top of eliminating fragmentation, particularly valuable for shared long system prompts or few-shot examples reused across a batch of otherwise-independent requests.

```python
class PagedKVCache:
    """Simplified illustration of PagedAttention's block-table indirection.
    Real vLLM implements this with custom CUDA kernels that gather
    across non-contiguous physical blocks efficiently during attention,
    plus reference counting for copy-on-write sharing -- omitted here."""

    def __init__(self, block_size: int, num_physical_blocks: int):
        self.block_size = block_size
        self.free_blocks = list(range(num_physical_blocks))
        self.block_tables: dict[int, list[int]] = {}   # seq_id -> [physical_block_ids]

    def append_token(self, seq_id: int, num_tokens_so_far: int):
        needs_new_block = (num_tokens_so_far % self.block_size) == 0
        if needs_new_block:
            physical_block = self.free_blocks.pop()   # grab any free block, not necessarily contiguous
            self.block_tables.setdefault(seq_id, []).append(physical_block)

    def fork_for_shared_prefix(self, parent_seq_id: int, child_seq_id: int):
        # Copy-on-write: child shares the SAME physical blocks initially.
        self.block_tables[child_seq_id] = list(self.block_tables[parent_seq_id])
```

vLLM, the serving framework built around PagedAttention, combines this memory-management scheme with the continuous batching described above, and the pairing of the two is why vLLM became, essentially overnight upon its release, one of the dominant open-source LLM inference servers — they attack the two biggest sources of wasted GPU capacity, scheduling idle time and memory fragmentation, simultaneously and largely independently of each other.

### Speculative Decoding

Everything so far has been about serving many requests efficiently; speculative decoding attacks a different axis entirely — making a *single* sequence's generation faster, by restructuring how many tokens get produced per expensive forward pass through the large model. The starting observation is that autoregressive generation is fundamentally sequential and, per the memory-bandwidth analysis at the top of this chapter, each single-token forward pass through a large model spends most of its time simply streaming the model's weights through HBM rather than doing useful arithmetic. That means if you could somehow verify *several* candidate tokens in the same amount of time it takes to generate *one*, you'd get a large reduction in wall-clock time almost for free, since the memory traffic to stream the weights is paid once regardless of how many token positions you evaluate against them in that pass.

Speculative decoding (Leviathan et al. 2023; Chen et al. 2023, both proposing essentially the same idea concurrently) exploits exactly this by using two models: a small, fast **draft model** — which could be a much smaller model from the same family, or a smaller "head" trained alongside the main model — and the large **target model** whose output distribution you actually want. At each round, the draft model generates several candidate tokens autoregressively, one at a time, fast because it's small and each of its own forward passes is cheap. Then, crucially, the target model evaluates *all* of those drafted tokens **in a single forward pass**, since verifying a proposed continuation is just a normal forward pass over that continuation's positions, which can be done in parallel across all the drafted positions at once — much like how a full prompt is processed in one shot during prefill — rather than one at a time. This single expensive pass yields the target model's true probability distribution at every one of those drafted positions simultaneously.

The final, and most theoretically important, piece is the acceptance/rejection scheme that decides which of the drafted tokens actually get kept. For each position, let `p(x)` be the target model's probability for the drafted token `x` and `q(x)` be the draft model's probability for that same token — the probability it assigned when it proposed `x`. The token is accepted with probability `min(1, p(x)/q(x))`: if the target model considers the drafted token at least as likely as the draft model did, it's accepted unconditionally (probability 1); if the target model considers it *less* likely than the draft model did, it's accepted only proportionally, with probability `p(x)/q(x)`, reflecting how much less the target model likes it. The moment a token is rejected, every drafted token after it in that round is discarded, since they were generated conditioned on now-discarded context, and a *replacement* token is resampled at that position from a specifically adjusted "residual" distribution, `max(0, p(x) - q(x))` renormalized to sum to 1 — the leftover probability mass in the target model's distribution that the draft model under-weighted.

This acceptance/rejection scheme is not a heuristic approximation — it is a form of rejection sampling specifically constructed so that the *marginal* distribution of the accepted-or-resampled tokens is mathematically identical to what you would get by sampling from the target model alone, token by token, with no draft model involved at all. This is the property that makes speculative decoding a genuinely lossless speedup technique: it changes nothing about the output distribution, and by extension nothing about quality in the aggregate statistical sense — it only changes how many expensive target-model forward passes are needed to produce a given number of tokens. The realized speedup depends heavily on the **acceptance rate** — how often the draft model's guesses actually match what the target model would have chosen — which in turn depends on how well-matched the draft model's output distribution is to the target model's (a draft model trained on similar data, or literally distilled from the target, tends to have much higher agreement than an unrelated small model) and on the entropy of the generation task itself: near-deterministic completions, like finishing a common phrase or emitting boilerplate code syntax, have very high acceptance rates and see the largest speedups, while highly creative or unpredictable generation sees smaller gains, since the draft model's guesses diverge from the target's more often.

```python
import numpy as np

def speculative_decoding_round(draft_probs: list[np.ndarray], target_probs: list[np.ndarray],
                                drafted_tokens: list[int], rng: np.random.Generator):
    """
    draft_probs[i], target_probs[i]: full vocab distributions at drafted
    position i, from the draft and target model respectively.
    drafted_tokens[i]: the token the draft model actually sampled at position i.

    Returns the list of accepted tokens plus exactly one resampled
    "correction" token at the first rejection, matching what
    target-model-only sampling would have produced in distribution.
    """
    accepted = []
    for i, tok in enumerate(drafted_tokens):
        p_target = target_probs[i][tok]
        q_draft = draft_probs[i][tok]
        accept_prob = min(1.0, p_target / max(q_draft, 1e-12))

        if rng.random() < accept_prob:
            accepted.append(tok)
            continue

        # Rejected: resample from the residual distribution max(0, p - q), renormalized.
        residual = np.clip(target_probs[i] - draft_probs[i], a_min=0, a_max=None)
        residual_sum = residual.sum()
        if residual_sum > 0:
            residual /= residual_sum
            correction_token = int(rng.choice(len(residual), p=residual))
        else:
            correction_token = int(np.argmax(target_probs[i]))
        accepted.append(correction_token)
        break   # everything drafted after a rejection is discarded

    return accepted
```

Speculative decoding composes cleanly with everything else in this chapter: the draft and target models can each independently be quantized for further memory/bandwidth savings, and the technique works underneath continuous batching and PagedAttention-style KV-cache management in modern serving stacks without conflicting with either — it is purely about restructuring the *sequence* of forward passes for a given request, orthogonal to how the serving system schedules and lays out memory across many concurrent requests.

## Summary: The Four Pillars of Modern LLM Inference Optimization

Quantization (shrinking what has to move through memory per byte), continuous batching (keeping the GPU saturated with useful work across many concurrent requests), PagedAttention (eliminating wasted, fragmented KV-cache memory and enabling cheap sharing), and speculative decoding (extracting more tokens per expensive forward pass) represent the four pillars of the modern LLM inference-optimization stack, and they are largely orthogonal to each other — a production serving system typically runs all four simultaneously. A strong interview answer to "how would you make LLM serving faster and cheaper" should be able to name and explain the actual mechanism behind each of them, tie the motivation for every single one back to the same root cause — decoding is memory-bandwidth-bound, not compute-bound — and be able to say concretely which combination you'd reach for first given a specific latency, throughput, or cost constraint, rather than reciting the names as an undifferentiated list of buzzwords.
