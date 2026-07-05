# Mixed Precision Training and Numerical Stability

## 1. Floating point from first principles: what exponent and mantissa bits actually buy you

Every floating-point format used in training splits its bits into three fields: a **sign** bit, an
**exponent** field, and a **mantissa** (significand/fraction) field. A value is reconstructed
roughly as `(-1)^sign × 1.mantissa × 2^(exponent - bias)`. The two fields that matter for this
discussion trade off against each other in a way that is completely determined by the bit layout,
not by any clever encoding trick:

- **Exponent bits determine dynamic range** — the ratio between the largest and smallest
representable magnitude. Each additional exponent bit roughly *squares* the representable range
(doubles the range of exponents, which is itself an exponent — so range grows doubly-exponentially
in exponent-bit count).
- **Mantissa bits determine precision** — the number of significant digits, and hence the relative
rounding error (machine epsilon) of any single representable value. Each additional mantissa bit
*halves* the relative rounding error (roughly `2^-mantissa_bits` as the relative quantization step
size, ULP).

This is a genuine trade-off, not a design choice you can dodge: for a fixed total bit budget (e.g.,
16 bits), giving a bit to the exponent field necessarily takes it away from the mantissa field,
buying more range at the cost of precision, or vice versa.

| Format | Sign | Exponent | Mantissa | Dynamic range (approx.) | Relative precision (approx. ULP) |
|---|---|---|---|---|---|
| fp32 | 1 | 8 | 23 | ~1e-38 to ~3e38 | ~1.2e-7 |
| fp16 (IEEE half) | 1 | 5 | 10 | ~6e-5 to ~6.5e4 | ~9.8e-4 |
| bf16 | 1 | 8 | 7 | ~1e-38 to ~3e38 (same as fp32) | ~7.8e-3 |
| fp8 E4M3 | 1 | 4 | 3 | ~2e-3 to ~448 | ~6.25e-2 |
| fp8 E5M2 | 1 | 5 | 2 | ~6e-5 to ~5.7e4 | ~1.25e-1 |

Read this table as the single most important artifact in this file. The two fp8 rows exist as a
deliberate pair for exactly the reason the trade-off above predicts: **E4M3** gives up range for
precision (3 mantissa bits, narrower range) and is used for values whose magnitude is comparatively
well-behaved and where precision matters most — typically forward-pass weights and activations.
**E5M2** gives up precision for range (only 2 mantissa bits, but 5 exponent bits matching fp16's
range) and is used where large dynamic range matters more than fine precision — typically gradients
in the backward pass, which are well known to have heavier-tailed, more extreme-magnitude
distributions than forward activations.

## 2. Why bf16 displaced fp16 as the default

fp16 and bf16 both use 16 bits total, but allocate them completely differently (5 exponent/10
mantissa vs. 8 exponent/7 mantissa), and the practical consequence of that difference is the entire
reason bf16 became the default on Ampere/Hopper-class GPUs and TPUs despite being, bit-for-bit, the
*less precise* of the two.

**fp16's problem: narrow dynamic range.** fp16's exponent field is only 5 bits (matching, not
coincidentally, fp8's E5M2 pattern), giving a representable range of roughly `6×10^-5` to
`6.5×10^4`. This sounds like a lot until you consider what actually flows through a transformer
during training: gradients, in particular, routinely take on values many orders of magnitude smaller
than the loss itself (vanishing-gradient-style shrinkage compounding through many layers), and it is
entirely normal for legitimate, useful gradient values to fall below fp16's minimum representable
magnitude — at which point they **underflow to exactly zero**, silently discarding a real training
signal rather than erroring out.

**The fp16-era fix: loss scaling.** Because the *relative* precision problem (mantissa bits) is not
what's failing here — it's specifically that small-magnitude values are falling outside the
representable range entirely — the standard mitigation is to artificially shift values into the
representable range before they get small enough to underflow, then shift them back. Concretely:
multiply the loss by a scalar `S` (e.g., `S = 2^15`) immediately before calling backward. Because
gradients are linear in the loss (by the chain rule), every gradient throughout the entire backward
pass is scaled by the same factor `S`, pushing what would have been sub-`6×10^-5` values up into
fp16's representable range. After backward completes, divide the accumulated gradients by `S` before
the optimizer step (this final division is done in fp32, so it does not reintroduce fp16's range
limits at the point where the values are being shrunk back down).

```
# Static loss scaling
S = 2**15
scaled_loss = loss * S
scaled_loss.backward()                      # every grad in the graph is now scaled by S
for p in model.parameters():
    p.grad = p.grad.float() / S             # unscale in fp32 before the optimizer touches it
optimizer.step()
```

**Dynamic loss scaling** (the practical version used everywhere, rather than a fixed `S`) adapts `S`
automatically: after each backward pass, check whether any gradient overflowed to `inf`/`NaN` (which
indicates `S` was too aggressive and pushed some *already large* gradient beyond fp16's *maximum*
representable value — the range problem cuts both ways). If an overflow is detected, **skip that
optimizer step entirely** (the corrupted gradients must not be applied) and halve `S`. If many
consecutive steps pass without overflow (a common heuristic: after some fixed window, e.g., 2000
steps, of no overflow), increase `S` (commonly by 2x) to push closer to the edge of usable range
again, since a larger `S` better protects the smallest gradients from underflowing.

```
class DynamicLossScaler:
    def __init__(self, init_scale=2**15, growth_interval=2000, growth_factor=2.0, backoff_factor=0.5):
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self._good_steps = 0

    def step(self, grads):
        overflowed = any(not torch.isfinite(g).all() for g in grads if g is not None)
        if overflowed:
            self.scale *= self.backoff_factor
            self._good_steps = 0
            return False                     # signal caller: SKIP this optimizer step
        self._good_steps += 1
        if self._good_steps >= self.growth_interval:
            self.scale *= self.growth_factor
            self._good_steps = 0
        return True                          # safe to apply this step's update
```

**bf16's fix: sidestep the problem instead of managing it.** bf16 uses the *same* 8-bit exponent
field as fp32, so its dynamic range is identical to fp32's — the range problem that motivates loss
scaling essentially does not arise, because any value fp32 could represent without
under/overflowing, bf16 can also represent without under/overflowing (it just represents it less
*precisely*, at ~7.8e-3 relative error instead of fp32's ~1.2e-7). This is why bf16 training
pipelines typically **do not need loss scaling at all** — a real operational simplification,
removing an entire class of hyperparameters (initial scale, growth interval, growth/backoff factors)
and an entire class of "step silently skipped due to overflow" failure modes from the training loop.

**The trade bf16 accepts in return.** bf16's 7 mantissa bits give coarser relative precision than
fp16's 10 — roughly 8x worse ULP. In practice, this has turned out to be a trade large-model
training is generally willing to make: large transformers trained in bf16 converge reliably without
loss scaling, and the coarser per-value precision has not been the dominant source of instability at
scale (unlike fp16's range limitation, which was a frequent, operationally painful source of NaN
losses and wasted steps in the fp16 era). This is the concrete reason — stated precisely, not just
"bf16 is better" — that bf16 became the default mixed-precision format once Ampere-class hardware
gave it native tensor-core support: **it eliminates an entire operational failure mode
(range-induced under/overflow) at the cost of a precision reduction that has empirically not been
the binding constraint for transformer training**, whereas fp16 keeps better precision but requires
actively managing a failure mode that bf16 doesn't have.

## 3. FP8: the next step down, and why it needs finer-grained scaling than a single factor per tensor

H100-class tensor cores add native FP8 support (both E4M3 and E5M2, Section 1), roughly doubling
matmul throughput again over bf16 and halving memory for the tensors stored in FP8. But FP8's
mantissa is only 2–3 bits — an order of magnitude coarser precision than bf16's
already-coarser-than-fp16 7 bits — and a single global scale factor (the loss-scaling-style trick
from Section 2, generalized to "tensor scaling": multiply an entire tensor by one scalar before
casting to fp8, divide back out after) runs into a **new** problem that global loss scaling for fp16
gradients didn't have to contend with as severely: **within a single large activation or weight
tensor, different elements can have magnitudes that differ by orders of magnitude from each other**
(e.g., a handful of outlier activation channels that are far larger than the typical element — a
well-documented phenomenon in transformer activations). A single scale factor for the whole tensor
has to accommodate the *largest* element to avoid overflow, which means every *smaller* element gets
pushed down into the bottom of FP8's tiny (2–3 bit) mantissa range, where the relative quantization
error is worst — precisely the elements a global scale factor fails to protect.

**Fine-grained (tile/block-wise) scaling** is the fix, and it's the mechanism
`..\OpenSource\007_DeepSeek_V3.md` describes concretely for its FP8 training recipe: rather than one
scale factor per tensor, compute a separate scale factor for each small **tile** of an activation
tensor (DeepSeek-V3 uses 1×128 element groups) and each small **block** of a weight tensor (128×128
blocks). Each tile/block gets its own scale, sized to that tile/block's own local maximum magnitude
— so an outlier in one tile does not force every *other* tile's elements down into fp8's
low-precision floor. This directly targets the failure mode described above: it localizes the
"protect against the largest element" cost to a small group of nearby elements (which tend to be
more homogeneous in magnitude than the tensor as a whole) rather than paying that cost globally.

```
# Conceptual difference between per-tensor and tile-wise fp8 quantization
def quantize_per_tensor(x, fp8_max=448.0):
    scale = fp8_max / x.abs().max()             # ONE scale for the whole tensor
    return (x * scale).to(fp8_e4m3), scale

def quantize_tilewise(x, tile_size=128, fp8_max=448.0):
    # x shape: [..., N], split last dim into tiles of `tile_size`
    tiles = x.reshape(*x.shape[:-1], -1, tile_size)
    tile_max = tiles.abs().amax(dim=-1, keepdim=True)
    scales = fp8_max / tile_max.clamp(min=1e-12)   # one scale PER TILE, not per tensor
    quantized = (tiles * scales).to(fp8_e4m3)
    return quantized.reshape_as(x), scales
```

**What DeepSeek-V3 additionally keeps in higher precision, and why.** Not everything runs in FP8.
`..\OpenSource\007_DeepSeek_V3.md` notes that certain accumulations — reductions where quantization
error would compound across many summed terms rather than staying local — and the optimizer state
itself are kept in bf16/fp32 rather than FP8. This is a direct instance of a general principle worth
stating precisely in an interview: **quantization error from a single low-precision multiply is
usually tolerable; quantization error accumulated across a long reduction (e.g., summing many
partial products, or updating optimizer state over many thousands of steps) compounds, and
compounding error is what actually degrades convergence over a long run — so the engineering
judgment is about *where in the computation graph error can compound* rather than a blanket "which
tensors are FP8."** Tensor-core hardware itself reflects this same principle even within a single
FP8 matmul: FP8 inputs are multiplied at FP8 precision, but the accumulation of the many products
within the matmul is still done in a higher-precision accumulator internally (typically fp32) — the
hardware does not compound rounding error across an entire dot product's worth of additions even
when the inputs are FP8.

## 4. Real numerical-instability failure modes and their observable symptoms

This section is the part of mixed-precision training most directly testable via "here's a loss curve
/ gradient-norm plot, diagnose it" interview questions.

**Overflow (activations or gradients hit `inf`).** In fp16 or FP8, a value whose true magnitude
exceeds the format's maximum representable value becomes `inf`. Once any single activation is `inf`,
it propagates almost immediately: any arithmetic operation involving an `inf` typically produces
`inf` or `NaN`, and this cascades through the rest of the forward pass and then through backward,
corrupting every downstream gradient. **Observable symptom:** the loss value itself becomes `NaN`
within one or two steps of the initial overflow — usually an abrupt, unmistakable event rather than
a gradual drift, which makes overflow the *easier* of the failure modes to detect (a simple
`torch.isfinite(loss)` check catches it immediately), but by the time it's caught, the step (or
several recent steps, if checkpointing/detection lag exists) needs to be skipped or rolled back.

**Underflow (small-magnitude values silently become zero).** Unlike overflow, underflow does **not**
crash anything — a value that rounds to zero is a perfectly valid, finite floating-point number, so
no `NaN`/`inf` check will ever catch it. **Observable symptom:** far subtler and harder to diagnose
— specific parameters (commonly ones deep in the network, receiving gradients that have already
shrunk through many layers of backward chain-rule multiplication) simply stop receiving meaningful
updates, which shows up as the loss curve plateauing earlier than expected, specific layers'
weight-update-norm tracking near zero while other layers' norms look healthy, or degraded downstream
task performance that doesn't correspond to any visible anomaly in the aggregate loss curve at all.
This is precisely the failure mode loss scaling (Section 2) exists to prevent for fp16, and
precisely the failure mode bf16's wider exponent range structurally avoids without needing loss
scaling — a good interview answer connects "why does bf16 not need loss scaling" directly back to
"underflow is silent and therefore dangerous, and bf16's range makes it far less likely to occur in
the first place."

**Gradient norm spikes as a leading indicator of impending divergence.** Engineers monitoring a live
training run watch the (typically per-step, sometimes per-layer) gradient norm as a leading
indicator, because a spike in gradient norm — a sudden jump to many multiples of the recent
running-average norm — very often *precedes* a visible loss spike or outright divergence by one or a
handful of steps, rather than coinciding with it. The mechanism: an unusually large parameter update
(driven by that outsized gradient) can push the model into a region of the loss landscape where the
*next* forward pass produces even larger activations, which produce even larger gradients on the
following backward pass — a positive feedback loop that, left unchecked, terminates in outright
numerical overflow within a small number of further steps. **This is exactly what gradient clipping
is a defense against**: capping the gradient norm to some fixed maximum (commonly 1.0) before the
optimizer step directly breaks this feedback loop by preventing any single step's update from being
disproportionately large, regardless of what produced the outsized gradient (a genuinely bad batch,
an emergent instability, or a precision artifact). A production monitoring setup should treat
"gradient norm crossed some multiple of its trailing moving average" as an alert condition worth
investigating *before* the loss curve itself shows a problem, not after.

**Loss-scaler skip-rate as a diagnostic signal (fp16-era, still relevant wherever fp16 or FP8 with
dynamic scaling is used).** The dynamic loss scaler from Section 2 logs, implicitly, how often it
skips an optimizer step due to detected overflow. An occasional skip (well under 1% of steps) is
normal and expected — it's the mechanism working as intended, occasionally probing the edge of
representable range and backing off. A **rising or persistently high skip rate**, however, is itself
a symptom worth escalating: it can indicate the model has entered a genuinely less numerically
stable regime (e.g., an LR that's too high for the current phase of training, or an
architectural/data change that increased activation magnitudes), separate from — and often preceding
— a more dramatic divergence event. This is a case where a metric that looks like "infrastructure
plumbing" (how the mixed-precision machinery is behaving) is actually a useful early-warning signal
about the *training dynamics themselves*, and a staff-level engineer should know to look at it
rather than treating it as pure implementation detail invisible to model-quality debugging.

**"The loss curve looks fine but downstream eval quietly degrades."** The hardest failure mode to
catch, because by construction it produces no crash and no obvious anomaly in the most commonly
watched metric (training loss). This is the plausible symptom of a precision choice that is
*technically* introducing systematic bias (e.g., an accumulation that should have used a
higher-precision accumulator but didn't, quietly and consistently rounding in a particular direction
over millions of steps) without producing anything dramatic enough to show up as a visible spike.
The mitigation is procedural rather than a single metric to watch: periodic, held-out evaluation
throughout training (not only at the end), and — when introducing any new precision recipe (e.g.,
adopting FP8 for the first time) — an explicit ablation against a known-good bf16 baseline on a
subset of training, comparing not just final loss but downstream benchmark performance, precisely
because loss alone has been shown (including in DeepSeek-V3's own reported validation of its FP8
recipe against a bf16 reference) to be an imperfect proxy for whether a precision change has
introduced a subtle quality regression.

## 5. Summary: the operational framing to carry into an interview

State the progression fp32 → fp16 → bf16 → fp8 not as "increasingly aggressive precision reduction"
in the abstract, but as a sequence of trades along the exponent/mantissa axis from Section 1, each
with a *specific, nameable* operational consequence: fp16's narrow range creates a silent-underflow
/ occasional-overflow problem that loss scaling manages at the cost of extra machinery and
occasional skipped steps; bf16 trades away precision to eliminate that range problem structurally,
becoming the default because the operational win outweighed the precision cost empirically; fp8
pushes precision down far enough that even bf16-style tensor-wide handling isn't sufficient,
requiring fine-grained (tile/block) scaling to keep local quantization error controlled, plus
deliberate retention of higher precision specifically at points in the computation where error
compounds (long reductions, optimizer state) rather than everywhere. Every numerical-instability
symptom in Section 4 — abrupt NaN loss, silent underflow-induced plateauing, leading-indicator
gradient-norm spikes, rising loss-scaler skip rates, and quietly degrading eval despite a
clean-looking loss curve — is a direct, predictable consequence of where a specific tensor sits on
this range/precision trade-off, not a mysterious or purely empirical phenomenon; being able to trace
a specific symptom back to a specific bit-allocation cause is the level of fluency this topic is
tested for at staff level.
