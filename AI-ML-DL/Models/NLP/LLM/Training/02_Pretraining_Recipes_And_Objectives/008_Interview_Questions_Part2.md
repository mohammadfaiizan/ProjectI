# Interview Questions — Part 2

Covers: optimizers and hyperparameters, multi-stage curricula (long-context extension, annealing), and ablation/research methodology. See `007_Interview_Questions_Part1.md` for the first 20 questions; no question is repeated across the two files.

---

## Q21: Derive AdamW's bias-correction terms from the moment update recursion, and explain concretely what breaks in the first several training steps if you omit them.

The first-moment update is \(m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t\), initialized at \(m_0 = 0\). Unrolling the recursion: \(m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i\). Taking an expectation under the simplifying assumption that the gradient's true mean is roughly stationary over this window, \(\mathbb{E}[g_i]\approx\mathbb{E}[g_t]\) for \(i\) near \(t\): \(\mathbb{E}[m_t] \approx \mathbb{E}[g_t](1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} = \mathbb{E}[g_t](1-\beta_1^t)\). So \(m_t\) is a biased estimator of the true gradient mean, biased low by exactly the factor \((1-\beta_1^t)\) — and this factor is small precisely when \(t\) is small (e.g., at \(t=1\), \(1-\beta_1^1 = 1-\beta_1 = 0.1\) for \(\beta_1=0.9\), meaning \(m_1\) is only 10% of the true gradient magnitude in expectation). The identical argument applies to \(v_t\) with \(\beta_2\) in place of \(\beta_1\). Bias correction divides out exactly this factor: \(\hat m_t = m_t/(1-\beta_1^t)\), \(\hat v_t = v_t/(1-\beta_2^t)\), recovering an unbiased estimate under the stationarity assumption, with both denominators approaching 1 (correction vanishing) as \(t\) grows.

Concretely, without bias correction, the very first update uses \(m_1 = (1-\beta_1)g_1\) and \(v_1=(1-\beta_2)g_1^2\) directly. The ratio \(m_1/\sqrt{v_1}\) that determines the update *direction and relative scale* actually simplifies to \(\frac{(1-\beta_1)g_1}{\sqrt{(1-\beta_2)}|g_1|} = \frac{1-\beta_1}{\sqrt{1-\beta_2}}\text{sign}(g_1)\) — a constant multiple of \(\text{sign}(g_1)\), independent of the gradient's actual magnitude, and with a constant factor that depends entirely on the specific \(\beta_1,\beta_2\) chosen (for \(\beta_1=0.9,\beta_2=0.95\): \(\frac{0.1}{\sqrt{0.05}}\approx0.447\), an arbitrary-looking scale with no principled connection to the intended per-parameter adaptive step size). Bias correction cancels both factors of \((1-\beta_1)\)-type terms via the denominators, restoring \(\hat m_1/\sqrt{\hat v_1} = g_1/|g_1| = \text{sign}(g_1)\) at step 1 — the update at step 1 collapses to (a properly-scaled) sign of the gradient rather than an uncorrected, oddly-scaled fraction of it. Practically, omitting bias correction means the first several dozen-to-hundred steps of training take systematically undersized (and \(\beta_1,\beta_2\)-choice-dependent, rather than principled) steps, right when the model is moving away from initialization — exactly the regime warmup (Q24) is separately trying to protect, so the two mechanisms (bias correction and warmup) are solving adjacent but distinct instabilities in the same early-training window.

---

## Q22: Implement AdamW from scratch, including decoupled weight decay and gradient clipping applied as part of the same optimizer step (global-norm clipping across all parameters passed to a single call).

```python
import numpy as np

class AdamWWithClipping:
    def __init__(self, param_shapes: dict[str, tuple], lr=3e-4, beta1=0.9, beta2=0.95,
                 eps=1e-8, weight_decay=0.1, max_grad_norm=1.0):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.wd, self.max_grad_norm = weight_decay, max_grad_norm
        self.m = {k: np.zeros(shape) for k, shape in param_shapes.items()}
        self.v = {k: np.zeros(shape) for k, shape in param_shapes.items()}
        self.t = 0

    def _clip_grads(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        total_sq = sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values())
        total_norm = total_sq ** 0.5
        scale = min(1.0, self.max_grad_norm / (total_norm + 1e-6))
        return {k: g * scale for k, g in grads.items()}

    def step(self, params: dict[str, np.ndarray],
              grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.t += 1
        grads = self._clip_grads(grads)          # clip BEFORE moment updates
        new_params = {}
        for k, theta in params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            # decoupled weight decay: independent of gradient/second-moment statistics
            theta = theta - self.lr * self.wd * theta
            theta = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            new_params[k] = theta
        return new_params
```

Two design points worth calling out unprompted: clipping is applied to the **raw gradients, once, globally across every parameter tensor, before any moment-buffer update** — clipping after computing \(m_t, v_t\) would be clipping an already-smoothed quantity rather than protecting the moment buffers themselves from being corrupted by one anomalously large batch, which defeats much of the point. And weight decay is applied as a **separate multiplicative shrinkage of `theta` itself**, never folded into `g` before it hits `self.m`/`self.v` — folding it into the gradient would make the decay's effective strength depend on that parameter's accumulated \(v_t\) (Q23), exactly the coupling AdamW is designed to avoid.

---

## Q23: Here is an AdamW implementation with a subtle but consequential bug. Identify it and explain why it silently degrades training rather than crashing.

```python
def broken_adamw_step(theta, grad, m, v, t, lr=3e-4, beta1=0.9, beta2=0.95,
                        eps=1e-8, weight_decay=0.1):
    grad = grad + weight_decay * theta        # <-- weight decay folded into grad
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v
```

The bug is on the first line: `weight_decay * theta` is added directly into `grad` before the moment updates, which is **L2-regularized Adam, not AdamW** — this is exactly the "decoupling" that Loshchilov & Hutter's paper identifies as the wrong way to combine weight decay with an adaptive-moment optimizer, described in `004_Optimizers_LR_Schedules_And_Hyperparameters.md` Section 1.4. It doesn't crash because it's not a type error or a shape mismatch — it's a numerically well-defined computation that produces plausible-looking, converging-looking loss curves in most cases, which is exactly why this class of bug is dangerous: the failure mode is *quality*, not *correctness in the crash sense*.

The concrete mechanism: because the decay term `weight_decay * theta` is now part of `grad`, it flows through the second-moment computation `v = beta2*v + (1-beta2)*grad**2` and gets divided by `sqrt(v_hat)` in the final update, exactly like the "real" gradient signal does. A parameter with a large accumulated `v` (because it has a genuinely large or noisy real-gradient history) has its decay contribution *shrunk* by that same large `sqrt(v_hat)` denominator — so the *effective* decay strength ends up small for exactly the parameters with the largest gradient-noise history, and comparatively larger for parameters with small, quiet gradient histories, entirely as a side effect of each parameter's own `v_hat`, with no relationship to what you'd actually want weight decay's strength to depend on. In practice this tends to under-regularize the parameters that would most benefit from stable, uniform shrinkage (typically the larger, more actively-updated weight matrices) while over-regularizing quieter parameters, producing a model that trains and appears to converge but with different (usually worse, less well-calibrated) generalization/regularization behavior than a correctly-decoupled implementation — a difference you would only reliably catch via a proper ablation comparing final eval loss/downstream benchmarks against a correct AdamW implementation, not by watching the training loss curve for a crash or an obvious divergence. The fix: apply weight decay as a direct multiplicative shrinkage of `theta`, outside the gradient/moment pipeline entirely, exactly as in Q22's `step` method.

---

## Q24: Explain, mechanistically, why learning-rate warmup specifically matters more at large batch size and large model scale than it does for a small model trained with a small batch size.

Two compounding effects, and a complete answer names both rather than gesturing at "big models are more unstable." First, the bias-correction argument from Q21: at step \(t=1\), \(v_1\) is estimated from a single gradient sample (or, at large batch size, a single *batch's* averaged gradient — still one sample of the underlying per-step gradient distribution as far as the moment-tracking recursion is concerned), and this estimate is maximally noisy and unreliable regardless of model or batch size. What changes with scale is the *consequence* of applying a large, tuned-for-later-in-training peak LR against that unreliable early estimate: at large model scale, the loss landscape near initialization is comparatively poorly-conditioned relative to the landscape the peak LR was actually tuned against (the landscape the model will occupy later in training, once it has moved meaningfully away from initialization) — an oversized early step at large scale is more likely to land somewhere the loss genuinely explodes, rather than just somewhere temporarily suboptimal, than the same relative misstep would at small scale.

Second, and more directly connected to batch size specifically: per `004_Optimizers_LR_Schedules_And_Hyperparameters.md` Section 4.2, the standard practice pairs a **larger batch size with a larger peak LR** (via the linear or sqrt scaling rules), on the premise that larger-batch gradient estimates are less noisy and can safely support a larger step. But that premise is exactly what's violated in the first few steps of training regardless of batch size — early-step noise in the *moment estimates* isn't fixed by a large batch size, because it's a function of how many *steps* of history \(m_t,v_t\) have accumulated, not how large any single step's batch is. So a large-batch/large-LR configuration is specifically pairing an aggressive step size with the exact regime (early steps) where the adaptive-moment machinery is least trustworthy, compounding the poor-early-conditioning problem from the previous paragraph. Warmup's job is to hold the LR down specifically during this window — giving the moment estimates time to accumulate a meaningful history and giving the model time to move into better-conditioned territory — before applying the full, aggressive peak LR that a large-batch configuration calls for. This is precisely why warmup length is typically tuned jointly with batch size and peak LR, not treated as an independent hyperparameter.

---

## Q25: What is the "critical batch size," and how does it determine whether you should use the linear or the square-root learning-rate scaling rule when increasing batch size?

The critical batch size (McCandlish et al., 2018, framed via the "gradient noise scale") marks the point past which increasing batch size stops buying proportional training-efficiency gains. Below it, each additional sample in a batch meaningfully reduces the gradient estimate's variance, and that reduced noise lets you safely take a correspondingly larger step — in this regime, doubling batch size and doubling learning rate (the **linear scaling rule**, \(\eta \propto B\)) roughly preserves the same effective per-token training progress, because the extra step size is justified by the genuinely lower estimate noise. Past the critical batch size, the gradient estimate is already precise enough that adding more samples mostly reduces an already-small noise floor rather than unlocking a correspondingly larger safe step size — curvature of the loss landscape, not gradient noise, becomes the binding constraint on how large a step can safely be taken, and continuing to scale LR linearly with batch size in this regime risks overshooting past what the landscape's curvature can tolerate, regardless of how precise the gradient estimate is. This is where the more conservative **square-root scaling rule** (\(\eta \propto \sqrt B\)) tends to hold better empirically — it still grows LR with batch size, but more cautiously, reflecting that gradient-noise reduction is no longer the dominant justification for a larger step.

Operationally, the critical batch size is not a fixed universal number — it depends on the specific model, data distribution, and point in training (it's known to grow over the course of training in many settings, meaning very large batch sizes become *more* justifiable later in training than they are early on) — so determining which scaling rule is appropriate at a candidate batch size requires empirically estimating where that batch size sits relative to the (model- and stage-specific) critical batch size, typically via the same small-scale proxy-run methodology used elsewhere in this module (`004_Optimizers_LR_Schedules_And_Hyperparameters.md` Section 4.2 and `006_Pretraining_Ablations_And_Research_Methodology.md`), rather than assumed a priori from either rule in isolation.

---

## Q26: Your cluster has 2048 GPUs sitting comparatively underutilized because your current global batch size is too small to parallelize across all of them efficiently at a reasonable per-replica micro-batch size. You want to increase batch size to use the cluster fully and cut wall-clock training time. Walk through what you need to check before doing this.

The core risk is that increasing batch size is not a free lever — it interacts directly with learning rate and with token efficiency, and doing it carelessly can either destabilize training or make training *more* token-hungry for the same final loss even while it's faster in wall-clock terms, which may or may not be the trade you actually want. Concretely, I'd walk through, in order: first, estimate (or look up, if already characterized for a similar model/data setup) roughly where the **critical batch size** (Q25) sits relative to my current and proposed target batch size — if the proposed batch size is still comfortably below critical, the linear scaling rule is a reasonable starting point for how much to raise peak LR alongside it; if it's at or past critical, I should expect diminishing (or negative) token-efficiency returns and should use the more conservative sqrt rule, and should go in expecting that this change trades some token efficiency for wall-clock speed, not that it's free on both axes.

Second, whatever LR change accompanies the batch-size increase needs re-validated warmup length (Q24) — a larger peak LR paired with the existing warmup schedule risks exactly the early-instability failure mode described there, so warmup duration should be revisited, not held fixed just because the old value "worked" at the old (batch size, peak LR) pair. Third, I would not make this change directly on the full frontier run — I'd validate the new (batch size, LR, warmup) triple at a smaller proxy scale first (the same proxy-run apparatus from `006_Pretraining_Ablations_And_Research_Methodology.md`), specifically checking that the loss trajectory at the new configuration tracks (or beats) the old configuration's trajectory at a matched token count, not just that it doesn't diverge — a configuration that trains stably but converges to a measurably worse loss at the same token budget is a silent failure this kind of comparison is designed to catch. Fourth, I'd sanity-check that the *rest* of the infrastructure (data-loading throughput, checkpoint cadence, gradient-clipping threshold, which may need proportional adjustment at the new gradient-noise regime) is compatible with the new batch size, since a batch-size change that's purely "correct" from an optimization standpoint can still be bottlenecked or destabilized by an infrastructure component tuned for the old regime.

---

## Q27: Explain what Lion and Muon each change relative to AdamW, and give an honest account of why most frontier labs still default to AdamW despite published evidence for both.

Lion (Chen et al., 2023) removes the second-moment (\(v_t\)) buffer entirely and replaces Adam's magnitude-normalized update with a pure **sign-based** update built from a momentum term: the parameter step is \(-\eta\,\text{sign}(c_t)\) where \(c_t\) is a momentum-smoothed gradient estimate, plus decoupled weight decay. Because there's no \(v_t\), Lion's optimizer state is half of AdamW's — one buffer per parameter instead of two — a direct, quantifiable memory saving at frontier parameter counts, where AdamW's ~14-bytes/parameter state (fp32 master weight + \(m\) + \(v\)) is itself a first-order driver of needing ZeRO/FSDP-style state sharding (`004_Optimizers_LR_Schedules_And_Hyperparameters.md` Section 1.5).

Muon (Jordan et al., 2024) targets the large 2D weight matrices specifically (leaving embeddings/norms on AdamW in most hybrid setups) and replaces Adam's *elementwise* adaptive scaling with an **orthogonalized-momentum** update: the momentum-averaged gradient matrix is projected toward its nearest orthonormal-row/column matrix via a fast Newton-Schulz iteration, and that orthogonalized matrix is applied as the update direction. The claim is that spreading the update evenly across a weight matrix's singular-value spectrum (rather than Adam's per-element rescaling, which can let a matrix's update be dominated by whichever elements happen to have small recent gradient variance) makes more efficient use of each optimization step for exactly the large dense matmuls that dominate transformer parameter counts, and reported step-count/wall-clock efficiency gains over AdamW on the runs tested (including some public speed-run-style benchmarks and at least one frontier-adjacent lab's disclosed usage) back this up at the scales tested.

The honest reason AdamW still dominates despite this: (1) track record specifically at the hundreds-of-billions-parameter, multi-trillion-token scale — AdamW has been validated there across many independently-run frontier efforts (DeepSeek-V3, Llama 3.1-405B, GPT-4-class training among them), with a correspondingly large body of transferable tuned-hyperparameter knowledge, while Lion's and Muon's strongest published results are, as of this writing, at meaningfully smaller scale and narrower coverage — and whether a new optimizer's advantage *transfers* to that regime is exactly the scale-transfer problem from `006_Pretraining_Ablations_And_Research_Methodology.md`, applying to optimizer choice as much as to architecture choice; (2) cost asymmetry — switching optimizers on a run costing tens of millions of dollars, based on evidence from a smaller-scale comparison, has a large downside (a subtle late-discovered instability or worse final loss) and a comparatively modest upside, which rationally biases conservative labs against the switch independent of how promising the published numbers look; (3) optimizer choice is entangled with the rest of the tuned recipe (LR schedule, warmup, weight decay, batch-size scaling heuristics are all tuned around AdamW's specific dynamics), so switching isn't a drop-in change, it's a re-derivation of a large fraction of the accumulated tuning. None of this means the newer research is wrong — some frontier-adjacent labs have begun disclosing Muon-family usage at meaningful scale, and this is a genuinely live, evolving area rather than a settled one.

---

## Q28: Implement a warmup-stable-decay (WSD) learning-rate schedule as a pure function, and explain in one or two sentences the specific operational advantage it has over a fixed-horizon cosine schedule.

```python
def wsd_lr(step: int, peak_lr: float, warmup_steps: int,
           decay_start_step: int, decay_steps: int,
           min_lr_ratio: float = 0.0) -> float:
    """Warmup -> Stable plateau at peak_lr -> Decay to (peak_lr * min_lr_ratio).
    decay_start_step is chosen dynamically, whenever training is decided to end soon --
    it need NOT be fixed at schedule-definition time, unlike cosine's total_steps."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step < decay_start_step:
        return peak_lr                                    # stable plateau
    decay_progress = min(1.0, (step - decay_start_step) / max(1, decay_steps))
    min_lr = peak_lr * min_lr_ratio
    # linear decay is the most common WSD choice; some variants use a short cosine tail
    return peak_lr - (peak_lr - min_lr) * decay_progress
```

WSD's operational advantage: because the "stable" plateau phase holds a constant LR rather than continuously decaying toward a pre-committed endpoint, the decision of *how long to keep training* can be made — or deferred, or revised upward if the loss curve still justifies it (the overtraining logic from `002_Scaling_Laws_And_Compute_Optimal_Training.md` Section 6) — independently of the schedule shape, and only once that decision is actually made does the short, bounded-cost decay phase (`decay_steps`, typically small relative to total training) get triggered; a fixed-horizon cosine schedule, by contrast, has to know the total step count \(T\) up front to shape its decay curve correctly, making an after-the-fact decision to extend training awkward without either accepting a distorted decay shape or restarting the decay in a way that isn't equivalent to having planned for the longer run from the start (`004_Optimizers_LR_Schedules_And_Hyperparameters.md` Section 2.3).

---

## Q29: Walk through the full three-stage pretraining curriculum — main pretraining, long-context extension, annealing — end to end, and explain why each later stage's compute share is small relative to the first, using a distinct mechanism for each stage.

Stage 1, main pretraining, consumes the overwhelming majority of the total compute budget (typically 90%+) at a comparatively short native context length, running the ordinary causal-LM objective over the bulk of the training corpus. The context length is kept short here specifically because attention's compute cost is quadratic in sequence length, and this stage processes the overwhelming majority of all training tokens — training at a long context throughout would multiply the dominant compute cost of the *entire* run, not just a small slice of it.

Stage 2, long-context extension, is cheap relative to Stage 1 because of a specific empirical claim: the model's core language competence doesn't need to be relearned to extend context, only its *positional* generalization does. RoPE's lowest-frequency rotation components have wavelengths long enough that, within a short native context, they never complete even a small fraction of a full rotation — the model has essentially never observed what those components look like at longer relative distances, which is the mechanistic cause of quality collapse when a model is naively run past its trained length. NTK-aware/YaRN-style rescaling fixes this by remapping the new maximum relative distance back into a rotation-angle range the model already saw, and once that positional recalibration is in place, comparatively few additional tokens (specifically at the new length) are needed to recover strong performance — DeepSeek-V3's disclosed figures put this stage at roughly 4-5% of main-pretraining compute (`..\..\OpenSource\007_DeepSeek_V3.md`, Section 3) despite taking the model from its native context all the way to 128K.

Stage 3, annealing/cooldown, is cheap for a *different* reason than Stage 2 — not because a mechanism does most of the work cheaply, but because its leverage comes from *where in the training trajectory* it's applied rather than from how many tokens it consumes. Two intertwined levers: decaying the LR to near-zero lets the model take much smaller, more precise steps and settle tightly into a low-loss basin rather than continuing to be pushed around by large per-step updates from noisy or idiosyncratic batches; and shifting the data mixture toward curated, high-quality sources during that same low-LR window means the model's *most recent* (and, combined with the small LR, disproportionately final-state-determining) gradient signal comes from exactly the kind of data downstream evaluations and products care about most. Because the LR is small during this stage, the mixture-shift intervention doesn't need to run for long to have an outsized effect on the final checkpoint relative to its tiny share of total compute — a structurally similar "small compute, disproportionate quality return" shape to Stage 2, arrived at for a different underlying reason (basin-settling and mixture polish, rather than positional recalibration).

---

## Q30: Explain, at a mechanism level, exactly why a RoPE-based model's quality collapses when run at a context length far beyond what it was trained on, and precisely what YaRN changes to fix it.

RoPE encodes relative position by rotating query and key vectors by an angle that's a function of absolute position and a set of geometrically-spaced frequencies (one pair of dimensions per frequency). Each frequency component has a characteristic wavelength — the absolute-position distance over which that component completes one full rotation. During training at a short native context length, the lowest-frequency components (longest wavelengths) never complete more than a small fraction of a rotation within any observed relative distance, because the maximum relative distance available during training (bounded by the native context length) is small relative to their wavelength. The model's attention mechanism has therefore never seen — and has no calibrated way to interpret — the rotation-angle patterns those low-frequency components would produce at the much larger relative distances a longer context introduces. Run the model naively past its native length, and those components produce out-of-distribution rotation angles the model was never trained to handle, which is the direct mechanistic cause of the sharp quality collapse observed when RoPE-based models are extended with no adjustment at all — it's specifically a positional-encoding generalization failure, not a general "the model forgot how to do language" failure, which is exactly why a targeted fix (rather than full retraining) works at all.

YaRN fixes this by rescaling frequencies **non-uniformly across the spectrum**, rather than applying one blanket interpolation factor everywhere: it ramps between two regimes based on how many rotations each frequency actually completes within the original training context — high-frequency components (which already completed many rotations within the original context, and are therefore not out-of-distribution at the new length) are left close to their original scale ("extrapolated"), while low-frequency components (the ones that were under-trained, having completed less than roughly one rotation originally) are rescaled by close to the full context-length ratio ("interpolated"), remapping their new maximum relative distance back into a rotation-angle range the model already saw during pretraining. YaRN additionally applies a small softmax-temperature correction to attention, motivated by the empirical observation that frequency rescaling alone tends to leave attention-score calibration subtly off, and a small temperature adjustment compensates for that residual miscalibration. The combination is why a comparatively small amount of continued training at the new length, using this rescaled scheme, is empirically sufficient to recover strong long-context performance rather than requiring anything close to a full retraining.

---

## Q31: You're planning a context-extension stage to take a model from its native 8K context to 128K. Walk through the decisions you need to make and the risks you'd flag before committing.

First decision: the positional-encoding rescaling scheme and its parameters — an NTK-aware or YaRN-style frequency ramp, with the interpolation/extrapolation boundary (in YaRN's framing, expressed via how many rotations each frequency completes within the original 8K context) chosen so the lowest-frequency components are correctly remapped to the new 128K maximum relative distance. This has to be validated, not assumed correct by formula alone — I'd want to check, at a smaller proxy scale first if my organization hasn't done this exact extension before, that the rescaled model's short-context (still-within-8K) performance isn't degraded by the rescaling itself, since a badly-tuned rescaling can distort the well-trained short-range frequencies as a side effect.

Second: the data used for this stage. It needs to be genuinely long documents (not just short documents padded or concatenated to look long, which wouldn't exercise real long-range dependency use), likely supplemented with synthetically constructed long-context tasks (e.g., retrieval-style tasks explicitly requiring information from far back in the context) — the composition and realism of this data directly determines whether the model that comes out the other side can actually *use* information anywhere in the new window, not merely tolerate a longer input without crashing or degrading numerically.

Third: how much compute to allocate. Based on the disclosed pattern across DeepSeek-V2/V3, Llama 3.1, and Qwen2.5, I'd expect this stage to be a small single-digit-to-low-double-digit percentage of main-pretraining compute — but I would not assume that ratio transfers without checking loss-curve and (more importantly) long-context-benchmark trajectory during the stage itself, since going 16x further (8K→128K) is a larger relative extension than some of those disclosed cases and might need more tokens to reach comparable quality at the far end of the new window, not just at the near end.

Fourth, and the risk I'd flag most explicitly to stakeholders: I would not accept "the model accepts 128K-token inputs without erroring" as evidence the extension worked. I'd insist on needle-in-a-haystack-style and long-document benchmarks specifically probing information placed at multiple positions throughout the new range (not just near the beginning, where the original 8K training already provides good coverage) before calling the extension complete, precisely because the extension stage's data distribution is narrower than main pretraining's and there's no architectural guarantee that positional generalization is uniform across the entire new window just because the mechanism is designed to make it possible in principle.

---

## Q32: What are the two distinct mechanisms bundled into an "annealing" or "cooldown" pretraining stage, and why is it useful to keep them conceptually separate even though they're usually applied together?

The two mechanisms are: (1) decaying the learning rate to a very small value, which is purely an optimization-dynamics lever — it lets the model take much smaller, more precise parameter-space steps late in training, settling more tightly into whatever loss basin the bulk of training has found, with much less risk of a late noisy or idiosyncratic batch knocking the model into a different (and not necessarily better) region; and (2) shifting the data mixture toward curated, high-quality, or product/eval-distribution-aligned sources, which is purely a *content* lever — independent of step size, it changes what signal the model is actually being shown.

Keeping them separate matters for two reasons. First, diagnostically: if an annealing stage underperforms expectations, knowing whether the LR-decay component or the mixture-shift component (or their interaction) is responsible requires having a mental model — and ideally an ablation — that isolates them, rather than treating "annealing" as one atomic intervention. Second, and more practically, because the LR is small during this stage, the mixture-shift component's *effect* is amplified relative to how much it would matter at a large LR — a curated-mixture nudge applied at full-strength LR would be just one more incremental part of an already-large, noisy update; applied at a near-zero LR, it disproportionately determines where the model's final weights actually settle, since there's little else pushing the weights around at the same time. That interaction — the mixture shift being disproportionately effective *because* it's paired with a small LR, not despite it — is exactly the kind of mechanistic precision that separates "I know annealing helps" from "I know why annealing helps and could reason about how to tune it," and is precisely why a WSD-style schedule (Q28) is a natural fit: it gives you a clean, deliberately-triggered window in which to run both levers together, once the timing decision to enter cooldown has actually been made.

---

## Q33: Walk through how you would validate a new MoE auxiliary-loss-free load-balancing scheme (in the style of DeepSeek-V3's bias-adjustment mechanism) before committing it to a frontier-scale training run, given your lab has an existing MoE training stack but has never used this specific balancing mechanism.

I'd treat this as a staged confidence-building pipeline, scaling the depth of validation to the fact that load-balancing mechanism choice is a large, hard-to-reverse-mid-run commitment (switching mid-run isn't realistically possible without restarting a large fraction of training) while the existing MoE stack means I'm not also validating basic MoE training feasibility from scratch.

Stage one: a small proxy-scale ablation (tens-of-millions to low-billions of activated parameters, matched routed/shared expert counts and top-k to the eventual target architecture as closely as feasible at that scale) comparing the new bias-adjustment mechanism directly against my existing auxiliary-loss-based balancing, at matched compute, with the specific things I'd instrument being: final loss (does the new mechanism reach comparable or better LM quality), measured expert-load distribution over training (does the bias control loop actually converge to balanced load, and how quickly, as a function of the step-size hyperparameter γ), and any training-stability signals (loss spikes, router collapse indicators) — since a genuinely new balancing mechanism could have failure modes the existing auxiliary-loss approach doesn't.

Stage two: because a control-loop mechanism's behavior (the γ step size, the window over which load is measured before adjusting bias) is exactly the kind of hyperparameter that might not transfer cleanly across scale (`006_Pretraining_Ablations_And_Research_Methodology.md` Section 4.2 — capacity and optimization-regime interactions are plausible here, since load-balancing dynamics interact with how many tokens per step are being routed, which changes with global batch size at different scales), I'd run a multi-point comparison across at least two or three proxy scales, specifically checking whether the loss gap and load-balance-convergence behavior are stable, growing, or shrinking as scale increases, rather than trusting a single-scale comparison.

Stage three, given the stakes of a frontier commitment: an intermediate "canary" run at a scale well above the proxy grid but still meaningfully cheaper than the full target (tens of billions of activated parameters, if the target is hundreds of billions) — specifically instrumented to catch the class of MoE-specific failure that might only manifest at real communication scale (does the bias mechanism's load-balancing interact badly with the actual expert-parallel all-to-all communication pattern at this larger scale, e.g., does it converge to a load distribution that's balanced in count but still creates a communication bottleneck due to *where* experts are physically placed) — before authorizing the full run. Given all of this is itself a cost/risk tradeoff (`006_...md` Section 5), I'd size the depth of this pipeline to the fact that a load-balancing-mechanism choice is close to irreversible mid-run and directly affects both training throughput and quality, which argues for more validation spend here than for, say, a comparatively easily-adjustable data-mixture percentage.

---

## Q34: Give three concrete, mechanistically distinct reasons an architectural change that measurably helps at 1B parameters could have no effect, or a negative effect, at 500B parameters.

First, an **optimization-regime interaction**: a change that helps primarily by improving convergence speed or stability in a small, comparatively noisy-gradient, early-training-dominated regime may simply have nothing left to fix once the model is large enough (and typically trained with larger batch sizes and correspondingly different optimizer dynamics, per `004_Optimizers_LR_Schedules_And_Hyperparameters.md` Section 4) that its optimization is already comparatively well-behaved for unrelated reasons — the mechanism the small-scale ablation was capturing may not even be present at the larger scale's optimization regime.

Second, a **capacity interaction**: any change that trades some representational flexibility for another benefit (a compression bottleneck, a routing constraint, a parameter-sharing scheme) can look like a clear net win at small scale, where capacity is the binding constraint and the efficiency gain from the change outweighs whatever capacity it costs — and can look like a net loss at large scale, where capacity is comparatively abundant and the same absolute efficiency gain no longer offsets the same capacity cost, because the larger model had capacity to spare regardless of the change.

Third, a **data-scale interaction**: some effects are bottlenecked by how much data the model has seen relative to its size (the \(D/N\) ratio), not by \(N\) in isolation — an ablation run at a proxy \(N\) and \(D\) that don't reflect the target run's actual \(D/N\) point (per the Chinchilla-optimal-ratio discussion in `002_Scaling_Laws_And_Compute_Optimal_Training.md`) may be answering a subtly different empirical question than the one the frontier run's real operating point poses, and a change that looks beneficial at one \(D/N\) ratio isn't guaranteed to look the same at a materially different one, independent of parameter count per se.

---

## Q35: What does μ-parameterization (μP / "Tensor Programs") actually guarantee about hyperparameter transfer across scale, and — just as importantly — what does it not cover?

μP derives specific scaling rules for how initialization variance, learning rate, and certain architectural constants (e.g., per-layer multipliers) should change as a function of model width, such that — under the theoretical framework the papers establish — the (near-)optimal values of the *covered* hyperparameters found by tuning at small width remain (approximately) optimal at much larger width. Practically, this means: tune learning rate (and the other μP-covered quantities) via a cheap sweep at small width, apply the μP scaling rule to translate those values to the target width, and skip re-tuning those specific hyperparameters at the expensive target scale — genuinely converting part of the scale-transfer problem from "hope it holds" (the position most other ablated changes are in, per `006_Pretraining_Ablations_And_Research_Methodology.md`) to "apply a derived rule with an actual theoretical guarantee, scoped to width scaling."

What it does *not* cover, and where I'd be careful not to overstate the guarantee: μP's theory is scoped to the specific set of hyperparameters and the specific scaling axis (width) its derivation addresses — it does not, by itself, provide a transfer guarantee for an arbitrary architectural change (a new attention variant, a new MoE routing mechanism, a new data mixture) that isn't one of the quantities the theory covers, nor does it necessarily cover scaling along other axes (depth, token count) with the same guarantee it provides for width. It also depends on the practitioner's model actually matching the parameterization conventions the theory assumes (specific initialization and per-layer-multiplier conventions) — a model that deviates from those conventions doesn't automatically inherit the transfer guarantee just because "μP" is invoked. So the honest framing is: μP meaningfully de-risks a *specific, named subset* of the hyperparameter-transfer problem, with real theoretical backing, and should be used wherever it applies — but it doesn't dissolve the broader scale-transfer problem this module keeps returning to, and shouldn't be cited as if it does for questions (architecture choice, data mixture, load-balancing mechanism design) outside its actual scope.

---

## Q36: Your 1B-parameter proxy ablation shows a new attention variant reduces loss by 3% relative to your current baseline, at matched compute. You have budget for exactly one 400B-parameter frontier run this year. How do you decide whether to adopt the new variant, and what's the minimum additional validation you'd insist on before saying yes?

I would not greenlight adoption directly from a single 1B-scale data point — a 3% loss reduction at 1B, extrapolated 400x in parameter count with no intermediate check, is exactly the unvalidated-extrapolation risk `006_Pretraining_Ablations_And_Research_Methodology.md` Section 4 is about, and an attention-mechanism change is precisely the kind of large, effectively-irreversible-mid-run commitment (per that file's Section 5 cost-asymmetry framing) that warrants more validation spend than a data-mixture percentage would.

Minimum additional validation before saying yes: first, a multi-point scaling curve rather than a single point — rerun the same comparison at two or three additional scales spanning at least another order of magnitude (e.g., 3B, 10B, maybe 30B if budget allows), and check whether the 3% gap is stable, growing, or shrinking as scale increases; a shrinking gap is a strong warning sign that the benefit may not survive another order-of-magnitude-plus extrapolation to 400B, even if it's never fully explained mechanistically. Second, I'd want a mechanistic story for *why* the new variant should help, and specifically whether that mechanism has any reason to interact with scale — if the claimed benefit is, say, improved cache efficiency or a genuinely scale-invariant inductive-bias argument, that's a materially stronger basis for confidence than "it produced a lower loss number at 1B with no accompanying explanation." Third, given that a single year's frontier-run budget is at stake, I'd push hard for at least one intermediate canary run — tens of billions of parameters, a real but much smaller commitment than the full 400B run — specifically to catch scale-dependent failure modes (optimization-regime or capacity interactions, per Q34) before the full budget is spent, even if that costs meaningful additional calendar time against the one-run-per-year constraint; if that canary isn't affordable within the timeline, I would flag explicitly to whoever owns the go/no-go decision that we are making this call on evidence from 1B extrapolated 400x, name the specific risk this entails, and let that be an informed, explicit risk acceptance rather than an implicit one.

---

## Q37: Implement a function estimating the gradient noise scale from a small set of measured gradient norms at two different batch sizes, and explain how the result would inform a critical-batch-size decision.

```python
import numpy as np

def estimate_gradient_noise_scale(grad_norm_sq_small_batch: float, batch_size_small: int,
                                    grad_norm_sq_large_batch: float, batch_size_large: int
                                    ) -> float:
    """Simplified two-batch-size estimator of the gradient noise scale B_noise,
    following the McCandlish et al. 2018 framing: the true (per-example)
    gradient variance trace can be estimated from how the measured squared
    gradient norm changes with batch size, since
        E[||g_B||^2] = ||g_true||^2 + trace(Sigma)/B
    Two measurements at different B let us solve for ||g_true||^2 and trace(Sigma),
    and B_noise = trace(Sigma) / ||g_true||^2 is the critical-batch-size estimate.
    """
    b1, b2 = batch_size_small, batch_size_large
    g1, g2 = grad_norm_sq_small_batch, grad_norm_sq_large_batch
    # g_i = g_true_sq + trace/b_i  =>  solve the 2x2 linear system
    # g1 - g2 = trace * (1/b1 - 1/b2)
    trace = (g1 - g2) / (1.0 / b1 - 1.0 / b2)
    g_true_sq = g1 - trace / b1
    if g_true_sq <= 0:
        raise ValueError("Noisy measurement produced a non-positive true-gradient estimate; "
                          "average over more measurement batches.")
    b_noise = trace / g_true_sq
    return b_noise
```

The estimated `b_noise` is a direct, model-and-data-specific estimate of the critical batch size (Q25): batch sizes well below `b_noise` are in the regime where the linear LR scaling rule is well-justified (gradient noise still the dominant limiting factor, so more samples still buy proportionally larger safe steps), while batch sizes approaching or exceeding `b_noise` should shift toward the more conservative sqrt scaling rule (curvature, not noise, becoming the binding constraint). In practice, this measurement is taken from the *training run itself* at a candidate operating point (comparing gradient norms measured with a small "probe" micro-batch against the full intended global batch, both drawn from the same training step's data) rather than purely from an external proxy-run sweep, precisely because — per `004_Optimizers_LR_Schedules_And_Hyperparameters.md` Section 4.3 — the critical batch size is known to shift over the course of training, so a single early-training measurement shouldn't be assumed to hold for the entire run without periodic re-estimation.

---

## Q38: Compare monitoring early loss-curve trajectory during a run against simply waiting for the run to finish before evaluating it. What specifically can trajectory monitoring catch that a final-loss comparison cannot, and what's the risk of over-relying on it?

Waiting for a run to finish gives you one clean, unambiguous number, uncontaminated by any mid-run confounds (schedule position, warmup-vs-decay phase, transient batch-composition effects) — but for a frontier-scale run, "wait for it to finish" means you only find out something went wrong (a bad hyperparameter choice, a data-pipeline bug, a scaling-law extrapolation that isn't holding) after the *entire* budget has already been spent, at which point there's nothing left to do but analyze the failure for next time. Given that a full frontier run cannot be meaningfully re-tried without re-spending the whole budget, this is a genuinely costly way to first learn about a problem.

Trajectory monitoring — fitting the loss-vs-tokens curve so far to the expected power-law shape and checking whether its *extrapolated* endpoint tracks what the proxy-scale scaling-law fit predicted for this point in training (`006_Pretraining_Ablations_And_Research_Methodology.md` Section 3) — gives an early, actionable read while there's still budget left to act on it: if the trajectory a meaningful fraction of the way through is trending measurably worse than the fitted expectation, that's real signal potentially triggering a hyperparameter intervention, a data-pipeline investigation, or in the most severe cases an abort-and-restart decision, all while most of the compute budget is still unspent. It specifically catches problems that manifest early and persist (a bad LR, a data bug, a numerically unstable configuration) — things a final-loss-only comparison would also eventually catch, just at maximal cost.

The risk of over-relying on it: trajectory shape early in training is itself noisy and schedule-position-dependent — a curve that looks worse than a baseline's during warmup or the early-plateau portion of a schedule can still cross over and finish better once both are compared at matched schedule position, and reading absolute loss (rather than the shape of the fitted extrapolation) at a fixed step without accounting for schedule position is a well-known way to draw the wrong conclusion mid-run. It's also fundamentally still an extrapolation — a trajectory that matches the fitted expectation at 10% of the way through a run is reassuring but not a guarantee the fit continues to hold at 100%, for the same reasons any scaling-law extrapolation carries residual risk (Q15). The right posture is to treat trajectory monitoring as a real, load-bearing risk-reduction tool specifically because it's one of the few levers available *during* an otherwise-irreversible run, while being explicit that it reduces, rather than eliminates, the uncertainty a final evaluation would resolve completely.

---

## Q39: You're the first staff research engineer hired at a new lab that intends to train its first 400B+-parameter frontier model within 18 months, with no prior large-model training experience in-house. Design the ablation-to-frontier-run pipeline you'd put in place, synthesizing the scaling-law, architecture-decision, hyperparameter, and methodology material in this module.

I'd structure this as a sequence of increasingly expensive, increasingly confidence-building phases, explicitly budgeting more calendar time and compute to the decisions that are most irreversible and least transferable across scale, per the cost-asymmetry framing in `006_Pretraining_Ablations_And_Research_Methodology.md` Section 5.

Phase 1 (months 1-4, small proxy models, tens-of-millions to low-billions of parameters): settle the architecture-decision axes from `003_Model_Architecture_Decisions_At_Pretraining_Time.md` — dense vs. MoE (given no in-house MoE experience and an 18-month deadline, I'd lean dense by default unless the product's inference-cost profile makes the MoE case compelling enough to justify the added execution risk, per that file's Section 1.2), attention variant (GQA as the default, given its near-zero risk and universal tooling support, unless target context length and serving concurrency make a stronger case worth the added risk of something like MLA), and a provisional native context length and extension-stage plan. In parallel, run the same proxy grid to fit an initial scaling-law curve (`002_...md` Section 5.1) and validate the optimizer/hyperparameter recipe (AdamW as the default, per the conservatism argument in `004_...md` Section 5.3 — this is not the deadline on which to bet a novel optimizer), including an initial critical-batch-size estimate (Q25/Q37).

Phase 2 (months 4-8, intermediate canary run(s), tens of billions of parameters): before locking anything in for the full run, validate the riskiest, least-reversible decisions from Phase 1 at a scale closer to (but still well below) target — specifically re-checking whether the fitted scaling curve's trajectory prediction is tracking reality at this larger scale (`006_...md` Section 3), whether the chosen attention/MoE configuration's training stability holds, and whether the batch-size/LR pairing chosen from Phase 1's estimate still looks right. This phase is explicitly there to catch the scale-transfer failures named in Q34 before they're baked into the full commitment — I would not skip it even under time pressure, given this is the org's first frontier run and has no accumulated institutional evidence that its proxy-scale conclusions transfer.

Phase 3 (months 8-9, final commit and launch): lock the \((N,D)\) split from the Chinchilla-style IsoFLOP fit refined using Phase 2's data, decide the deliberate-overtraining posture based on the org's actual projected deployment volume for this model (`002_...md` Section 6 — this requires an honest conversation with product about expected query volume, since the answer materially changes the \(N\) target), and finalize the multi-stage curriculum plan (main pretraining at the chosen native context, a budgeted long-context extension stage, and a WSD-schedule-based annealing/cooldown plan, per `005_...md`).

Phase 4 (months 9-16ish, the main run itself): active trajectory monitoring throughout (Q38) against the Phase-2-refined fit, with a pre-agreed intervention protocol (what deviation threshold triggers a hyperparameter change vs. an abort decision) established *before* the run starts, not improvised mid-run.

Phase 5 (months 16-18, extension and annealing stages, then ship): run the long-context extension and annealing stages as planned, with the long-context extension validated against needle-in-a-haystack-style benchmarks specifically (not just architectural context-length support) before declaring it complete.

Throughout, I would be explicit with leadership at every phase boundary about what has and hasn't actually been de-risked — a new lab's biggest risk on a first frontier attempt is treating small-scale ablation success as equivalent to frontier-scale confidence, which is exactly the gap this entire module, and this pipeline, is built to narrow rather than eliminate.

---

## Q40: Implement a function that, given a fixed downstream-quality bar (expressed as a target loss achievable by several candidate (N, D) configurations reaching that same bar) and a projected query volume, selects the total-cost-of-ownership-minimizing configuration from a candidate grid — and use it to show numerically that the answer changes as query volume grows.

```python
def tco_optimal_config(candidates: list[tuple[float, float]],
                        n_queries: float, avg_output_tokens: float,
                        cost_per_flop_train: float = 1.0,
                        cost_per_flop_infer: float = 1.0) -> tuple[float, float, float]:
    """candidates: list of (N, D) pairs, all assumed to reach the SAME target
    quality bar (e.g., each already validated via the scaling-law fit in
    002_Scaling_Laws_And_Compute_Optimal_Training.md to hit the bar).
    Returns the (N, D, total_cost) triple minimizing total lifetime cost:
        cost = 6*N*D*cost_per_flop_train + 2*N*n_queries*avg_output_tokens*cost_per_flop_infer
    """
    best = None
    for n, d in candidates:
        train_cost = 6.0 * n * d * cost_per_flop_train
        infer_cost = 2.0 * n * n_queries * avg_output_tokens * cost_per_flop_infer
        total = train_cost + infer_cost
        if best is None or total < best[2]:
            best = (n, d, total)
    return best

# Candidates all assumed (for illustration) to reach the same quality bar via
# different overtraining postures: smaller N paired with much larger D.
candidates = [
    (70e9, 1.5e12),    # near Chinchilla-optimal for 70B
    (30e9, 6e12),      # moderately overtrained
    (13e9, 12e12),     # heavily overtrained, Llama-3-8B-style
]

for n_queries in [1e6, 1e9, 1e11]:
    n, d, cost = tco_optimal_config(candidates, n_queries, avg_output_tokens=500)
    print(f"n_queries={n_queries:.0e} -> best N={n:.1e}, D={d:.1e}, total_flops={cost:.2e}")
```

Running this: at `n_queries=1e6`, the training-cost term dominates for all three candidates (inference is negligible at this volume), so the near-Chinchilla-optimal 70B/1.5T configuration wins — it's simply the cheapest to train among the three, and there aren't enough queries for its larger \(N\) to matter. At `n_queries=1e9`, the crossover region, the answer becomes sensitive to the exact `cost_per_flop` ratios and average output length, illustrating that there's a genuine middle regime where the decision isn't a landslide either way. At `n_queries=1e11` (a widely-deployed consumer/API-scale product), the heavily-overtrained 13B/12T configuration wins clearly — its much larger \(D\) is a one-time cost that's now dwarfed by the inference-cost term for either of the larger-\(N\) alternatives, exactly reproducing the Llama-3-8B-style argument (Q9, Q17) as a numerically swept result across a volume range rather than a single fixed scenario. The pedagogical point of sweeping `n_queries` rather than fixing it: the total-cost-optimal configuration is not a fixed property of a quality bar — it's a function of projected deployment volume, and any staff-level recommendation on this axis should be presented conditioned on a stated volume assumption, not as an unconditional "smaller-and-overtrained is always better" or "Chinchilla-optimal is always right" rule.
