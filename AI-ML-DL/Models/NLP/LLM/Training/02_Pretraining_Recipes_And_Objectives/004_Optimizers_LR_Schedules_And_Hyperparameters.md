# Optimizers, Learning-Rate Schedules, and Hyperparameters

The optimizer and schedule are the machinery that actually turns a fixed architecture and data mixture into a trained checkpoint; this file derives AdamW's update rule precisely, explains why warmup and batch size matter mechanistically rather than by convention, and gives an honest account of why the field has been slow to move past AdamW despite real competing research.

## 1. AdamW, derived from first principles

### 1.1 Why plain SGD is not the default for transformers

Plain stochastic gradient descent updates \(\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}\) using a single global learning rate applied identically to every parameter, regardless of how noisy or how differently-scaled that parameter's gradient is. Transformer training gradients are extremely heterogeneous across parameter groups — embedding rows, attention projections, and FFN weights see very different gradient magnitudes and variances over the course of training — and a single global step size that's well-tuned for one group is routinely poorly-scaled for another. Adam-family optimizers address this by maintaining **per-parameter adaptive step sizes**, estimated from that parameter's own recent gradient history, which is the single biggest reason they became (and remain) the default for transformer pretraining despite costing 2x the memory of plain SGD (Section 1.5).

### 1.2 Adam's moment estimates

At training step \(t\), given the gradient \(g_t = \nabla_\theta \mathcal{L}_t\), Adam (Kingma & Ba, 2015) maintains two running exponential moving averages per parameter: a first-moment estimate (mean gradient direction) and a second-moment estimate (mean squared gradient magnitude):

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(first moment, "momentum")}
\]
\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(second moment, "uncentered variance," elementwise square)}
\]

with typical defaults \(\beta_1 = 0.9\), \(\beta_2 = 0.95\)–\(0.999\) (large-LM pretraining commonly uses \(\beta_2\) around 0.95, lower than the vision-era default of 0.999, because faster adaptation to recent gradient statistics is empirically preferred over very long transformer runs — this specific choice is itself a hyperparameter tuned per training regime, not a fixed constant).

### 1.3 Bias correction, and why it's necessary

Both \(m_t\) and \(v_t\) are initialized at \(m_0 = v_0 = 0\). Unrolling the recursion for \(m_t\):

\[
m_t = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i
\]

At small \(t\) (early in training), this sum is dominated by the "missing" weight the zero-initialization implicitly assigns to steps before \(t=1\) — concretely, \(\mathbb{E}[m_t] = (1-\beta_1^t)\,\mathbb{E}[g_t]\) under a stationarity assumption, so \(m_t\) is a systematically **biased-toward-zero** estimate of the true gradient mean by a factor of exactly \((1-\beta_1^t)\), and the bias is worst exactly when \(t\) is small — i.e., in the first several dozen-to-hundred steps, precisely when getting the update direction and scale right matters for avoiding an early bad trajectory. Adam corrects for this explicitly:

\[
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
\]

Both correction factors approach 1 as \(t\) grows (so bias correction only matters early in training) and both denominators are close to 0 at \(t=1\) (so the correction is large exactly when the bias is large) — this is the correct fix rather than an approximation, given the zero-initialization and the stationarity assumption used to derive it.

### 1.4 The parameter update, and decoupled weight decay

Plain Adam's update (ignoring weight decay for a moment):

\[
\theta_t = \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

where \(\epsilon\) (typically \(10^{-8}\), sometimes larger, e.g. \(10^{-6}\)-\(10^{-5}\), at scale to avoid a different numerical failure mode discussed in Section 1.6) prevents division by zero when a parameter's gradient history has been consistently near-zero. The intuitive reading: each parameter's step is its momentum-smoothed gradient direction, rescaled by the inverse of its own recent gradient magnitude — parameters with large, noisy gradients get dampened steps; parameters with small, consistent gradients get amplified steps, adaptively and independently per parameter.

**Weight decay, done wrong (L2-regularized Adam).** The classical way to add weight decay to any gradient-descent method is to add an L2 penalty term \(\frac{\lambda}{2}\|\theta\|^2\) directly to the loss, so it contributes \(\lambda\theta\) to the gradient \(g_t\) before anything else happens: \(g_t \leftarrow g_t + \lambda\theta_{t-1}\). The problem specific to Adam: that decay term now flows *through* the adaptive-second-moment normalization along with the "real" gradient — a parameter with a large accumulated \(v_t\) (from a large or noisy real gradient history) gets its L2-decay term shrunk by the same \(1/\sqrt{\hat v_t}\) factor that dampens its real gradient, so the *effective* weight decay strength ends up parameter-dependent and coupled to gradient statistics in a way that has nothing to do with the actual regularization intent.

**AdamW (Loshchilov & Hutter, 2019): decouple weight decay from the gradient-based update entirely.** Rather than folding decay into \(g_t\), apply it as a separate, direct multiplicative shrinkage of the parameter itself, outside the adaptive-moment machinery:

\[
\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda \theta_{t-1}\right)
\]

(equivalently, and how it's usually implemented: \(\theta_{t-1} \leftarrow \theta_{t-1}(1 - \eta\lambda)\) as a separate step, then apply the ordinary Adam update on top). Now \(\lambda\) has a fixed, parameter-independent effect — every weight is shrunk toward zero by the same fraction \(\eta\lambda\) per step, regardless of that parameter's gradient-noise history, which is what "weight decay" is actually supposed to mean and is the reason essentially every modern transformer pretraining recipe uses AdamW rather than plain L2-regularized Adam.

```python
import numpy as np

class AdamW:
    def __init__(self, params_shape, lr=3e-4, beta1=0.9, beta2=0.95,
                 eps=1e-8, weight_decay=0.1):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)
        self.t = 0

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        # Decoupled weight decay: direct shrinkage, not folded into grad/v_hat.
        theta = theta - self.lr * self.wd * theta
        theta = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return theta
```

Note the two-line update at the end: weight decay and the adaptive-moment step are applied as genuinely separate operations, both scaled by \(\eta\) but through entirely different mechanisms — this separation is the entire content of the "decoupled" in AdamW and is the single most common thing an interviewer will ask you to point at precisely in a from-scratch AdamW implementation.

### 1.5 Memory cost, and why it matters at scale

AdamW stores two extra buffers per parameter (\(m\) and \(v\)), each typically kept in fp32 for numerical stability even when the model's working weights are bf16/fp16. Per parameter: the working weight (2 bytes, bf16), a fp32 master-weight copy for the optimizer's accumulation (4 bytes, common in mixed-precision recipes), plus \(m\) and \(v\) (4 bytes each, fp32) — roughly 14 bytes/parameter of state beyond gradients, which at frontier parameter counts (hundreds of billions) is itself measured in terabytes and is a first-order driver of why optimizer-state sharding (ZeRO/FSDP) is close to mandatory infrastructure, not an optimization, at that scale (see the memory-wall arithmetic worked through for GPT-3 in `..\..\GPT\003_GPT3.md`, Section 4). This memory cost is also the direct motivation for the memory-efficient optimizer research discussed in Section 5 (Lion in particular is explicitly pitched on this axis).

### 1.6 A numerical failure mode worth knowing precisely

If \(\epsilon\) is too small relative to the numerical precision of \(\sqrt{\hat v_t}\) (an issue that becomes more likely under low-precision training, e.g., bf16-heavy pipelines with imprecisely-computed \(v_t\)), a parameter whose gradient has been consistently near-zero for a long stretch can see \(\sqrt{\hat v_t}+\epsilon\) collapse toward a value dominated by rounding noise rather than genuine signal, producing an update that is effectively unbounded in direction relative to that parameter's real recent gradient — a classic source of isolated loss spikes at scale that is not a hyperparameter-tuning issue in the ordinary sense but a numerics issue in the optimizer's implementation. This is one reason some large-scale training recipes use a larger-than-textbook \(\epsilon\) (e.g., \(10^{-6}\) to \(10^{-5}\)) specifically at frontier scale, trading a small amount of per-parameter update precision for materially improved training stability.

### 1.7 A worked numeric table of AdamW's memory overhead by scale

Section 1.5 states the ~14-bytes-per-parameter figure qualitatively; tabulating it across scales makes the ZeRO/FSDP-sharding necessity concrete.

```python
def adamw_state_bytes(n_params: float, master_weight_bytes: int = 4,
                        moment_bytes: int = 4, working_weight_bytes: int = 2) -> float:
    """Total memory for one parameter's full mixed-precision AdamW footprint:
    working (bf16) weight + fp32 master weight + fp32 m + fp32 v."""
    return n_params * (working_weight_bytes + master_weight_bytes + 2 * moment_bytes)

for n in [7e9, 70e9, 175e9, 671e9]:
    total_gb = adamw_state_bytes(n) / (1024 ** 3)
    print(f"N={n:.0e} params -> {total_gb:.1f} GB of weights+optimizer state")
```

| Model scale | Approx. weights+optimizer-state memory (bf16 working + fp32 master/m/v) |
|---|---|
| 7B | ~91 GB |
| 70B | ~910 GB |
| 175B | ~2.28 TB |
| 671B (DeepSeek-V3 total params, illustrative) | ~8.75 TB |

None of these figures fit in a single accelerator's HBM even at the smallest end of this table once activations are added on top — which is precisely why ZeRO/FSDP-style state sharding across the data-parallel group is treated as close to mandatory infrastructure at this scale (Section 1.5), not an optional optimization, and why Lion's (Section 5.1) halved optimizer-state footprint is a real, quantifiable infrastructure lever rather than a marginal convenience.

## 2. Warmup and cosine decay

### 2.1 Why warmup specifically matters at this scale

At step \(t=1\), \(v_1 = (1-\beta_2)g_1^2\) is an estimate of gradient variance built from a single sample — an extremely noisy estimate with no averaging behind it yet. If the learning rate is already at its intended peak value at this point, the update \(\eta \hat m_1 / (\sqrt{\hat v_1}+\epsilon)\) is being scaled by a step-size estimate that has essentially no statistical grounding, and at the very start of training the model's weights are near their (often carefully-chosen, but still somewhat arbitrary relative to the loss landscape the trained model will eventually occupy) initialization — a regime where the loss landscape's curvature can differ sharply from what a fixed peak LR was tuned for later in training. Empirically, skipping warmup at large batch size / large model scale reliably produces early instability — loss spikes, or outright divergence — because the combination of "peak LR" and "unreliable adaptive-moment estimate" and "far-from-equilibrium initialization" compounds badly. **Linear warmup** — ramping the LR linearly from 0 (or a small value) up to the peak LR over some number of initial steps (commonly on the order of a few hundred to a couple thousand steps, or a small percentage of total training steps for a very long run) — gives the adaptive moment estimates time to accumulate a statistically meaningful history, and gives the model time to move away from initialization into a better-conditioned region of the loss landscape, before the full-strength learning rate is applied.

This is worth stating with the batch-size connection made explicit, because it is where warmup and Section 3's critical-batch-size discussion meet: **larger batch sizes amplify the need for warmup**, because a larger batch size is typically paired with a larger peak learning rate (Section 3.2), and a larger peak LR applied too early against noisy early-step statistics is exactly the failure mode described above, more severely.

### 2.2 Cosine decay

After warmup, the dominant schedule across frontier pretraining recipes decays the LR from its peak down to some small fraction of peak (often ~10%, sometimes decayed to near-zero) following a cosine curve over the remaining training steps:

\[
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\left(\pi \cdot \frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)\right)
\]

for \(t \geq t_{\text{warmup}}\), where \(T\) is the total planned number of training steps. The empirical motivation for cosine specifically over, say, linear or step decay is mostly pragmatic: it decays slowly near the peak (spending more steps near the effective LR that's been doing most of the useful optimization) and slowly near the floor (giving the model a long, gentle final settling period rather than an abrupt cutoff), with the steepest decay happening in the middle — a shape that has empirically produced strong final loss across a large number of published large-LM training recipes, though it is not derived from any first-principles optimality argument specific to transformer loss landscapes; it is an empirically-validated convention more than a theoretically necessary one.

```python
import math

def lr_at_step(step: int, peak_lr: float, warmup_steps: int, total_steps: int,
               min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return peak_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    min_lr = peak_lr * min_lr_ratio
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### 2.3 The known weakness of committing to a fixed \(T\) up front, and the WSD alternative

Cosine decay's schedule is a function of the *total planned step count* \(T\) — the schedule has to know in advance how long training will run in order to shape the decay curve correctly, because the cosine curve is defined relative to reaching its floor exactly at \(T\). This is operationally awkward for a lab that wants to keep the option open to extend training if the loss curve is still improving usefully at the originally-planned endpoint (exactly the decision Llama 3's overtraining strategy depends on being able to make — see `002_Scaling_Laws_And_Compute_Optimal_Training.md`, Section 6): re-committing to a longer cosine schedule after training has already substantially decayed the LR generally requires either accepting a worse final schedule shape or restarting the decay in a way that isn't equivalent to having planned for the longer run from the start. This is one of the practical motivations behind the **warmup-stable-decay (WSD)** schedule adopted in some recent training reports (e.g., MiniCPM; discussed further as the natural schedule to pair with the annealing/cooldown stage in `005_Curriculum_And_Multi_Stage_Pretraining.md`): hold the LR at a constant plateau for the bulk of training after warmup (rather than continuously decaying), and only run a short, steep decay-to-near-zero at the very end, whenever that end is actually decided. This decouples "how long do we train" from "what shape is our decay schedule" — the plateau can be extended indefinitely without needing to have pre-committed to a total step count, and the final decay is a short, bounded-cost operation applied once the endpoint is chosen.

## 3. Gradient clipping

### 3.1 Mechanism

Global-norm gradient clipping computes the norm of the *entire* gradient vector across all parameters, and rescales the whole vector (preserving its direction) if that norm exceeds a threshold \(c\) (commonly 1.0 for large-LM pretraining):

\[
g_t \leftarrow g_t \cdot \min\left(1, \frac{c}{\|g_t\|_2}\right)
\]

```python
def clip_grad_global_norm(grads: list[np.ndarray], max_norm: float = 1.0) -> list[np.ndarray]:
    total_sq = sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads)
    total_norm = total_sq ** 0.5
    scale = min(1.0, max_norm / (total_norm + 1e-6))
    return [g * scale for g in grads]
```

### 3.2 Why global norm, not per-parameter clipping

Clipping each parameter (or each layer) independently would distort the *relative* magnitudes of different gradient components — exactly the directional information the optimizer needs to take a sensible step. Global-norm clipping instead treats an anomalously large gradient batch (a common occurrence at scale: a rare data batch containing an unusual token sequence, a numerical near-instability in one layer, an unlucky sampling of a high-loss example) as a signal to shrink the *entire* step uniformly, preserving direction while bounding magnitude — this is specifically a defense against a single bad batch producing a destructively large parameter update, not a mechanism intended to change the optimizer's steady-state behavior on well-behaved batches (on a normal batch, the clip threshold typically isn't even reached).

## 4. Critical batch size and the batch-size/learning-rate relationship

### 4.1 The gradient noise scale framing

McCandlish et al. (2018, "An Empirical Model of Large-Batch Training") frame the batch-size question in terms of the **gradient noise scale**: at small batch size, each additional sample in the batch reduces gradient-estimate variance roughly proportionally, so increasing batch size buys a roughly proportional increase in the *useful* per-step progress (you can take a correspondingly larger learning-rate step because the gradient estimate is that much less noisy) — this is the regime where increasing batch size is nearly "free" in terms of total tokens needed to reach a given loss. Past a model- and data-dependent **critical batch size**, the gradient estimate is already precise enough that further batch-size increases buy little additional per-step benefit — each additional sample in the batch is now mostly reducing an already-small noise floor, so training with a batch far past the critical size wastes compute: you're consuming more tokens per step without a matching reduction in the number of steps needed, for no net FLOPs benefit and, past some point, for a genuine *increase* in total tokens required to reach a fixed loss target relative to training nearer the critical size and taking more (smaller-batch) steps.

### 4.2 How this connects to learning-rate scaling

Empirically, near the critical-batch-size regime, per-step learning rate should scale with batch size to keep the *effective* step size (in terms of parameter movement per unit of data seen) roughly constant — the two classical heuristics are the **linear scaling rule** (\(\eta \propto B\), well-justified when gradient noise, not curvature, is the dominant limiting factor, i.e., safely below the critical batch size) and the **square-root scaling rule** (\(\eta \propto \sqrt{B}\), a more conservative heuristic that tends to hold better as batch size approaches or exceeds the critical size, where gradient-noise reduction from more samples is no longer the dominant effect and step-size increases have to be more conservative to avoid overshoot). Neither is a universal law; both are heuristics whose validity range is itself a function of where the actual training regime sits relative to the critical batch size for that specific model/data/optimizer combination — which is precisely why labs run small-scale sweeps (the same proxy-run methodology as `002_Scaling_Laws_And_Compute_Optimal_Training.md` Section 5 and `006_Pretraining_Ablations_And_Research_Methodology.md`) to estimate a workable (batch size, peak LR) pairing before committing it to the full run, rather than deriving it purely from either scaling rule in isolation.

### 4.3 Why this matters operationally, not just theoretically

Batch size at frontier scale is not a free knob chosen purely for statistical efficiency — it interacts directly with **data parallelism degree** (larger global batch size is what lets you use more data-parallel replicas without each replica's local batch shrinking below an efficient GPU-utilization floor) and therefore with **training wall-clock time** for a fixed token budget. A lab under wall-clock pressure has a real incentive to push batch size up toward (and sometimes past) the critical batch size specifically to enable more data parallelism and shorter wall-clock training time, accepting some token-efficiency loss in exchange for calendar-time savings — a genuine, cost-quantifiable tradeoff between token efficiency and training duration, not simply "bigger batch is better" or "smaller batch is more efficient."

## 5. Newer optimizer research: Lion and Muon

### 5.1 Lion

Lion (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms," discovered via program search rather than hand-designed) replaces Adam's normalized-magnitude update with a much simpler **sign-based** update using only momentum, no second-moment estimate at all:

\[
c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad \theta_t = \theta_{t-1} - \eta\,\text{sign}(c_t) - \eta\lambda\theta_{t-1}
\]
\[
m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t
\]

(note the two different momentum-like buffers used at different points, per the paper — \(c_t\) built with \(\beta_1\) is used only to determine the update's *sign*, while \(m_t\) built with \(\beta_2\) is what's carried forward to the next step). Because there is no \(v_t\) buffer, Lion's optimizer-state memory is **half** of Adam's — one buffer per parameter instead of two — which is the headline practical claim: comparable or better quality than AdamW on the tasks the paper evaluated, at meaningfully lower optimizer memory footprint, which at frontier parameter counts is a real, quantifiable infrastructure saving (per Section 1.5's memory-wall arithmetic).

### 5.2 Muon

Muon (Jordan et al., 2024) targets 2D weight matrices specifically (not embeddings or norm parameters, which are typically left on AdamW even in a Muon-hybrid setup) and replaces the elementwise adaptive scaling of Adam with an **orthogonalized momentum update**: take the momentum-averaged gradient matrix, and instead of applying it directly (or elementwise-rescaled), project it toward the nearest matrix with orthonormal rows/columns via a fast approximate **Newton-Schulz iteration**, then apply that orthogonalized matrix as the update direction. The claimed benefit is that orthogonalizing the update spreads the update's effect more evenly across a weight matrix's singular-value spectrum than Adam's elementwise scaling does, which the authors argue produces a more efficient use of each optimization step particularly for the large dense matmuls that dominate transformer parameter counts — reported results (including some public large-scale training-speed comparisons, e.g., in NanoGPT-speed-run-style benchmarks and at least one frontier-adjacent lab's disclosed usage) show meaningful wall-clock/step-count efficiency gains over AdamW on the runs tested.

### 5.3 Why AdamW-family optimizers still dominate frontier pretraining despite this research, stated honestly

This is the question a staff interview is actually likely to probe, and the honest answer has to name real reasons, not dismiss the newer research:

- **Track record at frontier scale, specifically.** AdamW has been validated, at the hundreds-of-billions-of-parameters / multi-trillion-token scale, across an enormous number of independently-run frontier training efforts, with a correspondingly enormous accumulated body of tuned-hyperparameter folk knowledge (which \(\beta_2\), which \(\epsilon\), which warmup length works at which scale) that transfers reasonably well across labs and architectures. Lion and Muon's strongest published results are, as of this writing, at meaningfully smaller scale and narrower architecture/task coverage than the largest disclosed AdamW-trained frontier runs (DeepSeek-V3's 671B/14.8T-token run, Llama 3.1's 405B/15T+-token run, GPT-4-class training) — the risk of a new optimizer's advantage *not transferring* to that regime is exactly the scale-transfer problem covered in `006_Pretraining_Ablations_And_Research_Methodology.md`, and it applies to optimizer choice just as much as to architecture choice.
- **Cost asymmetry of being wrong.** Switching optimizers for a run costing tens of millions of dollars, based on evidence from a much smaller-scale comparison, is a bet with a large downside (a subtle instability or a worse final loss discovered only after most of the budget is spent) and a comparatively modest upside (some fraction of wall-clock/compute savings) relative to sticking with the best-understood option — this asymmetry alone rationally biases frontier labs toward conservatism on this specific axis, independent of how promising the new method's published numbers look.
- **Optimizer choice interacts with everything else in the recipe.** Learning-rate schedule, warmup length, weight-decay strength, gradient-clipping threshold, and batch-size scaling heuristics (Sections 2–4) are all typically tuned *around* AdamW's specific update dynamics; switching optimizers is not a drop-in change, it potentially invalidates a large amount of accumulated tuning intuition and requires re-deriving good values for every one of those interacting hyperparameters, which is itself a nontrivial ablation and validation cost.
- **The gap, honestly, may be closing rather than closed.** Lion's memory-saving case and Muon's step-efficiency case are both real, published, and not seriously disputed at the scales they've been tested — this is a case where "the field hasn't fully adopted X" should not be read as "X doesn't work," but as "the evidence at frontier scale specifically is thinner than the evidence for AdamW, and frontier labs are conservative about exactly the kind of bet where being wrong is extremely expensive and hard to detect until very late." Some frontier-adjacent labs have begun disclosing Muon-family usage at meaningful scale, and this is a genuinely live, evolving area rather than a settled one — a staff-level answer should reflect that state of flux rather than asserting either "AdamW is definitively optimal" or "the new methods have already won."

## 6. Summary of interacting hyperparameters

| Hyperparameter | What it controls | Key interaction |
|---|---|---|
| Peak LR (\(\eta_{\max}\)) | Step size at full training speed | Scales with batch size (Section 4.2); too high without warmup destabilizes early training |
| Warmup steps | Ramp from 0 to peak LR | Must be long enough for \(v_t\) to stabilize (Section 2.1); larger batch/peak-LR needs more warmup |
| \(\beta_1, \beta_2\) | Momentum / second-moment decay | Lower \(\beta_2\) (~0.95) common at LLM scale for faster adaptation; interacts with warmup length |
| Weight decay \(\lambda\) | Regularization strength | Decoupled in AdamW (Section 1.4); too high alongside a low final LR in decay/annealing (`005_...md`) can over-shrink weights right when the model should be settling |
| Grad clip norm \(c\) | Bounds worst-case step size | Rarely binds on normal batches; a safety net against rare bad batches, not a steady-state lever |
| Batch size \(B\) | Gradient-noise reduction, DP degree | Governed by critical batch size (Section 4.1); interacts with wall-clock/DP-degree tradeoffs |
| Total steps \(T\) / schedule shape | Where LR ends up decaying to, and when | Cosine needs \(T\) fixed up front; WSD (Section 2.3) decouples this, easing the overtraining decision in `002_...md` |

Every row in this table is best understood not as an independently-tunable dial but as a set of mutually-constraining choices whose *joint* validity at target scale is exactly what a proxy-scale ablation campaign (`006_Pretraining_Ablations_And_Research_Methodology.md`) is trying to de-risk before the full run commits to all of them at once.

## 7. A closing checklist

1. Can you derive AdamW's bias-correction factors from the moment-update recursion on demand (Section 1.3), and state precisely what breaks in the first several steps without them, rather than only knowing "bias correction exists"?
2. Can you state exactly why decoupled weight decay differs from L2-regularized Adam, and trace through the mechanism by which folding decay into the gradient makes its effective strength depend on each parameter's own \(v_t\) (Section 1.4)?
3. Can you explain why warmup matters more at large batch size specifically, connecting both the bias-correction argument and the batch-size/peak-LR pairing convention, rather than citing only one of the two (Section 2.1, Section 2.4's cross-reference)?
4. Given two measured gradient norms at two different batch sizes, can you actually estimate the gradient noise scale and critical batch size (Section 4.1), not just define the term?
5. Can you give an honest, two-sided account of Lion and Muon — what they change mechanically, and why frontier labs remain conservative about adopting them — without either dismissing the research or overstating its frontier-scale validation (Section 5)?
6. Can you compute, for a specific parameter count, the AdamW memory footprint that motivates state-sharding infrastructure (Section 1.7), rather than only citing "~14 bytes per parameter" as an unmotivated constant?

## 8. Quick-reference glossary

- **Bias correction** — dividing Adam's raw moment estimates by \((1-\beta^t)\) to remove the zero-initialization bias that is otherwise largest in the first several training steps (Section 1.3).
- **Decoupled weight decay** — applying weight decay as a direct multiplicative shrinkage of the parameter, outside the gradient/second-moment pipeline, so its strength doesn't depend on a parameter's own gradient-noise history (Section 1.4).
- **Global-norm gradient clipping** — rescaling the entire gradient vector (preserving direction) if its overall norm exceeds a threshold, protecting against a single anomalous batch rather than altering steady-state behavior (Section 3).
- **Gradient noise scale / critical batch size** — the batch-size regime boundary past which additional samples stop buying proportionally larger safe step sizes, marking the shift from linear to square-root LR scaling (Section 4.1-4.2).
- **Warmup-stable-decay (WSD)** — an LR schedule holding a constant plateau after warmup, deferring the decay-shape decision until training's actual endpoint is chosen, rather than committing to it up front as cosine decay requires (Section 2.3).
- **Sign-based update (Lion)** — an optimizer update using only the sign of a momentum-smoothed gradient, discarding the second-moment buffer entirely for a 2x optimizer-memory saving relative to AdamW (Section 5.1).
- **Orthogonalized momentum (Muon)** — an optimizer update that projects a momentum-averaged gradient matrix toward its nearest orthonormal matrix via Newton-Schulz iteration, targeting large 2D weight matrices specifically (Section 5.2).

## 9. See also

The batch-size/wall-clock and data-parallelism-degree interactions referenced in Section 4.3 connect directly to the training-infrastructure maturity constraints discussed in `003_Model_Architecture_Decisions_At_Pretraining_Time.md`. The WSD schedule introduced in Section 2.3 is the same mechanism used to structure the annealing/cooldown stage in `005_Curriculum_And_Multi_Stage_Pretraining.md`, Section 4.3. The question of whether a hyperparameter or optimizer choice validated at proxy scale actually transfers to frontier scale — raised throughout Section 5.3's discussion of Lion/Muon adoption risk — is the direct subject of `006_Pretraining_Ablations_And_Research_Methodology.md`. Worked, staff-level applications of this file's specific framework — implementing AdamW from scratch, debugging a broken decoupled-weight-decay implementation, and reasoning through a batch-size increase — are in `008_Interview_Questions_Part2.md`, Q21-Q28.
