# Scaling Laws and Compute-Optimal Training

Scaling laws convert "how big should this model be" from an intuition-driven guess into a fitted, extrapolable, but never fully certain quantitative bet — this file derives the laws, the FLOPs arithmetic they're built on, and the operational discipline required to use them responsibly at frontier stakes.

## 1. Why scaling laws exist as a research object at all

Training a frontier model is a decision made *before* you know how good the resulting model will be. A lab has to commit a training-compute budget — GPU-hours, dollars, wall-clock months of a scarce cluster — to a specific \((N, D)\) choice (parameter count, token count) months before that run finishes, at a cost that by the mid-2020s runs from single-digit millions to plausibly hundreds of millions of dollars for a single training run. Scaling laws are the field's answer to "how do we make that multi-million-dollar bet with quantitative evidence instead of intuition": fit a smooth, extrapolable relationship between (compute, model size, data size) and achieved loss using many *cheap* smaller runs, then use that fitted relationship to choose the configuration of the one expensive run you can actually afford to do once.

This file covers two generations of that research program — Kaplan et al. (2020) and its correction by Hoffmann et al. (2022, "Chinchilla") — derives the FLOPs approximation both papers build on, and then addresses the operational question a staff engineer actually gets asked: how do you use any of this in a real planning process, and what happens when the extrapolation is wrong.

## 2. The Kaplan et al. (2020) power laws

### 2.1 The empirical finding

Kaplan et al. ("Scaling Laws for Neural Language Models," OpenAI, Jan 2020) trained a large family of transformer LMs varying three quantities somewhat independently — non-embedding parameter count \(N\), dataset size \(D\) (tokens), and training compute \(C\) — and found that held-out test loss, as a function of any one of these with the others held large enough not to bottleneck, follows a **power law**:

\[
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}
\]

where \(N_c, D_c, C_c\) are fitted constants (roughly, "the scale at which loss would nominally hit 1 nat/token under the fit") and the exponents empirically came out in the range \(\alpha_N \approx 0.076\), \(\alpha_D \approx 0.095\), \(\alpha_C \approx 0.050\) (values as reported in the paper; treat exact digits as the paper's specific fit rather than universal constants — different tokenizers, architectures, and data distributions shift them). The qualitative content that matters far more than the exact exponents: **loss decreases smoothly and predictably as a power law over many orders of magnitude of scale, with no sign (in the ranges tested) of the smooth trend breaking down.** This smoothness is itself the finding that made "extrapolate from small models to predict a big one" a credible research strategy in the first place — if loss curves were noisy or discontinuous as a function of scale, extrapolation would be unjustifiable.

A power law is linear in log-log space: \(\log L = \log N_c^{\alpha_N} - \alpha_N \log N\), which is exactly why these relationships are fit and visualized as straight lines on log-log axes, and why the standard fitting procedure is ordinary least squares on logged data:

```python
import numpy as np

def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = a * x^b via least squares in log-log space.
    Returns (a, b). x, y must be positive (e.g., x=N, y=loss)."""
    log_x, log_y = np.log(x), np.log(y)
    b, log_a = np.polyfit(log_x, log_y, deg=1)   # slope, intercept
    return np.exp(log_a), b
```

### 2.2 The compute-allocation conclusion Kaplan et al. drew

The paper's most consequential downstream claim was about *how to spend a fixed compute budget* \(C\): given \(C\), what \((N, D)\) minimizes loss? Kaplan et al.'s fits implied the compute-optimal strategy allocates the large majority of *additional* compute to growing \(N\) (model size), with dataset size \(D\) growing much more slowly — the paper's own summary framing was that model size should grow roughly as \(N \propto C^{0.73}\) with a comparatively weak corresponding growth in tokens. Operationally, this was read across the field (correctly, given the paper's own numbers) as "if you get 10x more compute, spend most of it making the model bigger, not on more data" — and largely explains why GPT-3 (see `..\..\GPT\003_GPT3.md`, Section 3) was trained on a comparatively modest ~300B tokens (≈1.7 tokens per parameter at 175B params) rather than proportionally more.

### 2.3 What was wrong with this fit

Two methodological issues, both directly addressed by Hoffmann et al. (Section 3), are worth stating precisely rather than just asserting "Kaplan was wrong":

- **Learning-rate schedule confound.** Kaplan et al.'s smaller-model runs did not always fully decay the learning rate schedule to match the intended token budget for each run — some runs' LR schedules were tuned/decayed as if training would continue longer than the point at which loss was actually read off, which systematically inflates the loss measured at smaller \(D\) values relative to what a properly-decayed run at that same \(D\) would achieve. That bias distorts the fitted \(L(D)\) curve, in particular making bigger-model/less-data configurations look relatively better than they actually are.
- **Parameter-counting convention.** Kaplan et al. primarily fit against *non-embedding* parameter count. This is a reasonable modeling choice but not obviously the one you want when reasoning about *training FLOPs*, since a meaningful fraction of a model's forward/backward compute is embedding-independent in a different way than the fit implicitly assumes at small scale, where embedding parameters are a much larger fraction of the total than at 175B scale.

Neither issue invalidates the qualitative discovery (smooth, extrapolable power-law scaling); both distort the *quantitative* compute-allocation conclusion — which is exactly what Chinchilla revisited.

## 3. The \(C \approx 6ND\) FLOPs approximation, derived

Before the Chinchilla correction can even be stated, you need the standard approximation connecting training compute \(C\), parameter count \(N\), and token count \(D\), because both papers' compute-optimal frontiers are expressed as curves over \(C\) using exactly this relationship.

### 3.1 Forward pass: \(\approx 2N\) FLOPs per token

Consider a single dense matrix multiply inside a transformer: a \((d_{in} \times d_{out})\) weight matrix applied to a \(d_{in}\)-dimensional activation vector. Computing one output element requires \(d_{in}\) multiplications and \(d_{in} - 1 \approx d_{in}\) additions, i.e., \(\approx 2 d_{in}\) FLOPs (counting a multiply-add pair as 2 FLOPs, the standard convention in this literature); for all \(d_{out}\) output elements, that's \(\approx 2 d_{in} d_{out}\) FLOPs — exactly twice the number of parameters in that weight matrix. This "2 FLOPs per parameter per token" ratio holds for essentially every dense matmul in a transformer (QKV projections, attention output projection, the two FFN matrices), because every one of those operations is, per token, one matrix-vector product costing \(2 \times (\text{number of weights in that matrix})\) FLOPs. Summing over every parameter in the model (this is precisely why \(N\) is usually taken to mean *non-embedding* parameters for this approximation — embedding lookups are not matmuls and cost negligible FLOPs by comparison, and attention's \(O(n^2 d)\) score computation is typically small relative to the \(O(nd^2)\) matmul cost except at very long context, which is why the approximation is a *dense-matmul-dominated* approximation, not an exact accounting):

\[
C_{\text{forward}} \approx 2ND
\]

for a corpus of \(D\) tokens (one forward pass per token processed, and the "per token" figure of \(2N\) FLOPs summed over \(D\) tokens gives \(2ND\)).

### 3.2 Backward pass: \(\approx 4N\) FLOPs per token

Backpropagation through the same matmul requires computing two separate gradients: the gradient with respect to the *input activations* (needed to keep propagating backward into earlier layers) and the gradient with respect to the *weights* (needed for the optimizer step). Each of these is itself a matmul of the same size as the forward matmul — multiplying the upstream gradient by the transposed weight matrix (for the activation gradient) and by the transposed input activation (for the weight gradient) — so each costs \(\approx 2ND\) again, for a total backward cost of

\[
C_{\text{backward}} \approx 2 \times (2ND) = 4ND
\]

### 3.3 Total: \(C \approx 6ND\)

\[
C \approx C_{\text{forward}} + C_{\text{backward}} = 2ND + 4ND = 6ND
\]

This is the approximation used essentially everywhere in the scaling-law and compute-planning literature (both Kaplan et al. and Hoffmann et al. use it, as does every "how many FLOPs did training run X cost" back-of-envelope calculation across the per-model docs in this collection, e.g., GPT-3's reported \(3.14\times10^{23}\) FLOPs — see `..\..\GPT\003_GPT3.md`, Section 3 — is consistent with this formula given its 175B parameters and ~300B tokens: \(6 \times 175\text{e}9 \times 300\text{e}9 \approx 3.15\times10^{23}\)).

```python
def training_flops(n_params: float, n_tokens: float) -> float:
    """C ~ 6*N*D FLOPs approximation for one full pretraining pass over the data."""
    return 6.0 * n_params * n_tokens
```

Two caveats worth stating precisely, since this is a favorite place for an interviewer to probe whether "6ND" is understood as a derivation or just memorized: (a) it ignores attention's \(O(n^2 d)\) term, which becomes non-negligible relative to the \(O(nd^2)\) matmul term at very long context lengths — for context length \(n\) comparable to or exceeding \(d_{\text{model}}\), attention FLOPs are no longer a rounding error; (b) it is a *dense* approximation — for a mixture-of-experts model, \(N\) in this formula should be the *activated* parameter count per token, not total parameters, since only the activated experts' matmuls are actually computed for a given token (see `..\..\OpenSource\007_DeepSeek_V3.md` for a model where total and activated parameter counts differ by roughly 18x, making this substitution consequential rather than a rounding nuance).

## 4. The Chinchilla correction

### 4.1 Methodology: three independent estimators, one conclusion

Hoffmann et al. ("Training Compute-Optimal Large Language Models," DeepMind, 2022) re-ran the scaling-law fitting exercise with the LR-schedule confound (Section 2.3) fixed — every run's cosine schedule is decayed to match its own actual token budget rather than reused across different budgets — and used **three separate methodologies** to estimate the compute-optimal frontier, explicitly as a robustness check against any one method's fitting artifacts:

1. **Fix model sizes, vary training tokens (IsoFLOP-style by sweeping horizontally):** train a fixed set of model sizes, each for many different token budgets, and read off the loss-minimizing token count for each size at a series of fixed compute budgets.
2. **IsoFLOP profiles:** for several fixed compute budgets \(C\), train a range of model sizes \(N\) with \(D\) determined by \(C = 6ND\) (i.e., trace out a curve of constant compute), and find the \(N\) minimizing loss along each curve — this produces a parabola-shaped loss-vs-\(N\) curve at each fixed \(C\), whose minimum directly gives the compute-optimal \((N^*, D^*)\) at that \(C\).
3. **Parametric loss fit:** fit a single closed-form function of both \(N\) and \(D\) jointly, \(L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}\) (an irreducible loss floor \(E\) plus separate power-law penalty terms for under-sized model and under-sized data), then analytically solve for the \((N,D)\) minimizing this function subject to \(C=6ND\).

All three approaches converged on the same qualitative correction, which is the paper's headline result: **for compute-optimal training, \(N\) and \(D\) should scale at roughly the same rate with compute** — both close to \(N, D \propto C^{0.5}\) — rather than Kaplan et al.'s much more model-size-skewed allocation. The often-quoted summary heuristic derived from this — **roughly 20 training tokens per parameter at compute optimality** — is the practical number that reshaped the field's default parameter-to-token ratio: it reads directly off the fitted frontier's \(D^*/N^*\) ratio at the compute scales the paper examined, and is explicitly a *heuristic reading of the fit*, not itself a law with independent theoretical justification — the underlying law is the joint \(L(N,D)\) functional form and its fitted exponents, and "~20 tokens/param" is a convenient rule of thumb that approximately holds around the compute regime Chinchilla was fit on.

### 4.2 The headline empirical result

Applying the corrected frontier to GPT-3's training compute budget, Hoffmann et al. found that a **much smaller model trained on much more data** — their own Chinchilla model, 70B parameters on 1.4T tokens, compute-matched to Gopher's 280B-parameter / ~300B-token training run — outperformed the larger, less-data-rich model on a broad evaluation suite. This was the direct empirical demonstration, not just a re-fit curve, that the field had been systematically over-sizing models relative to their training data under the older Kaplan-style guidance. Restated in the GPT-3 case specifically (see `..\..\GPT\003_GPT3.md`, Section 7 and 12): a Chinchilla-optimal allocation of GPT-3's actual training compute would have called for a substantially smaller model than 175B trained on roughly 3.5T tokens, rather than 175B params on ~300B tokens (~1.7 tokens/param, roughly an order of magnitude below the ~20 tokens/param heuristic).

```python
def chinchilla_optimal_split(compute_budget_flops: float,
                              alpha: float = 0.5, tokens_per_param: float = 20.0
                              ) -> tuple[float, float]:
    """Given a fixed training-compute budget (FLOPs) and C = 6*N*D,
    return the compute-optimal (N, D) under the ~20-tokens-per-parameter
    Chinchilla heuristic (D = tokens_per_param * N).
    C = 6*N*D = 6*N*(tokens_per_param*N) = 6*tokens_per_param*N^2
    => N = sqrt(C / (6 * tokens_per_param))
    """
    n_params = (compute_budget_flops / (6.0 * tokens_per_param)) ** 0.5
    n_tokens = tokens_per_param * n_params
    return n_params, n_tokens
```

## 5. How labs actually use scaling laws operationally

The academic result ("fit a power law, read off the optimum") is only half the story; the other half is a real R&D process with its own risks, and this is the part staff-level discussion should center on rather than reciting the Chinchilla headline number.

### 5.1 Fitting a proxy scaling curve before committing to a frontier run

The standard operational pattern: before committing the full target compute budget \(C_{\text{target}}\) to one training run, a lab runs a **grid of much smaller proxy models** — typically spanning several orders of magnitude below the target scale (e.g., from tens of millions up to a few billion parameters, each at a matched \(D\) following \(C=6ND\)) — and fits the same \(L(N,D)\) or IsoFLOP relationship described in Section 4.1 to that proxy grid. This serves several concrete, distinct purposes beyond just "find the optimal \(N,D\)":

- **Choosing \((N, D)\) for the target run itself**, by extrapolating the fitted frontier out to \(C_{\text{target}}\) — this is the direct scaling-law application.
- **Validating architecture and hyperparameter choices at low cost before they're locked in at target scale** — this overlaps heavily with the ablation-methodology problem covered in `006_Pretraining_Ablations_And_Research_Methodology.md`, since the proxy grid used for fitting a scaling curve is frequently the *same* grid used to compare architectural variants (e.g., "does variant A's fitted scaling curve extrapolate to a lower loss at target compute than variant B's curve, even if A looks worse at small scale" is a materially different and more decision-relevant question than "which variant has lower loss at the small scale actually tested").
- **Estimating achievable loss at the target scale before the run starts**, which functions as a go/no-go gate and as an expectation-setting number for whoever is approving the compute spend — if the fitted curve predicts a loss substantially worse than a competitor's known achieved loss at similar compute, that is exactly the kind of signal that triggers a rethink of the recipe before, not after, spending the full budget.

### 5.2 The risk that the fitted law does not hold at the target scale

This is the sharpest, most staff-relevant risk in the entire operational picture, and it is a *structural* risk, not a hypothetical one: **a power-law fit over a proxy range of, say, \(10^7\) to \(10^9\) parameters is being extrapolated to \(10^{11}\)–\(10^{12}\) parameters — often several more orders of magnitude than the range it was fit on.** Nothing in the mathematics of curve-fitting guarantees a relationship that holds cleanly over the fitted range continues to hold, with the same exponents, arbitrarily far outside it. Concretely, several distinct failure modes are worth naming separately rather than lumping together as "the law might break":

- **Genuine regime change in the loss-vs-scale relationship.** The exponents themselves could differ at larger scale — nothing about a fit fixes the exponent as a universal physical constant; it is an empirical property of the specific architecture/data/optimizer regime tested, and there is no first-principles guarantee it is scale-invariant indefinitely. This is precisely the kind of correction Chinchilla itself represents relative to Kaplan et al., just one level higher: Chinchilla's own fit could, in principle, be superseded by a future correction at yet larger scale, and indeed the field has continued to refine scaling methodology (data-quality-aware scaling laws, scaling laws conditioned on data repetition, MoE-specific scaling laws, etc.) precisely because "the" scaling law is better understood as the current best fit for a given regime, not a settled constant of nature.
- **Data availability/quality becoming the binding constraint before compute does.** A fitted curve implicitly assumes a training corpus's marginal token is roughly as informative as the tokens already used in the fit. At the multi-trillion-token scale many frontier runs now target, this assumption can fail — either because the highest-quality easily-available data is exhausted well before \(D^*\) is reached (forcing either heavier deduplication/repetition of a smaller pool, which has its own, separately studied returns-diminishing effects, or inclusion of lower-quality tokens that don't carry the same marginal value the fit assumed), or because the *quality composition* of a much larger corpus is systematically different from the proxy runs' corpus. This is a real, practically binding constraint distinct from the pure model-vs-data-compute tradeoff the classical scaling-law literature isolates.
- **Hyperparameter and infrastructure interactions invisible at proxy scale.** Optimizer hyperparameters, batch size, and numerical-precision choices that are stable and near-optimal at proxy scale are not guaranteed to remain so at target scale (see `004_Optimizers_LR_Schedules_And_Hyperparameters.md`'s discussion of critical batch size and, separately, μ-parameterization-style approaches designed explicitly to make hyperparameters transfer predictably across scale) — a training run can underperform its fitted scaling-law prediction not because the loss-vs-compute law itself was wrong, but because the *hyperparameters* chosen for the target run were themselves mis-extrapolated.
- **The fit is confounded by exactly the kind of schedule/measurement artifacts Section 2.3 describes for Kaplan et al.** Fitting a clean scaling law from proxy runs requires real methodological discipline (matched, fully-decayed LR schedules across all proxy points; consistent tokenization and data mixture between proxy and target runs; consistent evaluation-loss measurement) — sloppy proxy-run methodology can produce a confidently-wrong extrapolation that looks statistically clean.

The honest operational posture at a frontier lab, given all of this, is that a scaling-law fit is treated as **the best available quantitative prior**, not a guarantee — it substantially de-risks the \((N,D)\) choice relative to guessing, but the gap between "de-risked" and "certain" is exactly where staff-level engineering judgment (how far outside the fitted range are we extrapolating; how much have we changed about data/architecture/optimizer relative to what the fit was measured on; do we have budget for a mid-scale canary run between the proxy grid and the full target, as discussed in `006_Pretraining_Ablations_And_Research_Methodology.md`) is expected to operate.

## 5.3 A numeric comparison across several compute budgets

It is worth tabulating how the Kaplan-style and Chinchilla-style allocations actually diverge in absolute numbers, since the qualitative claim ("Kaplan skews toward bigger models, Chinchilla splits more evenly") is easy to state and easy to under-appreciate quantitatively.

```python
def kaplan_style_split(compute_budget: float, model_exponent: float = 0.73) -> tuple[float, float]:
    """Illustrative reconstruction of a Kaplan-et-al.-style allocation, which
    skews additional compute much more heavily toward N than toward D.
    Not the paper's literal fitting procedure -- a simplified illustration
    of the qualitative N ~ C^0.73-style skew the paper's own summary implied."""
    # Calibrate against a reference point consistent with GPT-3: ~175B params
    # at ~3.14e23 FLOPs, then scale N with the given exponent.
    ref_c, ref_n = 3.14e23, 175e9
    n = ref_n * (compute_budget / ref_c) ** model_exponent
    d = compute_budget / (6.0 * n)
    return n, d

for c in [3.14e23, 1e24, 1e25, 1e26]:
    n_k, d_k = kaplan_style_split(c)
    n_c, d_c = chinchilla_optimal_split(c)   # from Section 4.2 above
    print(f"C={c:.1e}  Kaplan-style N={n_k:.2e} D={d_k:.2e}  |  Chinchilla N={n_c:.2e} D={d_c:.2e}")
```

Running this comparison shows the gap *widening* as compute grows, not staying fixed — at GPT-3's own compute budget the two allocations are calibrated to roughly agree by construction, but by \(10^{26}\) FLOPs (two orders of magnitude more compute), the Kaplan-style allocation has pushed \(N\) up substantially faster than Chinchilla's \(\sqrt{C}\)-scaling would, correspondingly starving \(D\) more severely relative to what Chinchilla's fit recommends. This growing divergence is exactly why the correction mattered increasingly *more*, not less, as the field's compute budgets kept scaling up through the early 2020s — a methodological bias that's small at the scale it was first estimated on can become a large practical error at a much bigger scale, which is itself a preview of the scale-transfer theme this module returns to directly in `006_Pretraining_Ablations_And_Research_Methodology.md`.

## 6. Inference-cost-aware "deliberate overtraining": a different objective function entirely

Everything above optimizes a single quantity: **training loss for a fixed training-compute budget.** This is not the same optimization problem as **minimizing total cost of ownership for a model that will be deployed and queried many times over its production lifetime**, and conflating the two is a common and consequential mistake.

### 6.1 Stating the two objectives precisely

*Training-compute-optimal*: given a fixed training FLOPs budget \(C\), choose \((N, D)\) with \(C = 6ND\) to minimize training loss \(L(N,D)\). This says nothing whatsoever about what happens after training finishes.

*Total-cost-of-ownership-optimal*: given a fixed **quality bar** (a target loss or downstream benchmark level) and a projected deployment volume (some number of inference queries, or inference-tokens, over the model's service lifetime), minimize

\[
\text{Total Cost} = \text{Cost}_{\text{train}}(N, D) + n_{\text{queries}} \times \text{Cost}_{\text{inference-per-query}}(N)
\]

The key structural fact that breaks the equivalence between the two objectives: **training cost is a one-time expenditure that scales with \(N \times D\) (via \(C=6ND\)), while inference cost per query scales with \(N\) alone** (for a dense model, ignoring cache and batching effects, inference FLOPs per generated token is \(\approx 2N\), i.e., just the forward-pass term from Section 3.1, since there's no backward pass at serving time) **and recurs on every single query for the model's entire deployment lifetime.** As \(n_{\text{queries}}\) grows — and for a widely-deployed product, it grows very large — the second term can dwarf the first, no matter how expensive the training run was, because it is being multiplied by a potentially enormous constant while training cost is paid exactly once.

This means that for a large enough \(n_{\text{queries}}\), the total-cost-minimizing strategy at a *fixed quality bar* is: **find the smallest \(N\) that reaches the target quality bar, and accept training it on far more tokens than the training-compute-optimal \(D^*\) for that \(N\) would call for** — because extra pretraining tokens are a one-time cost paid once at training time, while extra parameters are a cost paid on every future inference call. This is precisely, and explicitly, the reasoning Meta gives for Llama 3's smaller models (see `..\..\OpenSource\003_Llama3.md`, Section 5): the 8B model is trained on 15T+ tokens, roughly 75–100x past the ~150–200B tokens a Chinchilla-style compute-optimal recipe would prescribe for an 8B model, and this is explicitly framed in the paper as informed by observing that the smaller models' loss had not saturated even far past that point — i.e., there were still cheap gains available from continuing to overtrain, and those gains, paid for once, reduce the *serving* cost needed to hit a given quality bar for the rest of the model's deployed life. Notably, the 405B model in the same release is trained much closer to its own training-compute-optimal point — consistent with the overtraining logic being specifically a *smaller-model-for-cheaper-inference-at-scale* decision, not a blanket policy applied uniformly regardless of size or expected deployment volume.

### 6.2 A worked framing

```python
def total_cost_of_ownership(n_params: float, n_tokens: float,
                             n_inference_queries: float,
                             avg_output_tokens_per_query: float,
                             cost_per_flop_train: float,
                             cost_per_flop_infer: float) -> float:
    """Illustrative TCO comparison -- not a precise hardware-cost model.
    Training cost uses the 6ND approximation; inference cost per generated
    token uses the ~2N forward-pass-only approximation (Section 6.1)."""
    train_flops = 6.0 * n_params * n_tokens
    infer_flops = 2.0 * n_params * n_inference_queries * avg_output_tokens_per_query
    return train_flops * cost_per_flop_train + infer_flops * cost_per_flop_infer
```

Holding quality (and therefore, roughly, \(N\)) fixed, increasing \(D\) only ever increases the first term. The overtraining bet is that a fixed increase in \(D\) buys enough of a quality improvement to let you *decrease* \(N\) while holding the quality bar fixed — and because the second term scales with \(N\), not \(D\), that trade is a net win whenever \(n_{\text{queries}}\) is large enough that the second term dominates the total. This is exactly why the decision is inference-*volume*-dependent: a model expected to be queried a modest number of times (a narrow internal research checkpoint, a one-off benchmark submission) has little reason to overtrain past the training-compute-optimal point, since the second term never grows large enough to dominate; a model expected to serve as a widely-deployed consumer or API product has every reason to, and the more successful and widely-deployed the product becomes, the more that initial overtraining investment continues to pay off.

### 6.3 The general lesson

Chinchilla-style compute-optimal scaling laws answer a well-posed but narrower question than the one a lab actually faces when deciding how to train a model it intends to ship. Treat "\(N,D\) should be roughly compute-optimal" as the right default *only* when training compute is the dominant term in the real cost equation you care about — and treat deliberate, large-margin overtraining relative to that optimum as the correct response precisely when projected inference volume is large enough to flip which term in Section 6.1's total-cost equation dominates. Both are "compute-optimal" answers; they are compute-optimal for two different objective functions, and stating which objective function is actually being optimized is the single most important thing to get right before quoting a tokens-per-parameter ratio in an interview or in a real planning document.

## 7. A closing checklist

1. Given a compute budget, can you actually derive the Chinchilla-optimal \((N,D)\) split from an IsoFLOP-style fit (Section 4.1, Q7 of `007_Interview_Questions_Part1.md`), not just recite "~20 tokens per parameter" as a fixed constant?
2. Can you state precisely what methodological flaw in Kaplan et al.'s original fits (Section 2.3) caused it to skew toward larger models relative to Chinchilla's later correction, rather than only knowing that a correction happened?
3. Can you derive \(C \approx 6ND\) from the forward/backward FLOPs argument on demand (Section 3), including both caveats (attention's quadratic term at long context; activated vs. total parameters for MoE)?
4. Before trusting any scaling-law extrapolation for a real decision, can you name at least three distinct ways it could fail to hold at target scale (Section 5.2), not just gesture at "extrapolation is risky"?
5. Given a specific projected deployment volume, can you actually compute whether a smaller-overtrained or a larger-near-Chinchilla-optimal configuration wins on total cost of ownership (Section 6, worked numerically in Q17 of `007_Interview_Questions_Part1.md`), rather than only knowing the Llama-3-8B anecdote?
6. Can you state, precisely, why training-compute-optimal and total-cost-of-ownership-optimal are different objective functions (Section 6.3), including which quantity (\(N\) alone, vs. \(N\) and \(D\) jointly) each cost term in the TCO equation depends on?

## 7a. A note on what this file deliberately does not cover

Two adjacent topics are intentionally out of scope here, and it's worth naming them so the boundary is explicit rather than an accidental omission. First, *why* a given \((N,D)\) point should be realized as a dense model versus an MoE model, or paired with a particular attention mechanism, is a separate decision covered in full in `003_Model_Architecture_Decisions_At_Pretraining_Time.md` — this file's \(6ND\) arithmetic and compute-optimal frontier apply to either architecture family (with the activated-vs-total-parameter caveat from Section 3.3), but the architecture choice itself is a distinct constraint-satisfaction problem. Second, *how* a lab actually gains confidence that a small-scale scaling-law fit will hold at target scale — beyond the risk factors named qualitatively in Section 5.2 — is developed as a full research-methodology treatment, with statistical sizing and staged validation practices, in `006_Pretraining_Ablations_And_Research_Methodology.md`; this file states the risk, that one develops the response to it.

## 8. Quick-reference glossary

- **Power law** — a relationship of the form \(y = a x^{-\alpha}\), linear in log-log space, used throughout this file to describe how loss falls with scale (Section 2.1).
- **IsoFLOP profile** — a sweep of model sizes at a fixed total compute budget (with token count set by \(C=6ND\)), used to trace the compute-optimal frontier by finding the loss-minimizing model size at each fixed budget (Section 4.1).
- **Compute-optimal (training)** — the \((N,D)\) pair minimizing training loss for a fixed training-compute budget; says nothing about post-deployment inference cost (Section 6.1).
- **Total cost of ownership (TCO)** — training cost plus the full projected lifetime inference cost, the objective that actually governs whether "deliberate overtraining" is the right strategy (Section 6.1-6.2).
- **Deliberate overtraining** — training a smaller model on far more tokens than its own training-compute-optimal \(D^*\), to reduce recurring inference cost at a fixed quality bar, justified only when projected deployment volume is large (Section 6).
- **Proxy scale** — the small, affordable scale at which scaling-law fits and architectural ablations are actually run before being extrapolated to a frontier target (Section 5.1; developed fully in `006_Pretraining_Ablations_And_Research_Methodology.md`).

## 8a. One-paragraph summary

If only one idea from this file survives: fit before you commit, know exactly which objective function your fit is optimizing (training loss for a fixed budget, or total cost of ownership across a projected deployment lifetime — Section 6 — because these can prescribe very different \((N,D)\)), and hold the resulting number with exactly as much confidence as the extrapolation gap between your proxy scale and your target scale actually warrants, no more.

This is also the right frame for reading any specific model's disclosed \((N,D)\) point elsewhere in this collection: ask which objective function that lab was actually optimizing for — GPT-3's ~1.7 tokens/parameter reflects Kaplan-era training-compute guidance (`..\..\GPT\003_GPT3.md`), while Llama 3's 8B at 15T+ tokens reflects a total-cost-of-ownership bet made with the benefit of both Chinchilla's correction and a large projected deployment volume (`..\..\OpenSource\003_Llama3.md`) — rather than judging every disclosed ratio against a single fixed "correct" tokens-per-parameter number.

Read this way, the entire history covered in this file — Kaplan's original fits, Chinchilla's correction, and Llama 3's deliberate departure from Chinchilla-optimality for total-cost reasons — is not three competing answers to the same question, but three different, individually reasonable answers to three different, precisely-statable questions, and being able to state which question is actually in front of you is the single skill this file has been trying to build throughout.

Two immediately practical consequences follow from taking this framing seriously. First, when reviewing someone else's proposed \((N,D)\) configuration, the first question to ask is not "is this Chinchilla-optimal" but "what objective was this configuration optimized for, and is that the objective we actually care about here." Second, when presenting your own proposed configuration, stating the objective function explicitly — before the numbers — pre-empts exactly the kind of confusion that arises when a training-compute-optimal recommendation is evaluated against a total-cost-of-ownership expectation, or vice versa, which is a genuinely common source of miscommunication between research and product-facing stakeholders on real training-planning teams.

Neither consequence requires new mathematics beyond what Sections 1-6 already derive; both are entirely a matter of communication discipline layered on top of a correct technical foundation, which is itself worth noting explicitly — a substantial share of real-world scaling-law misapplication traces back not to a wrong formula but to an unstated or mismatched objective function between the people making and the people evaluating a training-scale decision.

This is a good note to end on precisely because it is the least mathematically glamorous point in the entire file, and also, in practice, the one most likely to actually prevent a costly mistake: derivations get checked by someone who knows the math, but an unstated objective function tends to pass silently through exactly the kind of review that would otherwise catch it.

Make it a habit, in any real planning document or interview answer touching this topic, to write the objective function down as its own explicit line before the numbers that follow from it.

## 9. See also

The FLOPs derivation in Section 3 is the direct basis for the KV-cache and inference-cost arithmetic used throughout `003_Model_Architecture_Decisions_At_Pretraining_Time.md` (e.g., its Section 5a cache-dominance check) and for the total-cost-of-ownership framing in Section 6 above. The scale-transfer risk named in Section 5.2 is developed into a full research-methodology treatment in `006_Pretraining_Ablations_And_Research_Methodology.md`, whose proxy-run apparatus is the same one Section 5.1 describes for fitting a scaling curve. The batch-size and learning-rate interactions that also have to be extrapolated alongside \((N,D)\) when moving from proxy to target scale are covered in `004_Optimizers_LR_Schedules_And_Hyperparameters.md`, Section 4. Worked, staff-level applications of this file's specific framework — deriving \(6ND\) live, fitting an IsoFLOP curve from data, and computing a total-cost-of-ownership comparison numerically — are in `007_Interview_Questions_Part1.md`, Q5-Q9 and Q17-Q18.
