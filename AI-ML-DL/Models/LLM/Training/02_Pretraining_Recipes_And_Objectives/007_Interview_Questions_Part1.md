# Interview Questions — Part 1

Covers: pretraining objectives, scaling laws, compute-optimal training, and architecture decision-making. See `007_Interview_Questions_Part1.md` (this file) and `008_Interview_Questions_Part2.md` for the full 40-question set; no question is repeated across the two files.

---

## Q1: Walk through exactly what "teacher forcing" means in causal LM pretraining, and explain what would go wrong — mechanically, not just "it would be worse" — if you removed it.

Teacher forcing means that at every position \(t\) during training, the context fed into the model is the ground-truth prefix \(x_1, \dots, x_{t-1}\) from the actual training document, not whatever the model itself would have predicted at those positions. Because of the causal attention mask, the model computes a full vector of next-token predictions for *every* position in a single forward pass — position \(t\)'s prediction conditions on positions \(1..t-1\), and because those are the ground-truth tokens (not sampled outputs), every position's loss term is well-defined and independent of any other position's prediction error within that same batch. This is what makes it possible to train on a full sequence in parallel: you're not running an actual autoregressive generation loop during training at all, you're computing \(T\) independent next-token classification problems simultaneously, using one shared set of hidden states.

Remove teacher forcing and you get something like a scheduled-sampling or fully autoregressive-rollout training scheme: at position \(t\), the model would condition on its own previously *sampled* tokens rather than ground truth. This breaks parallelism immediately — you can no longer compute all \(T\) positions' losses in one forward pass, because position \(t\)'s input now depends on position \(t-1\)'s *sampled output*, which itself required a forward pass to produce, so training becomes sequential exactly like inference-time generation, at a roughly \(T\times\) increase in the number of forward passes needed per training example. It also reintroduces exposure bias in the other direction: early in training, when the model's samples are close to random, the "context" being conditioned on is mostly noise, so the useful gradient signal is drowned out by garbage context, and training a large model this way from scratch would converge far slower, if at all, especially early in training when there's no good policy yet to sample from. Teacher forcing sidesteps this entirely by always supplying the true prefix, at the cost of a train/inference mismatch (train-time context is always correct; inference-time context is the model's own possibly-wrong output) that the field has empirically found to be a second-order concern at LLM pretraining scale rather than one requiring scheduled-sampling-style corrections.

---

## Q2: Compare BERT's masked-LM objective and GLM's autoregressive blank infilling specifically on the axis of "can the model produce a coherent multi-token span." Be precise about the mechanism, not just the conclusion.

BERT masks ~15% of positions and predicts each masked position **independently and simultaneously** in a single forward pass, conditioned only on the corrupted (masked) input — critically, if two masked positions are adjacent or otherwise part of what should be one coherent phrase, BERT's training objective has no path for position \(i\)'s prediction to condition on position \(i-1\)'s *predicted* (or even a plausible) value, because both are being predicted from the same fixed, masked context at the same time, with no ordering between them. The model is optimized to make each position's marginal prediction as accurate as possible against the single ground-truth token at that position — there is no term in the objective that rewards the *joint* plausibility of the two predictions together. In practice this means that if you tried to use BERT to generate an entire masked span at once (fill every mask with its highest-probability token, read independently), you can get locally-plausible-but-jointly-incoherent outputs — e.g., independently high-probability but mismatched fills like "New" and "Francisco" for two adjacent masked slots that should jointly read "New York."

GLM's blank infilling instead corrupts spans (not scattered single tokens) and generates each corrupted span's content **autoregressively**: token \(i\) of the span is predicted conditioning on the bidirectionally-encoded surviving context (Part A) *plus every token of the same span already generated* (the earlier part of Part B), because Part B's internal attention is causal. This means token \(i-1\)'s actual generated value is part of the conditioning context for token \(i\) — the same mechanism that makes ordinary GPT-style generation locally coherent (each new token conditions on everything generated so far) is preserved *within* each masked span. This is the precise mechanical reason GLM (and, via the same underlying idea, T5's span corruption, since its decoder is likewise autoregressive over span content) does not have BERT's joint-incoherence problem: the dependency BERT is missing (token \(i\) conditioning on token \(i-1\) of the same corrupted span) is exactly the dependency GLM's causal Part-B attention supplies. The full mechanism (Part A/Part B split, 2D positional encoding needed once spans are permuted) is derived in `..\..\OpenSource\010_GLM4.md`, Section 2, and summarized in `001_Pretraining_Objectives_Overview.md`, Section 4.

---

## Q3: You're designing a new foundation model that has to serve two very different downstream products from one pretrained checkpoint: a semantic-search/embeddings API, and a general chat assistant. Walk through how you'd think about the pretraining objective decision.

The first thing to name explicitly is that these two products want different properties from the base model's representations, and no single objective is unambiguously best for both — this is exactly the tension `001_Pretraining_Objectives_Overview.md` Section 6 frames as "not obviously the best objective for any single task in isolation, but chosen for the general-purpose case." Embeddings/semantic search wants a representation where a single vector (or a small number of them, pooled from the sequence) captures the *whole input's* meaning well enough that cosine similarity between two such vectors tracks semantic similarity — this is a task that bidirectional, full-context encoding is a naturally strong fit for, since there's no reason to artificially prevent an embedding-producing model from looking at the whole input at once. Chat, by contrast, is squarely a free-form-generation task, which (per Section 6 of that file) wants the structural train/inference symmetry, in-context-learning-friendliness, and infrastructure simplicity that causal decoder-only pretraining provides natively.

Given real engineering constraints, I would not try to force one from-scratch pretraining objective to be simultaneously optimal at both (e.g., a GLM-style unified objective is the "purest" attempt at this, but Section 4 of the overview file notes even GLM's own usage converged back toward the causal-generation regime in practice, which is a real data point against betting everything on a single unified objective covering both extremes equally well). The more standard, lower-risk path: pretrain the shared trunk with causal decoder-only LM (optimizing for the harder-to-retrofit property — strong general-purpose generation and in-context learning — as the primary objective, since chat-quality generation is much harder to bolt on after the fact than embedding quality is), and produce the embeddings product via a **lightweight adaptation stage on top of the same causal base**: extracting and pooling hidden states, optionally with a modest amount of bidirectional-attention fine-tuning or a contrastive fine-tuning objective (e.g., in the style of how modern sentence-embedding models are frequently built from decoder-only bases via such adaptation, rather than trained bidirectionally from scratch). This accepts a real but well-understood engineering cost (a separate small adaptation stage, and probably a separate served checkpoint or LoRA-style adapter for the embeddings product) in exchange for not compromising the primary chat product's pretraining recipe, and in exchange for reusing the same expensive base-model pretraining investment across both products rather than running two separate frontier-scale pretraining efforts.

---

## Q4: Implement, from scratch, the core mechanics of causal LM training: build a causal attention mask, and compute the shifted next-token cross-entropy loss, without using any framework's built-in causal-masking helper.

```python
import numpy as np

def causal_mask(seq_len: int) -> np.ndarray:
    """Returns an additive attention-bias mask: 0.0 where attention is allowed,
    -inf where it is forbidden. Position i may attend to positions 0..i."""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # 1s strictly above diagonal
    return np.where(mask == 1, -np.inf, 0.0)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def causal_self_attention(x: np.ndarray, Wq, Wk, Wv, Wo) -> np.ndarray:
    """x: (seq_len, d_model). Single-head, for clarity."""
    seq_len, d_model = x.shape
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = (q @ k.T) / np.sqrt(k.shape[-1]) + causal_mask(seq_len)
    attn = softmax(scores, axis=-1)
    return (attn @ v) @ Wo

def next_token_cross_entropy(logits: np.ndarray, input_ids: np.ndarray) -> float:
    """logits: (seq_len, vocab_size) unnormalized scores at every position.
    input_ids: (seq_len,) token ids for the same sequence.
    Position t's logits predict input_ids[t+1]; the last position has no target."""
    seq_len = input_ids.shape[0]
    shift_logits = logits[:-1]              # positions 0..seq_len-2 predict...
    shift_labels = input_ids[1:]            # ...tokens at 1..seq_len-1
    log_probs = shift_logits - np.log(np.sum(np.exp(shift_logits), axis=-1, keepdims=True))
    nll = -log_probs[np.arange(seq_len - 1), shift_labels]
    return float(np.mean(nll))
```

The two load-bearing pieces an interviewer is checking for: (1) the mask is **additive and applied before the softmax**, using `-inf` (or a large negative number in a fixed-precision setting) so that after exponentiation, forbidden positions contribute exactly zero probability mass — this is a strictly upper-triangular mask (`k=1` excludes the diagonal, since position \(i\) is allowed to attend to itself); (2) the **shift-by-one alignment** between logits and labels — `shift_logits[t]` (originally at sequence position \(t\)) is compared against `shift_labels[t]` which is `input_ids[t+1]`, and the final position is dropped because it has no next token to predict against within this sequence.

---

## Q5: Derive the \(C \approx 6ND\) FLOPs approximation from first principles. Where do the 2 and the 4 each come from?

Start with a single dense matmul inside the network: a weight matrix of shape \((d_{in}, d_{out})\) applied to one token's activation vector. Producing one output element requires \(d_{in}\) multiplications and (approximately) \(d_{in}\) additions to accumulate them, and the convention in this literature is to count a multiply-accumulate pair as 2 FLOPs — so producing all \(d_{out}\) output elements costs \(2 \cdot d_{in} \cdot d_{out}\) FLOPs, which is exactly \(2\times\) the number of weights in that matrix. Every dense matmul in a transformer (QKV projections, the attention output projection, both FFN matrices) has this same property: **forward-pass FLOPs per token, summed across all of a model's parameters \(N\), is \(\approx 2N\)** — "2" comes from counting a multiply and an accumulate-add as separate FLOPs for each of the \(N\) weight-activation products that have to be computed. Summed over a corpus of \(D\) tokens, forward-pass compute for the whole training corpus is \(C_{\text{fwd}} \approx 2ND\).

Backpropagation through that same matmul requires computing **two** separate gradients, each itself a matmul of the same size as the forward one: the gradient with respect to the *input activations* (so error can keep propagating backward into earlier layers — computed as (upstream gradient) times (transposed weight matrix)) and the gradient with respect to the *weights themselves* (computed as (transposed input activation) times (upstream gradient), needed for the optimizer step). Each of these costs \(\approx 2ND\) by the same logic as the forward pass, so backward-pass compute is \(C_{\text{bwd}} \approx 2 \times 2ND = 4ND\) — the "4" is exactly "two gradient computations, each costing the same \(2ND\) as one forward pass." Total: \(C = C_{\text{fwd}} + C_{\text{bwd}} \approx 2ND + 4ND = 6ND\).

Two caveats worth volunteering unprompted: this ignores attention's \(O(n^2 d)\) score-computation term, which is a reasonable approximation when context length \(n\) is small relative to \(d_{\text{model}}\) but becomes non-negligible at very long context; and for an MoE model, \(N\) in this formula must be the *activated* parameters per token, not total parameters, since only the routed (plus shared) experts actually run their matmuls for a given token — using total parameters for an MoE model in this formula would badly overstate its actual training FLOPs.

---

## Q6: Implement a function that fits a power law \(y = a x^b\) to a set of (model-size, loss) proxy-run data points via least squares in log space, and uses the fit to extrapolate predicted loss at a much larger target model size. Discuss, briefly, why this extrapolation is risky.

```python
import numpy as np

def fit_and_extrapolate_power_law(sizes: np.ndarray, losses: np.ndarray,
                                    target_size: float) -> tuple[float, float, float]:
    """Fits loss = a * size^b via OLS on log(size), log(loss), then
    extrapolates predicted loss at target_size.
    Returns (a, b, predicted_loss_at_target)."""
    log_x = np.log(sizes)
    log_y = np.log(losses)
    # log(y) = log(a) + b*log(x) -- linear regression in log-log space
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    b, log_a = np.linalg.lstsq(A, log_y, rcond=None)[0]
    a = np.exp(log_a)
    predicted = a * (target_size ** b)
    return a, b, predicted

# Example: proxy runs from 10M to 1B params, extrapolating to a 500B target
sizes = np.array([1e7, 3e7, 1e8, 3e8, 1e9])
losses = np.array([3.9, 3.55, 3.2, 2.95, 2.7])  # illustrative, decreasing loss
a, b, pred = fit_and_extrapolate_power_law(sizes, losses, target_size=5e11)
```

The extrapolation is risky for a structural reason, not just "noise": the fit is estimated over roughly two orders of magnitude of model size (\(10^7\) to \(10^9\)) and is being extrapolated to a target roughly two-to-three orders of magnitude *beyond* the fitted range. Nothing in the least-squares fitting procedure knows or enforces that the same exponent \(b\) continues to hold outside the range it was estimated on — \(b\) is an empirical property of the specific architecture/data/optimizer regime tested, not a physical constant, and there is real historical precedent for exactly this kind of correction (Chinchilla's re-fit of Kaplan et al.'s exponents, `002_Scaling_Laws_And_Compute_Optimal_Training.md` Sections 2–4). A staff-level answer should also flag the specific confounds that can corrupt the *fit itself* even before extrapolation risk is considered: LR schedules not properly decayed to match each run's own token budget, and inconsistent data mixture/tokenization between the proxy grid and the eventual target run.

---

## Q7: Given a fixed training-compute budget and a set of IsoFLOP proxy runs (several model sizes, each trained with tokens set so that \(6ND\) is held constant, with measured loss at each), write a function that finds the compute-optimal parameter count by fitting the characteristic IsoFLOP parabola and locating its minimum — rather than assuming any fixed tokens-per-parameter heuristic.

```python
import numpy as np

def isoflop_optimal_n(param_counts: np.ndarray, losses: np.ndarray) -> float:
    """Given several (N, loss) points all trained at the SAME fixed compute
    budget C (so D_i = C / (6*N_i) varies inversely with N_i across points),
    fit a quadratic to loss vs log(N) and return the N minimizing it.
    This is the IsoFLOP-profile method from Hoffmann et al. 2022, Approach 2."""
    log_n = np.log(param_counts)
    # loss ~ alpha*log(N)^2 + beta*log(N) + gamma  (locally parabolic near the minimum)
    coeffs = np.polyfit(log_n, losses, deg=2)
    alpha, beta, _ = coeffs
    if alpha <= 0:
        raise ValueError("Fit is not convex; need a wider or better-sampled IsoFLOP sweep.")
    log_n_star = -beta / (2 * alpha)     # vertex of the parabola
    return float(np.exp(log_n_star))

def isoflop_optimal_n_and_d(param_counts: np.ndarray, losses: np.ndarray,
                             compute_budget: float) -> tuple[float, float]:
    n_star = isoflop_optimal_n(param_counts, losses)
    d_star = compute_budget / (6.0 * n_star)
    return n_star, d_star
```

This is deliberately built to *not* assume a ~20-tokens-per-parameter heuristic — it derives the optimal split directly from the shape of the measured loss curve at the fixed compute budget, which is closer to what Hoffmann et al.'s actual IsoFLOP-profile methodology does (Approach 2 of the three approaches described in `002_Scaling_Laws_And_Compute_Optimal_Training.md`, Section 4.1). The "~20 tokens/parameter" figure is a convenient summary of *where the vertex tends to land* across the compute range Chinchilla examined — a real staff answer should be able to derive the optimal split from data directly rather than only quoting the heuristic, precisely because (per Q6) that heuristic's validity outside the fitted regime is not guaranteed.

---

## Q8: An engineer on your team says: "I ran 5 small proxy models from 50M to 2B parameters, fit a Kaplan-style scaling law, and it says we should build a 1 trillion parameter model with a comparatively small token budget. Let's greenlight the full run." What's your response?

Several concerns, and I'd raise all of them before signing off, roughly in order of severity. First, this is a two-and-a-half-order-of-magnitude proxy range being extrapolated four orders of magnitude further out (50M to 2B, then to 1T) — that's a much larger extrapolation gap than even Chinchilla's own re-fit was validated across, and per Q6/Q5's discussion, nothing guarantees the fitted exponents hold that far out. Second, and more specifically damning: "Kaplan-style" is a red flag on its own — Kaplan et al.'s specific methodology has a known, well-documented flaw (LR schedules not decayed to match each proxy run's own token budget, per `002_Scaling_Laws_And_Compute_Optimal_Training.md` Section 2.3) that biases the compute-allocation conclusion toward oversized models — precisely the failure mode that produced GPT-3's well-known undertraining relative to its parameter count. If this fit reproduces that same methodological choice, a "greenlight a huge model, comparatively little data" recommendation is exactly the kind of conclusion I'd expect from a stale methodology, not new evidence about this specific setup.

Concretely, before greenlighting anything at this scale, I'd want: (a) the fit re-run using Chinchilla-style methodology — properly decayed LR schedules per token budget, and ideally the IsoFLOP-profile approach (Q7) rather than a single Kaplan-style \(L(N)\) power-law fit, since the two methodologies have historically given materially different compute-allocation answers; (b) at least one intermediate-scale "canary" run (tens of billions of parameters, well above the 2B proxy ceiling but far below the 1T target) to check that the fitted curve's *trajectory* prediction is actually tracking reality partway to the target, using the early-loss-trajectory-monitoring practice described in `006_Pretraining_Ablations_And_Research_Methodology.md`, Section 3, before committing the full budget; (c) an explicit check of whether the proposed token budget is even achievable given available high-quality data — a 1T-parameter model's compute-optimal token count under any reasonable ratio is enormous, and if the "comparatively small token budget" figure in the recommendation is small specifically because the fit assumed a Kaplan-style skew, that's compounding one bias with what might also be a data-availability problem the team hasn't checked. This is precisely the operational risk `002_Scaling_Laws_And_Compute_Optimal_Training.md` Section 5.2 is about — treating a fit as a strong prior worth de-risking further, not as a green light in itself.

---

## Q9: Llama 3's 8B model is trained on 15T+ tokens — roughly 75-100x past what a Chinchilla-style compute-optimal recipe would prescribe for that parameter count. Is this a mistake? Justify your answer with the actual cost accounting.

No, and the reason requires being precise about which objective function is being optimized. Chinchilla-optimal is the answer to: "for a fixed *training*-compute budget, what \((N,D)\) minimizes training loss?" It says nothing about what happens after training finishes. The objective Meta is actually optimizing for a widely-deployed, long-lived product is closer to total cost of ownership: \(\text{Cost}_{\text{train}}(N,D) + n_{\text{queries}} \times \text{Cost}_{\text{inference-per-query}}(N)\). The structural asymmetry that makes overtraining rational: training cost scales with \(N \times D\) via \(C=6ND\) and is paid exactly **once**; inference cost per query scales with \(N\) alone (roughly \(2N\) FLOPs per generated token, the forward-pass-only term from Q5, since there's no backward pass at serving time) and is paid **on every one of the model's queries for its entire deployment lifetime**. For a model expected to be queried at the volume an 8B-class API/consumer product actually sees, the second term dwarfs the first for any \(n_{\text{queries}}\) past some threshold, no matter how many extra tokens you spend getting there.

Given that structure, the total-cost-minimizing strategy at a fixed target quality bar is: find the *smallest* \(N\) that reaches that quality bar, even if reaching it requires training on far more tokens than \(N\)'s Chinchilla-optimal \(D^*\) — because extra pretraining tokens are a one-time sunk cost, while extra parameters are a cost paid forever at serving time. Meta's own reported loss curves showed the 8B model's loss was still improving log-linearly well past the point Chinchilla-optimal training would have stopped, which is exactly the empirical justification for continuing (`..\..\OpenSource\003_Llama3.md`, Section 5) — every additional token spent training 8B further is cheap relative to the alternative of shipping a larger, Chinchilla-optimal-sized model that would cost more to serve for the rest of its life. Tellingly, the 405B model in the same release is trained much closer to its own training-compute-optimal point — consistent with overtraining being specifically a smaller-model-for-cheaper-inference-at-scale decision, applied where the inference-volume math justifies it, not a blanket claim that Chinchilla's guidance is wrong in general.

---

## Q10: Walk through how you'd decide between a dense and an MoE architecture for a new frontier model, given: your org has never trained an MoE model before, your primary product is a high-concurrency, latency-sensitive API serving hundreds of millions of queries per day, and you have 9 months until you need to ship.

I'd frame this explicitly as a constraint-satisfaction problem, not "which is theoretically better," because MoE's efficiency case (more total capacity at fixed inference-activated compute — `003_Model_Architecture_Decisions_At_Pretraining_Time.md`, Section 1.1) is real but comes bundled with costs that are specifically severe for exactly the situation described: no existing MoE training stack, and a hard 9-month deadline.

The two facts in tension: high query volume with a tight latency SLA is a real, quantifiable argument *for* MoE (more effective capacity at a fixed inference-cost point is directly valuable when you're serving at that scale) — but "never trained an MoE model before" means committing to MoE now means simultaneously building expert-parallel training infrastructure (custom all-to-all communication kernels, a load-balancing mechanism, and validating it doesn't have the router-collapse/expert-starvation instability modes that are specifically MoE's failure modes, not dense training's) *and* the corresponding expert-parallel serving infrastructure, on the critical path of a 9-month deadline. That is a large, unproven execution-risk stack layered on top of the scientific bet, and per `003_Model_Architecture_Decisions_At_Pretraining_Time.md` Section 1.2, when training-infrastructure risk and time-to-ship are the binding constraints, a dense model is the lower-execution-risk default even though it's not the theoretically more inference-efficient choice.

My actual recommendation: ship this cycle with dense, and address the inference-cost pressure via the overtraining lever instead (`002_Scaling_Laws_And_Compute_Optimal_Training.md`, Section 6) — pick the smallest dense model that can hit the quality bar and accept training it well past its Chinchilla-optimal token count, since that's a well-understood, low-execution-risk lever available immediately, with GQA (near-zero-risk, per `003_...md` Section 2.1) as the default attention mechanism to control KV-cache cost at the target concurrency. In parallel, I'd start a genuinely separate, smaller-scale MoE R&D effort — proxy-scale training-stack validation, load-balancing-mechanism development, serving-infrastructure prototyping — explicitly scoped to de-risk MoE for the *next* model generation rather than this one, following the staged-confidence-building pattern in `006_Pretraining_Ablations_And_Research_Methodology.md` Section 4.3. This treats the 9-month deadline as the binding constraint it is, without abandoning the MoE thesis — it defers the MoE commitment to a cycle where the infrastructure risk has actually been retired.

---

## Q11: You know MLA gives better KV-cache compression than GQA at a comparable quality bar. When would you deliberately choose GQA over MLA anyway?

Several concrete situations, and being able to name them (rather than just asserting "MLA is better, always use it") is the actual signal here. First, if the target context length and expected serving concurrency don't push KV-cache memory into being the binding serving-cost constraint in the first place — a moderate context length product at modest concurrency may simply never hit the regime where MLA's extra compression matters, in which case adopting it buys little while still paying its full implementation-risk cost (`003_Model_Architecture_Decisions_At_Pretraining_Time.md`, Section 2.1). Second, if my organization's training and serving infrastructure is built around standard MHA/GQA kernels and tooling (widely-supported in essentially every training and inference framework) and has no prior experience with a learned low-rank latent-compression attention mechanism, adopting MLA means taking on real implementation and validation risk — MLA is a more involved mechanism (a down-projection/up-projection structure with a decoupled-RoPE component, per `..\..\OpenSource\006_DeepSeek_V2.md`) that DeepSeek itself iterated on and validated across two model generations before it was proven at frontier scale; a first-time adopter is not starting from that same validated position. Third, serving-stack lock-in: a novel attention mechanism requires every downstream serving tool (KV-cache management, continuous-batching schedulers, speculative decoding) to be adapted to its specific cache layout, rather than reusing the mature ecosystem built around standard MHA/GQA — a real switching cost independent of MLA's technical merit. Fourth, and this connects directly to `006_Pretraining_Ablations_And_Research_Methodology.md`: MLA's claimed quality-vs-compression tradeoff was validated on DeepSeek's specific architecture, data mixture, and scale — whether that tradeoff holds for a *different* lab's model is exactly the kind of scale/setting-transfer claim that needs its own proxy-scale validation before being trusted, not assumed from a published number on a different system. Given all of that, I'd choose GQA whenever the KV-cache-cost pressure doesn't clearly justify absorbing MLA's implementation risk, and treat "our context length and concurrency targets make cache cost the dominant serving-cost driver, and we have the infrastructure maturity or runway to validate a new mechanism properly" as the actual bar for choosing MLA instead.

---

## Q12: A product team asks for a model that natively supports 1M-token context "from day one of pretraining." Walk through your response and what you'd actually plan instead.

I'd push back on "natively from day one" specifically, and explain the cost argument precisely rather than just saying no. Attention's compute (and, without context-parallel sharding, activation memory) scales quadratically in sequence length in the dominant \(O(n^2d)\) term; training the *entire* multi-trillion-token main pretraining run at 1M-token context rather than a much shorter native length would multiply the cost of that dominant compute stage across the whole run — a cost paid on every one of the trillions of tokens processed, which is a different and much larger commitment than paying it on the comparatively tiny number of tokens used in a dedicated later stage (`005_Curriculum_And_Multi_Stage_Pretraining.md`, Section 2). This is exactly why essentially no frontier model is pretrained end-to-end at its eventual maximum context length — DeepSeek-V2 pretrained at 4K and extended to 128K via YaRN, Llama 3 pretrained at 8K and extended to 128K in the 3.1 release, both via a separate, much cheaper stage (DeepSeek-V3's disclosed compute breakdown puts context extension at roughly 4-5% of main-pretraining compute — `..\..\OpenSource\007_DeepSeek_V3.md`, Section 3).

What I'd actually plan: pretrain at a short-to-moderate native context length (the same choice this module treats as standard, e.g., 4-8K) for the main run, then run a dedicated extension stage using a NTK-aware/YaRN-style RoPE frequency rescaling (`005_Curriculum_And_Multi_Stage_Pretraining.md`, Section 3.2) trained on genuinely long documents (plus likely some synthetically constructed long-context tasks) to reach 1M tokens. I'd flag two real risks specific to going as far as 1M rather than the more common 128K: first, the extension stage's data distribution (long documents, long-context synthetic tasks) is narrower than main pretraining's, and there's no guarantee the model's ability to *use* information anywhere in a 1M-token window is as uniform as its ability to use information in its native context — this has to be validated with needle-in-a-haystack-style and long-document benchmarks specifically at the long end of that range, not assumed from architectural support for the length; second, even with cache-efficient attention (GQA/MLA), serving at 1M context is a substantial KV-cache-memory commitment in its own right, and the concurrency/latency implications of that should be sized and budgeted before, not after, the extension stage locks in the target.

---

## Q13: Implement a simplified YaRN-style RoPE frequency rescaling function: given the original and target context lengths and a set of RoPE frequencies, apply a ramped interpolation that compresses low-frequency components more aggressively than high-frequency ones.

```python
import numpy as np

def rope_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """Standard RoPE frequency schedule: dim/2 frequencies, geometrically spaced."""
    i = np.arange(0, dim, 2)
    return 1.0 / (base ** (i / dim))

def yarn_rescale_frequencies(freqs: np.ndarray, orig_max_len: int, target_max_len: int,
                               beta_fast: float = 32.0, beta_slow: float = 1.0) -> np.ndarray:
    """Simplified YaRN-style rescaling. Low-frequency components (long wavelength,
    small freq value) are interpolated toward the scale factor s = target/orig
    (full compression); high-frequency components are left near their original
    scale (extrapolated, not compressed), with a ramp in between based on how
    many original-context rotations each frequency completes.

    beta_fast / beta_slow define the wavelength-based ramp boundaries in terms of
    number of rotations completed within orig_max_len, following YaRN's own framing.
    """
    scale = target_max_len / orig_max_len
    wavelengths = 2 * np.pi / freqs
    # number of full rotations this frequency completes within the original context
    n_rotations = orig_max_len / wavelengths

    # ramp: 0 => fully "extrapolate" (leave frequency unscaled, high-freq regime),
    #       1 => fully "interpolate" (divide frequency by scale, low-freq regime)
    ramp = np.clip((n_rotations - beta_slow) / (beta_fast - beta_slow), 0.0, 1.0)

    interpolated = freqs / scale
    return ramp * interpolated + (1 - ramp) * freqs
```

The key correctness properties an interviewer would check: (1) the interpolation factor is exactly `1/scale` where `scale = target_max_len / orig_max_len` — this is what makes the *lowest*-frequency component's effective wavelength, at the new maximum context length, map back into roughly the same rotation-angle range it saw during original pretraining; (2) the ramp is a function of **how many rotations each frequency actually completes within the original context**, not a linear function of frequency index — this matches YaRN's actual framing (frequencies that already complete many rotations within the original context are "high-frequency" and are safe to leave extrapolated; frequencies that complete less than one rotation are the ones that were badly under-trained and need full interpolation); (3) the two extremes (`ramp=0`, `ramp=1`) recover, respectively, "leave this frequency alone" and "fully compress this frequency by the context-length scale factor" — the mechanism is genuinely a smooth blend between those two regimes, not an all-or-nothing cutoff.

---

## Q14: What is a prefix-LM attention mask, and how does it relate to (but differ from) the attention pattern used in GLM's blank infilling?

A prefix-LM mask splits a sequence into a prefix \(x_1..x_k\) and a continuation \(x_{k+1}..x_T\): the prefix gets full bidirectional self-attention (every prefix position attends to every other prefix position, forward and backward, exactly as in BERT), the continuation gets ordinary causal attention (position \(i>k\) attends to \(1..i\)), and continuation positions may additionally attend back into the prefix, while prefix positions may never attend forward into the continuation. The loss is computed only over the continuation. Mechanically it's a strict generalization of causal LM's mask — causal LM is the degenerate case \(k=0\) (no bidirectional prefix at all) — that lets the "given" portion of an input get richer self-attention while still preserving a clean, purely-causal generation story for everything that has to actually be produced.

GLM's blank-infilling attention pattern (`..\..\OpenSource\010_GLM4.md`, Section 2; `001_Pretraining_Objectives_Overview.md`, Section 4) is a **specific instance** of exactly this general prefix-LM pattern: GLM's Part A (the surviving, uncorrupted context) plays the role of the bidirectional prefix, and Part B (the autoregressively-generated masked-span content) plays the role of the causal continuation, with the same "continuation may look back at the prefix, prefix may never look forward into the continuation" rule. What's specific to GLM on top of the general prefix-LM pattern: Part A isn't a contiguous leading block of the original sequence the way a prefix-LM's prefix typically is — it's the *complement* of several scattered, corrupted spans, stitched back together in original order with placeholders; and Part B's content is drawn from multiple, potentially-permuted spans rather than a single contiguous continuation, which is exactly why GLM additionally needs the two-dimensional positional encoding (position-id-1 for "which original slot," position-id-2 for "how far into this span's generation") that a simple, single-contiguous-block prefix-LM doesn't need at all — a prefix-LM's continuation positions can just use ordinary sequential position indices, since there's no span permutation to disambiguate.

---

## Q15: You're told "we fit a clean scaling law from our proxy runs and it looks great." Name three concrete, distinct ways this could still fail to predict the target-scale run's actual loss, and be specific about the mechanism for each.

First, **genuine regime change in the exponents themselves**: the fitted power-law exponent is an empirical property of the specific architecture/data/optimizer setup tested at the proxy scale, not a universal constant — there's no guarantee the same exponent describes the loss-vs-scale relationship several orders of magnitude further out, and this is precisely the kind of correction Chinchilla represented relative to Kaplan et al. one level down; a future correction at a scale beyond what either paper tested is entirely possible in principle (`002_Scaling_Laws_And_Compute_Optimal_Training.md`, Section 5.2).

Second, **data availability or quality becoming the binding constraint before compute does**: the fit implicitly assumes the corpus's marginal token at target-scale token counts is roughly as informative as the tokens the proxy runs were trained on. At multi-trillion-token target budgets, the highest-quality easily-available data may be exhausted well before the fitted \(D^*\) is reached, forcing either heavier repetition of a smaller high-quality pool (which has its own separately-studied diminishing-returns behavior) or dilution with lower-quality tokens the fit never accounted for — the loss-vs-\(D\) curve measured on the proxy runs' (likely cleaner, more curated, or simply differently-composed) data doesn't automatically describe what happens once the corpus composition itself has to change to hit the target token count.

Third, **hyperparameter and infrastructure interactions invisible at proxy scale**: optimizer settings, batch size, and numerical-precision choices that are stable and near-optimal at the proxy scale are not guaranteed to remain so at target scale — the critical-batch-size and LR-scaling-rule considerations in `004_Optimizers_LR_Schedules_And_Hyperparameters.md` Section 4 mean a target run can underperform its fitted scaling-law prediction not because the loss-vs-compute law itself was wrong, but because the hyperparameters chosen for the target run (extrapolated separately, often via a different heuristic like linear or sqrt batch-size scaling) were themselves mis-extrapolated, confounding the scaling-law prediction's accuracy with a completely separate hyperparameter-transfer failure.

---

## Q16: Here is a learning-rate schedule implementation a teammate wrote. It's producing a sudden loss spike right at the end of the warmup period. Find the bug.

```python
import math

def buggy_lr_schedule(step, peak_lr, warmup_steps, total_steps, min_lr_ratio=0.1):
    if step <= warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    min_lr = peak_lr * min_lr_ratio
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

The bug is at the warmup/decay boundary, `step == warmup_steps`. In the warmup branch, `step <= warmup_steps` includes `step == warmup_steps`, giving `peak_lr * warmup_steps / warmup_steps = peak_lr` at that exact step — so far so good, that's the intended peak. But because the condition is `<=` rather than `<`, the *next* step, `step == warmup_steps + 1`, falls into the cosine branch with `progress = 1/(total_steps - warmup_steps)`, a small positive number — that's also fine and continuous. The actual discontinuity is one step earlier than it looks: many training loops call the schedule function at `step = warmup_steps` from *both* contexts inconsistently (e.g., an off-by-one in how "step" is counted — 0-indexed vs. 1-indexed step counters, or whether the scheduler is queried before or after the optimizer step increments its counter), which is the single most common real-world source of exactly this symptom. But even taking the code at face value, there's a subtler and more concrete bug: at `step = warmup_steps` exactly, the function returns `peak_lr` (correct), but if `step` is ever passed as `warmup_steps` a second time due to a caller incrementing its counter *after* calling the scheduler rather than before (a common off-by-one), the model can receive `peak_lr` for two consecutive optimizer steps in a row right as warmup ends, followed by a sudden drop into the cosine curve's `progress≈0` value — which is numerically fine (\(\cos(0)=1\), so it evaluates to `peak_lr` again) — so the *true* bug is elsewhere: `warmup_steps` itself, if `step` starts at 0, means the model only receives `warmup_steps` distinct increasing LR values (0 through `warmup_steps-1` steps of ramp, `step=warmup_steps` already at peak) rather than the intended smooth ramp over the full warmup window, i.e., the ramp is effectively one step shorter than configured — a subtle but real off-by-one that compresses the last part of the intended warmup ramp into a single step, producing an unexpectedly large single-step LR jump right at the boundary rather than a smooth approach to peak. The fix: make the boundary condition and step-indexing convention explicit and consistent, e.g.:

```python
def fixed_lr_schedule(step, peak_lr, warmup_steps, total_steps, min_lr_ratio=0.1):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps   # step 0 -> 1/warmup_steps, not 0
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

using `step < warmup_steps` with `(step + 1)` in the numerator gives a clean, evenly-spaced ramp from `peak_lr/warmup_steps` up to exactly `peak_lr` at `step = warmup_steps - 1`, and the cosine branch then starts cleanly at `step = warmup_steps` with `progress = 0` (i.e., `cos(0) = 1`, giving exactly `peak_lr`) — a single, unambiguous handoff point with no double-counted or skipped step, and the added `min(1.0, ...)` guards against the schedule being queried past `total_steps` (e.g., during an extended-training decision, `002_...md` Section 6) from silently extrapolating the cosine argument past \(\pi\).

---

## Q17: Your model is projected to be queried 50 billion times over its production lifetime, averaging 500 output tokens per query. You have two candidate configurations reaching the same downstream quality bar: (A) 13B params trained on 10T tokens, (B) 70B params trained on 1.5T tokens (roughly Chinchilla-optimal for 70B). Which do you ship, and show the arithmetic.

Using the \(C\approx6ND\) training-FLOPs approximation and \(\approx2N\) inference-FLOPs-per-generated-token approximation (both from Q5):

Training FLOPs: A: \(6 \times 13\text{e}9 \times 10\text{e}12 = 7.8\text{e}23\). B: \(6 \times 70\text{e}9 \times 1.5\text{e}12 = 6.3\text{e}23\). B is cheaper to train, as expected for a near-Chinchilla-optimal configuration.

Inference FLOPs over the projected lifetime: total generated tokens = \(50\text{e}9 \times 500 = 2.5\text{e}13\) tokens. A: \(2 \times 13\text{e}9 \times 2.5\text{e}13 = 6.5\text{e}23\). B: \(2 \times 70\text{e}9 \times 2.5\text{e}13 = 3.5\text{e}24\).

Total (train + inference) FLOPs: A: \(7.8\text{e}23 + 6.5\text{e}23 = 1.43\text{e}24\). B: \(6.3\text{e}23 + 3.5\text{e}24 = 4.13\text{e}24\).

Configuration A — the smaller, deliberately overtrained model — has roughly one-third the total lifetime compute cost of B despite costing more to train, because B's inference cost (driven by its larger \(N\), recurring across all \(2.5\times10^{13}\) generated tokens) dominates its total far more than A's training-token excess costs A. This is exactly the Llama-3-8B-style argument from Q9 made quantitative: at this query volume, the inference term dominates the total-cost equation for B by more than 5x over its own training cost, while A's training cost and inference cost are much closer to balanced — meaning A is closer to being total-cost-optimal for *this specific deployment volume*, even though B is the one that's close to training-compute-optimal. I would ship A, and would flag that this conclusion is volume-dependent: at a much lower projected query volume, the crossover point moves and B's cheaper, near-Chinchilla training cost could dominate the comparison instead — which is exactly why this decision has to be re-derived per expected deployment volume, not read off a fixed rule.

---

## Q18: State, precisely and with an equation, the difference between "training-compute-optimal" and "total-cost-of-ownership-optimal." Why are these not the same optimization problem?

Training-compute-optimal answers: given a fixed training-compute budget \(C\), with \(C = 6ND\), choose \((N,D)\) to minimize training loss \(L(N,D)\). It is a statement purely about what happens *during* training, and terminates the moment training loss is measured — it has no term for anything that happens after the model is deployed.

Total-cost-of-ownership-optimal answers a different question: given a target quality bar and a projected deployment volume, minimize \(\text{Cost}_{\text{train}}(N,D) + n_{\text{queries}} \times \text{Cost}_{\text{inference-per-query}}(N)\), where \(\text{Cost}_{\text{train}} \propto 6ND\) and \(\text{Cost}_{\text{inference-per-query}} \propto 2N \times (\text{tokens per query})\) (from Q5). These are not the same optimization problem because the two cost terms scale with different quantities and are paid on different timelines: training cost is a function of *both* \(N\) and \(D\) and is paid exactly once; inference cost is a function of \(N\) **alone** (not \(D\) at all — the number of tokens a model was trained on has no bearing on its per-query serving cost) and is paid \(n_{\text{queries}}\) times, recurring for the model's entire deployment lifetime. Because \(D\) appears in one term and not the other, and because \(n_{\text{queries}}\) is a free multiplier on the term that doesn't depend on \(D\), the \((N,D)\) pair that minimizes the first (training-only) objective is generally *not* the pair that minimizes the second (total-cost) objective once \(n_{\text{queries}}\) is large enough — the total-cost-optimal strategy will, in general, prefer a smaller \(N\) than training-compute-optimality would choose, paid for with a larger \(D\) than training-compute-optimality would choose, exactly because shrinking \(N\) linearly shrinks the recurring term while growing \(D\) only affects the one-time term. Q9 and Q17 work through this concretely for Llama 3 and a hypothetical deployment scenario, respectively.

---

## Q19: Why does T5's span corruption use a single sentinel token to replace an entire corrupted span, rather than masking every token in the span individually the way BERT masks each selected token?

Two separate reasons, and they're worth distinguishing rather than citing just one. First, compute efficiency at the level of *decoder sequence length*: T5 is an encoder-decoder model, and the decoder's job is to reproduce, autoregressively, the content of every corrupted span concatenated together — if the encoder-side input retained a `[MASK]` for every individual corrupted token (as BERT's input does), the corrupted-and-uncorrupted content collectively would still be as long as the original sequence, and more importantly the *target* the decoder generates would need some other mechanism to know where each mask's content begins and ends when spans have variable length. Collapsing an entire span to one sentinel shortens the encoder's input (fewer positions to encode) and gives the decoder an unambiguous, minimal-length target to generate (sentinel, then span content, then next sentinel, ...) — directly reducing the number of decode steps needed per unit of corrupted content, a real wall-clock/FLOPs saving verified empirically in T5's own ablations.

Second, and mechanistically more interesting: because sentinel-collapsed spans are generated **autoregressively** by the decoder (token \(i\) of a span's content conditions on token \(i-1\) of the *same* span, already emitted), span corruption gets the same within-span joint-coherence property that GLM's Part-B mechanism gets (Q2) — a property BERT's independent-per-masked-position prediction structurally lacks. If T5 instead masked every token individually BERT-style, predicting each one independently and simultaneously from the same fixed corrupted context, it would reintroduce exactly BERT's joint-incoherence problem within a span, defeating a real part of the point of moving to contiguous-span corruption over scattered single-token masking in the first place. The sentinel-plus-autoregressive-decoding design is what buys both the compute saving and the coherence property simultaneously, and it's why "single-token replaces the whole span" is the mechanically necessary choice given that the decoder generates each span's content sequentially, not an arbitrary notational convenience.

---

## Q20: Implement a function that computes the per-sequence KV-cache memory footprint for a transformer, parameterized so it can represent plain MHA, GQA, and MLA-style compression, and use it to quantify how much memory GQA and MLA save relative to MHA at a long context length.

```python
def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int,
                    seq_len: int, batch_size: int, dtype_bytes: int = 2,
                    mla_latent_dim: int | None = None) -> int:
    """Standard MHA/GQA cache: store K and V per KV head, per layer, per token.
    bytes = 2 (K and V) * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes

    MLA-style cache: instead of per-head K/V, store a single shared low-rank
    latent vector per token per layer (mla_latent_dim), reconstructed into
    per-head K/V at attention time rather than cached per head.
    bytes = n_layers * mla_latent_dim * seq_len * batch_size * dtype_bytes
    (roughly -- ignores the small decoupled-RoPE key component for simplicity;
    see ..\\..\\OpenSource\\006_DeepSeek_V2.md for the full accounting.)
    """
    if mla_latent_dim is not None:
        return n_layers * mla_latent_dim * seq_len * batch_size * dtype_bytes
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes


def compare_cache_footprints(n_layers=60, head_dim=128, seq_len=128_000, batch_size=1):
    mha = kv_cache_bytes(n_layers, n_kv_heads=40, head_dim=head_dim,
                         seq_len=seq_len, batch_size=batch_size)               # 40 heads, no sharing
    gqa = kv_cache_bytes(n_layers, n_kv_heads=8, head_dim=head_dim,
                         seq_len=seq_len, batch_size=batch_size)               # 5:1 grouping
    mla = kv_cache_bytes(n_layers, n_kv_heads=0, head_dim=head_dim,
                         seq_len=seq_len, batch_size=batch_size, mla_latent_dim=512)
    gb = lambda b: b / (1024 ** 3)
    return {"MHA_GB": gb(mha), "GQA_GB": gb(gqa), "MLA_GB": gb(mla)}
```

Running `compare_cache_footprints()` at 128K context, 60 layers, head_dim 128, bf16 (2 bytes): MHA (40 KV heads) gives \(2 \times 60 \times 40 \times 128 \times 128000 \times 2\) bytes \(\approx 313\) GB per sequence; GQA (8 KV heads, 5:1 grouping) gives exactly \(1/5\) of that, \(\approx 62.7\) GB; MLA (512-dim latent) gives \(60 \times 512 \times 128000 \times 2 \approx 7.9\) GB. This makes the design-decision framing in `003_Model_Architecture_Decisions_At_Pretraining_Time.md` Section 2.1 concrete rather than qualitative: GQA already buys a 5x reduction here for a near-zero-risk, universally-supported mechanism, while MLA's learned compression buys a further roughly 8x on top of that — but note the comparison is only this dramatic *because* the context length is 128K and batch size (concurrency) is part of the multiplier; at a short context length these absolute numbers (and therefore the case for paying MLA's implementation-risk cost, per Q11) shrink substantially, which is exactly why the decision has to be made relative to the target context length and expected serving concurrency, not as an abstract ranking of the three mechanisms.
