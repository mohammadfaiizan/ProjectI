# Mixture-of-Experts Architectures and Neural Scaling Laws

## Part 1: Mixture of Experts

### The core motivation: decoupling parameter count from compute cost

In a standard dense transformer, every single parameter in every feed-forward block is used to process every single token. If you want a more capable model, the traditional lever is to make the model bigger — wider layers, more layers, bigger feed-forward blocks — but every one of those added parameters is now paid for, in FLOPs, on every token the model ever processes, whether at training time or at inference time. This is a rigid coupling: total capacity (parameter count) and per-token compute cost are forced to move together.

Mixture-of-Experts (MoE) architectures break that coupling. The idea is to replace the single dense feed-forward network (FFN) inside a transformer block with a large collection of smaller FFNs, called **experts**, and to route each token through only a small subset of them — commonly just one or two — rather than through all of them. This means you can grow the *total* parameter count of the model dramatically (by adding more experts) without growing the compute cost of processing any individual token, because most of those parameters simply are not touched for most tokens. The model gains capacity and specialization headroom while keeping the per-token compute (and, importantly for the KV-cache discussion elsewhere in this folder, the attention cost) essentially unchanged. This is usually described as **sparse activation**: the model is sparse in the sense that only a small fraction of its weights participate in any single forward pass, even though the total weight count can be an order of magnitude larger than an equivalently-fast dense model.

It is worth being precise about the vocabulary here because it shows up constantly in papers and interviews: a model like Mixtral 8x7B has roughly 47B **total** parameters (the sum of all experts plus shared components) but only around 13B **active** parameters per token (because only 2 of its 8 experts are used per token, per layer). The "8x7B" naming is itself a little misleading — it is not literally 8 independent 7B models, because the attention layers and other non-expert components are shared, but the naming convention persists because it is a useful shorthand for "8 experts, each roughly 7B-model-scale."

This active-vs-total split is exactly what makes MoE an attractive inference-cost lever, and it's worth spelling out precisely what you get and what you still pay for. The FLOPs spent computing a forward pass, and therefore decoding latency and compute cost per token, track the **active** parameter count — Mixtral 8x7B costs roughly what a dense ~13B model costs per token to run, which is a large latency and throughput win relative to a dense 47B model that would need all 47B parameters engaged on every token. Benchmark quality, on the other hand, tracks much closer to the **total** parameter count, since the model has 47B parameters' worth of learned, specialized capacity to draw on across different tokens and contexts even though any single token only exercises a slice of it — Mixtral 8x7B benchmarks competitively with dense models several times larger than its active parameter count, which is the entire point of the architecture. The cost that MoE does *not* remove, and that is easy to overlook, is memory: every expert's weights have to be resident somewhere accessible (in GPU VRAM, or sharded across several GPUs) regardless of whether a given token happens to route to it, because you don't know in advance which experts a batch of incoming tokens will need. So a Mixtral 8x7B deployment needs enough aggregate memory to hold all 47B parameters, even though it only pays the compute cost of about 13B of them per token — MoE trades compute for memory, it does not eliminate cost, and this is precisely why MoE deployments tend to be more memory-bandwidth- and VRAM-capacity-constrained than compute-constrained, which shapes decisions like how many GPUs to shard a model across and how aggressively to quantize expert weights to fit them in memory.

### Anatomy of an MoE transformer block

A typical MoE transformer block looks exactly like a dense transformer block, except that the single FFN sub-layer is replaced by two components: a small **router** (also called a gating network) and a bank of `N` expert FFNs, each structurally identical to what would have been the single dense FFN. The router looks at each token's hidden state and decides which expert(s) that token should be sent to. Concretely, the router is usually just a single learned linear layer mapping the hidden state to a vector of `N` logits (one per expert), followed by a softmax to turn those logits into a probability distribution over experts, and then a **top-k** selection that keeps only the `k` highest-probability experts and discards the rest. In production models, `k` is almost always 1 or 2 — Switch Transformer popularized top-1 routing specifically to minimize compute and communication overhead, while GShard, Mixtral, and DeepSeek-MoE use top-2 (or, in DeepSeek-V3's fine-grained scheme, top-k over a much larger pool of smaller experts) to give the model more redundancy and smoother gradients than top-1 routing allows.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A single expert is just a normal transformer FFN block."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        # x: (batch * seq_len, d_model) -- tokens flattened for routing
        logits = self.router(x)                          # (tokens, num_experts)
        probs = F.softmax(logits, dim=-1)                 # gating distribution
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)   # (tokens, top_k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize

        output = torch.zeros_like(x)
        for slot in range(self.top_k):
            expert_ids = topk_idx[:, slot]
            weight = topk_probs[:, slot].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = expert_ids == e
                if mask.any():
                    output[mask] += weight[mask] * self.experts[e](x[mask])

        return output, probs   # probs returned for the load-balancing loss below
```

This reference implementation loops over experts for clarity; real production implementations instead sort/group tokens by assigned expert and dispatch them as batched matrix multiplies (often across devices, since experts are typically sharded one-or-few-per-GPU), which is where most of the systems-engineering complexity of MoE training actually lives — the all-to-all communication needed to route tokens to the GPU holding their assigned expert, and back, is frequently the real bottleneck in large-scale MoE training rather than the FLOPs themselves.

### The load-balancing problem: "rich get richer" collapse

If you train the router above with nothing but the ordinary language-modeling loss, it tends to collapse in an unhelpful way. Early in training, random initialization means some experts get, by chance, slightly more traffic than others. Those experts then receive more gradient signal and become slightly better at whatever they've seen, which makes the router slightly more likely to route to them again, which gives them even more traffic and even more gradient signal. This is a classic rich-get-richer feedback loop, and left unchecked it leads to a small number of experts absorbing nearly all the tokens while the rest are starved of training signal and remain undertrained, useless dead weight. This is bad for two independent reasons: it wastes the very capacity you built MoE to exploit (an undertrained expert contributes nothing), and it creates a severe practical load-imbalance at the systems level, since experts are usually sharded across different accelerators — if one expert's GPU gets a hugely disproportionate share of tokens, that GPU becomes a straggler that the whole training step has to wait on, or excess tokens simply get dropped (more on that below).

The standard fix is an **auxiliary load-balancing loss**, added to the main training objective, whose purpose is purely to encourage the router to spread tokens roughly evenly across experts. The formulation introduced in the Switch Transformer paper (and used in essentially all subsequent MoE work in some form) works like this: for a batch of `T` tokens routed among `N` experts, define `f_i` as the fraction of tokens actually dispatched to expert `i` (a hard, discrete count), and `P_i` as the average router probability mass assigned to expert `i` across the batch (a soft, differentiable quantity). The auxiliary loss is proportional to the dot product of these two vectors, scaled by `N`:

```
L_aux = alpha * N * sum_i ( f_i * P_i )
```

The intuition behind this specific form is that it is minimized when both `f_i` and `P_i` are close to uniform (`1/N` each) — if this loss were zero it would mean every expert receives exactly `T/N` tokens and exactly `1/N` average routing probability. Because `f_i` is computed from a hard top-k decision (non-differentiable), it acts as a stop-gradient signal, while `P_i` carries the actual gradient back into the router's weights; multiplying them together means the loss actively penalizes exactly the failure mode described above — an expert that is receiving a disproportionate share of tokens (`f_i` large) *and* whose average routing probability is also large (`P_i` large) contributes a large penalty, pushing the router to spread its probability mass elsewhere. `alpha` is a small weighting hyperparameter (commonly around 0.01) chosen so the auxiliary loss nudges the router toward balance without meaningfully distorting the primary language-modeling objective.

```python
def load_balancing_loss(router_probs, topk_idx, num_experts):
    """
    router_probs: (tokens, num_experts) -- full softmax distribution from the router
    topk_idx:     (tokens, top_k)       -- which experts were actually selected
    """
    tokens = router_probs.shape[0]

    # f_i: fraction of tokens routed to each expert (hard assignment count)
    one_hot = F.one_hot(topk_idx, num_classes=num_experts).float()  # (tokens, top_k, N)
    tokens_per_expert = one_hot.sum(dim=(0, 1))                     # (N,)
    f = tokens_per_expert / tokens_per_expert.sum()

    # P_i: average router probability assigned to each expert (soft, differentiable)
    P = router_probs.mean(dim=0)                                    # (N,)

    return num_experts * torch.sum(f * P)
```

Beyond the auxiliary loss, real systems also enforce a hard **expert capacity** limit: since GPUs need statically-shaped tensors for efficient batched computation, each expert is typically given a fixed capacity (e.g., `capacity = (tokens_per_batch / num_experts) * capacity_factor`, with `capacity_factor` often around 1.25) representing the maximum number of tokens it will accept in a given batch. If the router sends more tokens to an expert than its capacity allows, the overflow tokens are **dropped** for that layer — meaning they skip the expert computation entirely (often passed through via a residual/skip connection instead) rather than being forcibly squeezed in. Token dropping is a deliberate, accepted trade-off: it bounds worst-case compute and memory per expert (essential for efficient hardware utilization) at the cost of occasionally giving some tokens a degraded, expert-free pass through that layer. Combined with a well-tuned load-balancing loss, in practice the drop rate is kept low enough that overall model quality is barely affected, but it is a real characteristic of how MoE training/inference actually behaves that is easy to forget if you have only read about MoE at the conceptual level.

### Expert-choice routing: flipping who does the choosing

Everything described so far is **token-choice routing**: each token independently computes a distribution over experts and picks its own top-k, which is exactly the setup that causes the rich-get-richer imbalance, since nothing stops many tokens from all preferring the same expert in the same batch. **Expert-choice routing** (Zhou et al., "Mixture-of-Experts with Expert Choice Routing") inverts this relationship: instead of each token choosing its experts, each expert independently chooses which tokens it wants to process, selecting its top-`c` tokens (by the same router-logit scores) up to a fixed capacity `c`, out of all the tokens in the batch. Because every expert independently fills its own fixed-size quota, perfect load balance is guaranteed by construction — there is no way for one expert to end up starving or overloaded, since the capacity per expert is a hard, symmetric constraint rather than an emergent property of many independent token-level decisions. This eliminates the need for an auxiliary load-balancing loss entirely, which removes one hyperparameter (`alpha`) and one source of tension with the primary training objective. The trade-off is a different one: because experts pick tokens rather than tokens picking experts, a given token is no longer guaranteed to be processed by any expert at all if it isn't in any expert's top-`c` (some tokens can be dropped entirely under heavy contention), and a token can, in principle, be picked up by a variable number of experts rather than a clean, fixed top-k per token — which changes the shape of the routing guarantees a system has to reason about, even though it solves the balance problem more directly than an auxiliary loss does. In practice, token-choice routing with an auxiliary loss (or DeepSeek-V3's bias-based alternative below) remains the more common choice in the largest production frontier models, but expert-choice is an important alternative point in the design space to be able to name and explain, since it reframes load balancing as a routing-direction choice rather than purely a loss-function patch.

### DeepSeek's refinements: fine-grained and shared experts, auxiliary-loss-free balancing

DeepSeek's MoE work (DeepSeek-MoE, then carried into DeepSeek-V2 and DeepSeek-V3) introduced two ideas worth knowing specifically because they represent a meaningful departure from the Mixtral-style "handful of large experts" design. The first is **fine-grained expert segmentation**: instead of a small number of large experts (e.g. 8, as in Mixtral), DeepSeek splits the same total FFN capacity into a much larger number of much smaller experts (DeepSeek-V3 uses 256 routed experts per layer, activating 8 per token) and correspondingly increases `k`. The argument is that this gives the router a finer-grained combinatorial space to compose from — instead of picking 2 out of 8 "generalist" experts, the model can pick 8 out of 256 much more specialized experts, allowing sharper specialization and more flexible combinations per token, which improved benchmark quality at matched compute in DeepSeek's ablations.

The second idea is **shared experts**: a small number of experts (DeepSeek-V3 uses 1) are not routed at all — every single token passes through them unconditionally, in addition to whichever routed experts it is assigned. The intuition is that some knowledge is genuinely common across essentially all tokens (basic syntax, very common patterns), and forcing that knowledge to be redundantly re-learned inside every routed expert wastes capacity; a shared expert lets the model factor out that common component once, freeing the routed experts to specialize on the genuinely token-specific residual.

The third, and most cited, refinement is DeepSeek-V3's **auxiliary-loss-free load balancing**. Rather than relying on the `L_aux` term above (which, being an additional loss term, always creates some tension with the primary language-modeling objective and requires tuning `alpha` to avoid distorting training), DeepSeek-V3 instead adds a learned per-expert bias term directly to the routing logits before the top-k selection. This bias is not trained by gradient descent at all; instead, after each training step (or batch of steps), the bias for each expert is nudged up if that expert received fewer tokens than the target balanced share, and nudged down if it received more, purely via a simple rule-based update, entirely outside the gradient computation. This achieves the same load-balancing goal — steering routing decisions toward balance — without ever injecting a competing gradient signal into the main loss, which DeepSeek reports removes a source of quality degradation that a poorly-tuned `alpha` can otherwise introduce, while still preventing the collapse behavior described earlier.

## Part 2: Scaling laws

### Kaplan et al. (2020): the original scaling laws

Once you accept that we mostly improve LLMs by throwing more parameters, more data, and more compute at them, the natural engineering question becomes: given a fixed compute budget, how should you split it between making the model bigger versus training it on more data? OpenAI's 2020 paper "Scaling Laws for Neural Language Models" (Kaplan et al.) was the first rigorous empirical answer. By training a large number of transformer models across a wide range of sizes and dataset sizes, the authors found that test loss follows remarkably clean **power-law** relationships with each of model size (`N`, number of non-embedding parameters), dataset size (`D`, number of training tokens), and compute (`C`), each considered while holding the others sufficiently large as not to be the bottleneck:

```
L(N) ≈ (N_c / N) ^ alpha_N
L(D) ≈ (D_c / D) ^ alpha_D
L(C) ≈ (C_c / C) ^ alpha_C
```

where `L` is the cross-entropy loss, and `N_c`, `D_c`, `C_c`, and the alpha exponents are empirically fit constants. The practically consequential conclusion the paper drew from fitting these curves was that, for a fixed compute budget, loss improves fastest by scaling up model size much more aggressively than dataset size — the paper's guidance implied you should grow `N` substantially faster than `D` as compute grows, and, somewhat counterintuitively, that it was often fine (compute-optimal, even) to stop training a large model well before it had converged on its training data, rather than train a smaller model to convergence. This conclusion directly shaped the first generation of very large models, most notably GPT-3 (175B parameters, trained on roughly 300B tokens) — a ratio of only about 1.7 tokens per parameter, which in hindsight (see below) was far too data-light relative to its parameter count.

### Chinchilla (Hoffmann et al., 2022): the correction

DeepMind's 2022 paper "Training Compute-Optimal Large Language Models" (Hoffmann et al.), which produced the model Chinchilla, revisited the Kaplan methodology with a more careful and much larger set of training runs and a critical methodological fix: the earlier analysis had not adequately accounted for learning-rate schedule effects (models trained with a schedule tuned for a different token budget than they actually used gave misleading loss curves), which had biased the original conclusion toward oversized models. Correcting for this, Hoffmann et al. found that compute-optimal training actually implies scaling model size and training data **in roughly equal proportion** as compute grows — not the aggressive size-over-data skew Kaplan's numbers suggested. The commonly cited practical takeaway is a target of roughly **20 tokens of training data per model parameter** for compute-optimal training (the actual Chinchilla model was 70B parameters trained on 1.4 trillion tokens, a ratio of exactly 20).

The demonstration model, Chinchilla, was trained with the same compute budget as DeepMind's earlier 280B-parameter Gopher model but was only 70B parameters, trained on proportionally more data — and it outperformed Gopher (and GPT-3) on a broad range of downstream benchmarks despite being 4x smaller, simply because it was allocated a compute-optimal split of the same total FLOP budget. This is the empirical result that made the paper so influential: it directly demonstrated that GPT-3, Gopher, MT-NLG, and other large contemporaneous models had been trained with a badly compute-suboptimal split — they were, in the terminology that stuck, **significantly undertrained relative to their size**, meaning that for the same compute budget spent training them, a smaller model trained on proportionally more data would have achieved lower loss.

```python
# Rough illustration of the Chinchilla compute-optimal allocation logic.
# (Toy numbers approximating the fitted relationship; real fits use the paper's
# exact coefficients, but the shape of the reasoning is what matters here.)

def chinchilla_optimal_split(compute_budget_flops):
    """
    Given a fixed training compute budget C (approximately C ~= 6 * N * D
    for a transformer trained with the standard forward+backward FLOP estimate),
    Chinchilla found the loss-minimizing split has N and D scale with roughly
    the same exponent in C -- i.e., grow model size and data together.
    """
    # C ~= 6 * N * D  =>  under the Chinchilla finding N* and D* both scale as C^0.5
    N_optimal = (compute_budget_flops / 6) ** 0.5
    D_optimal = (compute_budget_flops / 6) ** 0.5
    return N_optimal, D_optimal

N_opt, D_opt = chinchilla_optimal_split(compute_budget_flops=1e23)
print(f"N ~= {N_opt:.3e} params, D ~= {D_opt:.3e} tokens, ratio D/N ~= {D_opt/N_opt:.1f}")
```

### Why these laws matter for real engineering decisions

Scaling laws are not just a historical curiosity about two competing papers; they are one of the few tools that let you make a training-budget decision *before* spending millions of dollars on a run, because they let you extrapolate loss as a function of `N` and `D` from small, cheap pilot runs, and then answer questions like "for a fixed compute budget, would we get a better model by making it bigger or by feeding it more data?" without actually having to try both at full scale. This is precisely the reasoning that shaped the current generation of open-weight models: Llama 3's decision to train even its smallest 8B model on roughly 15 trillion tokens — a tokens-per-parameter ratio of well over 1000, vastly beyond the Chinchilla-optimal ~20 — is a deliberate departure from pure training-compute optimality, and it is a good example of why "compute-optimal" and "optimal" are not synonyms.

This departure is explained by a variant of the scaling-law question that accounts for **inference cost**, not just training cost. Chinchilla's notion of "optimal" only minimizes loss for a fixed *training* compute budget; it says nothing about the fact that a model, once trained, may be served billions of times over its deployment lifetime, and that inference cost scales with the model's parameter count on every single one of those calls. A smaller model that is deliberately "overtrained" well past the Chinchilla-optimal token count for its size can end up at a given loss/quality level while being dramatically cheaper to serve than a larger, Chinchilla-optimal model at that same loss level — and if you expect to serve the model at high volume for a long time, that lifetime inference-cost saving can vastly outweigh the extra up-front training compute spent overtraining it. This is exactly the calculus Meta made public reasoning about for Llama 3: it is cheaper, considering training plus a realistic amount of inference serving, to over-invest in training tokens for a smaller model than to train a larger, more "training-compute-optimal" one. This inference-aware framing is sometimes described informally as extending Chinchilla with a serving-cost term, and it is the concrete, practical reason why a well-informed 2024-era engineering team will choose model size independently from what pure Chinchilla compute-optimality would suggest, and why smaller, heavily-overtrained models like Llama 3 8B can outperform older, larger, undertrained models like the original GPT-3 175B on many benchmarks despite the parameter-count gap running in the "wrong" direction.

The broader lesson to carry out of both scaling-law results is methodological as much as numerical: they exist to let you reason quantitatively about compute allocation trade-offs (model size vs. data vs. inference volume) instead of guessing, and any time an interviewer asks "would you rather have a 2x bigger model or 2x more data," the correct answer is not a number pulled from memory but the reasoning process — fit a small-scale power law, account for your realistic inference volume, and solve for the allocation that minimizes total cost (training plus serving) at your target quality bar, not just training loss in isolation.
