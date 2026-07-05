"""
DeepSeek-V3 (2024) -- fine-grained MoE with auxiliary-loss-free, bias-based
load balancing.

This file demonstrates the mechanism that distinguishes V3's MoE router from
V2's (and from the standard Switch-Transformer-style MoE literature): instead
of adding an auxiliary load-balancing LOSS term to the training objective
(which competes with the LM loss and must be weighted carefully), V3 maintains
a per-expert BIAS term that is added to routing scores purely for the top-k
selection decision, and updates that bias after every step via a simple
feedback rule based on observed load -- overloaded experts get their bias
decremented (steered away from), underloaded experts get it incremented
(steered toward). No gradient ever flows through this bias update.

Also included: a minimal fine-grained MoE layer (many small routed experts +
one always-on shared expert, matching V3's N_r=256 / N_s=1 / top_k=8 design,
scaled down here for a runnable demo) and a toy Multi-Token Prediction (MTP)
head, since MTP is V3's other headline training-time contribution.

MLA (attention) is unchanged from DeepSeek-V2 -- see 006_DeepSeek_V2.py for
that implementation; this file focuses on what's NEW in V3.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fine-grained MoE with bias-adjusted (auxiliary-loss-free) load balancing
# ---------------------------------------------------------------------------

@dataclass
class MoEV3Config:
    d_model: int = 256
    d_ff_expert: int = 64        # small per-expert width (fine-grained); V3 uses a similarly narrow expert width relative to d_model
    n_routed_experts: int = 32   # V3 uses 256
    n_shared_experts: int = 1    # V3 uses 1 (down from V2's 2)
    top_k: int = 4               # V3 uses 8
    bias_update_speed: float = 0.01  # gamma: per-step bias adjustment magnitude


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))  # SwiGLU


class AuxLossFreeMoE(nn.Module):
    """
    Fine-grained routed-expert MoE + always-on shared expert, with V3-style
    bias-based load balancing instead of an auxiliary loss.

    Mechanism:
      1. Router produces raw affinity scores s_i(x) for each routed expert i
         (sigmoid-gated per DeepSeek-V3; a softmax-normalized sigmoid gate is
         used here for simplicity).
      2. Top-k selection uses BIASED scores: g_i = s_i(x) + b_i. The bias b_i
         is a per-expert scalar buffer, NOT a learned parameter -- it never
         receives a gradient.
      3. The *weight* applied to expert i's output uses the UNBIASED score
         (renormalized over the selected top-k), so the bias only steers WHICH
         experts are chosen, not how strongly a chosen expert's output counts.
      4. After the forward pass, `update_bias()` is called by the training
         loop: for each expert, compare its observed load (fraction of tokens
         routed to it in this step/batch) against the balanced target
         (top_k / n_routed_experts). Overloaded -> bias -= gamma.
         Underloaded -> bias += gamma. This is a plain feedback-control
         update, not a gradient step, and requires no backward pass.
    """

    def __init__(self, cfg: MoEV3Config):
        super().__init__()
        self.cfg = cfg
        self.shared_experts = nn.ModuleList(
            [Expert(cfg.d_model, cfg.d_ff_expert) for _ in range(cfg.n_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [Expert(cfg.d_model, cfg.d_ff_expert) for _ in range(cfg.n_routed_experts)]
        )
        self.router = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)

        # Per-expert routing bias: a plain buffer (no gradient), updated by a
        # feedback rule, not backprop. This is the whole "auxiliary-loss-free"
        # trick -- balancing lives entirely outside the computation graph.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_routed_experts))

        # Running load stats for the most recent forward pass (for update_bias).
        self._last_expert_counts = None
        self._last_n_tokens = None

    def forward(self, x: torch.Tensor):
        cfg = self.cfg
        b, t, d = x.shape
        flat = x.reshape(-1, d)  # [N, d]
        n_tokens = flat.shape[0]

        shared_out = sum(exp(flat) for exp in self.shared_experts)

        raw_scores = torch.sigmoid(self.router(flat))          # [N, n_routed], unbiased affinity
        biased_scores = raw_scores + self.expert_bias.unsqueeze(0)  # bias affects selection only

        topk_biased, topk_idx = biased_scores.topk(cfg.top_k, dim=-1)      # selection uses biased scores
        topk_unbiased = torch.gather(raw_scores, 1, topk_idx)              # weighting uses UNBIASED scores
        topk_weights = topk_unbiased / topk_unbiased.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        routed_out = torch.zeros_like(flat)
        expert_counts = torch.zeros(cfg.n_routed_experts, device=x.device)
        for slot in range(cfg.top_k):
            expert_idx = topk_idx[:, slot]              # [N]
            weight = topk_weights[:, slot].unsqueeze(-1)  # [N, 1]
            for e_id in expert_idx.unique():
                mask = expert_idx == e_id
                routed_out[mask] += weight[mask] * self.routed_experts[e_id](flat[mask])
                expert_counts[e_id] += mask.sum()

        # Stash load stats for the feedback bias update (called explicitly by
        # the training loop after backward(), analogous to an optimizer step
        # but entirely separate from it).
        self._last_expert_counts = expert_counts.detach()
        self._last_n_tokens = n_tokens

        out = (shared_out + routed_out).reshape(b, t, d)
        return out  # NOTE: no auxiliary loss returned -- balancing is not part of the loss

    @torch.no_grad()
    def update_bias(self):
        """
        Feedback-control bias update -- call once per training step, after the
        forward (and typically after optimizer.step(), though it is
        independent of gradient computation entirely).

        target load per expert if perfectly balanced: top_k / n_routed_experts
        of all (token, slot) assignments, i.e. top_k * n_tokens / n_routed_experts
        tokens per expert in expectation.
        """
        cfg = self.cfg
        if self._last_expert_counts is None:
            return
        target_load = cfg.top_k * self._last_n_tokens / cfg.n_routed_experts
        load_error = self._last_expert_counts - target_load  # >0 => overloaded, <0 => underloaded
        # sign(load_error) * gamma: overloaded experts' bias decreases (steer away),
        # underloaded experts' bias increases (steer toward). No gradient involved.
        self.expert_bias -= cfg.bias_update_speed * torch.sign(load_error)

    def load_imbalance_ratio(self) -> float:
        """max/mean expert load over the most recent forward pass -- purely diagnostic."""
        if self._last_expert_counts is None:
            return float("nan")
        counts = self._last_expert_counts
        return (counts.max() / counts.mean().clamp_min(1e-9)).item()


# ---------------------------------------------------------------------------
# Multi-Token Prediction (MTP) -- toy version
# ---------------------------------------------------------------------------

class MTPModule(nn.Module):
    """
    One MTP prediction depth: takes the main trunk's hidden state at position
    t together with the (embedded) ground-truth token at t+k-1, projects
    through a small transformer-ish block, and predicts the token at t+k.
    Real DeepSeek-V3 chains these sequentially (each depth's output feeds the
    next depth's input) to preserve a causal chain across the predicted
    horizon; this toy version keeps a single depth for clarity.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, trunk_hidden: torch.Tensor, next_token_embed: torch.Tensor):
        combined = self.combine(torch.cat([trunk_hidden, next_token_embed], dim=-1))
        h = self.norm(combined)
        return self.head(h)  # logits for token at t+k


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = MoEV3Config()
    moe = AuxLossFreeMoE(cfg)

    batch, seq_len = 4, 64
    x = torch.randn(batch, seq_len, cfg.d_model)

    print("=== Auxiliary-loss-free MoE (bias-based load balancing) ===")
    print(f"routed experts: {cfg.n_routed_experts}, shared experts: {cfg.n_shared_experts}, top_k: {cfg.top_k}")
    print(f"initial expert_bias (all zero): min={moe.expert_bias.min().item():.4f} max={moe.expert_bias.max().item():.4f}")

    # Simulate several training steps: forward -> (backward would go here) -> bias update.
    for step in range(20):
        out = moe(x)
        loss = out.pow(2).mean()   # stand-in for real LM loss
        loss.backward()            # gradients flow through routed/shared experts and the router, NOT through expert_bias
        moe.update_bias()          # feedback-control update, independent of the backward pass above
        for p in moe.parameters():
            if p.grad is not None:
                p.grad = None       # reset (no real optimizer here, just demonstrating the decoupling)

        if step in (0, 4, 9, 19):
            print(
                f"step {step:2d}: load_imbalance_ratio(max/mean)={moe.load_imbalance_ratio():.3f}  "
                f"bias_range=[{moe.expert_bias.min().item():.3f}, {moe.expert_bias.max().item():.3f}]"
            )

    print(
        "\nNote: expert_bias has no .grad and receives no optimizer step -- it is updated "
        "purely by update_bias()'s feedback rule, which is the entire 'auxiliary-loss-free' idea."
    )
    assert moe.expert_bias.grad is None, "bias must never receive a gradient"

    total_expert_params = sum(p.numel() for p in moe.routed_experts.parameters())
    active_expert_params = total_expert_params * cfg.top_k / cfg.n_routed_experts
    shared_params = sum(p.numel() for p in moe.shared_experts.parameters())
    total_params = total_expert_params + shared_params
    active_params = active_expert_params + shared_params
    print(f"\nTotal params in MoE module:  {total_params:,.0f}")
    print(f"Active params per token (~): {active_params:,.0f}  ({100 * active_params / total_params:.1f}% of total)")
    print(
        "\n(Real DeepSeek-V3: n_routed=256, n_shared=1, top_k=8, d_model=7168, "
        "671B total / 37B active params end-to-end, trained in FP8 on 2048 H800 GPUs.)"
    )

    print("\n=== Multi-Token Prediction (toy, single extra depth) ===")
    d_model, vocab_size = cfg.d_model, 1000
    mtp = MTPModule(d_model, vocab_size)
    trunk_hidden = torch.randn(batch, seq_len, d_model)   # main model's hidden state at position t
    next_token_embed = torch.randn(batch, seq_len, d_model)  # embedding of ground-truth token at t+1 (teacher-forced)
    mtp_logits = mtp(trunk_hidden, next_token_embed)
    print(f"trunk_hidden shape: {tuple(trunk_hidden.shape)}")
    print(f"MTP logits shape (predicting t+2 given trunk state at t and true token at t+1): {tuple(mtp_logits.shape)}")
    print(
        "MTP adds this extra next-next-token loss (lower weight) on top of the standard "
        "next-token loss, densifying the training signal per token."
    )
