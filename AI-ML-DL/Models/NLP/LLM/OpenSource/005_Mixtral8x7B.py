"""
Mixtral 8x7B (Jiang et al., 2023/2024) -- "Mixtral of Experts"

Demonstrates a real top-2 sparse mixture-of-experts (MoE) FFN layer: a learned linear
router, softmax + top-k expert selection, renormalized weighted combination of the
selected experts' outputs, and the Switch-Transformer-style auxiliary load-balancing
loss that keeps routing from collapsing onto a small subset of experts.

The backbone (RMSNorm, RoPE, GQA attention) is unchanged from the rest of this series
(see 001_Llama1.py / 002_Llama2.py) -- Mixtral's only structural departure is
replacing the single dense SwiGLU FFN at every layer with an MoEFeedForward layer.
Unlike Mistral 7B, Mixtral does NOT use sliding window attention; attention here is
standard (dense) causal GQA.

Real released config: dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, n_experts=8,
top_k=2, ~46.7B total parameters, ~12.9B active parameters per token. This file uses
small illustrative dimensions but computes the total-vs-active parameter distinction
exactly, so the ratio printed at the bottom is mechanically the same calculation that
gives Mixtral's real 46.7B / 12.9B split.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MixtralConfig:
    dim: int = 512
    n_layers: int = 2
    n_heads: int = 8
    n_kv_heads: int = 2       # GQA, same mechanism as Llama 2/3 / Mistral 7B
    n_experts: int = 8        # Mixtral's real config: 8 experts per MoE layer
    top_k: int = 2            # Mixtral's real config: top-2 routing
    vocab_size: int = 32_000
    ffn_multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    aux_loss_coef: float = 0.01  # standard small coefficient for the load-balancing loss


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len)
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
    *prefix, seq_len, n_heads, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(*prefix, seq_len, n_heads, head_dim // 2, 2))
    freqs = rope_freqs[:seq_len].view(1, seq_len, 1, head_dim // 2)
    x_out = torch.view_as_real(x_complex * freqs).reshape(*prefix, seq_len, n_heads, head_dim)
    return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """Standard dense causal GQA -- Mixtral does not use sliding window attention."""

    def __init__(self, cfg: MixtralConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        bsz, n_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        q = q.transpose(1, 2)
        k = self._repeat_kv(k.transpose(1, 2), self.n_rep)
        v = self._repeat_kv(v.transpose(1, 2), self.n_rep)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)


class Expert(nn.Module):
    """A single SwiGLU FFN expert -- architecturally identical to a normal dense FFN."""

    def __init__(self, cfg: MixtralConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * cfg.dim) / 3)
        hidden_dim = cfg.ffn_multiple_of * ((hidden_dim + cfg.ffn_multiple_of - 1) // cfg.ffn_multiple_of)
        self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFeedForward(nn.Module):
    """
    Top-k sparse mixture-of-experts FFN layer.

    Router: a single linear layer (dim -> n_experts) producing per-token logits.
    Selection: softmax over all n_experts, then keep only the top_k highest, then
    renormalize those top_k probabilities to sum to 1.
    Combination: output = sum_i (renormalized weight_i) * Expert_i(x), for i in the
    selected top_k experts -- the other (n_experts - top_k) experts perform zero
    compute for that token, which is the actual source of MoE's sparsity/efficiency.

    Also returns the Switch-Transformer-style auxiliary load-balancing loss:
        aux_loss = n_experts * sum_i (f_i * P_i)
    where f_i is the fraction of tokens in this batch whose top-k selection included
    expert i, and P_i is the average (full, pre-top-k) router probability assigned to
    expert i across the batch. This is minimized when both quantities are uniform
    (1/n_experts) across experts, giving the router a differentiable incentive to
    spread tokens evenly rather than collapsing onto a favored few.
    """

    def __init__(self, cfg: MixtralConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(cfg) for _ in range(cfg.n_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (n_tokens, dim) -- routing is per-token, not per-sequence
        n_tokens = x_flat.shape[0]

        router_logits = self.router(x_flat)                      # (n_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)          # full softmax, used for aux loss

        top_probs, top_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (n_tokens, top_k)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)        # renormalize to sum to 1

        output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            # token_mask[t] is True if this expert is among token t's selected top_k.
            token_mask = (top_idx == expert_id).any(dim=-1)  # (n_tokens,)
            if not token_mask.any():
                continue  # this expert performs zero compute for this batch -- the sparsity payoff
            # For each selected token, find its weight for *this* expert among its top_k slots.
            slot = (top_idx[token_mask] == expert_id).float().argmax(dim=-1)
            weight = top_probs[token_mask].gather(1, slot.unsqueeze(1)).squeeze(1)

            expert_out = expert(x_flat[token_mask])
            output[token_mask] += weight.unsqueeze(-1) * expert_out

        # --- Auxiliary load-balancing loss (Switch Transformer formulation) ---
        # f_i: fraction of tokens that selected expert i in their top_k.
        dispatch_mask = torch.zeros(n_tokens, self.n_experts, device=x.device)
        dispatch_mask.scatter_(1, top_idx, 1.0)
        f_i = dispatch_mask.mean(dim=0)          # (n_experts,)
        # P_i: average full-softmax router probability assigned to expert i.
        P_i = router_probs.mean(dim=0)           # (n_experts,)
        aux_loss = self.n_experts * torch.sum(f_i * P_i)

        return output.view(bsz, seq_len, dim), aux_loss


class MixtralBlock(nn.Module):
    def __init__(self, cfg: MixtralConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.moe_ffn = MoEFeedForward(cfg)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        moe_out, aux_loss = self.moe_ffn(self.ffn_norm(x))
        x = x + moe_out
        return x, aux_loss


class MixtralModel(nn.Module):
    def __init__(self, cfg: MixtralConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(MixtralBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        head_dim = cfg.dim // cfg.n_heads
        rope_freqs = precompute_rope_freqs(head_dim, 4096, cfg.rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tok_embeddings(tokens)
        aux_losses = []
        for layer in self.layers:
            x, aux_loss = layer(x, self.rope_freqs)
            aux_losses.append(aux_loss)
        x = self.norm(x)
        logits = self.output(x)
        total_aux_loss = torch.stack(aux_losses).mean()
        return logits, total_aux_loss

    def param_counts(self) -> tuple[int, int]:
        """Returns (total_params, active_params_per_token)."""
        total = sum(p.numel() for p in self.parameters())

        expert_params_per_layer = sum(p.numel() for p in self.layers[0].moe_ffn.experts[0].parameters())
        n_experts = self.cfg.n_experts
        top_k = self.cfg.top_k
        n_layers = self.cfg.n_layers

        # Dense parameters: everything except the expert FFN weights (embeddings, LM
        # head, attention, norms, router) -- these are fully active for every token.
        total_expert_params = expert_params_per_layer * n_experts * n_layers
        dense_params = total - total_expert_params

        # Active expert parameters: only top_k of n_experts run per token.
        active_expert_params = expert_params_per_layer * top_k * n_layers
        active = dense_params + active_expert_params
        return total, active


if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = MixtralConfig(dim=512, n_layers=4, n_heads=8, n_kv_heads=2, n_experts=8, top_k=2)
    model = MixtralModel(cfg)

    batch_size, seq_len = 2, 64
    tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    logits, aux_loss = model(tokens)

    print(f"Config: dim={cfg.dim}, n_layers={cfg.n_layers}, n_experts={cfg.n_experts}, "
          f"top_k={cfg.top_k}, n_heads={cfg.n_heads}, n_kv_heads={cfg.n_kv_heads}")
    print(f"Input tokens shape:  {tuple(tokens.shape)}")
    print(f"Output logits shape: {tuple(logits.shape)}")
    print(f"Auxiliary load-balancing loss (this batch): {aux_loss.item():.6f}")

    total_params, active_params = model.param_counts()
    print()
    print(f"Total parameters:            {total_params:,}")
    print(f"Active parameters per token: {active_params:,} "
          f"({100 * active_params / total_params:.1f}% of total)")
    print(f"Ratio total/active: {total_params / active_params:.2f}x "
          f"(real Mixtral 8x7B: 46.7B / 12.9B = {46.7 / 12.9:.2f}x)")

    # Illustrate the load-balancing loss's response to an imbalanced vs. balanced router.
    print()
    print("=" * 80)
    print("Load-balancing loss: balanced vs. imbalanced routing")
    print("=" * 80)
    n_tokens_demo = 256
    moe = MoEFeedForward(cfg)

    # Balanced case: router logits ~ i.i.d. noise -> roughly uniform routing.
    x_demo = torch.randn(1, n_tokens_demo, cfg.dim)
    _, aux_balanced = moe(x_demo)

    # Imbalanced case: bias the router's weights so expert 0 dominates the logits.
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] += 5.0  # push expert 0's logit far above the others
    _, aux_imbalanced = moe(x_demo)

    print(f"Auxiliary loss with roughly balanced routing:   {aux_balanced.item():.6f}")
    print(f"Auxiliary loss with expert-0-dominated routing: {aux_imbalanced.item():.6f}")
    print("(Imbalanced routing produces a higher auxiliary loss, which is exactly the "
          "gradient signal that discourages the router from collapsing onto a favored expert.)")
