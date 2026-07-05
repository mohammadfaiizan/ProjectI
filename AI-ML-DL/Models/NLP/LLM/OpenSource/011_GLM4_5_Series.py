"""
GLM-4.5 (Zhipu AI / Z.ai, 2025) -- illustrative implementation of the two
mechanisms that define this generation's architectural break from dense
GLM-4 (see 010_GLM4.py): (1) a top-k sparse mixture-of-experts FFN layer with
shared experts and a load-balancing loss, in the same family as the MoE
layers implemented in 005_Mixtral8x7B.py and 007_DeepSeek_V3.py but written
fresh here; and (2) a hybrid-reasoning wrapper module that can run the same
underlying model in a fast single-pass mode or an iterative extended-
reasoning mode with a configurable step budget, in the same spirit as the
thinking-budget pattern described (for Claude 3.7 Sonnet) in
Claude/005_Claude3_7_Extended_Thinking.md, written fresh for GLM-4.5's
single-checkpoint mode-toggle framing.

Reported real-world config (Zhipu's own disclosure, not independently
verified -- see 011_GLM4_5_Series.md Section 3): ~355B total parameters,
~32B active parameters per token. This file uses small illustrative
dimensions throughout but computes the total-vs-active parameter split
exactly the same way, so the ratio printed at the bottom is mechanically the
same calculation that gives the real ~9% active-parameter sparsity ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GLM45Config:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 8
    n_kv_heads: int = 2          # GQA, same mechanism used throughout this folder
    vocab_size: int = 128
    max_seq_len: int = 256
    n_routed_experts: int = 16   # illustrative scale; real GLM-4.5 reportedly routes over far more
    n_shared_experts: int = 1    # always-on expert(s), same DeepSeekMoE-style design used in 006/007
    top_k: int = 2
    d_ff_expert: int = 96
    aux_loss_coef: float = 0.01
    max_thinking_steps: int = 6  # configurable extended-reasoning step budget
    stop_threshold: float = 0.5  # "continue thinking" probability below which the model stops early
    answer_token_id: int = 1     # reserved vocab id marking "now produce the final answer"


# ---------------------------------------------------------------------------
# Backbone: RMSNorm + GQA attention + top-k MoE FFN
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class GQAAttention(nn.Module):
    """Standard causal GQA -- unchanged in kind from the attention used
    throughout this folder (see 009_Qwen2_5.py); GLM-4.5's headline departure
    from GLM-4 is the MoE FFN and the hybrid-reasoning wrapper below, not the
    attention mechanism itself."""

    def __init__(self, cfg: GLM45Config):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.wo(out)


class Expert(nn.Module):
    """A single SwiGLU FFN expert."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TopKMoE(nn.Module):
    """
    Top-k sparse MoE FFN with always-on shared experts, the fine-grained
    DeepSeekMoE-style pattern GLM-4.5 is reported to follow: every token is
    processed by ALL shared experts plus its top-k selected routed experts out
    of n_routed_experts, and the (n_routed_experts - top_k) unselected experts
    perform zero compute for that token -- the actual source of MoE sparsity.

    Router: softmax over routed-expert logits, then top-k selection,
    renormalized to sum to 1. Load balancing: a Switch-Transformer-style
    auxiliary loss (n_experts * sum_i f_i * P_i, minimized at uniform routing)
    -- written fresh here, but the same family of mechanism as the aux-loss
    version in 005_Mixtral8x7B.py. (Whether GLM-4.5 itself uses this
    auxiliary-loss formulation or a bias-based, auxiliary-loss-free scheme
    like DeepSeek-V3's -- 007_DeepSeek_V3.py -- is not confirmed by public
    disclosure at time of writing; see 011_GLM4_5_Series.md Section 2.)
    """

    def __init__(self, cfg: GLM45Config):
        super().__init__()
        self.cfg = cfg
        self.shared_experts = nn.ModuleList(
            Expert(cfg.d_model, cfg.d_ff_expert) for _ in range(cfg.n_shared_experts)
        )
        self.routed_experts = nn.ModuleList(
            Expert(cfg.d_model, cfg.d_ff_expert) for _ in range(cfg.n_routed_experts)
        )
        self.router = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        n_tokens = flat.shape[0]

        shared_out = sum(exp(flat) for exp in self.shared_experts)

        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_idx = torch.topk(router_probs, self.cfg.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(flat)
        for e_id in range(self.cfg.n_routed_experts):
            token_mask = (top_idx == e_id).any(dim=-1)
            if not token_mask.any():
                continue  # zero compute for this expert on this batch -- the sparsity payoff
            slot = (top_idx[token_mask] == e_id).float().argmax(dim=-1)
            weight = top_probs[token_mask].gather(1, slot.unsqueeze(1)).squeeze(1)
            routed_out[token_mask] += weight.unsqueeze(-1) * self.routed_experts[e_id](flat[token_mask])

        # Auxiliary load-balancing loss (Switch-Transformer formulation).
        dispatch_mask = torch.zeros(n_tokens, self.cfg.n_routed_experts, device=x.device)
        dispatch_mask.scatter_(1, top_idx, 1.0)
        f_i = dispatch_mask.mean(dim=0)
        p_i = router_probs.mean(dim=0)
        aux_loss = self.cfg.n_routed_experts * torch.sum(f_i * p_i)

        out = (shared_out + routed_out).reshape(b, t, d)
        return out, aux_loss

    def param_counts(self) -> tuple[int, int]:
        """(total_expert_params, active_expert_params_per_token)."""
        per_expert = sum(p.numel() for p in self.routed_experts[0].parameters())
        total_routed = per_expert * self.cfg.n_routed_experts
        active_routed = per_expert * self.cfg.top_k
        shared = sum(p.numel() for p in self.shared_experts.parameters())
        return total_routed + shared, active_routed + shared


class GLM45Block(nn.Module):
    def __init__(self, cfg: GLM45Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.moe = TopKMoE(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x))
        moe_out, aux_loss = self.moe(self.ffn_norm(x))
        x = x + moe_out
        return x, aux_loss


class GLM45Backbone(nn.Module):
    def __init__(self, cfg: GLM45Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(GLM45Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False):
        x = self.embed(input_ids)
        aux_losses = []
        for block in self.blocks:
            x, aux = block(x)
            aux_losses.append(aux)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        total_aux = torch.stack(aux_losses).mean()
        if return_hidden:
            return logits, total_aux, x
        return logits, total_aux

    def param_counts(self) -> tuple[int, int]:
        """(total_params, active_params_per_token) across the whole backbone."""
        total = sum(p.numel() for p in self.parameters())
        moe_total = moe_active = 0
        for block in self.blocks:
            t, a = block.moe.param_counts()
            moe_total += t
            moe_active += a
        dense = total - sum(
            sum(p.numel() for p in block.moe.routed_experts.parameters())
            for block in self.blocks
        )
        active = dense + moe_active
        return total, active


# ---------------------------------------------------------------------------
# Hybrid-reasoning wrapper: fast single-pass mode vs. iterative extended-
# reasoning mode with a configurable step budget.
# ---------------------------------------------------------------------------

class HybridReasoningWrapper(nn.Module):
    """
    Wraps a GLM45Backbone with two request-time modes, mirroring GLM-4.5's
    (and Claude 3.7's) single-checkpoint hybrid-reasoning design: one set of
    weights, a mode flag, NOT a router choosing between two different models
    (contrast GPT-5's disclosed system-of-models-plus-router framing,
    discussed in 011_GLM4_5_Series.md Section 1).

    - fast mode: one forward pass, produce the final-answer logits directly.
    - thinking mode: an iterative loop, up to `max_thinking_steps` (the
      configurable budget). At each step, the model inspects its own last
      hidden state through a small learned `continue_head` and produces a
      "continue thinking" probability. Below `stop_threshold`, thinking stops
      early -- exactly the behavior Claude 3.7 exhibits (converging on an
      answer within budget rather than always padding to the limit) -- and a
      reserved `answer_token_id` is appended before one final forward pass
      that produces the actual answer logits.
    """

    def __init__(self, backbone: GLM45Backbone, cfg: GLM45Config):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.continue_head = nn.Linear(cfg.d_model, 1)

    def fast_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.backbone(input_ids)
        return logits[:, -1, :]  # next-token logits at the final position -- direct answer, no deliberation

    @torch.no_grad()
    def think_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Iterative extended-reasoning loop. Returns (final_answer_logits, steps_used).
        Runs without gradients here since this models INFERENCE-time deliberation,
        not a training step (training the stopping policy is a post-training RL
        concern -- see the .md Section 6 -- not something this forward pass does).
        """
        seq = input_ids
        steps_used = 0
        for step in range(self.cfg.max_thinking_steps):
            logits, _, hidden = self.backbone(seq, return_hidden=True)
            last_hidden = hidden[:, -1, :]
            continue_prob = torch.sigmoid(self.continue_head(last_hidden)).item()
            steps_used += 1
            if continue_prob < self.cfg.stop_threshold:
                break  # the model "decided" it has thought enough -- early stop, not budget exhaustion
            next_thought_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_thought_token], dim=1)

        # Deliberation (if any) is complete; append the answer marker and do one
        # final forward pass to produce the actual response logits.
        answer_marker = torch.full((seq.shape[0], 1), self.cfg.answer_token_id, dtype=torch.long)
        seq = torch.cat([seq, answer_marker], dim=1)
        final_logits, _ = self.backbone(seq)
        return final_logits[:, -1, :], steps_used

    def forward(self, input_ids: torch.Tensor, mode: str = "fast"):
        if mode == "fast":
            return self.fast_forward(input_ids)
        elif mode == "thinking":
            return self.think_forward(input_ids)
        raise ValueError(f"unknown mode: {mode!r} (expected 'fast' or 'thinking')")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = GLM45Config()
    backbone = GLM45Backbone(cfg)
    model = HybridReasoningWrapper(backbone, cfg)

    batch, seq_len = 2, 12
    input_ids = torch.randint(2, cfg.vocab_size, (batch, seq_len))  # avoid ids 0/1 (reserved)

    print("=== GLM-4.5-style top-k MoE + hybrid-reasoning wrapper ===")
    print(f"n_routed_experts={cfg.n_routed_experts}, n_shared_experts={cfg.n_shared_experts}, "
          f"top_k={cfg.top_k}, n_layers={cfg.n_layers}")

    logits, aux_loss = backbone(input_ids)
    print(f"\ninput_ids shape: {tuple(input_ids.shape)}")
    print(f"backbone logits shape: {tuple(logits.shape)}")
    print(f"MoE auxiliary load-balancing loss (this batch): {aux_loss.item():.6f}")

    total_params, active_params = backbone.param_counts()
    print(f"\nTotal backbone parameters:            {total_params:,}")
    print(f"Active backbone parameters per token: {active_params:,} "
          f"({100 * active_params / total_params:.1f}% of total)")
    print(f"Ratio total/active: {total_params / active_params:.2f}x "
          f"(reported real GLM-4.5: ~355B / ~32B =~ {355 / 32:.2f}x)")

    print("\n=== Fast mode: single forward pass, no deliberation ===")
    fast_logits = model(input_ids, mode="fast")
    print(f"fast-mode next-token logits shape: {tuple(fast_logits.shape)}")

    print("\n=== Thinking mode: iterative deliberation with a step budget ===")
    print(f"max_thinking_steps={cfg.max_thinking_steps}, stop_threshold={cfg.stop_threshold}")
    for row in range(batch):
        single_input = input_ids[row : row + 1]
        think_logits, steps_used = model(single_input, mode="thinking")
        print(f"  example {row}: steps_used={steps_used}/{cfg.max_thinking_steps}, "
              f"final answer logits shape={tuple(think_logits.shape)}")

    print(
        "\n(Fast mode always costs exactly one forward pass over the input; thinking "
        "mode costs between 1 and max_thinking_steps extra forward passes plus one "
        "final answer pass -- the same 'thinking tokens cost extra, billable compute' "
        "tradeoff described for Claude 3.7 Sonnet in Claude/005_Claude3_7_Extended_Thinking.md, "
        "here applied to a request-time mode flag on a single MoE checkpoint rather than "
        "a router choosing between separate models as in GPT-5's system design.)"
    )
