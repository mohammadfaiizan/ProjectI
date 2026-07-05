"""
DeepSeek-V2 (2024) -- Multi-head Latent Attention (MLA) reference implementation.

This file demonstrates the core mechanism that distinguishes DeepSeek-V2 from a
standard MHA/GQA transformer: instead of caching full per-head K/V tensors, the
model down-projects the residual stream into a single small shared low-rank
latent vector, caches ONLY that latent (plus a small shared decoupled-RoPE key),
and reconstructs per-head K/V via learned up-projection matrices at attention
time. This is what shrinks the KV cache far below even aggressive GQA while
retaining full-rank-equivalent attention quality.

Also included: a minimal DeepSeekMoE FFN block (fine-grained routed experts +
always-on shared experts) so the file demonstrates both of V2's headline ideas.

Nothing here is optimized for production (no fused kernels, no KV-cache object,
no matrix-absorption inference trick) -- it is written to make the *mechanism*
of MLA legible and numerically checkable, not to be fast.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary position embedding (standard; applied only to the "decoupled" slice)
# ---------------------------------------------------------------------------

def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [seq_len, dim/2]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [seq_len, dim]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, seq_len, dim] (shared across heads, e.g. the decoupled MLA key);
    # cos/sin: [seq_len, dim] broadcasts naturally against the trailing two dims.
    return x * cos + rotate_half(x) * sin


def apply_rope_multihead(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, seq_len, n_heads, dim]; cos/sin: [seq_len, dim].
    # Insert singleton batch and head dims so cos/sin broadcast against the
    # seq_len dim (index 1) rather than the n_heads dim (index 2).
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Multi-head Latent Attention
# ---------------------------------------------------------------------------

@dataclass
class MLAConfig:
    d_model: int = 512          # hidden size (toy scale; V2 uses 5120)
    n_heads: int = 8            # number of attention heads (V2 uses 128)
    d_head: int = 32            # content dim per head (V2 uses 128)
    d_c: int = 64               # KV latent (compressed) dimension (V2 uses 512)
    d_c_q: int = 96              # query latent (compressed) dimension (V2 uses 1536)
    d_rope: int = 16            # decoupled RoPE dim, shared across heads (V2 uses 64)
    max_seq_len: int = 2048


class MultiHeadLatentAttention(nn.Module):
    """
    MLA as introduced in DeepSeek-V2.

    Cached state per token (what a real KV cache would store) is exactly:
        c_kv  in R^{d_c}       (compressed content latent, shared across ALL heads)
        k_rope in R^{d_rope}   (decoupled rotary key, shared across ALL heads)
    i.e. d_c + d_rope scalars per token -- independent of n_heads and d_head.
    Contrast with standard MHA, which must cache 2 * n_heads * d_head scalars
    per token (full per-head K and V).

    Per-head K/V are RECONSTRUCTED from the cached latent via learned
    up-projections (W_uk, W_uv) at attention time -- they are never
    themselves the cached quantity.
    """

    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        h, dh, dc, dcq, dr = cfg.n_heads, cfg.d_head, cfg.d_c, cfg.d_c_q, cfg.d_rope

        # --- KV path: down-projection to shared latent, then per-head up-projections ---
        self.W_dkv = nn.Linear(cfg.d_model, dc, bias=False)           # h_t -> c_t^{KV}
        self.W_uk = nn.Linear(dc, h * dh, bias=False)                  # c_t^{KV} -> k_t^{C} (all heads)
        self.W_uv = nn.Linear(dc, h * dh, bias=False)                  # c_t^{KV} -> v_t^{C} (all heads)

        # --- decoupled shared RoPE key (bypasses the latent bottleneck entirely) ---
        self.W_kr = nn.Linear(cfg.d_model, dr, bias=False)              # h_t -> k_t^{R}, ONE per token (shared across heads)

        # --- Query path: down-projection to a (separate) latent, then per-head up-projections ---
        self.W_dq = nn.Linear(cfg.d_model, dcq, bias=False)             # h_t -> c_t^{Q}
        self.W_uq = nn.Linear(dcq, h * dh, bias=False)                  # c_t^{Q} -> q_t^{C} (all heads, content)
        self.W_qr = nn.Linear(dcq, h * dr, bias=False)                  # c_t^{Q} -> q_t^{R} (all heads, rotary)

        self.W_o = nn.Linear(h * dh, cfg.d_model, bias=False)          # output projection

        cos, sin = build_rope_cache(cfg.max_seq_len, dr)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.scale = 1.0 / math.sqrt(dh + dr)

    def forward(self, h: torch.Tensor, use_cache: bool = False, past_latent=None):
        """
        h: [batch, seq_len, d_model]  -- residual stream input to this layer's attention.

        Returns: (output [batch, seq_len, d_model], cache) where cache, if requested,
        is exactly the (c_kv, k_rope) pair that a real server would persist -- this
        is the whole point of MLA: cache is latent-sized, not head-sized.
        """
        cfg = self.cfg
        b, t, _ = h.shape
        h_dim, dh, dr = cfg.n_heads, cfg.d_head, cfg.d_rope

        # ---- KV path ----
        c_kv = self.W_dkv(h)                                   # [b, t, d_c]   <-- THIS is what gets cached
        k_content = self.W_uk(c_kv).view(b, t, h_dim, dh)        # [b, t, heads, d_head] reconstructed per-head keys
        v_content = self.W_uv(c_kv).view(b, t, h_dim, dh)        # [b, t, heads, d_head] reconstructed per-head values

        k_rope = self.W_kr(h)                                    # [b, t, d_rope] <-- also cached (shared across heads)
        cos, sin = self.rope_cos[:t].to(h.device), self.rope_sin[:t].to(h.device)
        k_rope_rot = apply_rope(k_rope, cos, sin)                  # [b, t, d_rope]
        # broadcast the single shared rotary key across all heads
        k_rope_rot = k_rope_rot.unsqueeze(2).expand(b, t, h_dim, dr)

        k = torch.cat([k_content, k_rope_rot], dim=-1)             # [b, t, heads, d_head + d_rope]

        # ---- Query path (compressed for activation-memory reasons; never cached) ----
        c_q = self.W_dq(h)                                        # [b, t, d_c_q]
        q_content = self.W_uq(c_q).view(b, t, h_dim, dh)
        q_rope = self.W_qr(c_q).view(b, t, h_dim, dr)
        q_rope_rot = apply_rope_multihead(q_rope, cos, sin)          # per-head rotary query (each head gets its own slice)
        q = torch.cat([q_content, q_rope_rot], dim=-1)              # [b, t, heads, d_head + d_rope]

        v = v_content                                              # [b, t, heads, d_head]

        # ---- causal scaled dot-product attention over reconstructed K/V ----
        q = q.transpose(1, 2)   # [b, heads, t, d_head+d_rope]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)   # [b, heads, t, d_head]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [b, heads, t, t]
        causal_mask = torch.triu(torch.ones(t, t, device=h.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)                          # [b, heads, t, d_head]

        out = out.transpose(1, 2).reshape(b, t, h_dim * dh)
        out = self.W_o(out)

        cache = {"c_kv": c_kv, "k_rope": k_rope} if use_cache else None
        return out, cache

    def kv_cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """What a real serving system would persist per token per layer, in elements."""
        return (self.cfg.d_c + self.cfg.d_rope) * dtype_bytes

    @staticmethod
    def mha_equivalent_cache_bytes_per_token(n_heads: int, d_head: int, dtype_bytes: int = 2) -> int:
        return 2 * n_heads * d_head * dtype_bytes  # K and V, full per-head


# ---------------------------------------------------------------------------
# DeepSeekMoE FFN block (fine-grained routed experts + shared experts)
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    d_model: int = 512
    d_ff_expert: int = 128       # small per-expert intermediate size (fine-grained)
    n_routed_experts: int = 32   # V2 uses 160
    n_shared_experts: int = 2    # V2 uses 2
    top_k: int = 4               # V2 uses 6


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))  # SwiGLU


class DeepSeekMoE(nn.Module):
    """
    Fine-grained MoE: many small routed experts (top_k activated per token)
    plus a small number of always-on shared experts. Uses a standard
    auxiliary load-balancing loss here (V2-era); DeepSeek-V3's bias-based,
    auxiliary-loss-free scheme is implemented in 007_DeepSeek_V3.py.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.shared_experts = nn.ModuleList(
            [Expert(cfg.d_model, cfg.d_ff_expert) for _ in range(cfg.n_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [Expert(cfg.d_model, cfg.d_ff_expert) for _ in range(cfg.n_routed_experts)]
        )
        self.router = nn.Linear(cfg.d_model, cfg.n_routed_experts, bias=False)

    def forward(self, x: torch.Tensor):
        b, t, d = x.shape
        flat = x.reshape(-1, d)  # [N, d]

        # shared experts: always active, summed
        shared_out = sum(exp(flat) for exp in self.shared_experts)

        # routed experts: top-k gating
        logits = self.router(flat)                      # [N, n_routed]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.cfg.top_k, dim=-1)   # [N, top_k]
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize

        routed_out = torch.zeros_like(flat)
        for slot in range(self.cfg.top_k):
            expert_idx = topk_idx[:, slot]        # [N]
            weight = topk_probs[:, slot].unsqueeze(-1)  # [N, 1]
            for e_id in expert_idx.unique():
                mask = expert_idx == e_id
                routed_out[mask] += weight[mask] * self.routed_experts[e_id](flat[mask])

        # auxiliary load-balance loss: encourage uniform expert usage (V2-style; see V3 for the loss-free alternative)
        expert_usage = probs.mean(dim=0)  # [n_routed], mean routing probability per expert
        target = 1.0 / self.cfg.n_routed_experts
        aux_loss = ((expert_usage - target) ** 2).sum()

        out = (shared_out + routed_out).reshape(b, t, d)
        return out, aux_loss


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    mla_cfg = MLAConfig()
    mla = MultiHeadLatentAttention(mla_cfg)

    batch, seq_len = 2, 32
    h = torch.randn(batch, seq_len, mla_cfg.d_model)
    out, cache = mla(h, use_cache=True)
    print("=== Multi-head Latent Attention ===")
    print(f"input shape:  {tuple(h.shape)}")
    print(f"output shape: {tuple(out.shape)}")
    print(f"cached c_kv shape:   {tuple(cache['c_kv'].shape)}   ({mla_cfg.d_c} elements/token)")
    print(f"cached k_rope shape: {tuple(cache['k_rope'].shape)} ({mla_cfg.d_rope} elements/token)")

    mla_cache_bytes = mla.kv_cache_bytes_per_token()
    mha_cache_bytes = MultiHeadLatentAttention.mha_equivalent_cache_bytes_per_token(
        mla_cfg.n_heads, mla_cfg.d_head
    )
    print(f"\nKV cache per token per layer (fp16 bytes):")
    print(f"  MLA:            {mla_cache_bytes} bytes  ({mla_cfg.d_c} + {mla_cfg.d_rope} = {mla_cfg.d_c + mla_cfg.d_rope} elements)")
    print(f"  MHA-equivalent: {mha_cache_bytes} bytes  (2 * {mla_cfg.n_heads} heads * {mla_cfg.d_head} dim)")
    print(f"  reduction factor: {mha_cache_bytes / mla_cache_bytes:.1f}x")

    n_params_mla = sum(p.numel() for p in mla.parameters())
    print(f"\nMLA module parameter count: {n_params_mla:,}")

    print("\n=== DeepSeekMoE FFN (fine-grained experts + shared experts) ===")
    moe_cfg = MoEConfig()
    moe = DeepSeekMoE(moe_cfg)
    x = torch.randn(batch, seq_len, moe_cfg.d_model)
    moe_out, aux_loss = moe(x)
    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(moe_out.shape)}")
    print(f"aux load-balance loss: {aux_loss.item():.6f}")

    total_expert_params = sum(p.numel() for p in moe.routed_experts.parameters())
    active_expert_params = total_expert_params * moe_cfg.top_k / moe_cfg.n_routed_experts
    shared_params = sum(p.numel() for p in moe.shared_experts.parameters())
    print(f"\nTotal routed-expert params: {total_expert_params:,}  (all {moe_cfg.n_routed_experts} experts)")
    print(f"Active routed-expert params per token (~): {active_expert_params:,.0f}  (top_k={moe_cfg.top_k})")
    print(f"Always-active shared-expert params: {shared_params:,}")
    print(f"Total params in module: {total_expert_params + shared_params:,}")
    print(f"Active params per token (~): {active_expert_params + shared_params:,.0f}")
    print(
        "\n(For reference, real DeepSeek-V2 uses n_routed=160, n_shared=2, top_k=6, "
        "d_model=5120, giving 236B total / 21B active parameters end-to-end.)"
    )
