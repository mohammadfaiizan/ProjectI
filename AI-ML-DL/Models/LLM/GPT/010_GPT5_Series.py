"""
010_GPT5_Series.py -- Educational reconstruction of the two genuinely
disclosed system-level design ideas behind OpenAI's GPT-5 (released August
2025): (1) a real-time ROUTER dispatching each query to a "fast" model or a
deeper "reasoning" model within one unified system, and (2) SAFE COMPLETIONS,
a graduated (non-binary) response-shaping paradigm replacing a flat
refuse-or-comply decision.

SCOPE AND HONESTY NOTE (read this before reading the code): OpenAI has stated,
in its own GPT-5 launch materials and system card, that GPT-5 in ChatGPT is a
unified SYSTEM -- a fast/efficient default model plus a deeper reasoning model
(continuing the o1/o3 RL-trained extended-reasoning lineage), with a real-time
router deciding per query which component responds, based on task complexity,
conversation context, tool-use needs, and explicit user request. OpenAI has
also stated that GPT-5 introduces "safe completions": training the model to
produce the most helpful response achievable within safety constraints,
instead of a binary refuse/comply decision, explicitly to reduce both
over-refusal and under-refusal. Both of those two claims -- that a router
exists and that safe completions exists as a named paradigm -- are the
confirmed, disclosed facts this file dramatizes.

EVERYTHING else here -- the router's actual features and architecture, the
reasoning path's actual iterative mechanism, the safe-completion classifier's
actual training data and categories, and every numeric threshold -- is NOT
publicly known and is not claimed to resemble GPT-5's real internals. This
file implements one reasonable, illustrative design for each concept, purely
to make the CONCEPTS concrete:
  (a) the routing decision as a differentiable gate (a learned linear term
      over pooled embeddings, combined with interpretable heuristic features,
      passed through a sigmoid -- soft/differentiable for training, hard
      thresholded for dispatch at inference), continuing the pattern used for
      the router in 009_GPT4_1_And_GPT5.py in this same folder, but developed
      further here;
  (b) two sub-models that differ in actual compute performed, not just in
      parameter count: the fast path runs ONE forward pass; the reasoning
      path runs a shared "thinking block" for several ITERATIVE REFINEMENT
      steps over its hidden state before decoding, echoing (as an explicit,
      labeled analogy, not a mechanistic claim) the o1/o3-style idea that
      more test-time compute -- here, more refinement iterations rather than
      more generated tokens -- can be spent on harder inputs;
  (c) a SafeCompletionHead producing a graduated, three-way soft distribution
      over {full compliance, bounded-safe partial help, refuse} for a
      borderline/dual-use input, rather than a binary allow/refuse gate, as a
      concrete nod to Section 6 of 010_GPT5_Series.md.

Structural note carried over from this document series: query-level routing
across whole models (this file's ComplexityRouter) and token-level expert
routing inside a sparse MoE layer (the toy Top2MoEFeedForward in 006_GPT4.py,
reconstructing the UNCONFIRMED GPT-4 MoE rumor) are the same underlying
conditional-compute idea -- a gate deciding which sub-network/how much
compute handles a unit of work -- applied at different granularities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building block: a single ordinary causal transformer encoder layer,
# used identically inside both the fast path and the reasoning path. The
# ONLY difference between the two paths in this file is how many times (and
# how) this block (or a stack of it) is applied -- one forward pass for the
# fast path, several iterative refinement passes for the reasoning path --
# deliberately mirroring the point made in 010_GPT5_Series.md Section 2 that
# the disclosed GPT-5 innovation is a SYSTEM-level compute-allocation design,
# not a claimed new base-architecture primitive.
# ---------------------------------------------------------------------------

class CausalBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, ids):
        seq_len = ids.size(1)
        return self.token_embed(ids) + self.pos_embed[:, :seq_len, :]


# ---------------------------------------------------------------------------
# FAST PATH: one embedding pass, one stack of blocks, one forward pass to
# logits. Stands in for GPT-5's "fast, efficient default model."
# ---------------------------------------------------------------------------

class FastModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_layers=1, n_heads=2, max_len=64):
        super().__init__()
        self.embedder = TokenEmbedder(vocab_size, d_model, max_len)
        self.blocks = nn.ModuleList(CausalBlock(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.forward_passes_per_token = 1  # one shot, no iterative refinement

    def forward(self, ids):
        x = self.embedder(ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, prompt_ids, num_new_tokens):
        ids = prompt_ids.clone()
        for _ in range(num_new_tokens):
            next_id = self.forward(ids)[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


# ---------------------------------------------------------------------------
# REASONING PATH: the compute-distinguishing design of this file. A shared
# "thinking block" is applied to the hidden state for K ITERATIVE REFINEMENT
# steps (a weight-tied recurrent loop over the same block, i.e. compute
# proportional to K rather than to layer count alone) before the final
# hidden state is decoded. K is itself allowed to vary per call, standing in
# -- as an explicit, labeled analogy only -- for the o1/o3-lineage idea that
# a deeper reasoning process can spend a variable, larger test-time compute
# budget on a harder input. This is NOT a claim that GPT-5's real reasoning
# component works this way; OpenAI has disclosed no such mechanism.
# ---------------------------------------------------------------------------

class ReasoningModel(nn.Module):
    def __init__(self, vocab_size, d_model=48, n_heads=4, max_len=64,
                 min_iters=2, max_iters=6):
        super().__init__()
        self.embedder = TokenEmbedder(vocab_size, d_model, max_len)
        self.thinking_block = CausalBlock(d_model, n_heads)  # weight-tied, applied repeatedly
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.min_iters = min_iters
        self.max_iters = max_iters
        # A tiny learned "halting" signal: after each refinement iteration,
        # estimate whether further refinement is likely to help. This is a
        # simplified nod to adaptive-computation-time-style mechanisms in the
        # broader literature (Graves, 2016, "Adaptive Computation Time for
        # Recurrent Neural Networks") -- illustrative only, not a claim about
        # GPT-5's actual reasoning tier, which discloses no such mechanism.
        self.halt_probe = nn.Linear(d_model, 1)

    def forward(self, ids, num_iters=None, return_iters_used=False):
        x = self.embedder(ids)
        iters_used = 0
        n = num_iters if num_iters is not None else self.max_iters
        n = max(self.min_iters, min(n, self.max_iters))
        for _ in range(n):
            x = self.thinking_block(x)
            iters_used += 1
            # Illustrative early-halt: if the probe is confident (on the
            # pooled last-token state) further refinement is unnecessary,
            # stop early -- still never below min_iters. Purely a toy signal;
            # no claim this resembles any real halting mechanism.
            if iters_used >= self.min_iters:
                halt_logit = self.halt_probe(x[:, -1, :]).mean()
                if torch.sigmoid(halt_logit).item() > 0.95:
                    break
        logits = self.lm_head(self.ln_f(x))
        if return_iters_used:
            return logits, iters_used
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, num_new_tokens, num_iters=None):
        ids = prompt_ids.clone()
        total_iters = 0
        for _ in range(num_new_tokens):
            logits, iters_used = self.forward(ids, num_iters=num_iters, return_iters_used=True)
            total_iters += iters_used
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids, total_iters


# ---------------------------------------------------------------------------
# THE ROUTER. Illustrative design only (see module docstring): heuristic
# features (query length, count of "hard-query marker" tokens standing in
# for things like math notation / explicit "prove" or "step by step"
# phrasing / detected tool-call need) combined with a learned embedding-based
# term, producing a routing probability via sigmoid. Soft (differentiable)
# by default for training-time use; a hard threshold is applied for the
# actual dispatch decision at inference, matching how a production gate
# would typically be trained soft and deployed with a hard cutoff.
# ---------------------------------------------------------------------------

class ComplexityRouter(nn.Module):
    def __init__(self, vocab_size, d_model=32, hard_marker_start=180, threshold=0.5):
        super().__init__()
        self.hard_marker_start = hard_marker_start
        self.threshold = threshold
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_to_logit = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.embed_to_logit.weight)  # untrained demo starts purely heuristic
        # length (normalized) and hard-marker-count features, hand-set signs:
        # longer + more markers pushes toward the reasoning tier.
        self.heuristic_weight = nn.Parameter(torch.tensor([0.5, 1.5]))
        self.heuristic_bias = nn.Parameter(torch.tensor(-1.0))

    def extract_heuristic_features(self, ids):
        seq_len = ids.size(1)
        length_feature = torch.full((ids.size(0), 1), float(seq_len) / 32.0, device=ids.device)
        hard_marker_count = (ids >= self.hard_marker_start).float().sum(dim=1, keepdim=True)
        return torch.cat([length_feature, hard_marker_count], dim=1)

    def forward(self, ids):
        pooled_embed = self.embed(ids).mean(dim=1)
        heuristic_features = self.extract_heuristic_features(ids)
        embed_logit = self.embed_to_logit(pooled_embed).squeeze(-1)
        heuristic_logit = (heuristic_features * self.heuristic_weight).sum(dim=1) + self.heuristic_bias
        routing_logit = embed_logit + heuristic_logit
        routing_prob = torch.sigmoid(routing_logit)          # differentiable, soft signal
        route_to_reasoning = routing_prob >= self.threshold  # hard dispatch decision
        return routing_prob, route_to_reasoning


# ---------------------------------------------------------------------------
# SAFE-COMPLETIONS-STYLE OUTPUT WRAPPER. Illustrative design only. Rather
# than a binary allow/refuse gate, a small classifier produces a *graduated*
# soft distribution over three response strategies for a given input, and
# the wrapper blends/selects a shaped response accordingly -- a toy nod to
# GPT-5's disclosed "safe completions" paradigm (010_GPT5_Series.md Section
# 6): train toward the most helpful response achievable within constraints,
# rather than a flat comply-or-refuse decision. The three categories, the
# "sensitive marker" heuristic, and the redaction mechanism below are this
# file's own illustrative construction -- not a description of any actual
# OpenAI classifier, training data, or category taxonomy.
# ---------------------------------------------------------------------------

class SafeCompletionHead(nn.Module):
    RESPONSE_LEVELS = ("full_compliance", "bounded_safe_partial", "refuse")

    def __init__(self, vocab_size, d_model=32, sensitive_marker_start=190,
                 redact_token_id=1, refuse_token_id=2):
        super().__init__()
        self.sensitive_marker_start = sensitive_marker_start
        self.redact_token_id = redact_token_id
        self.refuse_token_id = refuse_token_id
        self.embed = nn.Embedding(vocab_size, d_model)
        # Maps a pooled representation of the request to logits over the
        # three graduated response strategies -- this is the "safety
        # classifier" stand-in. Zero-initialized so, before training, the
        # decision is driven by the interpretable heuristic bias below,
        # exactly for the same reason as the router's zero-init above: it
        # makes the untrained demo below behave illustratively.
        self.to_level_logits = nn.Linear(d_model, 3)
        nn.init.zeros_(self.to_level_logits.weight)
        nn.init.zeros_(self.to_level_logits.bias)

    def classify(self, ids):
        """Returns a soft distribution over response levels -- the graduated
        analogue of a binary allow/refuse gate. Also returns the count of
        'sensitive marker' tokens purely for readable demo output."""
        pooled = self.embed(ids).mean(dim=1)
        sensitive_count = (ids >= self.sensitive_marker_start).float().sum(dim=1)
        # Heuristic bias: more sensitive markers pushes mass away from full
        # compliance and toward bounded-safe-partial (not straight to
        # refusal) -- dramatizing the stated design goal that the *default*
        # response to a dual-use-looking request should be a graduated,
        # maximally-helpful-within-constraints answer, with hard refusal
        # reserved for the clearest cases rather than being the default
        # fallback for anything borderline.
        bias = torch.zeros(ids.size(0), 3, device=ids.device)
        bias[:, 0] = -1.0 * sensitive_count            # full_compliance drops
        bias[:, 1] = 0.8 * sensitive_count              # bounded_safe_partial rises
        bias[:, 2] = 0.15 * (sensitive_count - 2).clamp(min=0)  # refuse rises only once clearly severe
        logits = self.to_level_logits(pooled) + bias
        probs = F.softmax(logits, dim=-1)
        return probs, sensitive_count

    def shape_response(self, output_ids, level_probs):
        """Given a fully-generated candidate response and the graduated
        level distribution, produce the actually-returned response. Rather
        than hard-switching on argmax alone, the *dominant* level determines
        the response shape, but the full soft distribution is still surfaced
        -- illustrating that safe completions is a graduated decision even
        though any single turn ultimately renders one concrete response."""
        dominant = level_probs.argmax(dim=-1)
        shaped = output_ids.clone()
        for row in range(output_ids.size(0)):
            level = self.RESPONSE_LEVELS[dominant[row].item()]
            if level == "refuse":
                shaped[row, :] = self.refuse_token_id
            elif level == "bounded_safe_partial":
                # redact tokens at/above the sensitive-marker range, keep the
                # rest -- a toy stand-in for "answer helpfully around the
                # specific harmful operational detail while withholding it."
                mask = shaped[row] >= self.sensitive_marker_start
                shaped[row][mask] = self.redact_token_id
            # "full_compliance": leave the generated response untouched.
        return shaped, [self.RESPONSE_LEVELS[d.item()] for d in dominant]


# ---------------------------------------------------------------------------
# THE ORCHESTRATING SYSTEM: router + fast/reasoning tiers + safe-completions
# shaping, with an explicit override path mirroring GPT-5's confirmed
# dual-mode design (automatic routing by default; explicit tier/effort
# control available to a caller, e.g. via the API).
# ---------------------------------------------------------------------------

class GPT5StyleRoutedSystem(nn.Module):
    def __init__(self, vocab_size, fast_config, reasoning_config, router_kwargs=None,
                 safe_completion_kwargs=None):
        super().__init__()
        self.fast_model = FastModel(vocab_size, **fast_config)
        self.reasoning_model = ReasoningModel(vocab_size, **reasoning_config)
        self.router = ComplexityRouter(vocab_size, **(router_kwargs or {}))
        self.safe_head = SafeCompletionHead(vocab_size, **(safe_completion_kwargs or {}))

    def forward(self, prompt_ids, force_tier=None, fast_tokens=6, reasoning_tokens=6,
                reasoning_iters=None):
        if force_tier == "fast":
            routing_prob = torch.tensor([0.0])
            route_to_reasoning = torch.tensor([False])
        elif force_tier == "reasoning":
            routing_prob = torch.tensor([1.0])
            route_to_reasoning = torch.tensor([True])
        else:
            routing_prob, route_to_reasoning = self.router(prompt_ids)

        if route_to_reasoning.item():
            output_ids, iters_used = self.reasoning_model.generate(
                prompt_ids, num_new_tokens=reasoning_tokens, num_iters=reasoning_iters
            )
            tier_used, compute_note = "reasoning", f"{iters_used} thinking-block iterations total"
        else:
            output_ids = self.fast_model.generate(prompt_ids, num_new_tokens=fast_tokens)
            tier_used, compute_note = "fast", "1 forward pass per generated token"

        level_probs, sensitive_count = self.safe_head.classify(prompt_ids)
        shaped_ids, level_names = self.safe_head.shape_response(output_ids, level_probs)

        return {
            "output_ids": shaped_ids,
            "raw_output_ids": output_ids,
            "tier_used": tier_used,
            "compute_note": compute_note,
            "routing_prob": routing_prob.item(),
            "safe_completion_level": level_names[0],
            "safe_completion_probs": {
                name: round(p, 3) for name, p in zip(SafeCompletionHead.RESPONSE_LEVELS,
                                                      level_probs[0].tolist())
            },
            "sensitive_marker_count": int(sensitive_count[0].item()),
        }


def count_params(module):
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)

    VOCAB_SIZE = 200
    HARD_MARKER_START = 180        # ids in [180, 200) simulate "hard-query" (complexity) markers
    SENSITIVE_MARKER_START = 190   # ids in [190, 200) simulate "sensitive/dual-use" markers

    system = GPT5StyleRoutedSystem(
        vocab_size=VOCAB_SIZE,
        fast_config=dict(d_model=32, n_layers=1, n_heads=2, max_len=64),
        reasoning_config=dict(d_model=48, n_heads=4, max_len=64, min_iters=2, max_iters=6),
        router_kwargs=dict(d_model=32, hard_marker_start=HARD_MARKER_START, threshold=0.5),
        safe_completion_kwargs=dict(d_model=32, sensitive_marker_start=SENSITIVE_MARKER_START),
    )

    fast_params = count_params(system.fast_model)
    reasoning_params = count_params(system.reasoning_model)
    router_params = count_params(system.router)
    safe_head_params = count_params(system.safe_head)

    print("=== GPT-5-style toy system: router + fast/reasoning tiers + safe completions ===")
    print(f"fast model params:      {fast_params:,}  (1 forward pass per token)")
    print(f"reasoning model params: {reasoning_params:,}  (2-6 thinking-block iterations per token)")
    print(f"router params:          {router_params:,}")
    print(f"safe-completion head params: {safe_head_params:,}")

    def show(label, ids, force_tier=None):
        result = system(ids, force_tier=force_tier)
        print(f"\n-- {label} --")
        print(f"query ids: {ids.tolist()[0]}")
        print(f"routing probability (reasoning tier): {result['routing_prob']:.3f} "
              f"-> tier used: {result['tier_used']} ({result['compute_note']})")
        print(f"safe-completion distribution: {result['safe_completion_probs']} "
              f"-> selected level: {result['safe_completion_level']} "
              f"(sensitive markers detected: {result['sensitive_marker_count']})")

    # 1. Simple query: short, no complexity markers -> expect fast tier.
    simple_query = torch.randint(0, HARD_MARKER_START, (1, 5))
    show("Simple query (automatic routing)", simple_query)

    # 2. Complex query: several hard-query complexity markers -> expect
    #    reasoning tier, with multiple thinking-block iterations spent.
    complex_query = torch.cat([
        torch.randint(0, HARD_MARKER_START, (1, 3)),
        torch.randint(HARD_MARKER_START, SENSITIVE_MARKER_START, (1, 6)),
    ], dim=1)
    show("Complex query (automatic routing)", complex_query)

    # 3. Borderline dual-use-looking query: contains "sensitive marker"
    #    tokens (a superset of the hard-marker range in this toy vocab) ->
    #    expect the safe-completion head to shift mass toward the graduated
    #    "bounded_safe_partial" level rather than a flat refusal.
    borderline_query = torch.cat([
        torch.randint(0, HARD_MARKER_START, (1, 3)),
        torch.randint(SENSITIVE_MARKER_START, VOCAB_SIZE, (1, 3)),
    ], dim=1)
    show("Borderline dual-use-looking query (automatic routing + safe completions)", borderline_query)

    # 4. Explicit override, mirroring GPT-5's confirmed dual-mode design
    #    (automatic routing by default; explicit tier control available to a
    #    caller, e.g. via the API).
    print("\n-- Explicit tier override (bypassing the router) --")
    for forced in ("fast", "reasoning"):
        result = system(simple_query, force_tier=forced)
        print(f"forced tier='{forced}': tier used={result['tier_used']} ({result['compute_note']})")

    print("\nNOTE: the router's features/architecture, the reasoning path's iterative")
    print("mechanism, and the safe-completion classifier's categories are illustrative")
    print("designs for this file only -- not a description of GPT-5's actual, undisclosed")
    print("internals. See 010_GPT5_Series.md Sections 2, 6, and 11 for exactly what OpenAI")
    print("has and has not disclosed.")
