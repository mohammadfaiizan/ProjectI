"""
009_GPT4_1_And_GPT5.py -- Educational reconstruction of the query-level
ROUTER concept publicly associated with OpenAI's GPT-5 (2025) system design.

SCOPE: This is the most speculative file in this document series and should
be read that way. OpenAI has stated, in its own GPT-5 launch materials, that
GPT-5 in ChatGPT is a unified SYSTEM combining a fast/efficient default model
with a deeper reasoning model (continuing the o-series RL-trained-reasoning
lineage), with a real-time router deciding per query which underlying model
should respond -- removing the need for users to manually pick a model. That
system-design claim is the one specific, confirmed thing this file dramatizes.

EVERYTHING else here -- the router's actual features, architecture, training
method, and decision thresholds -- is NOT publicly known and is not claimed
to resemble GPT-5's real router. This file implements one reasonable,
illustrative router design (a small learned classifier over hand-engineered
complexity features, i.e. a heuristic-plus-learned-gate hybrid) purely to
make the ROUTING CONCEPT concrete: given an input, decide whether a "fast"
or "deep reasoning" sub-model should handle it, and dispatch accordingly.

Structural note carried over from the GPT-4/GPT-4.1/GPT-5 document series:
this query-level routing (gating across whole models) is the same underlying
idea as token-level expert routing inside a sparse MoE layer (see the toy
Top2MoEFeedForward in 006_GPT4.py, reconstructing the UNCONFIRMED GPT-4 MoE
rumor) -- a gate deciding which sub-network handles a unit of work, just at
a different granularity (per-query here vs. per-token there).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Two toy sub-models standing in for "fast, efficient default model" and
# "deeper reasoning model." Both are tiny causal transformers here; the
# only real difference in this toy is capacity (depth/width) and how many
# tokens the "reasoning" model is allowed to generate before answering,
# echoing the o1/o3-style test-time-compute distinction from the previous
# entry in this series -- GPT-5's stated design explicitly continues that
# lineage for its deeper tier.
# ---------------------------------------------------------------------------

class ToySubModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True)
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        seq_len = ids.size(1)
        x = self.token_embed(ids) + self.pos_embed[:, :seq_len, :]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=ids.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, num_new_tokens):
        ids = prompt_ids.clone()
        for _ in range(num_new_tokens):
            logits = self.forward(ids)[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


# ---------------------------------------------------------------------------
# The router. Illustrative design only (see module docstring): a small
# feature extractor over the input sequence, feeding a learned linear gate
# that outputs a routing probability. Combines cheap heuristic signals
# (sequence length, presence of "hard-query" marker tokens) with a learned
# component, as a plausible (not confirmed) design point in the space
# described in 009_GPT4_1_And_GPT5.md Section 2.
# ---------------------------------------------------------------------------

class ComplexityRouter(nn.Module):
    """Maps an input token sequence to a routing decision: FAST tier or
    DEEP-REASONING tier. Two complexity signals are combined:
      1. A heuristic feature vector (sequence length, count of tokens drawn
         from a designated "hard-query marker" id range -- standing in for
         signals like math symbols, explicit "prove"/"step by step"
         phrasing, or requested tool use in a real system).
      2. A learned embedding-based classifier over the same input, so the
         router is not purely rule-based.
    The two signals are combined and passed through a sigmoid to produce a
    routing probability; a threshold determines the dispatch decision.
    """

    def __init__(self, vocab_size, d_model=32, hard_marker_start=180, threshold=0.5):
        super().__init__()
        self.hard_marker_start = hard_marker_start  # ids >= this are "hard-query markers"
        self.threshold = threshold
        self.embed = nn.Embedding(vocab_size, d_model)
        # Learned embedding-based contribution to the routing logit. Zero-
        # initialized so that, before any training, the router's behavior is
        # driven purely by the interpretable heuristic term below -- this is
        # an initialization choice made so the untrained demo below actually
        # illustrates the intended routing behavior; a real trained router
        # would let gradient descent move this weight away from zero based
        # on labeled/graded routing data.
        self.embed_to_logit = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.embed_to_logit.weight)
        # Fixed, interpretable heuristic weights: more hard-query markers and
        # longer queries push the routing logit toward the deep-reasoning
        # tier. In a real system these would likely be learned jointly with
        # (or subsumed by) the embedding-based term rather than hand-set.
        self.heuristic_weight = nn.Parameter(torch.tensor([0.5, 1.5]), requires_grad=True)
        self.heuristic_bias = nn.Parameter(torch.tensor(-1.0), requires_grad=True)

    def extract_heuristic_features(self, ids):
        seq_len = ids.size(1)
        length_feature = torch.full((ids.size(0), 1), float(seq_len) / 32.0)  # normalized length
        hard_marker_count = (ids >= self.hard_marker_start).float().sum(dim=1, keepdim=True)
        return torch.cat([length_feature, hard_marker_count], dim=1)

    def forward(self, ids):
        pooled_embed = self.embed(ids).mean(dim=1)  # crude mean-pool of the query
        heuristic_features = self.extract_heuristic_features(ids)
        embed_logit = self.embed_to_logit(pooled_embed).squeeze(-1)
        heuristic_logit = (heuristic_features * self.heuristic_weight).sum(dim=1) + self.heuristic_bias
        routing_logit = embed_logit + heuristic_logit
        routing_prob = torch.sigmoid(routing_logit)
        route_to_deep = routing_prob >= self.threshold
        return routing_prob, route_to_deep


# ---------------------------------------------------------------------------
# The orchestrating system: router + two tiers + dispatch, with an explicit
# override path (mirrors GPT-5's confirmed dual-mode design: automatic
# routing by default, explicit effort/tier control available to a caller).
# ---------------------------------------------------------------------------

class RoutedReasoningSystem(nn.Module):
    def __init__(self, vocab_size, fast_config, deep_config, router_kwargs=None):
        super().__init__()
        self.fast_model = ToySubModel(vocab_size, **fast_config)
        self.deep_model = ToySubModel(vocab_size, **deep_config)
        self.router = ComplexityRouter(vocab_size, **(router_kwargs or {}))

    def forward(self, prompt_ids, force_tier=None, fast_tokens=6, deep_tokens=20):
        """force_tier: None -> automatic routing (default consumer-facing
        behavior). "fast" or "deep" -> explicit override (API-style manual
        control), bypassing the router entirely.
        """
        if force_tier == "fast":
            route_to_deep = torch.tensor([False])
            routing_prob = torch.tensor([0.0])
        elif force_tier == "deep":
            route_to_deep = torch.tensor([True])
            routing_prob = torch.tensor([1.0])
        else:
            routing_prob, route_to_deep = self.router(prompt_ids)

        if route_to_deep.item():
            output_ids = self.deep_model.generate(prompt_ids, num_new_tokens=deep_tokens)
            tier_used, tokens_spent = "deep_reasoning", deep_tokens
        else:
            output_ids = self.fast_model.generate(prompt_ids, num_new_tokens=fast_tokens)
            tier_used, tokens_spent = "fast", fast_tokens

        return {
            "output_ids": output_ids,
            "tier_used": tier_used,
            "tokens_spent": tokens_spent,
            "routing_prob": routing_prob.item(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 200
    hard_marker_start = 180  # ids in [180, 200) simulate "hard-query marker" tokens

    system = RoutedReasoningSystem(
        vocab_size=vocab_size,
        fast_config=dict(d_model=16, n_layers=1, n_heads=2, max_len=64),
        deep_config=dict(d_model=32, n_layers=3, n_heads=4, max_len=64),
        router_kwargs=dict(d_model=32, hard_marker_start=hard_marker_start, threshold=0.5),
    )

    fast_params = sum(p.numel() for p in system.fast_model.parameters())
    deep_params = sum(p.numel() for p in system.deep_model.parameters())
    router_params = sum(p.numel() for p in system.router.parameters())
    print("=== GPT-5-style toy routed system: fast tier + deep-reasoning tier + router ===")
    print(f"fast model params:   {fast_params:,}")
    print(f"deep model params:   {deep_params:,}")
    print(f"router params:       {router_params:,}")

    print("\n-- Automatic routing on a 'simple' query (no hard-query marker tokens) --")
    simple_query = torch.randint(0, hard_marker_start, (1, 5))
    result = system(simple_query, force_tier=None)
    print(f"query ids: {simple_query.tolist()[0]}")
    print(f"routing probability (deep): {result['routing_prob']:.3f} -> tier used: {result['tier_used']} "
          f"(tokens spent: {result['tokens_spent']})")

    print("\n-- Automatic routing on a 'complex' query (several hard-query marker tokens) --")
    complex_query = torch.cat([
        torch.randint(0, hard_marker_start, (1, 3)),
        torch.randint(hard_marker_start, vocab_size, (1, 6)),  # hard-query markers
    ], dim=1)
    result = system(complex_query, force_tier=None)
    print(f"query ids: {complex_query.tolist()[0]}")
    print(f"routing probability (deep): {result['routing_prob']:.3f} -> tier used: {result['tier_used']} "
          f"(tokens spent: {result['tokens_spent']})")

    print("\n-- Explicit override (API-style manual tier control, bypassing the router) --")
    for forced in ("fast", "deep"):
        result = system(simple_query, force_tier=forced)
        print(f"forced tier='{forced}': tier used={result['tier_used']} (tokens spent: {result['tokens_spent']})")

    print("\nNOTE: the router's features/architecture here are an illustrative design point,")
    print("not a description of GPT-5's actual (undisclosed) routing mechanism.")
