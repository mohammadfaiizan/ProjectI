"""
008_O1_O3_Reasoning_Models.py -- Educational reconstruction of the TEST-TIME
COMPUTE mechanism associated with OpenAI's o1/o3 reasoning models (2024-2025).

SCOPE AND HONESTY ABOUT WHAT THIS FILE IS AND IS NOT:

OpenAI has never disclosed o1/o3's architecture, and critically, the actual
RL training loop that produces o1/o3's reasoning behavior cannot be
faithfully reproduced here -- it requires real verifier infrastructure
(sandboxed code execution, symbolic math checking), large-scale rollout
generation, and a real policy-gradient training run against a real base
model, none of which is something a single educational script can do
meaningfully.

What THIS FILE demonstrates instead is the *inference-time mechanism* that
test-time compute scaling is actually built on top of, once an RL-trained
reasoning policy already exists: generate MULTIPLE candidate reasoning
continuations for the same prompt, SCORE each candidate with some value
signal, and SELECT the best one (or aggregate across them) before returning
a final answer. This is the "best-of-N at inference time" idea -- the
simplest concrete instance of "spend more inference compute to get a better
answer," which is the empirical phenomenon OpenAI's o1 announcement plots
(accuracy vs. reasoning-token / inference-compute budget).

WHAT IS TOY/STUBBED HERE, EXPLICITLY:
  - The "language model" generating reasoning continuations is a tiny random
    transformer, not a real trained reasoning model. Its outputs are
    linguistically meaningless; only the CONTROL FLOW (generate N candidates,
    score, select) is the point.
  - The "value model" that scores candidates is a small learned head trained
    on nothing (random-initialized), standing in conceptually for a real
    process/outcome reward model. In a real o1/o3-like system this would be
    (a) a learned reward/value model trained on verifier signal, and/or
    (b) a hard verifier itself (e.g., "does this candidate's final numeric
    answer match the known solution", "does this code pass the test suite").
    A toy rule-based verifier variant is included below for exactly that
    reason -- it is closer in spirit to the real RLVR reward signal than a
    learned value head is, precisely because it is a hard, checkable rule.

HOW THE REAL RL TRAINING LOOP (RLVR) WOULD DIFFER FROM STANDARD RLHF
(conceptual description only -- NOT implemented below, since it requires
real infrastructure this script cannot provide):

  Standard RLHF fine-tuning loop (GPT-3.5/4-era instruction tuning):
    1. Sample a prompt.
    2. Policy generates ONE full response.
    3. A learned reward model (trained on human pairwise preference labels
       over full responses) scores that single response.
    4. PPO update nudges the policy to raise the probability of
       higher-reward-model-scoring responses. The reward model is a proxy
       for subjective human preference; there is no ground-truth checker.

  RLVR loop associated with o1/o3 (as OpenAI describes it, at a conceptual
  level -- exact algorithm undisclosed):
    1. Sample a prompt FROM A VERIFIABLE DOMAIN (a math problem with a known
       answer, a coding problem with held-out tests).
    2. Policy generates MANY long reasoning rollouts for that SAME prompt
       (this is the inference-heavy part: each rollout can be thousands of
       tokens of intermediate reasoning before a final answer).
    3. Each rollout's FINAL ANSWER is checked by an automatic verifier
       (exact-match against the known solution, or executing generated
       code against tests) -- reward is objective/near-ground-truth, not a
       learned proxy for subjective preference.
    4. The reward for a correct-outcome rollout is propagated back across
       the reasoning trajectory that produced it (the specific credit-
       assignment mechanism -- e.g. treating the whole trajectory as one
       action for a trajectory-level reward, versus finer per-step credit
       assignment via a learned process reward model -- is not disclosed
       by OpenAI for o1/o3 specifically).
    5. Repeated at scale, over many verifiable-domain prompts, this
       optimizes the policy toward reasoning PROCESSES that tend to arrive
       at verifiably correct answers -- not toward outputs that merely look
       preferable to a human rater.

  The practical infrastructure delta: RLHF's dominant cost is pretraining;
  its RL loop is comparatively cheap (one rollout per prompt). RLVR's RL
  loop is itself inference-heavy (many long rollouts per prompt) and
  requires real verifier execution infrastructure (sandboxes, symbolic
  checkers) running at training-loop scale -- a materially different
  systems problem, discussed in the companion .md file, Section 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Toy autoregressive "reasoning" generator (stands in for a trained
# reasoning-RL policy; here it is just an untrained random transformer --
# only the sampling/branching control flow matters for this demo).
# ---------------------------------------------------------------------------

class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_heads=4, n_layers=2, max_len=128):
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
    def generate(self, prompt_ids, num_new_tokens, temperature=1.0):
        ids = prompt_ids.clone()
        for _ in range(num_new_tokens):
            logits = self.forward(ids)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


# ---------------------------------------------------------------------------
# Scoring mechanism 1: a learned toy "value model" (process/outcome-reward
# stand-in). In a real system this would be trained on verifier-derived
# labels; here it is a random-initialized head purely to demonstrate the
# scoring-and-selection control flow of best-of-N.
# ---------------------------------------------------------------------------

class ToyValueModel(nn.Module):
    """Scores a full (prompt + reasoning + answer) token sequence with a
    single scalar, standing in for a learned reward/value model that in a
    real RLVR system would be trained against verifier outcomes."""

    def __init__(self, vocab_size, d_model=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pool_proj = nn.Linear(d_model, d_model)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, ids):
        x = self.embed(ids)
        pooled = x.mean(dim=1)  # crude mean-pool "read the whole trajectory"
        pooled = torch.tanh(self.pool_proj(pooled))
        return self.value_head(pooled).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# Scoring mechanism 2: a hard, rule-based verifier -- closer in spirit to
# the real RLVR reward signal (exact-match / test-execution) than a learned
# value model is, precisely because it is a ground-truth check rather than
# a learned proxy. Here: does the candidate's generated sequence contain a
# specific target sub-pattern (standing in for "matches the known answer").
# ---------------------------------------------------------------------------

def rule_based_verifier(candidate_ids, target_token_id):
    """Toy verifiable reward: 1.0 if the target token id appears anywhere
    in the generated continuation, else 0.0. A real verifier would check a
    math answer against a ground-truth solution or execute code against
    tests -- this is a maximally simplified stand-in for "objectively
    checkable correctness," not a claim about how o1/o3's verifiers work.
    """
    return (candidate_ids == target_token_id).any(dim=1).float()


# ---------------------------------------------------------------------------
# The test-time-compute mechanism itself: best-of-N generation + selection.
# This is the part of the file that concretely demonstrates "spend more
# inference compute -> better answer," the empirical claim behind test-time
# scaling for o1/o3-style reasoning models.
# ---------------------------------------------------------------------------

def best_of_n_with_value_model(policy, value_model, prompt_ids, n_candidates, reasoning_tokens, temperature=1.0):
    """Generate N independent reasoning+answer rollouts from the same
    prompt, score each with the value model, return the highest-scoring
    rollout. This mirrors "high reasoning_effort" behavior conceptually:
    more candidates / more reasoning tokens per candidate = more inference
    compute spent = (empirically, in real reasoning models) higher expected
    answer quality, at proportionally higher latency and cost.
    """
    batch_prompt = prompt_ids.repeat(n_candidates, 1)
    candidates = policy.generate(batch_prompt, num_new_tokens=reasoning_tokens, temperature=temperature)
    scores = value_model(candidates)
    best_idx = scores.argmax().item()
    return candidates[best_idx : best_idx + 1], scores, best_idx


def best_of_n_with_verifier(policy, prompt_ids, target_token_id, n_candidates, reasoning_tokens, temperature=1.0):
    """Same generate-many-score-select pattern, but using a hard verifier
    (rule_based_verifier) instead of a learned value model -- the more
    RLVR-faithful scoring signal. If multiple candidates verify as correct,
    the first one found is returned (a real system might instead take a
    majority vote across verified-correct candidates, i.e. self-consistency
    decoding, another well-studied test-time-compute technique)."""
    batch_prompt = prompt_ids.repeat(n_candidates, 1)
    candidates = policy.generate(batch_prompt, num_new_tokens=reasoning_tokens, temperature=temperature)
    verified = rule_based_verifier(candidates, target_token_id)
    if verified.sum() > 0:
        chosen_idx = verified.nonzero()[0].item()
    else:
        chosen_idx = 0  # no verified-correct candidate found; fall back
    return candidates[chosen_idx : chosen_idx + 1], verified, chosen_idx


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 200
    policy = TinyCausalLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2, max_len=256)
    value_model = ToyValueModel(vocab_size=vocab_size, d_model=32)

    prompt_ids = torch.randint(0, vocab_size, (1, 6))
    print("=== o1/o3-style toy test-time-compute demo: best-of-N reasoning selection ===")
    print(f"prompt token ids: {prompt_ids.tolist()[0]}")

    print("\n-- Effect of scaling N (candidates) at fixed reasoning length --")
    for n in (1, 4, 16):
        best_seq, scores, best_idx = best_of_n_with_value_model(
            policy, value_model, prompt_ids, n_candidates=n, reasoning_tokens=10, temperature=1.0,
        )
        print(f"N={n:>2} candidates | best value-model score={scores.max().item():+.3f} "
              f"| chosen candidate idx={best_idx} | total reasoning tokens spent={n * 10}")

    print("\n-- Effect of scaling reasoning-token budget at fixed N (analogous to reasoning_effort) --")
    for reasoning_tokens in (4, 16, 32):
        best_seq, scores, best_idx = best_of_n_with_value_model(
            policy, value_model, prompt_ids, n_candidates=4, reasoning_tokens=reasoning_tokens, temperature=1.0,
        )
        print(f"reasoning_tokens={reasoning_tokens:>2} | best value-model score={scores.max().item():+.3f} "
              f"| total reasoning tokens spent={4 * reasoning_tokens}")

    print("\n-- Rule-based verifier selection (RLVR-style hard reward, not a learned proxy) --")
    target_token_id = 42
    best_seq, verified, chosen_idx = best_of_n_with_verifier(
        policy, prompt_ids, target_token_id=target_token_id, n_candidates=8, reasoning_tokens=20,
    )
    print(f"target token to verify against: {target_token_id}")
    print(f"per-candidate verified (1.0=correct, 0.0=incorrect): {verified.tolist()}")
    print(f"selected candidate idx: {chosen_idx} | any candidate verified correct: {bool(verified.sum() > 0)}")

    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value_model.parameters())
    print(f"\ntoy policy params: {policy_params:,}  |  toy value-model params: {value_params:,}")
    print("\nNOTE: this script demonstrates the INFERENCE-TIME best-of-N mechanism only.")
    print("The RL training loop (RLVR) that would produce a policy actually good at this")
    print("is described conceptually in the module docstring above, not implemented here.")
