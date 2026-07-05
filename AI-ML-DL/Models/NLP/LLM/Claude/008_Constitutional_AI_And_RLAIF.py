"""
008_Constitutional_AI_And_RLAIF.py

A runnable, toy-scale PyTorch implementation of the two-phase Constitutional
AI (CAI) pipeline from Bai et al., 2022, "Constitutional AI: Harmlessness
from AI Feedback":

  PHASE 1 (SL-CAI): a policy model generates a response, an AI "feedback
  model" critiques that response against a sampled principle from an
  explicit written constitution, the response is revised in light of the
  critique, and the policy is then supervised-fine-tuned (real
  cross-entropy gradient steps) on these self-revised (prompt, response)
  pairs.

  PHASE 2 (RL-CAI / RLAIF): the fine-tuned policy generates pairs of
  candidate responses; instead of a human choosing the better one, the same
  "AI evaluates against a principle" logic is used to generate a preference
  label; these AI-generated preference pairs train a Bradley-Terry reward
  model via real gradient descent; the reward model is then used exactly as
  an RLHF reward model would be, to drive a simplified PPO-style clipped
  policy-gradient update of the policy.

Because no real pretrained language model is assumed, "critique" and
"revise" are implemented behind a `ConstitutionalFeedbackModel` abstraction
whose interface mirrors what a real LLM-backed implementation would expose
(prompt the model to critique text against a principle; prompt it to revise;
prompt it to judge a pair). The toy backend implemented here is a
deterministic function of a small synthetic token vocabulary (some tokens
are tagged as "harmful", some as "safe/explanatory", some as "bare
refusal") purely so the entire pipeline -- critique, revision, preference
labeling, reward-model training, and policy fine-tuning -- is genuinely
runnable and its effects genuinely measurable, without requiring an actual
pretrained LLM. Swapping `ConstitutionalFeedbackModel` for a class that
calls a real LLM API would not require changing any other part of this
file: that is the point of exposing it as an abstraction.

None of the token semantics, model sizes, or training hyperparameters below
are disclosed facts about any Anthropic model or training run; they exist
solely to make the mechanics of the CAI pipeline itself concrete and
inspectable.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Toy vocabulary and constitution.
# --------------------------------------------------------------------------- #

VOCAB_SIZE = 40
PAD_ID = 0
BOS_ID = 1

HARMFUL_TOKENS = set(range(3, 9))        # 6 tokens standing in for harmful content
SAFE_EXPLAIN_TOKENS = set(range(9, 15))  # 6 tokens standing in for helpful, harm-aware explanation
BLUNT_REFUSAL_TOKENS = set(range(15, 18))  # 3 tokens standing in for a bare, unexplained refusal
NEUTRAL_TOKENS = list(range(18, VOCAB_SIZE))  # generic filler content

PROMPT_LEN = 6
RESPONSE_LEN = 10


def _frac(tokens: Sequence[int], subset: set) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in subset) / len(tokens)


@dataclass
class Principle:
    """One clause of the written constitution: a name, a human-readable
    description, and a violation-scoring function (higher = worse) that
    the AI feedback model consults when critiquing, revising, or judging
    preferences. In a real system this scoring is done by prompting an
    LLM with the principle's text; here it is a direct function of the toy
    token vocabulary so violations are unambiguous and the pipeline's
    effect on them is directly measurable."""

    name: str
    description: str
    violation_fn: Callable[[Sequence[int]], float]
    weight: float = 1.0


CONSTITUTION: List[Principle] = [
    Principle(
        name="avoid_harm",
        description="The response must not include harmful content.",
        violation_fn=lambda toks: _frac(toks, HARMFUL_TOKENS),
        weight=2.0,
    ),
    Principle(
        name="non_evasive",
        description="The response must not be a bare refusal; if declining, it should explain why.",
        violation_fn=lambda toks: 1.0
        if (_frac(toks, BLUNT_REFUSAL_TOKENS) > 0 and _frac(toks, SAFE_EXPLAIN_TOKENS) == 0)
        else 0.0,
        weight=1.0,
    ),
    Principle(
        name="be_helpful",
        description="The response should contain substantive, explanatory content.",
        violation_fn=lambda toks: max(0.0, 0.3 - _frac(toks, SAFE_EXPLAIN_TOKENS)),
        weight=1.0,
    ),
]


def total_violation(tokens: Sequence[int]) -> float:
    """Aggregate constitutional violation score across all principles --
    used only for reporting/inspection in this file, never directly as a
    training signal (that would defeat the point of the pipeline, which
    trains only via self-critique/revision and AI-generated preferences)."""
    return sum(p.weight * p.violation_fn(tokens) for p in CONSTITUTION)


# --------------------------------------------------------------------------- #
# Minimal decoder-only transformer, used both as the POLICY model and (in a
# second instantiation) as the backbone for the REWARD model.
# --------------------------------------------------------------------------- #


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PolicyLM(nn.Module):
    """The generative model undergoing Constitutional AI training. Its
    initial weights are deliberately biased toward occasionally producing
    HARMFUL_TOKENS, standing in for an already-helpful-but-not-yet-harmless
    base model -- the documented starting point for the SL-CAI phase in the
    original paper."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 3, d_ff: int = 256):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(PROMPT_LEN + RESPONSE_LEN + 8, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        self._bias_toward_harm()

    def _bias_toward_harm(self) -> None:
        with torch.no_grad():
            for t in HARMFUL_TOKENS:
                self.head.bias[t] += 2.0  # simulates an unaligned baseline tendency

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class RewardModel(nn.Module):
    """Bradley-Terry-style scalar reward model: encodes a full
    (prompt + response) token sequence and outputs a single scalar score.
    Trained purely from AI-generated preference pairs (Phase 2), exactly as
    a conventional RLHF reward model would be trained from human preference
    pairs -- only the label source differs."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, d_ff: int = 256):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(PROMPT_LEN + RESPONSE_LEN + 8, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        last_hidden = x[:, -1, :]  # causal model: last position has seen the whole sequence
        return self.reward_head(last_hidden).squeeze(-1)  # (B,)


# --------------------------------------------------------------------------- #
# Sampling / log-prob utilities shared by SFT and RL.
# --------------------------------------------------------------------------- #


@torch.no_grad()
def sample_response(policy: PolicyLM, prompt_ids: torch.Tensor, length: int, temperature: float = 1.0) -> torch.Tensor:
    idx = prompt_ids.clone()
    for _ in range(length):
        logits = policy(idx)[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx[:, prompt_ids.shape[1]:]


def sequence_log_prob(policy: PolicyLM, full_ids: torch.Tensor) -> torch.Tensor:
    """Sum log-probability the given policy assigns to the RESPONSE portion
    of `full_ids` (the prompt portion is treated as fixed context, not
    generated), via teacher-forced next-token prediction."""
    logits = policy(full_ids[:, :-1])
    response_logits = logits[:, PROMPT_LEN - 1: PROMPT_LEN + RESPONSE_LEN - 1, :]
    targets = full_ids[:, PROMPT_LEN: PROMPT_LEN + RESPONSE_LEN]
    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=1)


def random_prompts(n: int, rng: random.Random) -> torch.Tensor:
    rows = [[BOS_ID] + [rng.choice(NEUTRAL_TOKENS) for _ in range(PROMPT_LEN - 1)] for _ in range(n)]
    return torch.tensor(rows, dtype=torch.long)


# --------------------------------------------------------------------------- #
# The "language model call" abstraction: critique, revise, and preference
# judgment. In production this class would issue actual prompted calls to
# an LLM (the same policy model, or a separate feedback model); here it is
# implemented deterministically over the toy vocabulary, but every method
# signature is written the way an LLM-call wrapper's would be.
# --------------------------------------------------------------------------- #


class ConstitutionalFeedbackModel:
    def critique(self, response: List[int], principle: Principle) -> str:
        score = principle.violation_fn(response)
        flagged = [i for i, t in enumerate(response) if t in HARMFUL_TOKENS] if principle.name == "avoid_harm" else []
        return (
            f"[critique vs '{principle.name}'] violation={score:.2f}"
            + (f" flagged_positions={flagged}" if flagged else "")
        )

    def revise(self, response: List[int], critique_note: str, principle: Principle, rng: random.Random) -> List[int]:
        revised = list(response)
        if principle.name == "avoid_harm":
            for i, t in enumerate(revised):
                if t in HARMFUL_TOKENS:
                    revised[i] = rng.choice(list(SAFE_EXPLAIN_TOKENS))
        elif principle.name == "non_evasive" and principle.violation_fn(revised) > 0:
            revised = revised + [rng.choice(list(SAFE_EXPLAIN_TOKENS)) for _ in range(2)]
        elif principle.name == "be_helpful" and principle.violation_fn(revised) > 0:
            revised = revised + [rng.choice(list(SAFE_EXPLAIN_TOKENS))]
        revised = (revised + [PAD_ID] * RESPONSE_LEN)[:RESPONSE_LEN]
        return revised

    def judge_preference(self, resp_a: List[int], resp_b: List[int], principle: Principle) -> Tuple[str, str]:
        score_a, score_b = principle.violation_fn(resp_a), principle.violation_fn(resp_b)
        if score_a == score_b:
            score_a, score_b = total_violation(resp_a), total_violation(resp_b)
        preferred = "a" if score_a <= score_b else "b"
        rationale = (
            f"[AI preference vs '{principle.name}'] violation(a)={score_a:.2f} "
            f"violation(b)={score_b:.2f} -> preferred={preferred}"
        )
        return preferred, rationale


# --------------------------------------------------------------------------- #
# PHASE 1: SL-CAI -- generate, critique, revise, then supervised fine-tune.
# --------------------------------------------------------------------------- #


def run_sl_cai_phase(
    policy: PolicyLM,
    feedback_model: ConstitutionalFeedbackModel,
    prompts: torch.Tensor,
    constitution: List[Principle],
    rng: random.Random,
) -> List[Tuple[List[int], List[int]]]:
    dataset = []
    for i in range(prompts.shape[0]):
        prompt_row = prompts[i: i + 1]
        response = sample_response(policy, prompt_row, RESPONSE_LEN, temperature=1.0)[0].tolist()
        for principle in constitution:  # one critique-revise round per constitutional principle
            critique_note = feedback_model.critique(response, principle)
            response = feedback_model.revise(response, critique_note, principle, rng)
        dataset.append((prompt_row[0].tolist(), response))
    return dataset


def supervised_finetune(policy: PolicyLM, dataset: List[Tuple[List[int], List[int]]], epochs: int, lr: float) -> List[float]:
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    prompts = torch.tensor([p for p, _ in dataset], dtype=torch.long)
    responses = torch.tensor([r for _, r in dataset], dtype=torch.long)
    full = torch.cat([prompts, responses], dim=1)

    losses = []
    for _ in range(epochs):
        logits = policy(full[:, :-1])
        response_logits = logits[:, PROMPT_LEN - 1: PROMPT_LEN + RESPONSE_LEN - 1, :]
        targets = full[:, PROMPT_LEN: PROMPT_LEN + RESPONSE_LEN]
        loss = F.cross_entropy(response_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# --------------------------------------------------------------------------- #
# PHASE 2: RL-CAI / RLAIF -- AI preference labels, Bradley-Terry reward
# model training, then a PPO-style policy update using that reward model.
# --------------------------------------------------------------------------- #


@dataclass
class PreferencePair:
    prompt: List[int]
    chosen: List[int]
    rejected: List[int]
    principle: str
    rationale: str


def generate_preference_pairs(
    policy: PolicyLM,
    feedback_model: ConstitutionalFeedbackModel,
    prompts: torch.Tensor,
    constitution: List[Principle],
    rng: random.Random,
) -> List[PreferencePair]:
    pairs = []
    for i in range(prompts.shape[0]):
        prompt_row = prompts[i: i + 1]
        resp_a = sample_response(policy, prompt_row, RESPONSE_LEN, temperature=1.6)[0].tolist()
        resp_b = sample_response(policy, prompt_row, RESPONSE_LEN, temperature=1.6)[0].tolist()
        principle = rng.choice(constitution)
        preferred, rationale = feedback_model.judge_preference(resp_a, resp_b, principle)
        chosen, rejected = (resp_a, resp_b) if preferred == "a" else (resp_b, resp_a)
        pairs.append(PreferencePair(prompt_row[0].tolist(), chosen, rejected, principle.name, rationale))
    return pairs


def train_reward_model(reward_model: RewardModel, pairs: List[PreferencePair], epochs: int, lr: float) -> List[Tuple[float, float]]:
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
    prompts = torch.tensor([p.prompt for p in pairs], dtype=torch.long)
    chosen = torch.tensor([p.chosen for p in pairs], dtype=torch.long)
    rejected = torch.tensor([p.rejected for p in pairs], dtype=torch.long)
    full_chosen = torch.cat([prompts, chosen], dim=1)
    full_rejected = torch.cat([prompts, rejected], dim=1)

    history = []
    for _ in range(epochs):
        r_chosen = reward_model(full_chosen)
        r_rejected = reward_model(full_rejected)
        # Bradley-Terry pairwise preference loss -- identical in form to
        # standard RLHF reward-model training; only the labels' provenance
        # (AI feedback vs. human feedback) differs.
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (r_chosen > r_rejected).float().mean().item()
        history.append((loss.item(), acc))
    return history


def ppo_style_update(
    policy: PolicyLM,
    reward_model: RewardModel,
    prompts: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    clip_eps: float = 0.2,
) -> Tuple[float, float]:
    """One simplified PPO-clipped-surrogate update: sample responses from a
    frozen snapshot of the current policy, score them with the reward
    model, then update the live policy to increase the probability of
    above-baseline-reward responses (and decrease below-baseline ones),
    clipping the importance-sampling ratio for stability -- the same basic
    pattern RLHF uses, just with a reward model trained on AI-generated
    rather than human-generated preference labels."""
    old_policy = copy.deepcopy(policy)
    old_policy.eval()

    with torch.no_grad():
        responses = sample_response(old_policy, prompts, RESPONSE_LEN, temperature=1.0)
        full = torch.cat([prompts, responses], dim=1)
        old_logp = sequence_log_prob(old_policy, full)
        rewards = reward_model(full)
        baseline = rewards.mean()
        advantage = rewards - baseline

    new_logp = sequence_log_prob(policy, full)
    ratio = torch.exp(new_logp - old_logp)
    surrogate1 = ratio * advantage
    surrogate2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    loss = -torch.min(surrogate1, surrogate2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), rewards.mean().item()


# --------------------------------------------------------------------------- #
# Demonstration.
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    torch.manual_seed(0)
    rng = random.Random(0)

    policy = PolicyLM()
    feedback_model = ConstitutionalFeedbackModel()

    train_prompts = random_prompts(48, rng)
    eval_prompts = random_prompts(16, rng)

    def mean_violation(model: PolicyLM, prompts: torch.Tensor, temperature: float = 1.0) -> float:
        with torch.no_grad():
            responses = sample_response(model, prompts, RESPONSE_LEN, temperature=temperature)
        return sum(total_violation(r.tolist()) for r in responses) / responses.shape[0]

    print("=== PHASE 0: base (helpful-only, not yet constitutionally trained) policy ===")
    base_violation = mean_violation(policy, eval_prompts)
    print(f"Mean constitutional violation score (lower is better): {base_violation:.3f}\n")

    print("=== PHASE 1: SL-CAI -- generate -> critique -> revise -> supervised fine-tune ===")
    sl_cai_dataset = run_sl_cai_phase(policy, feedback_model, train_prompts, CONSTITUTION, rng)
    print(f"Built {len(sl_cai_dataset)} self-revised (prompt, response) training pairs.")
    print("Example critique/revise trace for one training prompt:")
    example_prompt = train_prompts[0:1]
    example_response = sample_response(policy, example_prompt, RESPONSE_LEN, temperature=1.0)[0].tolist()
    print(f"  initial response tokens: {example_response}")
    for principle in CONSTITUTION:
        note = feedback_model.critique(example_response, principle)
        example_response = feedback_model.revise(example_response, note, principle, rng)
        print(f"  {note} -> revised: {example_response}")

    sft_losses = supervised_finetune(policy, sl_cai_dataset, epochs=60, lr=3e-3)
    print(f"\nSFT loss: start={sft_losses[0]:.4f} end={sft_losses[-1]:.4f}")

    post_sl_cai_violation = mean_violation(policy, eval_prompts)
    print(f"Mean constitutional violation after SL-CAI (lower is better): {post_sl_cai_violation:.3f}\n")

    print("=== PHASE 2: RL-CAI / RLAIF -- AI preference labels -> reward model -> PPO-style update ===")
    preference_pairs = generate_preference_pairs(policy, feedback_model, train_prompts, CONSTITUTION, rng)
    print(f"Generated {len(preference_pairs)} AI-labeled preference pairs. Example rationale:")
    print(f"  {preference_pairs[0].rationale}")

    reward_model = RewardModel()
    rm_history = train_reward_model(reward_model, preference_pairs, epochs=150, lr=3e-3)
    print(
        f"\nReward model Bradley-Terry loss: start={rm_history[0][0]:.4f} end={rm_history[-1][0]:.4f}; "
        f"pairwise accuracy: start={rm_history[0][1]:.2f} end={rm_history[-1][1]:.2f}"
    )

    rl_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    print("\nRunning PPO-style updates against the AI-feedback-trained reward model:")
    for step in range(10):
        loss, mean_reward = ppo_style_update(policy, reward_model, train_prompts, rl_optimizer)
        print(f"  step {step:02d}: policy_loss={loss:+.4f} mean_reward={mean_reward:+.4f}")

    final_violation = mean_violation(policy, eval_prompts)
    print(
        f"\nMean constitutional violation: base={base_violation:.3f} -> "
        f"after SL-CAI={post_sl_cai_violation:.3f} -> after RL-CAI={final_violation:.3f}"
    )
    print(
        "Note: mean_reward under the learned reward model climbs steadily across "
        "PPO steps even in runs where the true constitutional-violation metric "
        "wobbles slightly rather than monotonically improving further from its "
        "already-low post-SL-CAI level -- a small, concrete instance of the "
        "general reward-model-imperfection/over-optimization risk: the policy is "
        "optimizing the learned proxy (the reward model), not the true target "
        "(the constitution) directly, and the two are not guaranteed to move in "
        "lockstep once the easy violations have already been eliminated."
    )
    print(
        "\nThroughout this pipeline, no human ever labeled a preference pair or "
        "wrote a per-response critique -- both were produced by the "
        "'ConstitutionalFeedbackModel' consulting the explicit, written "
        "CONSTITUTION list. The only human input was authoring that list of "
        "principles, which is exactly the auditability trade-off file 008's "
        "markdown discusses: the training target is inspectable text, not an "
        "implicit human-preference dataset."
    )
