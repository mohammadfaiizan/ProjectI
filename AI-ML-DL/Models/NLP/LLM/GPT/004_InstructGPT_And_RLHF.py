"""
InstructGPT / RLHF (Ouyang et al., 2022) -- illustrative PyTorch mechanics.
============================================================================

This file deliberately does NOT implement "a GPT model". The base transformer
architecture in InstructGPT is unchanged from GPT-3 (see Section 2 of the
companion markdown doc, 004_InstructGPT_And_RLHF.md). What is genuinely new
in this paper is the POST-TRAINING PROCEDURE:

  1. A reward model (RM): a transformer backbone with its LM head replaced by
     a scalar head read off the last non-padding token's hidden state,
     trained with a Bradley-Terry pairwise preference loss on human-ranked
     completions.
  2. A PPO-style RL fine-tuning step that updates the SFT policy against the
     reward model, under a KL penalty back to a frozen SFT reference model
     (to bound reward-model overoptimization / reward hacking), using PPO's
     clipped surrogate objective and a value-function baseline.

This module demonstrates both mechanisms end to end in minimal, self-contained
form:

  - RewardModel:     backbone + scalar reward head, trained via
                      bradley_terry_loss().
  - PolicyModel:      the same backbone reused as an autoregressive LM,
                      plus a value head for a PPO baseline.
  - ppo_rlhf_loss():  a SIMPLIFIED, illustrative PPO-style update -- per-token
                      KL penalty against a frozen reference model, a
                      KL-shaped reward, a reward-minus-value advantage
                      (a simplification of full GAE(lambda)), and the PPO
                      clipped surrogate objective plus a value-function MSE
                      loss.

This is NOT a production RLHF trainer. There is no rollout/generation loop
sampling against a real environment, no multi-epoch minibatching over a
replay buffer of rollouts, no full Generalized Advantage Estimation, and no
distributed actor/learner split (see Section 4 of the markdown doc for what
the real infrastructure looks like). The goal here is to make the DISTINCTIVE
RLHF mechanisms -- the Bradley-Terry loss and the KL-penalized PPO update --
concrete and runnable, not to reproduce OpenAI's training system.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Minimal self-contained causal-transformer backbone.
#
# Both the policy/SFT model and the reward model are, per Section 2 of the
# markdown doc, "the same GPT-family architecture up to the output head" --
# so a single backbone class is reused by both PolicyModel and RewardModel
# below, and only the head differs (LM head + value head vs. scalar reward
# head). That head-swap is the actual architectural novelty of this paper.
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention with a causal
    mask (no cross-attention -- decoder-only, GPT-family style)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, d_model)
        key_padding_mask: (B, T) with 1 = real token, 0 = padding. Optional.
        """
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_heads, T, d_head)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, T, T)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if key_padding_mask is not None:
            # Mask out padded KEY positions so no query attends to padding.
            pad = (key_padding_mask == 0).view(B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(pad, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        out = attn_probs @ v  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm causal self-attention + position-wise FFN, GELU activation --
    the standard GPT-family decoder block (Section 2 of the markdown doc)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerBackbone(nn.Module):
    """Shared GPT-family decoder-only backbone, reused by both PolicyModel
    and RewardModel below. This class alone is architecturally identical to
    (a tiny version of) the GPT-3 backbone InstructGPT starts from -- there
    is no InstructGPT-specific change here."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Returns final hidden states, shape (B, T, d_model)."""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, key_padding_mask=attention_mask)
        return self.ln_f(x)


def last_token_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    """Gather the hidden state at each sequence's LAST NON-PADDING token.

    This is exactly the pooling operation the reward model's scalar head is
    applied to (Section 2 of the markdown doc): r_theta(x, y) is read off the
    transformer's hidden state at the final token of the (prompt, response)
    sequence, not a mean-pooled or first-token representation.
    """
    B, T, C = hidden_states.shape
    if attention_mask is None:
        last_idx = torch.full((B,), T - 1, device=hidden_states.device, dtype=torch.long)
    else:
        last_idx = attention_mask.sum(dim=1).long() - 1
        last_idx = last_idx.clamp(min=0)
    batch_idx = torch.arange(B, device=hidden_states.device)
    return hidden_states[batch_idx, last_idx]  # (B, C)


# ---------------------------------------------------------------------------
# 2. Reward Model + Bradley-Terry pairwise preference loss.
# ---------------------------------------------------------------------------


class RewardModel(nn.Module):
    """GPT-family backbone + scalar reward head.

    The LM head is replaced by a single linear projection to a scalar,
    applied to the hidden state at the last non-padding token of the
    (prompt, response) sequence -- see Section 2 of the markdown doc. This
    head-swap is the only architectural novelty InstructGPT introduces
    relative to GPT-3's own decoder-only transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.backbone = TransformerBackbone(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Returns a scalar reward per sequence, shape (B,)."""
        hidden = self.backbone(input_ids, attention_mask=attention_mask)
        pooled = last_token_hidden(hidden, attention_mask)
        return self.reward_head(pooled).squeeze(-1)


def bradley_terry_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """Pairwise preference loss under the Bradley-Terry choice model.

    The Bradley-Terry model posits that the probability a human prefers
    completion y_w ("chosen"/"winner") over y_l ("rejected"/"loser") for the
    same prompt x is a logistic function of the difference between the two
    completions' latent scalar reward scores:

        P(y_w > y_l | x) = sigmoid(r_theta(x, y_w) - r_theta(x, y_l))

    Training the reward model to maximize the log-likelihood of the observed
    human rankings under this model is equivalent to minimizing:

        L(theta) = -E[(x, y_w, y_l) ~ D] [ log( sigmoid(r_w - r_l) ) ]

    where r_w = r_theta(x, y_w), r_l = r_theta(x, y_l). This is implemented
    via F.logsigmoid rather than log(sigmoid(...)) for numerical stability:
    log(sigmoid(z)) underflows to -inf in naive float32 computation for
    very negative z, whereas F.logsigmoid is computed with a numerically
    stable formulation (softplus-based) that avoids this.

    Args:
        reward_chosen:   (B,) scalar reward model scores for the preferred completions.
        reward_rejected: (B,) scalar reward model scores for the dispreferred completions.

    Returns:
        Scalar loss (mean negative log-likelihood over the batch).
    """
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


# ---------------------------------------------------------------------------
# 3. Policy model (SFT-initialized RL policy) with LM head + value head.
# ---------------------------------------------------------------------------


class PolicyModel(nn.Module):
    """GPT-family backbone + LM head (the RL policy, initialized from an SFT
    checkpoint in the real pipeline) + a linear value head on the pooled
    final hidden state, used only as a PPO advantage baseline."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.backbone = TransformerBackbone(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.backbone.token_emb.weight  # weight tying, standard GPT convention
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Returns (logits, value):
            logits: (B, T, vocab_size) next-token distribution at each position.
            value:  (B,) scalar value-function baseline for the whole sequence,
                    read off the last non-padding token's hidden state (same
                    pooling convention as the reward model).
        """
        hidden = self.backbone(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)
        pooled = last_token_hidden(hidden, attention_mask)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value


def token_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """log pi(a_t | s_t) for a_t = input_ids[:, t], i.e. the log-probability
    the model assigns to each ACTUAL token in the sequence, under a
    teacher-forced forward pass. This is the standard operation needed to
    recompute rollout log-probs under the policy, the reference model, and
    (across PPO epochs) the updated policy again.

    Returns: (B, T) per-token log-probabilities.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# 4. Simplified PPO-style RLHF policy update.
# ---------------------------------------------------------------------------


def ppo_rlhf_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rm_reward: torch.Tensor,
    value_pred: torch.Tensor,
    attention_mask: torch.Tensor = None,
    beta: float = 0.02,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
):
    """Simplified, illustrative RLHF-PPO policy-update loss.

    This captures the MECHANISM described in Section 6 of the companion
    markdown doc, not a production PPO trainer. Real InstructGPT-style PPO
    differs in several ways this function deliberately simplifies:
      - Full Generalized Advantage Estimation (GAE(lambda)) over a per-token
        reward/value trajectory, instead of a single terminal reward and a
        single (reward - value) advantage per whole completion.
      - Multiple PPO epochs / minibatches replaying the same batch of
        rollouts, rather than a single gradient computation.
      - An additional PPO-ptx pretraining log-likelihood loss term mixed in
        to reduce the "alignment tax" (omitted here; see markdown Section 6).
      - In practice, the value function is sometimes a separate model rather
        than a head sharing the policy's backbone.

    Args:
        new_log_probs:  (B, T) log pi_theta(a_t|s_t) under the CURRENT policy
                         being optimized (grad-enabled).
        old_log_probs:  (B, T) log pi_theta_old(a_t|s_t) under the policy that
                         generated the rollout (detached / no grad).
        ref_log_probs:  (B, T) log pi_SFT(a_t|s_t) under the frozen SFT
                         reference model (detached / no grad).
        rm_reward:      (B,) scalar reward model score r_theta(x, y) for each
                         completed rollout (detached / no grad).
        value_pred:     (B,) value-function baseline prediction for each
                         rollout, from the CURRENT policy's value head
                         (grad-enabled, so it receives gradients from the
                         value loss below).
        attention_mask: (B, T) 1 for real completion tokens, 0 for padding.
        beta:           KL penalty coefficient.
        clip_eps:       PPO clip range (typically ~0.2).
        vf_coef:        weight on the value-function MSE loss term.

    Returns:
        dict with total_loss (grad-enabled) and detached scalars for logging:
        policy_loss, value_loss, kl_mean, shaped_reward_mean.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(new_log_probs)
    mask = attention_mask.float()
    token_counts = mask.sum(dim=1).clamp(min=1.0)  # (B,) real tokens per sequence

    # (a) Per-token KL penalty, approximated the standard practical way as a
    # log-probability difference between the rollout-generating policy and
    # the frozen reference, summed over the completion -> a per-sequence KL
    # estimate. We use old_log_probs (not new_log_probs) here because that is
    # the policy whose *sampling distribution* actually produced this
    # rollout and is what the reward was conceptually shaped against.
    per_token_kl = (old_log_probs - ref_log_probs) * mask
    kl_estimate = per_token_kl.sum(dim=1)  # (B,)

    # (b) KL-shaped reward: RM(x, y) minus beta * KL(policy || reference).
    # This is the term that keeps the policy from overoptimizing the reward
    # model (reward hacking / Goodhart's Law -- see markdown Section 6).
    shaped_reward = rm_reward - beta * kl_estimate  # (B,)

    # (c) Advantage estimate: reward-minus-value baseline.
    # NOTE: this is a simplification of full GAE(lambda), which would combine
    # a trajectory of per-token rewards and value estimates with a
    # bias/variance tradeoff parameter lambda. Here the whole completion is
    # treated as a single step with one terminal shaped reward and one value
    # estimate, which is enough to illustrate the mechanism but not what a
    # production PPO-for-RLHF trainer actually computes.
    advantage = (shaped_reward - value_pred).detach()  # (B,), no grad through the advantage
    advantage_per_token = advantage.unsqueeze(1).expand_as(new_log_probs)  # broadcast to tokens

    # (d) PPO clipped surrogate objective (Schulman et al., 2017), applied
    # per token and masked for padding:
    #   ratio_t(theta) = exp(new_log_prob_t - old_log_prob_t)
    #   L^CLIP = mean_t[ min(ratio_t * A_t, clip(ratio_t, 1-eps, 1+eps) * A_t) ]
    ratio = torch.exp(new_log_probs - old_log_probs)  # (B, T)
    unclipped = ratio * advantage_per_token
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage_per_token
    surrogate = torch.min(unclipped, clipped) * mask
    # We MAXIMIZE the clipped surrogate, i.e. MINIMIZE its negation.
    policy_loss = -(surrogate.sum(dim=1) / token_counts).mean()

    # Value-function regression toward the (detached) shaped-reward target.
    value_loss = F.mse_loss(value_pred, shaped_reward.detach())

    total_loss = policy_loss + vf_coef * value_loss

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "kl_mean": kl_estimate.mean().detach(),
        "shaped_reward_mean": shaped_reward.mean().detach(),
    }


# ---------------------------------------------------------------------------
# 5. Demo / sanity check.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    VOCAB_SIZE = 64
    D_MODEL = 32
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 64
    MAX_SEQ_LEN = 20
    BATCH = 8
    SEQ_LEN = 12

    # --- Instantiate policy (SFT-initialized RL policy), a frozen reference
    # model (a copy simulating the SFT snapshot), and a separate reward model.
    policy_model = PolicyModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN)
    reference_model = copy.deepcopy(policy_model)  # simulates: frozen SFT snapshot
    for p in reference_model.parameters():
        p.requires_grad_(False)
    reference_model.eval()

    reward_model = RewardModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN)

    def count_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    print("=== InstructGPT / RLHF illustrative mechanics ===")
    print(f"Policy model parameters:    {count_params(policy_model):,}")
    print(f"Reference model parameters: {count_params(reference_model):,} (frozen copy of policy)")
    print(f"Reward model parameters:    {count_params(reward_model):,}")
    print()

    # ------------------------------------------------------------------
    # (i) Reward model training step: Bradley-Terry pairwise preference loss.
    # ------------------------------------------------------------------
    chosen_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    rejected_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    pref_mask = torch.ones(BATCH, SEQ_LEN)

    r_chosen = reward_model(chosen_ids, attention_mask=pref_mask)
    r_rejected = reward_model(rejected_ids, attention_mask=pref_mask)
    rm_loss = bradley_terry_loss(r_chosen, r_rejected)

    print("[Reward Model] Bradley-Terry pairwise preference loss")
    print(f"  loss = {rm_loss.item():.4f}")
    print(f"  mean r_chosen   = {r_chosen.mean().item():.4f}")
    print(f"  mean r_rejected = {r_rejected.mean().item():.4f}")
    print()

    rm_loss.backward()  # sanity check: gradients flow into the reward model
    rm_grad_norm = sum(
        p.grad.norm() ** 2 for p in reward_model.parameters() if p.grad is not None
    ) ** 0.5
    print(f"  reward-model gradient norm after backward(): {rm_grad_norm.item():.4f}")
    print()

    # ------------------------------------------------------------------
    # (ii) Simplified PPO-style policy update.
    # ------------------------------------------------------------------
    # Stand-in "rollouts": in a real pipeline these are sampled autoregressively
    # from the policy against real prompts (temperature/top-p decoding). Here
    # we substitute random token sequences purely to exercise the loss
    # mechanics end to end.
    rollout_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    rollout_mask = torch.ones(BATCH, SEQ_LEN)

    # old_log_probs / old value: recorded at ROLLOUT-COLLECTION time, before
    # any gradient step in this PPO iteration -- hence no_grad and detached.
    with torch.no_grad():
        old_logits, _old_value = policy_model(rollout_ids, attention_mask=rollout_mask)
        old_log_probs = token_log_probs(old_logits, rollout_ids)

        ref_logits, _ = reference_model(rollout_ids, attention_mask=rollout_mask)
        ref_log_probs = token_log_probs(ref_logits, rollout_ids)

        rm_reward = reward_model(rollout_ids, attention_mask=rollout_mask)

    # new_log_probs / value_pred: recomputed WITH gradients enabled -- this is
    # what the PPO update actually backpropagates through. On the very first
    # minibatch of the very first PPO epoch these are close to old_log_probs
    # (ratio ~= 1), which is expected, correct behavior, not a bug.
    new_logits, value_pred = policy_model(rollout_ids, attention_mask=rollout_mask)
    new_log_probs = token_log_probs(new_logits, rollout_ids)

    ppo_out = ppo_rlhf_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        rm_reward=rm_reward,
        value_pred=value_pred,
        attention_mask=rollout_mask,
        beta=0.02,
        clip_eps=0.2,
        vf_coef=0.5,
    )

    print("[PPO update] simplified RLHF policy-update components")
    print(f"  policy (clipped surrogate) loss: {ppo_out['policy_loss'].item():.4f}")
    print(f"  value-function MSE loss:         {ppo_out['value_loss'].item():.4f}")
    print(f"  mean per-sequence KL(policy||ref) estimate: {ppo_out['kl_mean'].item():.4f}")
    print(f"  mean KL-shaped reward:            {ppo_out['shaped_reward_mean'].item():.4f}")
    print(f"  total loss (policy + vf_coef * value): {ppo_out['total_loss'].item():.4f}")

    ppo_out["total_loss"].backward()
    policy_grad_norm = sum(
        p.grad.norm() ** 2 for p in policy_model.parameters() if p.grad is not None
    ) ** 0.5
    print(f"  policy-model gradient norm after backward(): {policy_grad_norm.item():.4f}")

    ref_grads = [p.grad for p in reference_model.parameters() if p.grad is not None]
    print(f"  reference-model gradients present: {len(ref_grads) > 0} (expected False -- frozen)")
