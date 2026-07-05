"""
DeepSeek-R1 (2025) -- GRPO (Group Relative Policy Optimization) core mechanic,
plus a toy RLVR (RL with Verifiable Rewards) task and a toy SFT-distillation
demo.

This file demonstrates the mechanism that distinguishes GRPO from PPO: instead
of training a separate critic/value network to produce a baseline for the
advantage estimate, GRPO samples a GROUP of candidate outputs for the SAME
prompt, scores each with a verifiable (exactly checkable) reward, and uses the
group's own mean reward as the baseline. The advantage for each sample is its
reward minus the group mean, normalized by the group's standard deviation.
This requires no critic network, no critic optimizer, and no separate value
training dynamics.

The RLVR task here is deliberately toy but structurally faithful to the real
one: a "policy" must generate a sequence of digit-tokens that, when summed,
equals a target value hidden in the prompt. The reward is an exact,
mechanically-checkable function of the output (sum matches target) -- no
learned reward model anywhere in the loop, which is the entire point of RLVR
versus RLHF.

Also included: a toy distillation demo showing that once an RL-trained
"teacher" policy produces reasoning traces, a smaller "student" model can
absorb the resulting behavior via ordinary supervised fine-tuning (cross-
entropy against the teacher's outputs) -- no RL required for the student.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Toy RLVR task: policy must emit a sequence of "digit" tokens summing to a
# target encoded in the prompt embedding. Reward is EXACT-MATCH (verifiable),
# not a learned reward model.
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    vocab_size: int = 10        # digits 0-9
    seq_len: int = 4            # output length (4 digits per completion)
    d_model: int = 64
    group_size: int = 8         # G: number of sampled completions per prompt
    clip_eps: float = 0.2       # PPO-style clip range
    kl_coef: float = 0.02       # beta: KL penalty coefficient against reference policy


class TinyPolicy(nn.Module):
    """
    A minimal autoregressive "policy": given a target-sum prompt embedding,
    emit a sequence of digit-token logits at each of seq_len positions,
    conditioned causally on previously sampled tokens.

    Stands in for DeepSeek-R1's actual policy (the 671B MoE model) -- the
    algorithmic mechanics of GRPO below are identical regardless of policy
    network size.
    """

    def __init__(self, cfg: GRPOConfig):
        super().__init__()
        self.cfg = cfg
        self.prompt_proj = nn.Linear(1, cfg.d_model)               # encode scalar target sum
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rnn = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, target_sum: torch.Tensor, tokens: torch.Tensor = None, sample: bool = False):
        """
        target_sum: [batch] float, the value the emitted digit sequence should sum to.
        tokens: [batch, seq_len] previously generated tokens, for teacher-forced log-prob
                scoring (used when re-scoring a completion under the current policy).
        sample: if True, autoregressively sample a fresh completion instead of scoring `tokens`.

        Returns: logits [batch, seq_len, vocab_size], and if sample=True also the sampled tokens.
        """
        b = target_sum.shape[0]
        h0 = self.prompt_proj(target_sum.unsqueeze(-1)).unsqueeze(0)  # [1, b, d_model] initial RNN hidden state

        if sample:
            generated = []
            logits_list = []
            hidden = h0
            prev_tok = torch.zeros(b, dtype=torch.long, device=target_sum.device)  # BOS = token 0
            for _ in range(self.cfg.seq_len):
                emb = self.token_embed(prev_tok).unsqueeze(1)          # [b, 1, d_model]
                out, hidden = self.rnn(emb, hidden)
                logits = self.head(out.squeeze(1))                     # [b, vocab_size]
                probs = F.softmax(logits, dim=-1)
                prev_tok = torch.multinomial(probs, 1).squeeze(-1)      # sample next digit
                generated.append(prev_tok)
                logits_list.append(logits)
            tokens_out = torch.stack(generated, dim=1)                 # [b, seq_len]
            logits_out = torch.stack(logits_list, dim=1)                # [b, seq_len, vocab_size]
            return logits_out, tokens_out
        else:
            assert tokens is not None
            bos = torch.zeros(b, 1, dtype=torch.long, device=tokens.device)
            inp_tokens = torch.cat([bos, tokens[:, :-1]], dim=1)        # shift right (teacher forcing)
            emb = self.token_embed(inp_tokens)                          # [b, seq_len, d_model]
            out, _ = self.rnn(emb, h0)
            logits = self.head(out)                                     # [b, seq_len, vocab_size]
            return logits, tokens


def verifiable_reward(tokens: torch.Tensor, target_sum: torch.Tensor) -> torch.Tensor:
    """
    The RLVR reward function: EXACT, mechanically checkable, no learned model.
    Reward = 1.0 if the emitted digits sum exactly to the target, else a small
    partial-credit shaping term based on closeness (kept simple; real R1 uses
    a stricter accuracy+format reward, see the .md for the actual design).
    """
    digit_sum = tokens.float().sum(dim=-1)               # [batch]
    exact = (digit_sum == target_sum).float()
    closeness = 1.0 - (digit_sum - target_sum).abs() / 10.0
    return torch.clamp(exact + (1 - exact) * closeness * 0.1, min=0.0, max=1.0)


def grpo_step(policy: TinyPolicy, ref_policy: TinyPolicy, target_sums: torch.Tensor, cfg: GRPOConfig):
    """
    One GRPO update for a batch of prompts. For each prompt, sample a GROUP of
    G completions, score each with the verifiable reward, compute the
    group-relative advantage (reward minus GROUP MEAN, divided by group std --
    no critic network anywhere), and optimize the PPO-style clipped objective
    plus a KL penalty against a frozen reference policy.
    """
    n_prompts = target_sums.shape[0]
    G = cfg.group_size

    # Expand each prompt G times to form the sampling group.
    expanded_targets = target_sums.repeat_interleave(G)          # [n_prompts * G]

    with torch.no_grad():
        _, sampled_tokens = policy(expanded_targets, sample=True)     # [n_prompts*G, seq_len]
        rewards = verifiable_reward(sampled_tokens, expanded_targets)  # [n_prompts*G]

    rewards_grouped = rewards.view(n_prompts, G)                  # [n_prompts, G]
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)          # <-- the GRPO baseline: empirical group mean, NOT a learned critic
    group_std = rewards_grouped.std(dim=1, keepdim=True).clamp_min(1e-4)
    advantages = (rewards_grouped - group_mean) / group_std          # [n_prompts, G], group-normalized advantage
    advantages = advantages.view(-1)                                  # [n_prompts*G]

    # Re-score the sampled completions under the CURRENT policy (with grad) and the reference policy (no grad).
    logits, _ = policy(expanded_targets, tokens=sampled_tokens)        # [n_prompts*G, seq_len, vocab]
    with torch.no_grad():
        ref_logits, _ = ref_policy(expanded_targets, tokens=sampled_tokens)

    log_probs = F.log_softmax(logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    token_log_probs = torch.gather(log_probs, 2, sampled_tokens.unsqueeze(-1)).squeeze(-1).sum(dim=1)          # [n_prompts*G]
    ref_token_log_probs = torch.gather(ref_log_probs, 2, sampled_tokens.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    # Importance ratio: policy that generated the samples IS the current policy here (on-policy,
    # single inner step), so ratio ~= 1 at the first update; kept for structural fidelity to PPO/GRPO's clipped objective.
    old_log_probs = token_log_probs.detach()
    ratio = torch.exp(token_log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    # KL penalty against the frozen reference policy (keeps the policy from drifting too far).
    kl = (token_log_probs.exp() * (token_log_probs - ref_token_log_probs)).mean()
    loss = policy_loss + cfg.kl_coef * kl

    return loss, rewards.mean().item(), advantages.std().item()


# ---------------------------------------------------------------------------
# Toy distillation: SFT of a smaller "student" on a larger RL-trained
# "teacher's" generated traces -- no RL for the student.
# ---------------------------------------------------------------------------

def distillation_sft_step(student: TinyPolicy, teacher: TinyPolicy, target_sums: torch.Tensor):
    """
    Generate reasoning "traces" (here, just digit sequences) from the
    RL-trained teacher, then train the student via ordinary cross-entropy
    SFT to imitate them. This is the entire distillation mechanism DeepSeek-R1
    uses to transfer reasoning behavior into smaller dense models -- no RL,
    no reward function, no critic, just supervised imitation of the teacher's
    own outputs.
    """
    with torch.no_grad():
        _, teacher_tokens = teacher(target_sums, sample=True)   # [batch, seq_len], teacher's traces

    student_logits, _ = student(target_sums, tokens=teacher_tokens)  # teacher-forced on teacher's own tokens
    loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        teacher_tokens.reshape(-1),
    )
    return loss, teacher_tokens


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = GRPOConfig()
    policy = TinyPolicy(cfg)
    ref_policy = TinyPolicy(cfg)
    ref_policy.load_state_dict(policy.state_dict())   # reference policy starts identical to the policy
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    print("=== GRPO: RL with Verifiable Rewards, no critic network ===")
    print(f"group size (G): {cfg.group_size}, seq_len: {cfg.seq_len}, vocab_size: {cfg.vocab_size}")

    n_params_policy = sum(p.numel() for p in policy.parameters())
    print(f"policy parameter count: {n_params_policy:,}  (no separate critic network exists in this loop at all)")

    batch_size = 32
    for step in range(30):
        target_sums = torch.randint(5, 30, (batch_size,)).float()   # random target sums for each prompt
        loss, mean_reward, adv_std = grpo_step(policy, ref_policy, target_sums, cfg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 6 == 0 or step == 29:
            print(f"step {step:2d}: loss={loss.item():.4f}  mean_group_reward={mean_reward:.4f}  advantage_std={adv_std:.4f}")

    print(
        "\nmean_group_reward should trend upward across steps: the policy is learning to hit the "
        "target sum purely from the verifiable reward signal, with the group mean as its only baseline."
    )

    print("\n=== Distillation: SFT-only transfer from RL-trained teacher to a smaller student ===")
    # "Smaller" student: fewer RNN hidden units, illustrating the size asymmetry (toy scale).
    student_cfg = GRPOConfig(d_model=16)
    student = TinyPolicy(student_cfg)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    n_params_teacher = sum(p.numel() for p in policy.parameters())
    n_params_student = sum(p.numel() for p in student.parameters())
    print(f"teacher (RL-trained) parameter count: {n_params_teacher:,}")
    print(f"student (SFT-only) parameter count:   {n_params_student:,}  ({n_params_student / n_params_teacher:.1%} of teacher)")

    for step in range(20):
        target_sums = torch.randint(5, 30, (batch_size,)).float()
        d_loss, teacher_tokens = distillation_sft_step(student, policy, target_sums)
        student_optimizer.zero_grad()
        d_loss.backward()
        student_optimizer.step()
        if step % 5 == 0 or step == 19:
            print(f"distill step {step:2d}: student cross-entropy loss = {d_loss.item():.4f}")

    print(
        "\nThe student never runs RL, never samples a group, never sees a reward function -- it only "
        "ever does supervised cross-entropy against the teacher's sampled traces. This is the mechanism "
        "behind DeepSeek-R1-Distill-{Qwen,Llama}-* : the expensive RLVR step happens once at teacher scale."
    )
