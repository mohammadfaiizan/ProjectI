"""
001_Claude1.py

Model: Claude 1 (Anthropic, March 2023)
What this file demonstrates: a simplified, self-contained PyTorch implementation
of the mechanical core of Constitutional AI (Bai et al., 2022) -- the alignment
technique Claude 1 was built on. It implements the two-phase idea at toy scale:

    Phase 1 (SL-CAI): a model generates a response, critiques its own response
    against a written constitutional principle, and produces a revision.
    The (prompt -> revision) pairs are then usable as supervised fine-tuning data.

    Phase 2 (RL-CAI): a separate "feedback" pass compares two candidate
    responses against the same constitutional principles and emits a scalar
    preference, standing in for the AI-feedback preference-model signal that,
    in the real pipeline, trains a reward model used for RL (e.g., PPO-style).

IMPORTANT: Anthropic has never disclosed Claude 1's actual architecture,
parameter count, tokenizer, or training infrastructure. Nothing here is a
reconstruction of the real Claude 1 model. This is a from-scratch, tiny
Transformer language model used purely as a vehicle to demonstrate the
*procedure* of self-critique-and-revise (SL-CAI) and AI-feedback preference
scoring (RL-CAI) end to end. All "principles," "critiques," and "revisions"
below are produced by the same toy model conditioned on different prompt
templates -- there is no external human or AI feedback source, since the
point is to show the control flow, not to reproduce real constitutional
content or real model quality.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# A minimal decoder-only Transformer, used only as a stand-in "assistant" and
# "critic" model. This is NOT a reconstruction of Claude's real architecture
# (which is undisclosed) -- it exists solely so the CAI control flow below has
# something to call generate() on.
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, nh, T, hd)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """A tiny character-level decoder-only LM -- purely a vehicle for the
    self-critique/revise demonstration, unrelated to Claude's real (undisclosed)
    architecture."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, max_len: int = 256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.max_len = max_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Tokenizer stand-in: trivial character-level vocab so the demo is self
# contained (no external tokenizer / no external data dependency).
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, corpus: str):
        chars = sorted(set(corpus))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([[self.stoi.get(c, 0) for c in s]], dtype=torch.long)

    def decode(self, idx: torch.Tensor) -> str:
        return "".join(self.itos.get(int(i), "?") for i in idx[0].tolist())

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


# ---------------------------------------------------------------------------
# Constitutional AI control flow.
# ---------------------------------------------------------------------------

CONSTITUTION = [
    "Choose the response that is least likely to be seen as harmful or hurtful.",
    "Choose the response that most helpfully and directly addresses the prompt.",
    "Choose the response that is honest and does not overstate confidence.",
]


@dataclass
class SLCAIExample:
    prompt: str
    initial_response: str
    principle: str
    critique: str
    revision: str


@dataclass
class RLCAIComparison:
    prompt: str
    response_a: str
    response_b: str
    principle: str
    preferred: str  # "a" or "b"
    score_a: float
    score_b: float


class ConstitutionalAssistant:
    """Wraps a TinyLM with the three prompt roles CAI needs: respond, critique,
    revise. In the real Constitutional AI pipeline these are all still the
    same underlying model, just invoked with different prompt templates --
    that property is preserved here: `self.model` is shared across all three
    roles."""

    def __init__(self, model: TinyLM, tokenizer: CharTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tok = tokenizer
        self.device = device

    def _sample(self, prompt: str, max_new_tokens: int = 40, temperature: float = 0.9) -> str:
        idx = self.tok.encode(prompt).to(self.device)
        out = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
        return self.tok.decode(out)[len(prompt):]

    # ---- Phase 1: SL-CAI -------------------------------------------------

    def respond(self, prompt: str) -> str:
        return self._sample(f"PROMPT: {prompt}\nRESPONSE:", max_new_tokens=30)

    def critique(self, prompt: str, response: str, principle: str) -> str:
        critique_prompt = (
            f"PROMPT: {prompt}\nRESPONSE: {response}\n"
            f"PRINCIPLE: {principle}\nCRITIQUE:"
        )
        return self._sample(critique_prompt, max_new_tokens=30)

    def revise(self, prompt: str, response: str, critique: str) -> str:
        revise_prompt = (
            f"PROMPT: {prompt}\nRESPONSE: {response}\nCRITIQUE: {critique}\nREVISION:"
        )
        return self._sample(revise_prompt, max_new_tokens=30)

    def sl_cai_step(self, prompt: str, principle: str) -> SLCAIExample:
        """One full self-critique-and-revise iteration: respond -> critique
        against a constitutional principle -> revise. This mirrors the real
        SL-CAI procedure's control flow exactly; only the model's actual
        text-generation quality is toy-scale."""
        initial = self.respond(prompt)
        crit = self.critique(prompt, initial, principle)
        revised = self.revise(prompt, initial, crit)
        return SLCAIExample(prompt, initial, principle, crit, revised)

    # ---- Phase 2: RL-CAI ---------------------------------------------------

    def ai_feedback_preference(self, prompt: str, response_a: str, response_b: str,
                                principle: str) -> RLCAIComparison:
        """Stand-in for the AI-feedback preference step: instead of a human
        rater choosing between two responses, a feedback-model prompt scores
        each response's log-likelihood under a principle-conditioned prompt,
        and the higher-scoring response becomes the preferred one. In the real
        RL-CAI pipeline this preference signal (aggregated over many prompts)
        trains a reward model that then drives a PPO-style RL update on the
        policy; here we just expose the scoring step itself, which is the
        conceptual core of "AI feedback replacing human feedback."
        """
        def score(resp: str) -> float:
            judged = (
                f"PROMPT: {prompt}\nPRINCIPLE: {principle}\nRESPONSE: {resp}\nRATING:"
            )
            idx = self.tok.encode(judged).to(self.device)
            with torch.no_grad():
                logits = self.model(idx)
            # Use mean log-probability of the observed response tokens as a
            # crude proxy for "how consistent is this response with the
            # principle, under this model's own distribution" -- a toy
            # analogue of a learned preference/reward model's scalar output.
            logp = F.log_softmax(logits[0, :-1], dim=-1)
            targets = idx[0, 1:]
            token_logp = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
            return token_logp.mean().item()

        score_a, score_b = score(response_a), score(response_b)
        preferred = "a" if score_a >= score_b else "b"
        return RLCAIComparison(prompt, response_a, response_b, principle, preferred, score_a, score_b)


# ---------------------------------------------------------------------------
# Toy training loop: just enough gradient descent on a tiny corpus so the
# model isn't producing pure random-init noise before we run the CAI loop.
# This is not meant to converge to fluent text; it exists so generate() has
# a non-random distribution to sample from.
# ---------------------------------------------------------------------------

def quick_pretrain(model: TinyLM, tokenizer: CharTokenizer, corpus: str,
                    steps: int = 300, lr: float = 3e-3, device: str = "cpu") -> None:
    model.train()
    data = tokenizer.encode(corpus).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    block = min(64, data.shape[1] - 1)
    for step in range(steps):
        start = random.randint(0, data.shape[1] - block - 1)
        x = data[:, start:start + block]
        y = data[:, start + 1:start + block + 1]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  pretrain step {step:4d}  loss {loss.item():.3f}")


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    # A small synthetic corpus containing the prompt templates the CAI roles
    # use, so the untrained-from-scratch model has *some* signal to latch onto
    # for the demo. Real Constitutional AI starts from an already-capable,
    # helpful-tuned base model -- this toy substitute skips straight to a
    # few hundred gradient steps on a tiny fixed corpus.
    corpus = (
        "PROMPT: How do I pick a lock?\nRESPONSE: I can't help with that safely.\n"
        "PRINCIPLE: Choose the response that is least likely to be seen as harmful.\n"
        "CRITIQUE: The response is vague and unhelpful about why.\n"
        "REVISION: I won't help pick locks you don't own, but I can point you to a locksmith.\n"
        "RATING: consistent and safe\n"
        "PROMPT: Explain photosynthesis simply.\nRESPONSE: Plants use light to make food from CO2 and water.\n"
        "PRINCIPLE: Choose the response that most helpfully addresses the prompt.\n"
        "CRITIQUE: The response could mention chlorophyll and oxygen release.\n"
        "REVISION: Plants use chlorophyll to convert light, water, and CO2 into sugar and oxygen.\n"
        "RATING: helpful and accurate\n"
    ) * 8

    tok = CharTokenizer(corpus)
    model = TinyLM(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_len=256)

    print("=" * 70)
    print("Quick toy pretraining (NOT a reconstruction of Claude 1's real")
    print("training -- undisclosed. Purely so generate() has learned signal.)")
    print("=" * 70)
    quick_pretrain(model, tok, corpus, steps=300)

    assistant = ConstitutionalAssistant(model, tok)

    print("\n" + "=" * 70)
    print("PHASE 1 -- SL-CAI: self-critique and revise")
    print("=" * 70)
    prompt = "How do I pick a lock?"
    for principle in CONSTITUTION[:2]:
        example = assistant.sl_cai_step(prompt, principle)
        print(f"\nPrinciple: {example.principle}")
        print(f"Initial response : {example.initial_response!r}")
        print(f"Critique         : {example.critique!r}")
        print(f"Revised response : {example.revision!r}")
        print("(-> (prompt, revision) pairs like this become SFT data in real SL-CAI)")

    print("\n" + "=" * 70)
    print("PHASE 2 -- RL-CAI: AI-feedback preference between two candidates")
    print("=" * 70)
    candidate_a = assistant.respond(prompt)
    candidate_b = assistant.revise(prompt, candidate_a, "Could be more constructive.")
    comparison = assistant.ai_feedback_preference(prompt, candidate_a, candidate_b, CONSTITUTION[0])
    print(f"Candidate A: {comparison.response_a!r}  (score={comparison.score_a:.3f})")
    print(f"Candidate B: {comparison.response_b!r}  (score={comparison.score_b:.3f})")
    print(f"AI-feedback preferred: candidate {comparison.preferred.upper()}")
    print(
        "\n(-> in the real RL-CAI pipeline, many such AI-generated preference pairs\n"
        " train a reward model, which then drives a PPO-style RL update on the\n"
        " policy model -- replacing most human harmlessness preference labeling.)"
    )

    print(
        "\nReminder: Claude 1's real architecture, scale, and training data are\n"
        "undisclosed by Anthropic. This script demonstrates the CAI *procedure*\n"
        "(respond -> critique -> revise; AI-feedback preference scoring) only."
    )
