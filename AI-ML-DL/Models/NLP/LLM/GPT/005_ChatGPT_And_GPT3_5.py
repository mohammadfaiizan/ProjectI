"""
ChatGPT / GPT-3.5 (OpenAI, 2022) -- reference PyTorch implementation.

What this file demonstrates
----------------------------
ChatGPT's actual mechanical delta over InstructGPT (see the companion
`004_InstructGPT_And_RLHF.md` for the SFT -> reward-model (Bradley-Terry) ->
PPO-with-KL-penalty pipeline itself, which is NOT re-implemented here) is
that its supervised fine-tuning data is DIALOGUE-FORMATTED: multi-turn
conversations with distinguishable `system` / `user` / `assistant` roles,
rather than InstructGPT's single-turn "prompt in, completion out" pairs.

This file implements exactly that delta, end to end, on top of a small,
otherwise-ordinary decoder-only causal transformer:

1. `Role` -- the three conversational roles (system / user / assistant).
2. A toy word-level tokenizer with special role/turn-delimiter tokens
   (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`) -- not a real BPE,
   just enough vocabulary machinery to render/inspect a conversation as a
   flat integer sequence.
3. `render_conversation`: flattens a list of (role, text) turns into ONE
   token-id sequence a decoder-only model can consume, and records which
   token spans belong to assistant turns.
4. `build_loss_mask`: the mechanically important, distinctive piece. SFT on
   dialogue data must train the model to predict the ASSISTANT's tokens
   only -- not the user's turns, not the system turn, not the role
   delimiters themselves. The mask is 1 on assistant-turn content tokens
   (including the assistant turn's terminating `<|end|>` token) and 0
   everywhere else.
5. A compact pre-norm decoder-only causal transformer (learned absolute
   position embeddings, multi-head causal self-attention, GELU FFN) that
   consumes the rendered sequence and produces next-token logits. This is
   intentionally minimal -- the point of this file is the dialogue
   formatting/masking mechanism above, not re-deriving transformer-block
   mechanics (see `001_GPT1.py` for a more detailed, annotated block-level
   walkthrough of that part).
6. `masked_cross_entropy`: standard next-token cross-entropy, but weighted
   by the (shifted) loss mask from (4), so gradient signal flows only
   through assistant-turn token predictions.

This file is self-contained: only `torch`, `math`, `dataclasses`, and
`enum` are used. It is illustrative reference code, not a production
training script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Conversational roles
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """The three role types introduced by ChatGPT's dialogue format.

    InstructGPT's training data has no notion of "role" at all -- it is
    flat (prompt, completion) pairs. ChatGPT's SFT data is a sequence of
    role-tagged turns, and the model must learn turn-taking and
    role-conditioned behavior (e.g. "assistant" turns should follow
    "user" turns and should honor any standing "system" instruction).
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# A conversation is just a list of (role, text) turns, in order.
Turn = tuple  # (Role, str)
Conversation = list  # list[Turn]


# ---------------------------------------------------------------------------
# 2. Toy tokenizer with special role/turn-delimiter tokens
# ---------------------------------------------------------------------------

# Special tokens that delimit roles and turn boundaries. In a real system
# these would be reserved IDs in a BPE vocabulary (as, e.g., the Chat
# Completions message format is internally rendered into some analogous
# token scheme); here they are just entries in a tiny toy vocabulary.
ROLE_TOKEN = {
    Role.SYSTEM: "<|system|>",
    Role.USER: "<|user|>",
    Role.ASSISTANT: "<|assistant|>",
}
END_TOKEN = "<|end|>"
UNK_TOKEN = "<|unk|>"
PAD_TOKEN = "<|pad|>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, END_TOKEN] + list(ROLE_TOKEN.values())


class ToyTokenizer:
    """A minimal whitespace/word-level tokenizer, purely for illustration.

    Not a real BPE tokenizer (no merges, no subword handling) -- the point
    of this file is the role/turn structure imposed ON TOP of tokenization,
    not tokenization itself, so a trivial word-level vocab is sufficient.
    The vocabulary is built once from a fixed corpus of words plus the
    special tokens above, so token ids are stable and reproducible.
    """

    def __init__(self, corpus_words: list[str]):
        vocab = list(SPECIAL_TOKENS)
        seen = set(vocab)
        for w in corpus_words:
            if w not in seen:
                seen.add(w)
                vocab.append(w)

        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token: dict[int, str] = {i: tok for tok, i in self.token_to_id.items()}
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.end_id = self.token_to_id[END_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode_word(self, word: str) -> int:
        return self.token_to_id.get(word, self.unk_id)

    def encode_text(self, text: str) -> list[int]:
        return [self.encode_word(w) for w in text.split()]

    def role_token_id(self, role: Role) -> int:
        return self.token_to_id[ROLE_TOKEN[role]]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(i, UNK_TOKEN) for i in ids]


# ---------------------------------------------------------------------------
# 3. Rendering a conversation into a single flat token sequence
# ---------------------------------------------------------------------------

def render_conversation(
    conversation: Conversation, tok: ToyTokenizer
) -> tuple[list[int], list[tuple[int, int]]]:
    """Flatten (role, text) turns into one token-id sequence.

    Layout per turn:  <role_token> w1 w2 ... wk <|end|>

    e.g. a whole rendered conversation looks like:
        <|system|> be concise <|end|> <|user|> hi there <|end|>
        <|assistant|> hello how can i help <|end|> ...

    Returns
    -------
    ids : list[int]
        The full flattened token sequence.
    assistant_spans : list[(start, end)]
        Half-open [start, end) index ranges into `ids` covering ONLY the
        content tokens + terminating <|end|> of each assistant turn. The
        role-delimiter token itself (<|assistant|>) is deliberately
        EXCLUDED from the span -- it is a fixed, always-known control
        token the model should condition on, not something we need to
        train it to *predict* as if it were assistant-authored content.
    """
    ids: list[int] = []
    assistant_spans: list[tuple[int, int]] = []

    for role, text in conversation:
        ids.append(tok.role_token_id(role))          # e.g. <|assistant|>
        content_start = len(ids)
        ids.extend(tok.encode_text(text))              # turn content
        ids.append(tok.end_id)                          # <|end|>
        content_end = len(ids)                          # exclusive

        if role == Role.ASSISTANT:
            assistant_spans.append((content_start, content_end))

    return ids, assistant_spans


# ---------------------------------------------------------------------------
# 4. Loss mask: train ONLY on assistant-turn tokens
# ---------------------------------------------------------------------------

def build_loss_mask(seq_len: int, assistant_spans: list[tuple[int, int]]) -> torch.Tensor:
    """Build a 0/1 mask of shape (seq_len,): 1 at positions that are part of
    an assistant turn's content (or its terminating <|end|>), 0 everywhere
    else -- including system tokens, user tokens, and every role-delimiter
    token (<|system|>, <|user|>, <|assistant|>).

    This is the mechanically distinctive piece of SFT on dialogue data: the
    autoregressive LM loss at position i (predicting token i+1) should only
    contribute to the gradient when the TARGET token (i+1) is a token the
    assistant is responsible for producing. We must not train the model to
    imitate predicting the user's next message -- that would optimize the
    model to "predict what a human says next," which is not the SFT
    objective (the objective is "produce a good assistant reply given
    whatever the human said").
    """
    mask = torch.zeros(seq_len, dtype=torch.float32)
    for start, end in assistant_spans:
        mask[start:end] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 5. A compact pre-norm decoder-only causal transformer
# ---------------------------------------------------------------------------

@dataclass
class DialogueTransformerConfig:
    vocab_size: int
    block_size: int = 128
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 4 * 64
    dropout: float = 0.1
    init_std: float = 0.02

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    """Standard multi-head scaled dot-product self-attention with a causal
    mask -- no cross-attention sub-layer (decoder-only, same convention as
    every model in this GPT-lineage series)."""

    def __init__(self, cfg: DialogueTransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        causal_mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = self.causal_mask[:t, :t]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.out_proj(out)
        return self.resid_dropout(out)


class FeedForward(nn.Module):
    """Position-wise GELU FFN, 4x expansion."""

    def __init__(self, cfg: DialogueTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """PRE-norm residual block: x = x + SubLayer(LayerNorm(x)).

    Pre-norm (rather than GPT-1's original post-norm) is the convention
    used by essentially every GPT-2-and-later model in this series, and is
    what's assumed here as the "presumed-unchanged" architectural baseline
    for the GPT-3.5-class base model ChatGPT's SFT/RLHF stages build on top
    of (see the accompanying .md, Section 2: no architectural novelty is
    disclosed for GPT-3.5, so this file uses the standard modern-GPT
    pre-norm convention rather than inventing something new).
    """

    def __init__(self, cfg: DialogueTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DialogueTransformer(nn.Module):
    """Token + learned position embeddings, N pre-norm blocks, final LN,
    and a weight-tied LM head producing next-token logits over the toy
    vocabulary (including the special role/turn tokens)."""

    def __init__(self, cfg: DialogueTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        assert t <= self.cfg.block_size, (
            f"sequence length {t} exceeds block_size {self.cfg.block_size}"
        )
        positions = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = x @ self.token_embed.weight.T  # weight-tied LM head
        return logits


# ---------------------------------------------------------------------------
# 6. Masked cross-entropy: gradient only from assistant-turn predictions
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Next-token cross-entropy, masked to assistant-turn targets only.

    logits:    (batch, seq_len, vocab_size) -- model output at every position.
    input_ids: (batch, seq_len)             -- the rendered token sequence.
    loss_mask: (batch, seq_len)             -- 1 where position i is an
               assistant-turn token (from build_loss_mask), 0 elsewhere.

    Standard next-token setup: logits at position i predict the token at
    position i+1, so both the labels AND the mask must be shifted left by
    one. The mask element that matters at shifted position i is whether the
    TARGET token (input_ids[i+1], i.e. what's actually being predicted) is
    an assistant token -- not whether the token we're conditioning FROM is.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous().float()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)

    masked_sum = (per_token_loss * shift_mask).sum()
    denom = shift_mask.sum().clamp(min=1.0)
    return masked_sum / denom


# ---------------------------------------------------------------------------
# Parameter counting helper
# ---------------------------------------------------------------------------

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Illustrative smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # --- Build a toy multi-turn conversation ---
    conversation: Conversation = [
        (Role.SYSTEM, "you are a concise helpful assistant"),
        (Role.USER, "what is the capital of france"),
        (Role.ASSISTANT, "the capital of france is paris"),
        (Role.USER, "and italy"),
        (Role.ASSISTANT, "the capital of italy is rome"),
    ]

    # Build the toy vocabulary from every word actually used above, so
    # encode/decode round-trips cleanly (no <|unk|> tokens in this demo).
    corpus_words = [w for _, text in conversation for w in text.split()]
    tok = ToyTokenizer(corpus_words)

    print("=== ChatGPT/GPT-3.5 dialogue-formatting smoke test ===")
    print(f"Toy vocab size: {tok.vocab_size}")

    # --- Render the conversation into a flat token sequence ---
    ids, assistant_spans = render_conversation(conversation, tok)
    seq_len = len(ids)
    print(f"\nRendered token sequence length: {seq_len}")
    print("Rendered tokens:")
    print(" ".join(tok.decode(ids)))
    print(f"Assistant turn spans (token index ranges): {assistant_spans}")

    # --- Build the assistant-turn-only loss mask ---
    loss_mask = build_loss_mask(seq_len, assistant_spans)
    print("\nLoss mask (1 = trained on, i.e. assistant content + its <|end|>):")
    print(loss_mask.int().tolist())
    print("Aligned view (token -> mask):")
    for token_str, m in zip(tok.decode(ids), loss_mask.int().tolist()):
        print(f"  {token_str:>12s} : {m}")

    # --- Run the rendered sequence through the tiny transformer ---
    cfg = DialogueTransformerConfig(
        vocab_size=tok.vocab_size,
        block_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=4 * 32,
        dropout=0.1,
    )
    model = DialogueTransformer(cfg)

    input_ids = torch.tensor([ids], dtype=torch.long)          # (1, seq_len)
    batch_mask = loss_mask.unsqueeze(0)                          # (1, seq_len)

    logits = model(input_ids)
    print(f"\nLogits shape: {tuple(logits.shape)}  (batch, seq_len, vocab_size)")

    loss = masked_cross_entropy(logits, input_ids, batch_mask)
    print(f"Masked SFT loss (assistant-turn tokens only): {loss.item():.4f}")

    # For contrast, show what the (incorrect-for-SFT) unmasked loss would be,
    # i.e. training on every token including user/system turns and role
    # delimiters -- this is exactly the thing build_loss_mask exists to avoid.
    full_mask = torch.ones_like(batch_mask)
    unmasked_loss = masked_cross_entropy(logits, input_ids, full_mask)
    print(f"(For contrast) unmasked loss over ALL tokens: {unmasked_loss.item():.4f}")

    # --- Parameter counts ---
    print("\n=== Parameter counts (toy illustrative config) ===")
    print(cfg)
    print(f"Total trainable parameters: {count_parameters(model):,}")
