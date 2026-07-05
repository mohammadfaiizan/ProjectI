"""
007_GPT4o_Multimodal.py -- Educational reconstruction of GPT-4o (OpenAI, 2024)
"natively multimodal" mechanism.

SCOPE: OpenAI has disclosed that GPT-4o is "a single model trained end-to-end
across text, vision, and audio," but has published no architecture details --
no tokenizer specification, no layer/parameter counts, no loss formulation.
This file does NOT reproduce GPT-4o. It implements one concrete, defensible
way to satisfy the *stated design constraint* ("one model, one shared
sequence, all modalities") so the engineering tradeoffs are inspectable:

  - Text is tokenized normally (integer ids into a text vocabulary).
  - Images are patch-encoded then VECTOR-QUANTIZED into discrete codebook
    ids, i.e. treated as an image "vocabulary" -- not projected as
    continuous embeddings. This is the discrete-tokenization design
    discussed in 007_GPT4o_Multimodal.md Section 2, chosen specifically
    because it is the design that supports *generating* a modality (not
    just reading it) from the same softmax used for text, which is what
    "any-to-any" (text out AND audio out) requires architecturally.
  - Audio is likewise frame-encoded and vector-quantized into discrete
    codebook ids at a fixed frame rate, illustrating the rate-mismatch
    problem: an audio clip produces far more tokens per second of raw
    content than a comparable text utterance would as subwords.
  - All three token streams share ONE embedding table (a single unified
    vocabulary: text subwords + image codes + audio codes, each range
    given a disjoint id block) and are concatenated with modality-type
    tag embeddings into ONE causal sequence fed to ONE transformer stack,
    with ONE output softmax that can emit text, image, or audio tokens.

This directly contrasts with 006_GPT4.py's MultimodalTokenFuser, which
represents the older "bolt-on adapter" pattern (separate continuous vision
encoder feeding into a text-only decoder that can only ever output text).
The difference between those two files IS the architectural point this
document makes about the GPT-4 -> GPT-4o transition.

Also included: a small streaming/incremental generation loop, to make the
low-latency-decoding story concrete (each modality's tokens can be decoded
and emitted to the user as soon as they're generated, rather than waiting
for a full-response round trip through separate ASR/LLM/TTS services).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Discrete tokenizers for image and audio (toy vector quantization)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Maps continuous feature vectors to the nearest of `num_codes` learned
    codebook vectors, returning discrete indices. This is the mechanism
    (VQ-VAE/VQGAN/SoundStream-family) that lets a non-text modality be
    represented as tokens from a finite vocabulary, exactly like text
    subwords -- the prerequisite for putting all modalities in one softmax.
    """

    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim) * 0.02)

    def forward(self, x):
        # x: (batch, seq, code_dim) continuous features -> nearest codebook id
        b, s, d = x.shape
        flat = x.reshape(-1, d)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.t()
                + self.codebook.pow(2).sum(1))
        ids = dist.argmin(dim=-1).view(b, s)
        return ids


class ImageTokenizer(nn.Module):
    """Patchify -> linear feature -> VQ -> discrete image token ids."""

    def __init__(self, image_size=32, patch_size=8, in_channels=3, code_dim=32, num_codes=64):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_channels * patch_size * patch_size
        self.feature_proj = nn.Linear(patch_dim, code_dim)
        self.vq = VectorQuantizer(num_codes, code_dim)
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, images):
        b, c, h, w = images.shape
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p).contiguous()
        patches = patches.view(b, c, -1, p, p).permute(0, 2, 1, 3, 4).reshape(b, -1, c * p * p)
        features = self.feature_proj(patches)
        return self.vq(features)  # (batch, num_patches) discrete ids


class AudioTokenizer(nn.Module):
    """Toy fixed-rate frame encoder -> VQ -> discrete audio token ids.
    Models a neural-codec-style audio tokenizer (SoundStream/EnCodec family):
    a raw waveform chunk is grouped into frames, each frame becomes one
    token. Frame rate here is deliberately small for the toy example; real
    neural audio codecs commonly run tens of tokens per second per codebook.
    """

    def __init__(self, samples_per_frame=64, code_dim=32, num_codes=64):
        super().__init__()
        self.samples_per_frame = samples_per_frame
        self.feature_proj = nn.Linear(samples_per_frame, code_dim)
        self.vq = VectorQuantizer(num_codes, code_dim)

    def forward(self, waveform):
        # waveform: (batch, num_samples) -> (batch, num_frames)
        b, n = waveform.shape
        n_frames = n // self.samples_per_frame
        frames = waveform[:, : n_frames * self.samples_per_frame].view(b, n_frames, self.samples_per_frame)
        features = self.feature_proj(frames)
        return self.vq(features)


# ---------------------------------------------------------------------------
# Unified multimodal sequence assembly
# ---------------------------------------------------------------------------

class ModalityBlockIds:
    """Disjoint id ranges within one shared embedding table, i.e. one
    unified vocabulary spanning text + image + audio -- the core structural
    difference from the bolt-on-adapter pattern in 006_GPT4.py."""

    def __init__(self, text_vocab_size, num_image_codes, num_audio_codes):
        self.text_start = 0
        self.image_start = text_vocab_size
        self.audio_start = self.image_start + num_image_codes
        self.total_vocab = self.audio_start + num_audio_codes


class UnifiedMultimodalTransformer(nn.Module):
    """One embedding table, one causal decoder stack, one output softmax,
    shared across text/image/audio token streams. Modality-type embeddings
    are added so the model can distinguish "this position is an image
    token" from "this position is a text token" even though both are drawn
    from the same underlying transformer computation.
    """

    def __init__(self, text_vocab_size, num_image_codes, num_audio_codes,
                 d_model=64, n_layers=4, n_heads=4, d_ff=128, max_seq_len=512):
        super().__init__()
        self.ids = ModalityBlockIds(text_vocab_size, num_image_codes, num_audio_codes)
        self.token_embed = nn.Embedding(self.ids.total_vocab, d_model)
        self.modality_embed = nn.Embedding(3, d_model)  # 0=text, 1=image, 2=audio
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        self.decoder_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True)
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.ids.total_vocab, bias=False)

    def build_sequence(self, text_ids=None, image_ids=None, audio_ids=None):
        """Offsets each modality's local ids into the shared vocabulary,
        tags each position with its modality, and concatenates into one
        causal sequence: [image tokens, audio tokens, text tokens]."""
        pieces, modality_tags = [], []
        if image_ids is not None:
            pieces.append(image_ids + self.ids.image_start)
            modality_tags.append(torch.full_like(image_ids, 1))
        if audio_ids is not None:
            pieces.append(audio_ids + self.ids.audio_start)
            modality_tags.append(torch.full_like(audio_ids, 2))
        if text_ids is not None:
            pieces.append(text_ids + self.ids.text_start)
            modality_tags.append(torch.full_like(text_ids, 0))
        return torch.cat(pieces, dim=1), torch.cat(modality_tags, dim=1)

    def forward(self, text_ids=None, image_ids=None, audio_ids=None):
        seq_ids, modality_tags = self.build_sequence(text_ids, image_ids, audio_ids)
        x = self.token_embed(seq_ids) + self.modality_embed(modality_tags)
        x = x + self.pos_embed[:, : x.size(1), :]

        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        for layer in self.decoder_layers:
            x = layer(x, src_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # spans the ENTIRE unified vocabulary: text+image+audio
        return logits, seq_ids, modality_tags


# ---------------------------------------------------------------------------
# Streaming generation: emit tokens (of any modality) as soon as produced,
# illustrating the low-latency contrast with a serial ASR->LLM->TTS pipeline.
# ---------------------------------------------------------------------------

@torch.no_grad()
def stream_generate(model, prime_ids, prime_modality_tags, steps=8):
    """Greedy incremental decode. Each yielded step models "one token ready
    to hand to the client" -- for a real system this is what lets audio
    output start playing before the full response has finished generating,
    which is where most of the perceived latency win over a chained
    ASR -> LLM -> TTS pipeline actually comes from.
    """
    seq_ids = prime_ids.clone()
    modality_tags = prime_modality_tags.clone()
    for _ in range(steps):
        x = model.token_embed(seq_ids) + model.modality_embed(modality_tags)
        x = x + model.pos_embed[:, : x.size(1), :]
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        for layer in model.decoder_layers:
            x = layer(x, src_mask=causal_mask)
        logits = model.lm_head(model.ln_f(x))
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Infer which modality block the sampled id falls in, purely so the
        # streaming loop can tag it correctly for the next step's embedding.
        if next_id.item() >= model.ids.audio_start:
            tag = 2
        elif next_id.item() >= model.ids.image_start:
            tag = 1
        else:
            tag = 0
        next_tag = torch.full_like(next_id, tag)

        seq_ids = torch.cat([seq_ids, next_id], dim=1)
        modality_tags = torch.cat([modality_tags, next_tag], dim=1)
        yield next_id.item(), tag
    return


if __name__ == "__main__":
    torch.manual_seed(0)

    text_vocab_size = 500
    num_image_codes = 64
    num_audio_codes = 64
    batch_size = 2

    image_tok = ImageTokenizer(image_size=32, patch_size=8, code_dim=32, num_codes=num_image_codes)
    audio_tok = AudioTokenizer(samples_per_frame=64, code_dim=32, num_codes=num_audio_codes)

    images = torch.randn(batch_size, 3, 32, 32)
    waveform = torch.randn(batch_size, 64 * 20)  # 20 frames of audio
    text_ids = torch.randint(0, text_vocab_size, (batch_size, 10))

    image_ids = image_tok(images)
    audio_ids = audio_tok(waveform)

    print("=== GPT-4o-style toy model: unified discrete multimodal sequence ===")
    print(f"image tokens per sample: {image_ids.shape[1]}")
    print(f"audio tokens per sample: {audio_ids.shape[1]}  (rate-mismatch: same wall-clock "
          f"content as far fewer text tokens would need)")
    print(f"text tokens per sample:  {text_ids.shape[1]}")

    model = UnifiedMultimodalTransformer(
        text_vocab_size=text_vocab_size,
        num_image_codes=num_image_codes,
        num_audio_codes=num_audio_codes,
        d_model=64, n_layers=4, n_heads=4, d_ff=128, max_seq_len=512,
    )

    logits, seq_ids, modality_tags = model(text_ids=text_ids, image_ids=image_ids, audio_ids=audio_ids)
    print(f"\nfused sequence length:  {seq_ids.shape[1]} (image+audio+text)")
    print(f"unified vocab size:     {model.ids.total_vocab} (text block + image block + audio block)")
    print(f"output logits shape:    {tuple(logits.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total toy model params: {total_params:,}")

    print("\n--- streaming incremental decode (single-sample) ---")
    prime_ids = seq_ids[:1]
    prime_tags = modality_tags[:1]
    modality_names = {0: "text", 1: "image", 2: "audio"}
    for step, (tok_id, tag) in enumerate(stream_generate(model, prime_ids, prime_tags, steps=6)):
        print(f"  step {step}: emitted token id={tok_id} modality={modality_names[tag]} "
              f"(client could begin rendering/playing this immediately)")
