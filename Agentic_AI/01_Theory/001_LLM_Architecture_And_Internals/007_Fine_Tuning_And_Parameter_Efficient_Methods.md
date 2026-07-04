# Fine-Tuning and Parameter-Efficient Methods

## Why Fine-Tune At All

A pretrained LLM has already learned an enormous amount about language, facts, and reasoning patterns from its training corpus, but that knowledge is generic by construction — it was optimized to predict the next token across a huge, undifferentiated mixture of internet text, books, and code. Fine-tuning is the process of taking that general-purpose model and further training it on a smaller, more targeted dataset so its behavior shifts toward a specific distribution: following instructions, adopting a particular tone, producing structured outputs, specializing in a narrow domain like legal or medical text, or mimicking a proprietary style of response. The pretrained weights are not a fixed artifact you only consume through prompting; they are a starting point that gradient descent can continue to move, and the central question this chapter answers is: how much of that starting point do you actually need to move, and at what cost?

It helps to be precise about what "fine-tuning" means as distinct from other ways of adapting a model. Prompting and in-context learning (covered elsewhere in this repo) change the model's behavior without touching a single weight — you're steering a fixed function by changing its input. Retrieval-augmented generation changes what information is available to the model at inference time, again without touching weights. Fine-tuning is fundamentally different: it changes the function itself by updating parameters through backpropagation on a new dataset. That makes it strictly more powerful in what it can accomplish (it can teach genuinely new behaviors and even suppress old ones) and strictly more expensive and riskier (it requires compute, data curation, evaluation infrastructure, and carries the risk of catastrophic forgetting — degrading capabilities the model had before you started).

## Full Fine-Tuning and Its Real Cost

The most straightforward approach is full fine-tuning: take every weight in the pretrained network and let the optimizer update all of them using gradients computed from your new dataset. Conceptually this is no different from pretraining itself — same architecture, same loss (next-token cross-entropy, typically), just a different, usually much smaller and more curated, dataset and typically a much lower learning rate so you nudge the model rather than overwrite what it already knows. The appeal is obvious: with every parameter free to move, the model has the maximum possible flexibility to fit the new data well, and empirically full fine-tuning generally gets the best achievable task performance when you have enough data and compute to do it properly.

The problem is memory, and it's worth working through the arithmetic explicitly because this is one of the most commonly tested "do you actually understand this or did you just read a blog post" interview questions. Training with the Adam optimizer (or AdamW, which nearly everyone uses for LLMs) requires more than just storing the weights themselves. Adam maintains two additional exponential moving average buffers per parameter: the first moment estimate (a running average of the gradient, i.e., momentum) and the second moment estimate (a running average of the squared gradient, used to adapt the per-parameter learning rate). Both of these buffers are the same shape as the parameter tensor they track. That means for every one parameter you're training, Adam needs two more numbers of the same size sitting in memory throughout training, even though those numbers are never used for inference.

Concretely, for a model with mixed-precision training (the standard setup), the accounting looks like this per parameter:

- Weights, kept in FP16/BF16 for the forward and backward pass: 2 bytes
- Gradients, computed in FP16/BF16: 2 bytes
- A master copy of the weights kept in FP32 for numerically stable optimizer updates: 4 bytes
- Adam's first moment (FP32): 4 bytes
- Adam's second moment (FP32): 4 bytes

That totals roughly 16 bytes per parameter just for what the DeepSpeed ZeRO paper calls "model states," before you've accounted for a single activation. For a 7B-parameter model, that's 7 billion times 16 bytes, which is 112 GB — and that number has nothing to do with batch size or sequence length yet. A single 80 GB A100 cannot hold that on its own; full fine-tuning a 7B model already requires either multiple GPUs with some form of model or optimizer-state sharding (ZeRO, FSDP), or aggressive memory-saving tricks like offloading optimizer states to CPU RAM, or giving up and dropping to a smaller model.

On top of model states, you need activation memory: the intermediate outputs of every layer that get cached during the forward pass so they're available for gradient computation during the backward pass. Activation memory scales with batch size, sequence length, hidden dimension, and number of layers, and for long-context fine-tuning it can dwarf the model-state memory entirely — this is exactly why techniques like gradient checkpointing (recomputing activations during the backward pass instead of storing them, trading compute for memory) and FlashAttention (which avoids materializing the full attention score matrix) matter as much for fine-tuning as they do for pretraining.

```python
def full_finetune_memory_estimate(num_params_billion: float, seq_len: int,
                                    batch_size: int, hidden_dim: int, num_layers: int):
    """Rough, order-of-magnitude memory estimate for full fine-tuning with AdamW.
    This is illustrative, not exact -- real numbers depend on framework overhead,
    activation checkpointing choices, and attention implementation."""
    params = num_params_billion * 1e9

    weights_fp16 = params * 2
    grads_fp16 = params * 2
    master_weights_fp32 = params * 4
    adam_m = params * 4
    adam_v = params * 4
    model_state_bytes = weights_fp16 + grads_fp16 + master_weights_fp32 + adam_m + adam_v

    # Extremely rough activation estimate: proportional to batch * seq_len * hidden * layers
    activation_bytes = batch_size * seq_len * hidden_dim * num_layers * 2 * 4  # fp16, few tensors/layer

    total_gb = (model_state_bytes + activation_bytes) / (1024 ** 3)
    return {
        "model_state_gb": model_state_bytes / (1024 ** 3),
        "activation_gb_rough": activation_bytes / (1024 ** 3),
        "total_gb_rough": total_gb,
    }

print(full_finetune_memory_estimate(7, seq_len=2048, batch_size=4, hidden_dim=4096, num_layers=32))
# model_state_gb ~= 112 GB, well before activations are even added
```

This is precisely the gap that parameter-efficient fine-tuning (PEFT) methods exist to close: if you can get most of the benefit of fine-tuning while updating only a small fraction of the parameters, you eliminate the optimizer-state blowup almost entirely, because Adam's extra buffers only need to exist for the parameters that are actually trainable.

## Instruction Tuning: Fine-Tuning With a Specific Purpose

Before going deeper into PEFT mechanics, it's worth placing instruction tuning correctly in this picture, because it's a common source of interview confusion. Instruction tuning is not a different algorithm from fine-tuning — it is fine-tuning (full or parameter-efficient) applied to a specific kind of dataset: pairs of instructions and desired responses, such as ("Summarize this article in three bullet points," followed by an appropriate three-bullet summary), or ("Write a Python function that reverses a linked list," followed by correct code). A base, purely pretrained model is a raw completion engine — asked "What is the capital of France?" it might just as plausibly continue with "What is the capital of Germany?" because that's a statistically reasonable continuation of a list of trivia questions found somewhere in its training data. It has no learned notion that it is supposed to answer, stop, and wait. Instruction tuning is the step that teaches the model the conversational contract: recognize an instruction, produce a direct, on-topic, appropriately terminated response, and adopt something resembling a helpful assistant persona.

Datasets for instruction tuning range from human-written demonstrations (expensive but high quality, as used in the original InstructGPT work) to synthetically generated instruction-response pairs (as in Self-Instruct or Alpaca-style datasets, where a strong model generates training examples for a weaker one), to curated mixtures of both spanning many task types — question answering, summarization, coding, reasoning, safety refusals — so the resulting model generalizes to instructions it has never literally seen. The loss function is unchanged: it is still next-token cross-entropy, just computed (often exclusively) over the response tokens rather than the instruction tokens, so the model isn't wasting gradient signal learning to predict the human's side of the conversation.

The academic ancestor of this whole approach is Google's FLAN line of work (Wei et al., "Finetuned Language Models Are Zero-Shot Learners," and later Flan-T5/Flan-PaLM), which demonstrated something that was not obvious at the time: if you take dozens of existing NLP datasets — translation, summarization, natural language inference, sentiment classification, and so on — reformat every one of them into a natural-language instruction plus response, and fine-tune a single pretrained model on the resulting mixture, the model doesn't just get better at those specific tasks, it gets measurably better at *following instructions for entirely unseen task types*. This is the empirical basis for calling instruction tuning "multi-task" fine-tuning: the goal is never to master any one dataset, it's to expose the model to enough surface variety in how instructions are phrased and structured that it learns the general skill of "parse an instruction, do the thing." Super-NaturalInstructions pushed this idea further by assembling over 1,600 distinct NLP tasks with expert-written task definitions into a single benchmark and training mixture, showing that generalization to held-out tasks continues to improve as the number of distinct training tasks grows — even holding the total number of training examples roughly fixed — which is direct evidence that task *diversity*, not just data volume, is what drives instruction-following generalization. Modern industrial instruction-tuning mixtures (used in models like Llama, Mistral, and Claude) descend from this lineage but blend in far more open-ended chat, coding, tool-use, and multi-turn conversation data than the FLAN-style academic mixtures, since real assistant usage looks far less like a classic NLP benchmark task than the FLAN datasets did.

It's important to be clear about where instruction tuning stops and where the next chapter's material — RLHF, DPO, and preference-based alignment — begins. Instruction tuning (also called supervised fine-tuning, or SFT, in the alignment literature) teaches the model *what a good response looks like* via direct demonstration: here is an instruction, here is the correct completion, minimize the difference. It cannot teach subtler notions of relative quality — that response A is better than response B even though both are individually plausible completions — because that requires comparative human judgment, not a single gold-label target. That comparative signal, and the reinforcement-learning or direct-preference-optimization machinery used to exploit it, is the subject of the next chapter. In production alignment pipelines, instruction tuning (SFT) is almost always stage one, providing a reasonable starting policy, with preference-based methods layered on top to further refine tone, helpfulness, and safety in ways demonstration data alone struggles to capture.

## Low-Rank Adaptation (LoRA)

### The Core Hypothesis

LoRA, introduced by Hu et al. in 2021, starts from an empirical observation about over-parameterized networks: when you fine-tune a large pretrained model on a downstream task, the *update* to the weights — not the weights themselves, but the delta the optimizer wants to apply — tends to have a low "intrinsic rank." In other words, even though a weight matrix might be, say, 4096 x 4096 (over 16 million entries), the meaningful change that fine-tuning wants to make to that matrix can be well-approximated by a much lower-dimensional structure, something more like the information content of a rank-8 or rank-16 matrix. This isn't a claim that the pretrained weights themselves are low-rank — they're not — it's a claim specifically about the *update* ΔW that fine-tuning would otherwise compute and add to the frozen base weight W₀.

If that hypothesis holds, and empirically it holds surprisingly well across a wide range of models and tasks, it suggests a very direct optimization: instead of letting ΔW be a full-rank, densely parameterized matrix that the optimizer is free to fill in arbitrarily (which is what full fine-tuning does implicitly), constrain ΔW from the start to be the product of two much smaller matrices, and only train those.

### The Math

For a frozen pretrained weight matrix W₀ ∈ R^(d×k) (say, a query or value projection inside an attention layer), full fine-tuning would learn an unconstrained update ΔW ∈ R^(d×k) and compute the adapted weight as W₀ + ΔW. LoRA instead factorizes:

```
ΔW = B @ A
```

where B ∈ R^(d×r), A ∈ R^(r×k), and r is the rank, chosen to be much smaller than min(d, k) — commonly somewhere between 4 and 64 for LLM fine-tuning, versus a d and k that are typically in the thousands. The forward pass for a linear layer becomes:

```
h = W0 @ x + (B @ A) @ x = W0 @ x + B @ (A @ x)
```

Notice that A projects the input down into an r-dimensional bottleneck, and B projects that bottleneck back up into the original output dimension. The number of trainable parameters drops from d*k (for full ΔW) to r*(d+k) — for a 4096 x 4096 matrix with r=8, that's roughly 16.7 million parameters for the full update versus about 65,000 for the LoRA factorization, a reduction of over two orders of magnitude for that single matrix. Applied across all the targeted matrices in a 7B model, LoRA fine-tuning commonly trains well under 1% of the total parameter count.

Crucially, because Adam's optimizer state is proportional to the number of *trainable* parameters, that same reduction applies to the memory blow-up discussed earlier: with 99%+ of parameters frozen, there's no gradient, no momentum buffer, and no variance buffer to store for them. This is the direct mechanism by which LoRA turns a 112 GB full fine-tuning memory footprint for a 7B model into something that comfortably fits on a single high-end consumer or prosumer GPU.

Initialization matters here and is a detail worth remembering: A is initialized with small random values (typically sampled from a Gaussian), while B is initialized to all zeros. This guarantees that ΔW = BA = 0 at the very start of training, so the adapted model is numerically identical to the base pretrained model before any fine-tuning steps happen — training starts from exactly the pretrained model's behavior and gradually diverges, rather than starting from a randomly perturbed version of it.

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """A linear layer with a frozen base weight and a trainable low-rank
    LoRA adapter running alongside it. This mirrors how libraries like
    Hugging Face's PEFT wrap existing nn.Linear layers."""

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False  # freeze the pretrained weight entirely

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank  # alpha/r scaling, discussed below

        # A: (rank, in_features) projects down into the bottleneck
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        # B: (out_features, rank) projects back up
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B=0 => delta_W = 0 at init

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)                      # frozen W0 @ x
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling           # W0 @ x + (alpha/r) * B @ A @ x

    def trainable_parameters(self):
        return [self.lora_A, self.lora_B]


# Wrapping an existing projection layer
base_q_proj = nn.Linear(4096, 4096, bias=False)
lora_q_proj = LoRALinear(base_q_proj, rank=8, alpha=16)

trainable = sum(p.numel() for p in lora_q_proj.trainable_parameters())
frozen = sum(p.numel() for p in lora_q_proj.base_layer.parameters())
print(f"trainable: {trainable:,}  frozen: {frozen:,}")
# trainable: 65,536   frozen: 16,777,216  -- roughly 0.4% of this matrix's parameters
```

### Merging at Inference Time

One of LoRA's most practically important properties is that the adapter can be merged back into the base weight after training with zero added inference cost or latency. Because the adapted weight is simply W₀ + (alpha/r)·BA, and matrix addition is associative, you can precompute `W_merged = W0 + scaling * (B @ A)` once, offline, and deploy `W_merged` exactly as you would have deployed a fully fine-tuned weight matrix — same shape, same inference-time compute graph, no extra matrix multiplications at serving time. This is a significant advantage over some other PEFT methods (like prefix tuning, discussed below) that do add inference-time overhead because they can't be algebraically folded into existing weights.

The alternative to merging is to keep the adapter separate and add its contribution at inference time via the `B @ (A @ x)` path shown in the code above. This costs a small amount of extra compute and memory per forward pass, but it buys you the ability to swap adapters per request without reloading or re-merging the base model — which is exactly what makes multi-LoRA serving possible.

### Practical Hyperparameters

Two hyperparameters dominate LoRA configuration decisions. The rank `r` controls the capacity of the adapter — how expressive an update it can represent. Small ranks (4–16) are sufficient for many tasks, especially style adaptation, narrow-domain adaptation, or instruction tuning on top of an already-capable base model; larger ranks (32–128) are used when the target task is more different from the pretraining distribution or when more behavioral change is genuinely needed. Going arbitrarily high on rank erodes LoRA's efficiency benefit and, past a certain point, tends not to improve quality much further, which is itself evidence for the low intrinsic rank hypothesis the method is built on.

The alpha parameter is a scaling factor applied to the LoRA update (shown as `alpha/r` in the code above), and it functions similarly to a learning-rate multiplier specifically for the adapter's contribution. A common practical convention is to set alpha to roughly twice the rank (e.g., r=8, alpha=16), though this is a heuristic rather than a law, and many teams tune it directly like any other hyperparameter.

The third major decision is which weight matrices to target. The original LoRA paper found that applying it just to the attention query and value projection matrices (Wq and Wv) captured most of the benefit at very low parameter cost, leaving the key and output projections, and the feed-forward/MLP layers, untouched. In current practice, particularly for QLoRA-style fine-tuning discussed next, it has become common to instead target *all* linear layers in the transformer block — Q, K, V, output projection, and both up and down projections in the MLP — because the extra parameter cost is still small relative to full fine-tuning, and empirically this "target everything" approach often closes more of the gap to full fine-tuning quality. The right choice is workload-dependent: attention-only targeting is a reasonable default when compute is tightly constrained, while all-linear-layer targeting is the more common recommendation when the goal is to get as close as possible to full fine-tuning performance.

### Multi-LoRA Serving

Because a LoRA adapter is small (often tens of megabytes, versus tens of gigabytes for the base model) and its contribution can be added or removed cheaply, it becomes practical to serve many different fine-tuned "personalities" from a single loaded copy of the base model. A multi-tenant inference server can keep one frozen base model resident in GPU memory and hot-swap between dozens or hundreds of different LoRA adapters — one per customer, one per task, or one per fine-tuned "skill" — loading whichever adapter a given request needs and applying it via the unmerged `B @ (A @ x)` path. Systems like S-LoRA and the multi-adapter serving support in vLLM implement exactly this pattern: a batch of concurrent requests can even be routed through *different* adapters simultaneously in the same forward pass, since the adapter computation is cheap enough to apply per-request inside an otherwise shared, batched computation. This is a major cost advantage over the alternative of fully fine-tuning and hosting a separate multi-billion-parameter model per customer — you get most of the customization benefit of full fine-tuning while paying for only one base model's worth of GPU memory, plus a small adapter footprint per tenant.

## QLoRA: Fine-Tuning Even Larger Models on a Single GPU

QLoRA (Dettmers et al., 2023) pushes the same underlying idea — freeze the base model, train only a small low-rank adapter — one step further by also shrinking the memory footprint of the frozen base weights themselves through quantization, without meaningfully sacrificing fine-tuning quality. The combination is what makes it possible to fine-tune a 65B-parameter model on a single 48 GB GPU, or a 7B model comfortably on a single consumer GPU with under 12 GB of memory — configurations that would be entirely out of reach for full fine-tuning and difficult even for standard LoRA at full precision.

QLoRA rests on four technical ingredients working together:

**4-bit NormalFloat (NF4) quantization of the frozen base model.** Rather than quantizing weights into a generic 4-bit integer format, QLoRA uses a data type specifically designed around the empirical fact that pretrained neural network weights tend to be approximately normally distributed around zero. NF4 is an "information-theoretically optimal" quantization scheme for such data: instead of evenly spaced quantization bins (as in plain integer quantization), it places bin boundaries at the quantiles of a standard normal distribution, so that each of the 16 representable 4-bit values covers an equal amount of probability mass under a normal distribution rather than an equal amount of numeric range. This concentrates representational precision where the weight values actually are (near zero) instead of wasting representational capacity on the sparse tails, which is exactly why NF4 preserves fine-tuning quality far better than naive 4-bit integer quantization at the same bit width.

**Keeping the LoRA adapters themselves in higher precision.** Only the frozen base weights are quantized to 4-bit. The trainable LoRA matrices A and B are kept in a higher-precision format (typically BFloat16), and the forward pass dequantizes the relevant slice of base weights on the fly to compute in that higher precision, adds the LoRA contribution, and discards the dequantized copy. Gradients only ever flow into the LoRA parameters, never into the frozen 4-bit base weights, so no precision-sensitive optimizer state needs to exist for the quantized parameters at all.

**Double quantization.** Standard quantization schemes store a scaling constant (and often a zero-point) per block of weights so that dequantization can recover approximately the right numeric range for that block; the smaller the block, the better the accuracy, but the more scaling constants you need to store, and those constants themselves consume non-trivial memory at scale. QLoRA quantizes the quantization constants a second time — treating the (already small) FP32 per-block scaling factors as their own dataset and quantizing them into a lower-precision representation. This second layer of compression is reported to save on average around 0.37 bits per parameter across the whole model, which sounds small per-parameter but adds up meaningfully at the multi-billion-parameter scale QLoRA targets.

**Paged optimizers.** Even with a quantized base model and a tiny trainable adapter, GPU memory usage during fine-tuning is not perfectly flat — certain operations, particularly gradient checkpointing recomputation on long sequences, can produce sudden, transient memory spikes that would otherwise cause an out-of-memory crash even though average memory usage is comfortably within budget. QLoRA uses NVIDIA's unified memory feature to designate the optimizer state as "pageable," meaning that if a spike would exceed available GPU memory, the relevant pages are automatically and transparently moved to CPU RAM and paged back in when needed, exactly analogous to how an operating system pages virtual memory to disk under pressure. This converts what would be a hard crash into a graceful, if momentarily slower, continuation of training.

```python
def nf4_quantize_block(weights_fp32, nf4_levels):
    """Simplified illustration of NF4-style quantization for one block of weights.
    Real NF4 uses 16 precomputed quantile-based levels derived from a standard
    normal distribution; this shows the mechanism, not the exact production values."""
    import numpy as np

    absmax = np.max(np.abs(weights_fp32))          # per-block scaling constant
    normalized = weights_fp32 / absmax              # values now roughly in [-1, 1]

    # Snap each normalized weight to its nearest representable NF4 level
    quantized_indices = np.array([
        np.argmin(np.abs(nf4_levels - w)) for w in normalized
    ])
    return quantized_indices, absmax  # absmax is what double quantization compresses further


def nf4_dequantize_block(quantized_indices, absmax, nf4_levels):
    return nf4_levels[quantized_indices] * absmax


# NF4's 16 levels are asymmetric quantiles of N(0,1), denser near zero -- shown schematically
nf4_levels_schematic = [-1.0, -0.70, -0.52, -0.39, -0.28, -0.18, -0.09, 0.0,
                          0.08, 0.16, 0.25, 0.34, 0.44, 0.56, 0.72, 1.0]
```

Why does this combination work as well as it does? The intuition is that the frozen base model doesn't need to be numerically precise — it just needs to be *close enough* to the original pretrained weights that the model's existing knowledge and capabilities are preserved, since none of the actual learning during fine-tuning happens in those weights anyway. All of the adaptation, all of the gradient-based learning, happens in the small, full/higher-precision LoRA matrices. Quantization error introduced into the frozen backbone acts a bit like a fixed, small perturbation to the starting point rather than an error that compounds during training, which is why QLoRA fine-tuning has been shown empirically to match full 16-bit LoRA fine-tuning quality remarkably closely, despite the base model living in 4 bits.

## Other PEFT Methods

LoRA and QLoRA dominate current practice, but it's worth knowing the earlier and adjacent PEFT family, both because interviewers sometimes probe the landscape broadly and because understanding the alternatives clarifies exactly what makes LoRA's approach distinctive.

**Adapters** (Houlsby et al., 2019) predate LoRA and take a different structural approach: rather than modifying existing weight matrices, adapters insert small new bottleneck feed-forward modules directly into the architecture, typically after the attention sublayer and after the MLP sublayer within each transformer block. Each adapter module down-projects the sublayer's output into a small bottleneck dimension, applies a nonlinearity, up-projects back to the original dimension, and adds the result back in via a residual connection. Only the adapter modules' parameters are trained; everything else stays frozen. The key practical difference from LoRA is that adapters sit *in the computation path sequentially* — every forward pass must run through them — which adds inference latency that cannot be eliminated by merging, since the nonlinearity between the down- and up-projections prevents the adapter from being algebraically folded back into the surrounding frozen weights the way LoRA's purely linear update can be.

**Prefix tuning** (Li & Liang, 2021) takes yet another approach: instead of modifying any weights at all, it prepends a sequence of trainable, continuous ("soft") vectors to the keys and values that attention operates over, at every layer of the network. These prefix vectors don't correspond to any real token or embedding in the vocabulary — they're free parameters optimized directly by gradient descent to steer the frozen model's behavior, effectively acting as a learned, per-layer conditioning signal that every attention operation in the network attends to. Because a fresh set of prefix vectors is injected at every layer, prefix tuning has more capacity to influence deep, layer-specific behavior than a method that only touches the input.

**Prompt tuning** (Lester et al., 2021) is prefix tuning's simpler sibling: it learns continuous vectors too, but prepends them only once, at the input embedding layer, rather than injecting fresh vectors into every layer's keys and values. This means prompt tuning has dramatically fewer trainable parameters than prefix tuning, but it also relies entirely on the frozen network's own layers to propagate whatever signal those input vectors encode all the way through the depth of the model — which is why prompt tuning was empirically found to need a sufficiently large base model to be competitive; on smaller models the signal from a handful of input-layer vectors doesn't propagate with enough strength to meaningfully change deep-layer behavior.

The throughline that distinguishes all three of these from LoRA is *where the trainable capacity lives and how it interacts with the frozen network*. Adapters add new sequential layers into the forward pass. Prefix and prompt tuning add new *tokens'* worth of learned context that the frozen network attends to. LoRA instead adds a low-rank correction directly to the existing weight matrices, running in parallel to (not in sequence after, and not as an input to) the frozen computation. That parallel, purely linear structure is exactly why LoRA can be merged into the base weights with zero inference-time cost, which the other three methods generally cannot do as cleanly, and it's the primary reason LoRA (and QLoRA) became the default choice in most production PEFT pipelines.

## When to Use What: A Production Decision Framework

Choosing among full fine-tuning, LoRA/QLoRA, prompt-based PEFT, and simply investing more in prompting or retrieval-augmented generation is ultimately a cost-benefit judgment that depends on a handful of concrete factors, and a senior engineer should be able to reason through it explicitly rather than reflexively reaching for the most sophisticated tool.

Start by asking whether the problem can be solved without touching weights at all. If the model already has the requisite knowledge and capability and the issue is that it doesn't have the right *information* at generation time — it doesn't know your company's internal documentation, or today's data — that's a retrieval problem, and RAG will outperform fine-tuning at a fraction of the cost and with the enormous practical advantage that the underlying knowledge base can be updated by editing documents rather than retraining a model. Fine-tuning is a poor tool for injecting volatile factual knowledge; it's a good tool for changing *behavior* — format, tone, task-specific reasoning patterns, domain-specific style, following a particular structured schema reliably. If better prompting (few-shot examples, more explicit instructions, chain-of-thought scaffolding) closes the gap, that should usually be exhausted first, since it requires no training infrastructure, no data collection pipeline, and no risk of degrading other capabilities.

Once you've concluded that weight updates are genuinely needed, data volume and diversity are the next filter. Full fine-tuning tends to need substantially more data to avoid overfitting a huge number of free parameters, and it delivers its biggest advantage over PEFT specifically when the target behavior is quite far from the base model's pretraining distribution — heavy domain shift, a fundamentally different output format at scale, or when you're trying to distill a genuinely new skill rather than adjust an existing one. If you have a modest dataset (a few thousand to a few tens of thousands of examples), which describes the overwhelming majority of real production fine-tuning efforts, LoRA is almost always the better starting point: it trains faster, is far less prone to catastrophic forgetting of general capabilities (because most of the network is literally frozen and cannot forget anything), and is dramatically cheaper to iterate on.

Cost and hardware constraints often make the decision for you before quality considerations even enter the picture. If you don't have access to a multi-GPU cluster with fast interconnect, full fine-tuning of anything beyond a small model is simply off the table, and QLoRA's ability to fine-tune large models on a single GPU becomes the deciding factor rather than one option among several.

Latency and serving architecture matter more than people initially expect. If a merged LoRA adapter is used, there is zero inference-time latency penalty versus a fully fine-tuned model, which removes latency as a consideration entirely in the single-tenant case. But if you need multi-tenancy — many customers or many task variants served from shared infrastructure — LoRA's small adapter footprint and hot-swappability make it not just cost-efficient but often the *only* practical option, since hosting a fully fine-tuned multi-billion-parameter model per tenant is rarely economical at any reasonable scale. Prompt and prefix tuning are rarely the right production default today; they were important stepping stones in PEFT research and remain useful in constrained settings (e.g., extremely tight parameter budgets, or scenarios needing per-task conditioning vectors that must remain separable from model weights for auditability), but LoRA generally dominates them on the combination of quality, tooling maturity, and the zero-latency merge property.

The pragmatic default for most production teams in 2026 looks like this: reach for better prompting and RAG first; if that's insufficient and you have a moderate, well-curated dataset, use LoRA (QLoRA if hardware-constrained); reserve full fine-tuning for cases with large, high-quality datasets, substantial compute budgets, and a genuine need to shift the model far from its pretrained behavior, such as building a foundation-model derivative that a whole product line depends on.
