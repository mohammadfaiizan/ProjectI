## Supervised Fine-Tuning (SFT)

### 0. Where This Sits in the Pipeline

Every modern instruction-following model is produced by a pipeline with roughly this shape: pretrain on raw text with a next-token-prediction objective, then post-train in one or more stages that convert "a model that predicts plausible continuations of internet text" into "a model that behaves like a helpful assistant." Supervised fine-tuning (SFT) is almost always the first post-training stage, and it is the least conceptually exotic one -- it uses exactly the same loss function as pretraining.

What changes between pretraining and SFT is the data distribution, plus one masking detail in how the loss is computed over that data. Despite that apparent simplicity, SFT is not a minor preprocessing step to rush past on the way to RLHF or DPO. It sets the initialization point for every later stage: it is the policy that RL fine-tuning starts from, the reference distribution that KL penalties are measured against, and the behavioral scaffold that later preference data is sampled around. Its data-quality choices have an outsized effect on final model behavior relative to how simple the SFT objective itself is.

This file covers SFT mechanically (Sections 1-2), then spends most of its depth on the two things a staff-level engineer actually needs to reason about precisely: why SFT alone is a structurally insufficient objective for producing a good assistant (Section 3), and how SFT data quality and diversity choices propagate into downstream model behavior in ways that are easy to underestimate (Sections 4-5).

### 1. The Mechanics: Continued Pretraining on Curated Pairs

A pretrained base model is a function `p_theta(x_t | x_<t)` -- a distribution over the next token given all previous tokens, trained by maximizing the log-likelihood of naturally occurring text under a strictly causal, left-to-right autoregressive factorization. SFT does not change this factorization or the model architecture at all.

What SFT changes is the *data*. Instead of training on arbitrary web, book, or code text, you train on a curated set of `(prompt, response)` pairs, where each pair is a demonstration of the behavior you want: a human, or a stronger model (Section 5.4), wrote or approved the response as a good answer to the prompt.

A training example is built by concatenating the prompt and response into a single token sequence, usually wrapped in a chat template that marks role boundaries with special tokens:

```
<|user|> Explain why the sky is blue. <|assistant|> The sky appears blue because ... <|end|>
```

This concatenated sequence is fed through the model exactly as in pretraining, and the loss is standard next-token cross-entropy over the whole sequence:

```
L(theta) = - sum_t log p_theta(x_t | x_<t)
```

The one mechanically important difference from raw pretraining is **loss masking**: the sum over `t` runs only over token positions belonging to the response (the assistant's turn), not the prompt or any system/role-marker tokens. In code, this is typically implemented with a per-token label mask:

```python
# input_ids: the full concatenated (prompt + response) token sequence
# response_start: index of the first response token
labels = input_ids.clone()
labels[:response_start] = -100          # -100 is the standard "ignore this position" sentinel
                                          # for torch's F.cross_entropy / nn.CrossEntropyLoss

logits = model(input_ids).logits         # forward pass sees the *unmasked* full sequence
loss = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, vocab_size),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
```

Why mask the prompt at all, given that the model still needs to *condition* on it? Because conditioning and predicting are different roles for the same tokens. The model must see the prompt as context -- it flows through causal self-attention exactly as any other token would -- but you do not want gradient to push the model toward becoming better at *predicting the user's question*. That would spend training signal on modeling the prompt distribution, which is arbitrary and not the behavior you are trying to shape, instead of on modeling the response distribution, which is exactly the behavior you want. This is a purely computational-graph distinction: masked tokens still participate in the forward pass and attention computation; they simply contribute zero gradient via the loss.

In multi-turn conversation data this generalizes to masking every user/system turn and computing loss only over assistant turns, across the whole conversation. There is a genuine engineering choice about whether to compute loss over *every* assistant turn in a multi-turn example or only the final one. Masking-in all assistant turns is the more common production choice, since throwing away supervision on earlier turns is wasteful, and the model needs practice generating good responses conditioned on realistic -- not just ground-truth -- conversation history.

A second mechanical detail that recurs throughout post-training is **packing versus padding**. Pretraining corpora are typically packed: many documents concatenated back-to-back into fixed-length sequences with document-boundary tokens, to avoid wasting compute on padding. SFT examples are much shorter and highly variable in length, so naively packing several *unrelated* prompt-response pairs into one sequence risks cross-contamination through attention unless attention is explicitly masked to block cross-example attention (a block-diagonal attention mask, or per-example position-id resets). Getting this wrong is a real, easy-to-miss bug: without a proper block mask, a packed example's response tokens can attend to a *different, unrelated* example's prompt tokens earlier in the packed sequence, silently corrupting the training signal in a way that will not show up as a crash -- only as unexplained quality degradation.

### 1.1 A Worked Example of Loss Masking, With Numbers

It helps to walk through a tiny concrete case rather than only reasoning about masking abstractly. Suppose the tokenized sequence for a training example is:

```
index:   0    1    2    3    4    5    6    7    8    9
token:  <u>  Why   is  the  sky  blue <a>  Ray leigh scatter ... <end>
```

where `<u>` marks the start of the user turn and `<a>` marks the start of the assistant turn. Tokens 0-5 are the prompt (including the role marker), and tokens 6 onward are the response. The `labels` tensor used for the loss is `[-100, -100, -100, -100, -100, -100, 6, 7, 8, 9, ...]` where the surviving entries are simply the *next* token's id shifted appropriately by the causal shift shown in the code snippet above (predicting token `t+1` from the hidden state at position `t`). Concretely: the loss has a term for "given `<u> Why is the sky blue <a>`, predict `Ray`," a term for "given `... <a> Ray`, predict `leigh`," and so on through the end-of-turn token -- but zero loss terms for "given `<u>`, predict `Why`" or any other prompt-internal prediction.

Two consequences of this that are easy to get wrong in a first implementation: (1) the causal shift means the label at position `t` should be the *token id at position t+1*, not the token id at position `t` itself -- an off-by-one error here silently trains the model to predict the current token from itself, which will not crash but will produce a badly degraded model; and (2) the end-of-turn/end-of-sequence token itself should typically be *included* in the loss (with a real label, not masked), because the model needs to learn *when to stop generating* just as much as it needs to learn what to generate -- a model trained without loss on the stop token will tend to ramble past where a good response should end.

### 1.2 Typical Hyperparameters, as a Sanity-Check Reference

Exact figures vary by lab and are often undisclosed for frontier models, but the following orders of magnitude are broadly consistent with public recipes (OpenAssistant, Alpaca/Vicuna-style open replications, and the qualitative figures InstructGPT reports) and are useful as a sanity check when reasoning about a de novo SFT run: dataset size on the order of `10^4` to low `10^6` examples; 1-3 epochs (rarely more, given the overfitting risk in Section 4); peak learning rate roughly `10x` to `100x` smaller than the pretraining peak LR (commonly in the `1e-6` to `1e-5` range for a large model, versus `~1e-4`-`3e-4` at pretraining scale); a short linear or cosine warmup followed by decay to zero over the SFT run rather than the long, slowly-decaying schedules used in pretraining; and effective batch sizes far smaller than pretraining batch sizes, since the dataset itself is orders of magnitude smaller and large-batch training would run out of unique data within a fraction of an epoch's worth of steps.

### 1.3 Sequence Packing With a Block-Diagonal Attention Mask

Because SFT examples are short and variable-length, naively padding every example out to the longest sequence in a batch wastes enormous amounts of compute on positions that carry no signal. The fix is to pack several examples into one fixed-length sequence, but this requires the attention mask to prevent tokens from one packed example attending to tokens from a different packed example -- otherwise the model silently learns spurious cross-example dependencies. A minimal illustration:

```python
def build_packed_batch(examples, max_len):
    """examples: list of (prompt_ids, response_ids) tuples to pack into one sequence."""
    input_ids, labels, block_ids = [], [], []
    for i, (prompt_ids, response_ids) in enumerate(examples):
        input_ids += prompt_ids + response_ids
        labels += [-100] * len(prompt_ids) + response_ids
        block_ids += [i] * (len(prompt_ids) + len(response_ids))   # which example each token belongs to

    input_ids, labels, block_ids = input_ids[:max_len], labels[:max_len], block_ids[:max_len]

    # attention_mask[i, j] = True only if tokens i and j belong to the same packed example
    # AND j <= i (causal). This is what actually prevents cross-example attention leakage.
    block_ids_t = torch.tensor(block_ids)
    same_block = block_ids_t.unsqueeze(0) == block_ids_t.unsqueeze(1)
    causal = torch.tril(torch.ones(len(block_ids), len(block_ids), dtype=torch.bool))
    attention_mask = same_block & causal
    return torch.tensor(input_ids), torch.tensor(labels), attention_mask
```

The `same_block & causal` mask is the crux: `causal` alone (the ordinary lower-triangular mask every decoder uses) is not sufficient once multiple unrelated examples share one sequence, because it would happily let a later example's tokens attend to an earlier, unrelated example's tokens simply because they occur earlier in the packed sequence. Some training frameworks instead handle this by resetting position ids at each example boundary and relying on a specialized flash-attention "varlen" kernel that accepts per-sequence length metadata rather than materializing the full mask -- functionally equivalent, but far more memory-efficient at scale, since materializing an `O(n^2)` boolean mask per batch is itself a nontrivial memory cost at long packed-sequence lengths.

### 1.4 Chat-Template Consistency: a Silent Failure Mode

A failure mode specific to SFT (and to every later stage that reuses the same policy) that is easy to overlook until it bites you in production: the chat template used to format `(prompt, response)` pairs during SFT training must be *exactly* the template used at inference time, down to whitespace and special-token placement, because the model has learned a distribution conditioned on those exact token sequences, not on the semantic content of "a user turn." If the serving stack applies a template with a different role-marker convention, a missing or extra newline, or a different system-prompt insertion point than what training used, the model is being asked to generate conditioned on an out-of-distribution prompt format -- this typically does not crash anything, it simply degrades quality in a way that is easy to misattribute to "the model is bad at this task" rather than "the harness re-templated the input differently than training expected." This is a mundane-sounding but genuinely common source of quality regressions when a model is fine-tuned by one team and served through infrastructure maintained by a different team, and it is worth naming explicitly as a staff-level operational concern, not just a research one.

### 2. Why "Supervised" Undersells What's Happening -- the Imitation-Learning Framing

It is worth being precise about vocabulary, because it clarifies the failure mode in Section 3. SFT is, in the classical ML sense, exactly **behavioral cloning** applied to text generation: you are training a policy (the language model, viewed as a policy over token-generation actions) to match a fixed dataset of demonstrated trajectories (the human-written responses), using maximum likelihood.

This framing matters because behavioral cloning has well-studied failure modes in the imitation-learning literature -- compounding error under distribution shift, and the implicit assumption that demonstrations exhaustively cover the states the policy will encounter at deployment -- and these transfer directly to language-model SFT (Section 3.3). Knowing this literature lets you reason about SFT's limitations from first principles rather than from memorized folklore about "SFT models are worse than RLHF models."

A useful table for keeping the distinction crisp:

| Property | SFT (behavioral cloning) | Preference optimization (RLHF/DPO) |
|---|---|---|
| Supervision signal | One demonstrated response per prompt | A comparison between two or more sampled responses |
| Objective | Match the demonstration exactly (MLE) | Increase the margin between preferred and dispreferred responses |
| Trains on model's own samples? | No -- always conditions on ground-truth prefixes | Yes (on-policy RL) or on off-policy samples (DPO on a fixed preference set) |
| Can express "these are all okay, but this one is better"? | No | Yes, directly |
| Failure mode if data is inconsistent | Model hedges between inconsistent modes | Preference model can still learn a consistent ordering from noisy comparisons |

### 3. Why SFT Alone Is Structurally Insufficient

There are several distinct arguments here, and a strong interview answer distinguishes them rather than gesturing vaguely at "RLHF is just better."

**3.1 Maximum likelihood on demonstrations is mode-covering, not mode-seeking, and it targets the wrong distribution entirely.** Standard MLE training minimizes the forward KL divergence `KL(p_data || p_theta)` between the empirical data distribution and the model. Forward KL is *mode-covering*: wherever the data distribution puts nonzero mass, the loss penalizes the model heavily for putting near-zero mass there, because `log p_theta(x)` blows up toward negative infinity as `p_theta(x) -> 0` for an `x` the data says is likely. So the model is pushed to spread probability mass broadly enough to cover everything in the training set, including any noise or inconsistency across how different demonstrations were written.

But the actual target you want to hit is not "the empirical distribution of things labelers happened to write." It is "the distribution of responses a human would judge as *best*." For almost every nontrivial prompt there are many valid responses of very different quality, only one of which was written down as *the* demonstration. SFT has no mechanism to express "this response was fine, but this other hypothetical response would have been much better" -- it only ever sees one exemplar per prompt and has no contrastive signal explaining *why* that exemplar was chosen over alternatives. A preference-based method (Files 002/003) directly optimizes "prefer good responses over worse ones sampled from the model's own distribution," which is a fundamentally different and more informative training signal than "reproduce this one written response."

**3.2 Writing a demonstration and identifying the best of several candidates are different (and differently reliable) human tasks.** A labeler asked to demonstrate a good response is producing their own generation, under time pressure, typically without sampling and comparing multiple candidates first. A labeler asked to *compare* several already-generated candidates is doing a psychologically easier and more reliable task -- discriminating between options -- than unconstrained generation-and-self-assessment. This is a large part of why preference-comparison data (Section 6 of File 002) tends to be cheaper to collect at scale and more internally consistent than demonstration data: you are asking humans to do the thing they are comparatively good at (recognizing a better answer when they see one) rather than the thing they are comparatively worse at (reliably generating the objectively best answer from scratch, every time, under time pressure).

**3.3 Exposure bias and compounding error under distribution shift.** Behavioral cloning trains the model exclusively on prefixes drawn from the ground-truth demonstration distribution -- at training time the model always conditions on a *correct* prefix, either the true prompt or, in multi-turn data, the human-written prior turns. At inference time the model conditions on its *own* previously generated tokens, which may already contain small errors relative to what a human would have written.

Because the model was never trained on the conditional distribution "given that I have already made a small mistake, what should I generate next to recover," small errors early in a generation can compound rather than self-correct. This is the classic exposure-bias problem from sequence-to-sequence learning, and it is structurally the same problem regardless of decoder architecture. RL fine-tuning, by contrast, trains on rollouts sampled from the *policy's own* distribution, so it directly sees, and gets penalized or rewarded for, the actual trajectories the model produces -- including its own errors. This is one of the more underrated reasons RL-based post-training outperforms pure SFT, independent of the preference-modeling argument in 3.1.

**3.4 SFT has no notion of "the model already knows this, leave it alone" versus "the model needs to learn this."** Every gradient step from an SFT example nudges every parameter touched by that example's forward pass, regardless of whether the model already assigned the correct token high probability (small but nonzero gradient, still perturbs weights) or was confidently wrong (large, useful gradient). Vanilla SFT has no built-in mechanism to preferentially learn from informative examples and leave already-mastered behavior undisturbed. This is part of the mechanistic story behind the forgetting problem in Section 4, and part of why some post-training recipes lean on more targeted, on-policy signals -- RL, or rejection-sampling-plus-SFT ("RAFT"/"ReST"-style pipelines, where you sample many completions from the current policy, keep only the best-scoring ones under a reward or verifier, and SFT on those) -- that only reinforce the model where it is *actually* deficient relative to some standard, rather than re-deriving all demonstrated behavior from scratch on every gradient step.

**3.5 The ceiling-effect argument.** An SFT model's quality is bounded above by the quality and consistency of whoever, or whatever, wrote the demonstrations, because the training signal *is* "match this specific text." If demonstrations are inconsistent -- different labelers with different styles, different notions of "helpful," different levels of rigor on hard technical questions -- the model is being asked to match a moving target. Maximum-likelihood training under an inconsistent target distribution produces a model that hedges between the inconsistent modes rather than confidently producing any one of them: a concrete, mechanistic explanation for the hedging, wishy-washy tone often observed in SFT-only models, distinct from any explicit "be balanced" instruction in the data.

Preference-based post-training partially escapes this ceiling because a reward or preference model can, in principle, learn a *smoother, more consistent* notion of quality across many comparisons than any single demonstration exhibits, and RL can push the policy toward responses that no single demonstration ever contained but which the learned quality signal endorses.

None of this means SFT is dispensable -- see Section 6 -- but a staff-level answer to "why not just do more SFT" needs to be this precise: it is not merely "SFT models are worse." It is "the SFT objective is imitation of a fixed demonstration distribution, which cannot by construction express a preference ordering over the many valid responses to a prompt, cannot train on the model's own error modes, and is capped by demonstration consistency" -- three separate, separately-fixable-or-not limitations, not one vague deficiency.

### 4. Catastrophic Forgetting During SFT

**4.1 The mechanism.** Catastrophic forgetting is the general neural-network phenomenon where gradient updates optimized for a new, narrower data distribution overwrite parameter configurations that encoded previously learned, broader behavior. There is no architectural mechanism in a standard transformer that protects "knowledge acquired during pretraining" from being perturbed by fine-tuning gradients, because both stages update the exact same parameters via the exact same kind of gradient descent.

Pretraining exposes the model to an enormous, extremely diverse distribution -- trillions of tokens spanning nearly every domain, register, and implicit task present in web-scale text. SFT datasets are, by comparison, tiny (tens of thousands to low millions of examples) and comparatively narrow, reflecting whatever set of task types, response lengths, and stylistic conventions the curators chose. Fine-tuning on a narrow distribution for even a few epochs can measurably shift the model's behavior away from capabilities that were well-represented in pretraining but under-represented, or absent, in the SFT set.

**4.2 Concrete manifestations, not just an abstract worry.** In practice, forgetting during SFT shows up as several distinct symptoms:

- **Degraded few-shot in-context learning.** The pretrained model's raw ability to pick up a task pattern from examples in the prompt can regress after heavy SFT on a fixed instruction-following format, because the model has been pushed toward always responding in "assistant mode" rather than flexibly adapting to whatever pattern the prompt establishes.
- **Narrowed output diversity and stylistic collapse.** The model defaults to whatever formatting conventions dominate the SFT set -- bullet points, a fixed disclaimer template, a characteristic opening phrase -- even when a more natural, varied response would serve the prompt better.
- **Calibration degradation.** A pretrained model's next-token probabilities are reasonably well-calibrated as frequency estimates over its training corpus. Heavy SFT tends to sharpen the output distribution toward the SFT targets, which is part of why post-SFT models are typically less well-calibrated probability estimators even before any RL stage is applied.
- **Outright knowledge regressions** on tasks not represented in the SFT mix, especially with a small SFT set trained for many epochs, since the model can start overfitting to the SFT distribution's surface patterns rather than retaining pretraining's breadth.

**4.3 Mitigations, and why each works mechanistically.**

- **Few epochs, low learning rate.** SFT datasets are small enough that the model can memorize them within a handful of epochs. Production recipes typically run one to three epochs at a learning rate one to two orders of magnitude below the pretraining peak LR, so that the smallest perturbation to pretrained weights sufficient to instill the target behavior is applied, rather than letting the optimizer aggressively re-carve the loss landscape toward the SFT distribution.
- **Rehearsal / data mixing.** Mixing a small fraction of pretraining-distribution data, or a broader and more diverse instruction set, into the SFT batch acts as a regularizer that keeps gradients from moving exclusively in the direction the narrow SFT set would otherwise dictate. This is mechanically the same idea as the pretraining-loss mixing term used later in PPO (File 007's PPO-ptx), just applied one stage earlier.
- **Parameter-efficient fine-tuning (LoRA, adapters).** Restricting trainable parameters to a small, low-rank subset of the full weight matrices, or to inserted adapter modules, mechanically limits how far the model's function can move from its pretrained behavior, since the vast majority of pretrained weights are frozen and cannot be perturbed at all. This trades some fine-tuning expressiveness for a strong, built-in forgetting guardrail, and is a common choice when a lab wants many task- or customer-specific fine-tunes of the same base model without each one degrading general capability.
- **KL-to-base-model regularization even during SFT.** Some pipelines add an explicit penalty term keeping the fine-tuned model's output distribution close to the pretrained model's distribution on a held-out set of general-domain prompts -- a more direct, differentiable version of "don't forget" than epoch/LR tuning alone, at the cost of an additional forward pass through the frozen base model plus an extra hyperparameter.
- **Diverse SFT data as its own forgetting mitigation.** This connects directly to Section 5: a broad, diverse SFT mixture that itself spans many task types and domains is less likely to induce narrow overfitting than a small, homogeneous one, simply because the target distribution SFT is imitating is closer in breadth to the target distribution pretraining already represented, so there is less for the model to forget in the first place.

### 5. SFT Data Quality, Diversity, and How They Propagate Downstream

**5.1 Quality over quantity is a real, empirically supported claim, not just a slogan.** Several public results (most notably the LIMA line of work, "Less Is More for Alignment") show that a small number -- on the order of one thousand -- of extremely carefully curated, high-quality, stylistically consistent demonstrations can produce an SFT model competitive with models trained on orders of magnitude more, noisier data.

The mechanistic reading is consistent with the ceiling-effect argument in 3.5: if pretraining has already exposed the model to essentially all the world knowledge and linguistic competence it will need, the *marginal* job of SFT is not to teach new facts but to teach a narrow, specific behavioral pattern -- format, tone, "how to be an assistant rather than a document-completer." A small, consistent set of demonstrations of that pattern is a more sample-efficient teacher for a narrow behavioral shift than a large, inconsistent one, precisely because inconsistency actively degrades the signal rather than merely diluting it.

**5.2 Diversity across task types and formats is a distinct axis from quality, and both matter.** A high-quality but narrow SFT set -- excellent demonstrations, but only for short factual Q&A -- will produce a model that is excellent at short factual Q&A and poor at generalizing to instruction types not represented in training: multi-step reasoning, long-form writing, code generation, following complex multi-constraint instructions, appropriate refusal. This is a straightforward generalization argument: the model can only be expected to extend "be a helpful assistant" behavior to instruction types that are either represented in the SFT mixture, or close enough in instruction-space that the pretrained model's general capability transfers cleanly on its own.

Production SFT mixtures are therefore deliberately constructed across many axes simultaneously -- task type (open QA, closed QA, summarization, code, creative writing, classification, extraction, multi-turn dialogue, tool use), response length, difficulty, and refusal/safety-relevant edge cases -- and getting the *relative proportions* right is a genuine, empirically-tuned engineering problem. File 006 covers the multi-task mixture-balancing problem in more depth, since it is related to, but not identical to, the pretraining-mixture-balancing problem.

**5.3 Demonstration style bleeds directly into model personality, for better and worse.** Because SFT is literally training the model to imitate the surface form of the demonstrations, every consistent stylistic tic in the SFT data -- a habit of prefacing answers with an enthusiastic acknowledgment, defaulting to bulleted lists even for narrative answers, a particular hedging register, a fixed disclaimer template on sensitive topics -- becomes a learned default behavior of the model, often more durably than the writers of the labeling guidelines intended, because the model cannot distinguish "this labeler's personal writing habit" from "the deliberate house style we want."

This is a genuinely double-edged property. It is *how* labs deliberately instill a desired assistant voice: SFT is, among other things, the mechanism by which "sound like our brand of assistant" gets baked in. But it is equally the mechanism by which unintended, undesired stylistic artifacts get baked in just as durably -- and it is one reason later RLHF/DPO stages sometimes have to actively work *against* tics the SFT stage over-instilled (excessive hedging, reflexive over-listing, sycophantic praise of the user's question) rather than simply building on top of them.

**5.4 Synthetic and distilled SFT data: a cost-effective but risk-laden source.** Because human-written demonstrations are expensive and slow to collect at the scale modern instruction-tuning mixtures require, a large and growing share of production SFT data is generated by *other, usually stronger, models* -- either via self-instruct-style pipelines (a strong model generates candidate instructions and responses, which are then filtered and curated) or via direct distillation (collecting a strong teacher's completions on a fixed prompt set as SFT targets for a smaller student, discussed mechanically in File 008).

This is far cheaper per example than human authorship and can straightforwardly transfer a strong teacher's capabilities and stylistic conventions to a smaller or earlier-stage student. But it inherits the ceiling-effect problem in a new form: the student's SFT ceiling is now bounded by the teacher's quality and idiosyncrasies rather than by human demonstrators', and any systematic bias, factual-error pattern, or stylistic quirk of the teacher is faithfully reproduced, sometimes amplified, since a student trained purely on a teacher's outputs has no exposure to the teacher's own uncertainty or to cases where the teacher itself is wrong. When this loop is iterated across generations of models trained substantially on prior models' synthetic outputs rather than on a continually refreshed supply of grounded, human- or verifiably-checked data, the failure mode is sometimes called **model collapse** or **imitation degeneration**.

**5.5 Deduplication, contamination, and format leakage.** Two further data-hygiene issues are worth naming precisely, because they are easy to get wrong at scale:

- **Benchmark contamination.** SFT prompts that overlap, even partially, with public evaluation benchmarks inflate reported capability without reflecting genuine generalization -- a real and recurring practical concern given how many SFT prompt sources are scraped or aggregated from public data.
- **Format leakage.** A narrow, over-represented response format in the SFT set gets over-generalized by the model to prompts where that format is a poor fit -- a direct downstream consequence of the diversity argument in 5.2.

**5.6 Refusal and boundary-setting data is its own delicate sub-mixture.** Every production SFT mixture includes examples that demonstrate declining or redirecting a request -- disallowed content, requests the model lacks the tools or context to fulfill safely, ambiguous requests that need clarification first. This sub-mixture is delicate in both directions: too little of it, and later safety training has to do more work from a worse starting point (and the model may have already learned harmful-completion patterns from the rest of the SFT mixture that now need to be trained away rather than never instilled); too much, or too blunt an instantiation of it (a small number of nearly-identical refusal templates repeated across many examples), and the model overgeneralizes the refusal *pattern* to superficially similar but benign prompts, a direct instance of the format-leakage mechanism in 5.5 applied specifically to refusal behavior, and a documented, recurring complaint about over-cautious assistant models. Getting the refusal sub-mixture's size, diversity of phrasing, and diversity of *triggering conditions* right -- so the model learns the underlying judgment rather than a shallow lexical trigger -- is a genuinely hard, iterative data-curation problem, not a one-time checklist item.

### 6. SFT's Real Job in the Larger Pipeline

Given Section 3's critique, it is worth being precise about what SFT is actually for, because "insufficient alone" is not the same claim as "unnecessary." SFT does three load-bearing jobs that no later stage substitutes for:

1. **It converts a document-completion model into a model that reliably adopts a conversational, instruction-following *format* at all** -- stopping cleanly, addressing the user's actual request rather than continuing the prompt as if it were a document, using role structure correctly. Without this, RL or preference optimization has no sensible policy to start from; RL fine-tuning a raw pretrained model directly against a reward signal is a far harder exploration problem, since the model would have to discover the entire "behave like an assistant" pattern via sparse reward, rather than starting from a policy that already exhibits it.
2. **It provides the initialization point -- the policy -- and, typically, the reference distribution for the KL penalty, for every subsequent RLHF/DPO stage** (Files 002 and 003). The SFT model is not a discarded scaffold; it is a first-class artifact that later stages are mechanically anchored to.
3. **It is the cheapest, most controllable lever for instilling specific, known-good behavior directly.** When you already know exactly what a good response looks like for a class of prompts -- always cite sources for factual claims, always ask a clarifying question before executing a destructive action -- writing demonstrations and running SFT is a far more sample-efficient and interpretable way to instill that behavior than hoping a preference model and RL discover and reinforce it indirectly.

The staff-level framing to hold in your head: SFT is necessary-but-not-sufficient because it optimizes the wrong objective -- imitate a demonstration -- for the actual goal, which is to produce the response humans would most prefer among many valid options. But it is also foundational and irreplaceable, because every later stage in the pipeline depends on the behavioral scaffold and initialization it provides. The practical failure mode worth being able to diagnose in an interview is a team that treats "collect more SFT data" as a universal lever for every quality problem, when the actual bottleneck is a preference-ordering problem that SFT structurally cannot express. Being able to say precisely *why* SFT cannot express that ordering (Section 3) is the difference between reciting "RLHF is better than SFT" and actually understanding the mechanism.

### 6.1 SFT on Structured Outputs: Tool Calls and Reasoning Traces

Two increasingly important categories of SFT data deserve separate mention because their loss-masking and formatting considerations are slightly different from plain prose responses. **Tool-use / function-calling data** trains the model to emit a structured call (typically JSON or a JSON-like DSL naming a function and arguments) at the appropriate point in a response, sometimes interleaved with a tool result injected back into the context and a subsequent model turn that incorporates it. The masking convention here typically treats the model's own emitted function-call tokens as loss-bearing (the model must learn to produce them) while treating the *tool's response*, once it is injected back into context, as unmasked context exactly like a user turn -- the model needs to condition on the tool output but is not being trained to generate it, since the tool produced it, not the model.

**Reasoning-trace SFT** -- training on long chain-of-thought traces, either human-written or (far more commonly in current practice) distilled from a stronger reasoning model's outputs -- is mechanically identical plain-vanilla SFT (mask the prompt, compute loss over every generated token including the reasoning trace and the final answer), but it is worth flagging as its own category because it is the concrete mechanism behind the "reasoning distillation" result covered in depth in File 005: a smaller model's ability to perform extended chain-of-thought reasoning can be substantially instilled via ordinary SFT on a larger RL-trained reasoning model's traces, without the smaller model ever itself undergoing the RL training that produced those traces in the first place. This is a striking, if by-now well-established, empirical fact: SFT, the least exotic post-training stage, is sufficient to transfer a *behavior pattern* (extended deliberation, self-checking, backtracking) that was originally discovered via a much more expensive RL search process, as long as suficiently many high-quality example traces of that behavior exist to imitate.

### 6.2 Full Fine-Tuning Versus Parameter-Efficient SFT: a Practical Comparison

| Dimension | Full fine-tuning | LoRA / adapter-based SFT |
|---|---|---|
| Trainable parameters | 100% of model weights | Typically well under 1% (low-rank update matrices) |
| Forgetting risk | Higher -- every weight can move | Lower -- frozen backbone bounds how far behavior can shift |
| Optimizer memory (Adam states) | ~2x model size, in addition to weights/gradients | Proportional only to the small trainable adapter parameter count |
| Expressiveness / ceiling on behavior change | Higher -- can in principle instill any behavior representable by the architecture | Lower -- constrained to whatever the low-rank update subspace can express |
| Typical use case | Primary production post-training runs at a lab that owns the base model | Many cheap task/customer-specific fine-tunes of one shared base model; rapid iteration; serving many variants efficiently via adapter-swapping |
| Merge-ability with other fine-tunes | Nontrivial -- requires the techniques in File 008 | Often easier -- multiple LoRA adapters can be combined, swapped, or algebraically composed with less risk of destructive interference, precisely because each one occupies a small, distinct subspace of weight-space |

The choice is not purely a cost decision: at frontier-lab scale, where the SFT stage is feeding into a large, carefully tuned RLHF/DPO pipeline downstream and every increment of quality matters, full fine-tuning is the default despite its cost, because the marginal capability ceiling of LoRA-style updates is a real constraint you do not want to accept for your primary production model. LoRA-style approaches earn their keep primarily in the long tail of specialized, customer-, or task-specific fine-tunes layered on top of an already-strong shared base model, where forgetting-avoidance and cheap multiplicity of variants matter more than squeezing out the last few points of quality on any single variant.

### 7. How You'd Actually Evaluate an SFT Checkpoint

Held-out cross-entropy loss (equivalently, perplexity on masked response tokens of a held-out demonstration set) is the cheapest signal to compute and the one you should track every training step, but it is a weak proxy for what you actually care about, for exactly the reason in Section 3.1: lower held-out loss just means the model matches the held-out demonstrations more closely, and the held-out demonstrations have the same "only one exemplar per prompt" limitation as the training set. A model that has genuinely overfit to the SFT distribution's stylistic idiosyncrasies can have excellent held-out loss and still be a worse assistant than a slightly-higher-loss checkpoint that generalizes better.

Because of this, production evaluation of an SFT checkpoint almost always supplements loss with some form of **pairwise win-rate evaluation**: sample completions from the candidate checkpoint and from a reference (an earlier checkpoint, a competitor model, or the previous production model) on a fixed held-out prompt set, and have either human raters or a strong LLM judge state a preference, reporting the win rate. This is mechanically the same comparison primitive used to build reward-model training data in File 002, applied here as an evaluation tool rather than a training-data source -- worth noticing explicitly, since it means "the eval methodology for SFT" and "the data-collection methodology for the next pipeline stage" are the same underlying operation applied to different purposes. A staff-level detail worth flagging: LLM-judge win rates are subject to known biases (verbosity bias, position bias, self-preference bias when the judge and a candidate model share training lineage) discussed in depth in File 004, so a rigorous evaluation protocol randomizes response order, controls for length, and, where the stakes justify it, validates a sample of judge decisions against human raters rather than trusting the judge signal uncritically.

A third, complementary axis is targeted capability regression testing: running the SFT checkpoint against the same benchmark suite used to characterize the base pretrained model (general knowledge, reasoning, coding benchmarks) specifically to catch the catastrophic-forgetting failure mode of Section 4 before it reaches a later pipeline stage where it becomes entangled with RLHF/DPO effects and harder to attribute to the SFT step specifically.

### 8. Common Interview Traps on This Topic

- **Confusing "the model learns new facts during SFT" with what's actually happening.** SFT datasets are far too small to teach broad new world knowledge relative to what pretraining already provides; SFT's job is behavioral/format shaping, not knowledge injection. A candidate who describes SFT as "teaching the model information" rather than "teaching the model a response *pattern*" is signaling a shallow model of what the stage does -- though it is fair and correct to note that SFT *can* inject narrow, specific factual associations if they are repeatedly present in the demonstrations (e.g., a consistent company-specific fact pattern), just not broad general knowledge.
- **Treating "loss masking" as optional or a minor implementation detail.** Failing to mask the prompt tokens does not merely add noise; it actively trains the model to become better at *predicting user questions*, which is both wasted capacity and can measurably degrade instruction-following behavior if done at scale, since gradient budget that should shape response behavior is instead spent on an unrelated objective.
- **Conflating SFT with instruction tuning as if they were different things.** In current usage they largely denote the same mechanism (supervised training on prompt-response pairs); "instruction tuning" more often emphasizes the *diversity/multi-task* framing of the data mixture (File 006) while "SFT" is the more mechanism-focused term, but neither term implies a different loss function.
- **Assuming more SFT epochs monotonically improves quality.** Section 4 makes the opposite case: past a small number of epochs, additional SFT training on a fixed, narrow dataset tends to compound overfitting and forgetting rather than adding generalizable capability, and the empirically useful lever is usually data quality/diversity, not epoch count.
- **Not being able to say precisely why RLHF/DPO outperforms SFT.** "RLHF is better" is not an answer; "the SFT objective can only imitate a fixed demonstration and has no contrastive preference signal, whereas RLHF/DPO directly optimize a preference ordering over the model's own sampled outputs" is the answer a staff interview is looking for.

### 9. Quick-Reference Summary

- SFT loss: standard causal-LM cross-entropy, masked to response tokens only (`ignore_index=-100` on prompt/system tokens).
- SFT is behavioral cloning: it minimizes forward KL to a fixed demonstration distribution, which is mode-covering, not preference-ranking.
- Structural insufficiency has (at least) three independent causes: no contrastive signal among valid responses, exposure bias from never training on the model's own rollouts, and a quality ceiling set by demonstration consistency.
- Catastrophic forgetting is a real, measurable risk because SFT and pretraining update identical parameters via identical gradient descent, with no built-in protection for pretraining-acquired behavior.
- Primary mitigations for forgetting: few epochs, low LR, rehearsal/data mixing, parameter-efficient tuning, explicit KL-to-base regularization.
- Data quality and consistency matter more than raw volume past a fairly modest dataset size (the LIMA result); diversity across task types is a distinct, equally necessary axis.
- Demonstration style is learned as durably as demonstration content -- SFT is the primary lever for both intentional house style and unintentional stylistic artifacts.
- Synthetic/distilled SFT data is now the dominant source at scale, with model-collapse risk as the corresponding downside to watch for.
- SFT's irreplaceable role: establishing assistant-format behavior, providing the RL/DPO initialization and reference policy, and cheaply instilling specific known-good behaviors (including refusal boundaries) directly.
- Evaluate with held-out loss as a cheap proxy, but treat pairwise win-rate (human or LLM-judge) and targeted capability-regression testing as the signals that actually matter for production decisions.
- Chat-template mismatch between training and serving is a mundane but real, recurring source of unexplained quality regressions -- treat template exactness as a correctness requirement, not a cosmetic detail.
- Reasoning-trace SFT is the concrete mechanism behind distilling RL-trained reasoning behavior into smaller models (File 005) -- the receiving model never runs RL itself, it imitates traces via the same masked cross-entropy loss described in Section 1.

This file's cross-references forward: File 002 covers what replaces "one demonstration per prompt" with an explicit preference signal (reward modeling plus PPO); File 003 covers the direct-preference-optimization family that removes the separate reward model and RL loop entirely; File 006 covers the multi-task mixture-balancing problem for SFT/instruction-tuning data in more depth; and File 008 covers the distillation mechanics referenced in Sections 5.4 and 6.1 in full.

Read together, Sections 3 and 6 are the two halves of the single idea this file most wants you to leave with: SFT's limitation and SFT's necessity are not in tension, they are two descriptions of the same fact -- it is a behavioral-cloning stage doing a behavioral-cloning-shaped job extremely well, sitting underneath stages whose job is the different, preference-ranking-shaped problem that behavioral cloning cannot solve by construction.

If you take only one diagnostic question away from this file for use in an interview or on the job, make it this one: for any observed model-quality problem, ask "is this a *coverage* problem (the model has never seen behavior like this demonstrated) or a *preference* problem (the model has seen both good and bad versions of this and needs to learn to prefer the good one)?" -- the first is an SFT-data problem, fixable by adding targeted demonstrations; the second is structurally outside what SFT alone can fix, no matter how much demonstration data you add, and needs the machinery in Files 002 and 003 instead.
