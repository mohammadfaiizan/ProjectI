## Model Merging, Distillation, and Model Souping

### 0. Scope of This File

This file covers three related, but mechanically distinct, techniques for getting more value out of models you already have, without running a full fresh training process: knowledge distillation (compressing a large teacher's behavior into a smaller student), model souping (averaging the weights of multiple independently fine-tuned checkpoints), and model merging more broadly (algebraically combining multiple specialized fine-tunes into one model via task-vector arithmetic). All three are, in a real sense, cheaper alternatives to full retraining, and all three have appeared already in this module as forward references -- File 001, Section 6.1 and File 005, Section 6 both lean on distillation mechanics this file now derives properly; File 006, Section 4.3 previews model merging as an alternative to resolving the generality-versus-depth tension entirely within one training run.

### 1. Knowledge Distillation

**1.1 The basic setup.** Distillation (Hinton, Vinyals, and Dean, 2015, "Distilling the Knowledge in a Neural Network") trains a smaller **student** model to reproduce a larger, already-trained **teacher** model's behavior, using the teacher's output distribution as the training target rather than (or in addition to) the ground-truth hard labels alone. The key insight motivating this over ordinary supervised training on hard labels: a teacher's full output probability distribution over all classes/tokens carries strictly more information than just its single argmax prediction, and that extra information -- how confidently second-best and third-best options were ranked, which wrong answers the teacher considered plausible versus implausible -- is a genuinely useful training signal Hinton et al. termed **"dark knowledge."**

**1.2 Temperature-scaled softmax.** To make the teacher's distribution more informative as a training target, distillation typically applies a **temperature** `T > 1` to soften the teacher's (and the student's, symmetrically) output distribution before comparing them:

```
softmax_T(z)_i = exp(z_i / T) / sum_j exp(z_j / T)
```

At `T = 1` this is the ordinary softmax. As `T` increases, the distribution flattens -- probabilities that were near-zero for confidently-ruled-out classes become non-negligible, surfacing the relative ordering and relative confidence the teacher placed on options it didn't select, information that is nearly invisible in the sharp, near-one-hot distribution an ordinary (low-temperature) softmax produces for a well-trained, confident model.

**1.3 The full distillation loss.** The standard distillation objective combines a **soft-label term** (matching the teacher's temperature-softened distribution) with an ordinary **hard-label term** (matching the true label, at normal temperature), weighted by a coefficient `alpha`:

```
L_KD = alpha * T^2 * KL( softmax_T(z_teacher) || softmax_T(z_student) )   +   (1 - alpha) * CE(y_true, softmax(z_student))
```

The `T^2` factor multiplying the soft-label term is not a stylistic choice -- it corrects for the fact that the gradient of the soft-label cross-entropy term with respect to the student's logits scales as `1/T^2` once temperature scaling is applied, so multiplying the loss term by `T^2` keeps the *relative magnitude* of the soft-label gradient comparable to the hard-label gradient's magnitude regardless of what `T` is chosen -- without this correction, increasing `T` to extract more dark knowledge would simultaneously and unintentionally shrink the soft-label term's effective contribution to the overall gradient.

**1.4 A full, runnable implementation:**

```python
def distillation_loss(student_logits, teacher_logits, true_labels, T=4.0, alpha=0.7):
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

    hard_loss = F.cross_entropy(student_logits, true_labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

Note that the teacher's logits are computed once, with `torch.no_grad()` (the teacher is frozen throughout distillation, exactly as a reward model or reference model is frozen during RLHF, File 002) -- distillation's computational shape is "one extra frozen forward pass per training step," a modest, well-understood overhead relative to ordinary supervised training.

**1.5 Where reasoning-trace SFT (File 001/005) fits on this spectrum.** It is worth being precise about a distinction that is easy to blur: the reasoning-distillation technique in File 005, Section 6 -- SFT on a teacher's generated reasoning traces -- is a **degenerate, hard-label-only special case** of the general distillation framework above, not the full soft-distribution-matching technique. Training a student on a teacher's *sampled text output* via ordinary cross-entropy is equivalent to setting `alpha = 0` in the loss above and using the teacher's greedily-or-sampled-decoded sequence as the hard label, discarding the teacher's full token-level probability distribution (the dark knowledge) entirely in favor of just its realized sample. This is simpler to implement (no need for the student and teacher to share a tokenizer/vocabulary, no need to keep the teacher loaded and queryable during the student's training run at all, since the traces can be pre-generated once and reused indefinitely) and is why it's the dominant practical technique for transferring reasoning behavior across model families or even across labs' models, at the cost of discarding the extra soft-label signal Section 1.1-1.3 describe as distillation's core advantage over plain hard-label training. **Full logit-level distillation** (matching the complete output distribution, not just a sampled sequence) requires the teacher and student to share a tokenizer and be queryable at training time, and is used more in settings where teacher and student are from the same model family/lab and both are available as live, queryable models throughout training.

### 1.6 Distillation as a Deployment-Cost Lever, Independent of Any Alignment Motivation

It's worth being explicit that distillation's original and still most common motivation has nothing to do with alignment or behavior transfer specifically -- it is, first and foremost, a **model-compression** technique: a large, expensive-to-serve teacher's competence is compressed into a smaller, cheaper-to-serve student, trading a (hopefully small) capability gap for a substantial reduction in inference cost, latency, and memory footprint. This deployment-economics motivation is independent of, and predates, the specific reasoning-distillation application in Section 1.5 and File 005 -- the same core technique (Sections 1.1-1.4) is used whenever a lab wants to ship a smaller model that captures as much as possible of a larger, already-trained model's competence, for cost reasons alone, with no reasoning-specific or RL-discovered-behavior angle involved at all. Keeping this general-purpose framing in view alongside the reasoning-specific application prevents a common narrowing of the concept, where "distillation" gets discussed as if it were invented for, or only applicable to, reasoning-trace transfer.

### 2. Model Souping: Weight Averaging Across Fine-Tuned Checkpoints

**2.1 The technique.** Model souping (Wortsman et al., 2022, "Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time") is disarmingly simple: take several independently fine-tuned checkpoints, all initialized from the *same* pretrained base model but trained with different hyperparameters, data orderings, or minor data variations, and **average their weights directly**, parameter by parameter:

```python
def uniform_soup(checkpoints):
    # checkpoints: list of state_dicts, all with identical architecture/parameter shapes
    soup = {}
    for key in checkpoints[0].keys():
        soup[key] = sum(ckpt[key] for ckpt in checkpoints) / len(checkpoints)
    return soup
```

The resulting averaged model is used directly for inference -- no ensembling at inference time (which would multiply inference cost by the number of checkpoints), just one model with averaged weights, at the exact same inference cost as any single one of the constituent checkpoints.

**2.2 Why this works at all -- the loss-landscape argument.** Naively, averaging the weights of two differently-trained neural networks should produce a nonsensical model, since neural network loss landscapes are generally understood to have many distinct, differently-labeled symmetric minima (permuting neurons within a layer produces an equivalent-function but differently-parameterized model), and averaging two arbitrary minima's parameters would land somewhere between two unrelated, incompatible solutions with no guarantee of being good at either. Model souping works specifically because its constituent checkpoints are **not** arbitrary, independently-initialized minima -- they all start from the *same* pretrained initialization and undergo comparatively small further fine-tuning, which empirically tends to keep them within the same broad region ("basin") of the loss landscape, connected by a path of similarly-low loss (a phenomenon called **linear mode connectivity**). When two solutions lie in the same basin and are linearly connected by a low-loss path, the straight-line average of their parameters is itself likely to lie on or near that low-loss path, rather than falling into a high-loss region between two unrelated basins -- this is the precise, citable mechanism (not mere coincidence) behind why weight averaging of same-initialization fine-tunes tends to produce a model at least as good as, and often better than, any individual constituent, since averaging over independent, different-in-detail fine-tuning noise can act as a variance-reduction mechanism analogous to ensembling, but realized in parameter space rather than in prediction space.

**2.3 Why this fails outside these conditions.** The linear-mode-connectivity argument in 2.2 is explicitly conditioned on the checkpoints sharing an initialization and staying within one basin -- averaging weights of models trained from *different* random initializations, or fine-tuned so aggressively/divergently that they've moved into genuinely different basins, typically produces a materially worse model than either constituent, because there is no reason to expect a straight-line path between two unrelated basins to stay in a low-loss region; the interpolated point can land in a high-loss "barrier" between them. This is the single most important boundary condition to state precisely in an interview: **souping is not "averaging any two good models" -- it is specifically "averaging multiple fine-tunes of the same base model that have not diverged too far from a shared basin."**

**2.4 Greedy souping.** Rather than always averaging every available checkpoint uniformly, Wortsman et al.'s **greedy soup** procedure sorts candidate checkpoints by their individual held-out validation performance (best first), then iteratively adds each subsequent checkpoint to the running average *only if doing so does not decrease held-out performance*, skipping checkpoints that would hurt the current soup:

```python
def greedy_soup(checkpoints_sorted_by_val_perf, eval_fn):
    soup = checkpoints_sorted_by_val_perf[0]
    soup_size = 1
    for ckpt in checkpoints_sorted_by_val_perf[1:]:
        candidate = {k: (soup[k] * soup_size + ckpt[k]) / (soup_size + 1) for k in soup}
        if eval_fn(candidate) >= eval_fn(soup):
            soup, soup_size = candidate, soup_size + 1
    return soup
```

This directly guards against 2.3's failure mode on a per-checkpoint basis -- a checkpoint that has drifted too far from the others' shared basin, or that is simply lower-quality, is empirically detected via the held-out check and excluded, rather than blindly included and allowed to drag down the average.

### 3. Model Merging Beyond Simple Averaging: Task Vectors

**3.1 The task-vector construction.** Ilharco et al. (2023, "Editing Models with Task Arithmetic") introduces a reframing that generalizes souping into a full algebra: define a **task vector** as the *difference* between a fine-tuned model's weights and the shared pretrained base's weights:

```
tau_task = theta_finetuned - theta_base
```

This vector, living in the same parameter space as the model itself, is interpreted as encoding "the specific behavioral change fine-tuning on this task induced," relative to the shared base. Because it is a vector, it inherits vector-space operations, and the paper's central empirical claim is that these operations correspond to intuitive, useful edits to model behavior:

```
theta_new = theta_base + lambda * tau_task              # ADD a task's capability (scaled by lambda)
theta_new = theta_base - lambda * tau_task               # SUBTRACT/remove a task's capability
theta_new = theta_base + sum_i (lambda_i * tau_task_i)     # COMBINE multiple tasks' capabilities into one model
```

Ordinary uniform model souping (Section 2.1) is recovered as the special case of the combine operation where every `tau_task_i` comes from a fine-tune of the *same* task (different hyperparameters/seeds rather than genuinely different tasks) and every `lambda_i = 1/N` -- task arithmetic is a strict generalization, allowing genuinely different tasks' vectors to be combined, scaled individually, or subtracted, rather than only uniformly averaging same-task variants.

**3.2 A minimal implementation:**

```python
def task_vector(theta_finetuned, theta_base):
    return {k: theta_finetuned[k] - theta_base[k] for k in theta_base}

def apply_task_vectors(theta_base, task_vectors, lambdas):
    merged = {k: v.clone() for k, v in theta_base.items()}
    for tau, lam in zip(task_vectors, lambdas):
        for k in merged:
            merged[k] = merged[k] + lam * tau[k]
    return merged
```

**3.3 Why "subtraction" (negating a task vector) is a genuinely useful, non-obvious operation.** Beyond combining capabilities, subtracting a task vector -- `theta_base - lambda * tau_task` -- is used to *remove* or *suppress* a specific behavior a fine-tune induced, without retraining: if a fine-tune on toxic or undesired content produced a task vector capturing "the direction in weight space that induces this behavior," subtracting a scaled version of that same direction from a deployed model can reduce the corresponding behavior, a lightweight, training-free intervention distinct from any of the RLHF/DPO/RLVR techniques in Files 002-005, operating directly in weight space rather than via any further gradient-based training at all.

### 3.4 A Worked Numeric Illustration

Consider a single scalar parameter (standing in for one weight in a real model, for illustration) with base value `theta_base = 1.0`. A coding fine-tune shifts this parameter to `1.3` (task vector `tau_code = +0.3`), and a math fine-tune shifts the same parameter to `1.2` (task vector `tau_math = +0.2`) -- both fine-tunes agree in direction, so a combined model `theta_base + tau_code + tau_math = 1.5` plausibly captures both tasks' intended shift on this parameter without much loss, consistent with Section 3.1's combine operation working well when task vectors agree. Now suppose a third, safety-tuned fine-tune shifts the *same* parameter to `0.7` (task vector `tau_safety = -0.3`) for reasons specific to that fine-tune's own objective -- naively summing all three gives `theta_base + 0.3 + 0.2 - 0.3 = 1.2`, which partially cancels the safety fine-tune's intended shift on this parameter without actually resolving *which* of the conflicting directions should have won, an example, at the scale of one parameter, of exactly the sign-conflict problem Section 4.1 describes at the scale of a full model with millions of such conflicts scattered across its parameters. TIES's elect-sign step would instead look at the (trimmed) sign distribution across the three vectors on this parameter (`+, +, -`), elect the majority sign (`+`), and average only the two agreeing contributions (`(0.3 + 0.2)/2 = 0.25`), discarding the safety vector's disagreeing contribution at this specific position entirely rather than letting it partially and un-interpretably cancel the majority signal.

### 4. Handling Conflicts When Merging Many Task Vectors: TIES and DARE

**4.1 The problem uniform combination runs into at scale.** Simply summing many task vectors (Section 3.1's combine operation) works reasonably for a small number of not-too-dissimilar tasks, but degrades as more, more diverse task vectors are combined, for two related reasons: **sign conflicts** (different task vectors pushing the same parameter in opposite directions, which partially or fully cancel when summed, losing both tasks' intended effect on that parameter) and **redundant/interfering small-magnitude noise** (many parameters in any given task vector are the result of fine-tuning noise rather than a task-relevant signal, and summing many such vectors' noise components can accumulate into a meaningfully corrupting perturbation even though no single vector's noise was individually large).

**4.2 TIES-Merging (Yadav et al., 2023).** Directly targets both failure modes with a three-step procedure applied before combining task vectors: **Trim** -- zero out all but the top-k% largest-magnitude values in each task vector, discarding the low-magnitude entries most likely to be noise rather than task-relevant signal; **Elect sign** -- for each parameter position, determine the majority sign across all (trimmed) task vectors being merged, resolving Section 4.1's sign-conflict problem by an explicit voting rule rather than letting conflicting contributions silently cancel; **Merge (disjoint mean)** -- for each parameter position, average only the contributions from task vectors that agree with the elected sign at that position (excluding, rather than including with a canceling opposite sign, the disagreeing ones):

```python
def ties_merge(task_vectors, trim_frac=0.8):
    trimmed = [trim_to_topk(tau, keep_frac=1 - trim_frac) for tau in task_vectors]  # zero out smallest values
    merged = {}
    for k in trimmed[0]:
        stacked = torch.stack([tau[k] for tau in trimmed])            # shape: (num_tasks, *param_shape)
        sign = torch.sign(stacked.sum(dim=0))                          # elected majority sign per position
        agreeing_mask = (torch.sign(stacked) == sign.unsqueeze(0))     # which task-vectors agree with the sign
        agreeing_values = stacked * agreeing_mask
        counts = agreeing_mask.sum(dim=0).clamp(min=1)
        merged[k] = agreeing_values.sum(dim=0) / counts                # mean over only the agreeing contributions
    return merged
```

**4.3 DARE (Drop And REscale; Yu et al., 2024).** Takes a related but distinct approach to the same underlying noise problem: **randomly** (rather than magnitude-selectively, as TIES's trim step does) drops a large fraction of each task vector's parameters to zero, then **rescales** the surviving parameters by `1/(1 - drop_rate)` to preserve the task vector's expected magnitude despite the random pruning. The random-drop-plus-rescale combination is reported to reduce cross-task interference similarly to TIES's more deliberate trim step, with the interesting empirical finding that a surprisingly high drop rate (a large majority of parameters dropped) can be tolerated with little quality loss, consistent with the general observation that fine-tuning deltas tend to be highly redundant -- most individual parameter changes contribute only a small, individually-dispensable amount to the fine-tune's overall behavioral shift, which is exactly the kind of redundancy that both TIES's trimming and DARE's random dropping are separately exploiting from different angles.

### 4.4 Evaluating a Merged Model Before Shipping It

Because merging offers no theoretical guarantee of quality the way a validated training run's loss curve does, a merged model needs its own dedicated evaluation pass before being trusted, structured around the specific failure modes Sections 2-4 identify: per-source-task held-out evaluation (does the merged model retain each constituent fine-tune's original capability, measured on that task's own held-out set, not just an aggregate score across all tasks); a check for unexpected interference on tasks *not* represented in any constituent fine-tune (since Section 4.1's noise-accumulation argument implies merging can degrade general capability even on unrelated tasks, not just trade off between the merged tasks themselves); and, where safety- or refusal-relevant fine-tunes are among the constituents, a specific check that merging hasn't diluted or partially canceled that fine-tune's intended effect (exactly Section 3.4's worked illustration, now framed as a pre-ship evaluation requirement rather than only a mechanism to understand). A merged model that passes an aggregate benchmark check but has not been evaluated this way can silently ship with a materially weakened safety fine-tune, entirely invisible in a general capability score.

### 4.5 Self-Distillation and Ensemble Distillation as Related Variants

Two variants of the basic teacher-student setup (Section 1) worth being able to name alongside standard distillation: **self-distillation**, where a model is distilled from an earlier or larger-capacity version of *itself* (or, in the iterated setting, from its own past checkpoint), used both as a training-efficiency technique and, more speculatively, as a possible contributor to improved generalization via the same soft-label dark-knowledge argument in Section 1.1 even when teacher and student share an architecture; and **ensemble distillation**, where the teacher is not a single model but an ensemble of several independently trained models, whose averaged (or otherwise combined) output distribution is distilled into a single student -- a way to capture much of an ensemble's accuracy benefit (ensembles reliably outperform any single constituent model on average) without paying the ensemble's full multi-model inference cost at serving time, since the distilled student is a single model at ordinary single-model inference cost. Both variants use the identical loss machinery from Section 1.3-1.4; what varies is only what serves as the teacher's output distribution.

### 5. When Merging Works, and When It Doesn't

| Condition | Merging tends to work | Merging tends to fail |
|---|---|---|
| Shared base model / architecture | Yes -- required | No shared parameter space to merge in at all if bases differ |
| Task similarity | Related, complementary tasks (e.g., merging a coding fine-tune with a math fine-tune) | Highly dissimilar or actively conflicting objectives (e.g., merging a maximally-verbose fine-tune with a maximally-terse one) |
| Magnitude of each fine-tune's task vector | Small-to-moderate deltas, staying within a shared basin (Section 2.2) | Large, aggressive fine-tunes that have moved far from the shared initialization into a different basin |
| Number of task vectors combined | A handful, or many if using TIES/DARE-style conflict resolution | Many, combined via naive uniform summation with no conflict handling |
| Architecture-sensitive components | Dense feed-forward/attention weights merge comparably well | Components with permutation symmetry across differently-trained checkpoints (e.g., independently-trained MoE expert assignments) may need alignment/permutation-matching before merging makes sense at all |

### 5.1 A Worked Diagnostic Scenario

Suppose a team merges a coding-specialist fine-tune and a long-context-summarization fine-tune (both from the same base model) via uniform task-vector addition, and observes the merged model performs well on coding but noticeably worse than the standalone summarization fine-tune on long-document summarization specifically. Working through Section 5's table: first, check task similarity -- coding and summarization are not obviously conflicting objectives, so this alone doesn't predict failure; second, check the magnitude of each task vector -- if the summarization fine-tune involved a comparatively aggressive, high-learning-rate adaptation (common when adapting to a very different input-length regime than the base model's typical training distribution), its task vector may have larger-magnitude, more basin-departing deltas than the coding fine-tune's, meaning a uniform, unweighted combination effectively under-weights summarization's needed contribution relative to how much it actually needed to move the base model's weights. The fix suggested directly by Section 3.1's algebra: don't default to `lambda_i = 1/N` for every task vector -- treat the individual scaling coefficients as tunable hyperparameters (searched via held-out per-task evaluation, echoing Section 4.4's evaluation practice), giving the summarization task vector a larger relative weight to compensate for its larger basin-departure, rather than concluding merging has failed outright and abandoning it for a full joint retraining run.

### 5.2 Merging Across Different Architectures or Tokenizers: Why It Doesn't Work

Worth stating as an explicit boundary, since it is a natural follow-up question: none of the techniques in this file apply when the models being combined do not share an identical parameter space -- different architectures (even different layer counts or hidden dimensions of an otherwise similar transformer family), or the same architecture with different tokenizers/vocabularies (making embedding and output-head parameters fundamentally non-comparable position-by-position), have no well-defined notion of "the same parameter" to average or subtract at all. This is precisely why distillation (Section 1), which operates at the level of output *distributions* rather than raw *parameters*, remains the only technique in this file applicable across genuinely different architectures or tokenizers -- a fact worth stating explicitly to preempt the natural but incorrect follow-up question of whether TIES or DARE could be adapted to merge, say, a Llama-family and a Qwen-family checkpoint directly.

### 6. Merging Versus Full Retraining or Distillation: a Cost-Benefit View

Merging's core value proposition is cost: producing a new, multi-capability model via a handful of tensor-arithmetic operations on already-existing checkpoints costs a tiny fraction of what a fresh joint training run over a combined dataset (File 006's mixture-design problem) or a full distillation pipeline (Section 1) would cost, since it requires no additional gradient computation or data curation at all beyond whatever produced the original constituent checkpoints. The corresponding limitation is a ceiling on integration quality: a merged model's capabilities are bounded by how well the underlying task vectors' directions in weight space actually compose (Section 5's conditions), whereas a joint training run or distillation process can, in principle, learn genuinely new, integrated representations that serve multiple objectives jointly rather than being limited to linearly combining separately-learned deltas. The practical decision, mirroring File 006, Section 4.3's framing: merging is the right tool for combining capabilities cheaply when the source fine-tunes are reasonably compatible (Section 5) and a modest quality ceiling is an acceptable tradeoff for cost and speed; a fresh joint training run or a dedicated distillation pipeline is the right tool when the target capability combination is demanding enough, or the source fine-tunes divergent enough, that merging's ceiling is likely to bind in practice.

### 6.1 A Note on Compute Accounting

To make Section 6's cost claim concrete: producing a merged model via task-vector arithmetic requires essentially zero additional GPU-hours beyond whatever produced the constituent fine-tunes -- the merge operation itself is a handful of tensor additions/subtractions over the model's parameters, executable on a single machine in the time it takes to load the checkpoints from disk, in sharp contrast to a joint retraining run (which requires the full training compute budget of File 006's mixture-design problem, applied to a combined dataset) or a distillation pipeline (which requires however many forward/backward passes the student's training schedule specifies, plus the teacher's frozen forward passes at every step, per Section 1.4). This multiple-orders-of-magnitude compute gap is the single strongest practical argument for trying merging first, with the explicit understanding (Section 6's ceiling argument) that it may not reach joint-training-quality integration, and falling back to joint retraining or distillation only if a merging-based evaluation pass (Section 4.4) reveals the ceiling is actually binding for the specific capability combination at hand.

### 7. Common Interview Traps

- **Claiming model souping works for any two models.** Section 2.3's basin/initialization condition is the load-bearing caveat; averaging weights of independently-initialized models is not souping's claimed regime and typically produces a badly broken model.
- **Confusing distillation's soft-label mechanism with the reasoning-trace-SFT technique in File 005.** Section 1.5 makes this precise: trace-based reasoning distillation is a hard-label-only degenerate case, not full logit-level distillation, and the distinction matters for what infrastructure (a live, queryable teacher versus pre-generated traces) each requires.
- **Forgetting the `T^2` correction in the distillation loss.** A common, easy-to-miss implementation bug; without it, changing the temperature hyperparameter has an unintended side effect on the soft/hard loss balance.
- **Treating task-vector subtraction as equivalent to RLHF-style unlearning or safety training.** Section 3.3's weight-space subtraction is a lightweight, training-free heuristic operating on a specific fine-tune's captured direction; it is not a substitute for the RL/preference-based techniques in Files 002-005 and provides no guarantee of removing a behavior as robustly as targeted retraining would.
- **Presenting TIES/DARE as solving merging's fundamental limitations rather than mitigating a specific noise/conflict problem.** Both are real, effective, and worth naming specifically, but Section 5's basin/similarity conditions still bound what any conflict-resolution scheme can achieve when the underlying task vectors are too dissimilar or too large in magnitude.
- **Assuming merging works across different architectures or tokenizers.** Section 5.2 is explicit that this is a hard boundary, not a matter of degree -- there is no shared parameter space to operate on at all, and distillation is the correct fallback technique specifically because it works at the level of output distributions rather than raw parameters.
- **Defaulting to uniform weighting (`lambda_i = 1/N`) for every task vector without considering that different fine-tunes may have departed the shared basin by very different amounts.** Section 5.1's worked scenario is the direct illustration of why this default can silently under-serve a task with a larger, more basin-departing delta.
- **Shipping a merged model on the strength of an aggregate benchmark score alone.** Section 4.4's per-source-task and interference-specific evaluation practice exists precisely because an aggregate score can hide a materially weakened safety or refusal fine-tune within an otherwise-fine-looking merge.

### 8. Quick-Reference Summary

- Knowledge distillation trains a student to match a temperature-softened teacher distribution (capturing "dark knowledge" beyond the argmax label), combined with an ordinary hard-label loss, with a `T^2` correction term keeping the two loss components' gradient scales comparable across different temperature choices.
- Reasoning-trace SFT (Files 001/005) is a hard-label-only degenerate case of full distillation, trading away the soft-label signal for the practical convenience of not needing a live, queryable teacher during student training.
- Model souping averages the weights of multiple same-base-model fine-tunes directly, working because such fine-tunes tend to stay within a shared, linearly-connected loss-landscape basin (linear mode connectivity) -- not because weight averaging is generically sensible across arbitrary models.
- Greedy souping adds checkpoints to a running average only when doing so doesn't hurt held-out performance, directly guarding against including a checkpoint that has drifted too far from the shared basin.
- Task vectors (fine-tuned minus base weights) generalize souping into a full algebra: adding, scaling, subtracting, and combining multiple tasks' vectors in weight space, with uniform souping recovered as the special case of averaging same-task variants.
- TIES-Merging (trim low-magnitude noise, elect a majority sign per parameter, average only agreeing contributions) and DARE (random dropping plus rescaling) both address the sign-conflict and noise-accumulation problems that plague naive summation of many, more diverse task vectors.
- Merging works best for related tasks on a shared base with moderate-magnitude deltas, and degrades or fails outright for dissimilar bases, highly divergent fine-tunes, or many conflicting objectives combined without conflict resolution.
- Merging trades a much lower cost than full joint retraining or distillation for a ceiling on integration quality bounded by how well the underlying weight-space directions actually compose -- the right tool when source fine-tunes are compatible and a modest ceiling is acceptable, not a universal replacement for joint training.
- Merging requires shared architecture and tokenizer/vocabulary; distillation is the correct fallback whenever teacher and student differ in either, since it operates on output distributions rather than raw, position-aligned parameters.
- Self-distillation and ensemble distillation reuse the identical distillation loss machinery, varying only what serves as the teacher's output distribution -- a model's own past checkpoint, or a combined ensemble's output, respectively.
- Uniform task-vector weighting (`1/N`) is a default, not a rule -- fine-tunes that departed the shared basin by different amounts often need individually-tuned scaling coefficients, found via per-task held-out evaluation.
- Producing a merged model costs orders of magnitude less compute than a joint retraining run or a distillation pipeline, making it the reasonable first attempt whenever the source fine-tunes are plausibly compatible, with joint retraining or distillation as the fallback once a rigorous evaluation pass shows merging's ceiling is actually binding.
- A merged model's pre-ship evaluation must include per-source-task held-out checks and an explicit check for unintended weakening of any safety/refusal-relevant constituent fine-tune, not just an aggregate benchmark score.

### 8.1 A Final Checklist Before Relying on a Merge or a Distilled Student

- Do all constituent checkpoints share an identical base model, architecture, and tokenizer (Section 5.2)? If not, stop -- use distillation instead, not a merging technique.
- Have task-vector magnitudes been compared across constituents, and has uniform weighting been checked against per-task-tuned weighting (Section 5.1)?
- Has the merge been evaluated per-source-task on held-out data, not just via one aggregate score (Section 4.4)?
- If a safety- or refusal-relevant fine-tune is among the constituents, has its specific effect been verified to survive the merge intact (Section 3.4, Section 4.4)?
- For a distilled student, is the distillation using full soft-label matching (requiring a live, queryable, tokenizer-matched teacher) or hard-label trace imitation (requiring only pre-generated text), and does the infrastructure in place actually match which of the two was intended (Section 1.5)?
- Has the `T^2` gradient-scale correction been included in the distillation loss if temperature scaling is used at all (Section 1.3)?

### 9. Cross-References

File 001, Section 6.1 and File 005, Section 6 both rely on the hard-label distillation mechanics this file's Section 1.5 makes precise. File 006, Section 4.3 previews model merging as an alternative to resolving the generality-versus-depth tension within a single training mixture; this file's Sections 3-5 supply the actual mechanics of how that alternative works and where its limits are. File 002's frozen-reference/frozen-reward-model pattern is the same "keep a second model resident and query it without updating it" computational shape this file's Section 1.4 notes recurs in distillation's frozen-teacher forward pass. File 007's alignment-tax discussion and this file's Section 3.3 task-vector-subtraction technique are worth contrasting directly: both are ways of thinking about a model's behavior as movable in a structured direction, but the alignment tax describes an emergent, hard-to-fully-control side effect of gradient-based preference optimization, while task-vector subtraction is a deliberate, explicit, training-free weight-space edit -- different tools for related, but not identical, "change what this model does without a full retraining run" goals.

Across this entire module (Files 001-008), the recurring meta-lesson is that every technique examined trades a specific, nameable limitation of a simpler predecessor for a specific, nameable new cost or constraint of its own: SFT's imitation ceiling motivates preference optimization; RLHF's learned-proxy fragility motivates DPO's directness and RLVR's ground-truth exactness; RLHF's human-labeling cost motivates RLAIF's AI-feedback substitution; and full joint retraining's cost motivates this file's merging and distillation shortcuts. A staff-level command of this module is less about memorizing any one technique in isolation and more about being able to produce this whole chain of "what specific problem did this solve, and what did it cost" reasoning on demand, for any pair of techniques an interviewer names.

### 9.1 A Closing Practical Note

The techniques in this file are unusual within the module in one respect worth flagging explicitly: every other file (001-007) describes a *training-time* intervention, requiring gradient computation, data curation, and a full training run's worth of infrastructure. Distillation still requires a training run (albeit against a fixed, frozen teacher rather than a moving reward signal), but souping and task-vector merging require none at all -- they are post-hoc, inference-time-cost-free operations on already-existing weights. This makes them a distinctive, disproportionately cheap lever in a staff engineer's toolkit specifically for the "we already have several good models and need one model that captures what's good about each of them, quickly" problem, as opposed to the "we need a model to newly acquire a capability it doesn't have at all" problem, which remains squarely the domain of the training-time techniques in Files 001-005.
