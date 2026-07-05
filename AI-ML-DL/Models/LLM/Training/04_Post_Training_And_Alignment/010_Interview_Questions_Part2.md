## Interview Questions -- Part 2

## Q1: Explain ORPO's odds-ratio mechanism, and precisely state what it removes relative to DPO that the other DPO-family variants (IPO, KTO) do not.

ORPO combines an ordinary SFT cross-entropy loss on the preferred response with an additional preference-ranking penalty expressed via the **log odds ratio** between the preferred and dispreferred response's sequence-level probabilities, computed entirely from the model currently being trained -- with no separate frozen reference model anywhere in the expression:

```
odds(y|x) = pi_theta(y|x) / (1 - pi_theta(y|x))
L_ORPO = L_SFT(y_chosen) - lambda * log sigmoid( log(odds(y_chosen|x) / odds(y_rejected|x)) )
```

DPO, IPO, and KTO all still require a frozen reference model (typically the SFT checkpoint) to compute the log-ratio or implicit-reward term their losses depend on, and all three are explicitly *second*-stage techniques run after a separate SFT stage has already produced that reference. ORPO's distinguishing move is removing the reference model from the loss entirely -- the "contrast" in its preference term is between the currently-training model's own odds for the chosen versus rejected response, not between the model and any fixed external reference -- which also lets it collapse SFT and preference-tuning into a single training stage, run directly from a base or lightly-adapted model, rather than requiring SFT to run to completion first as a prerequisite. So the precise claim: IPO and KTO each modify one specific aspect of DPO (the loss shape, or the pairing requirement, respectively) while keeping the reference-model-plus-separate-SFT-stage structure intact; ORPO is the one variant among these that removes that structure altogether.

## Q2: Implement model souping, including the greedy variant that only merges a checkpoint if it doesn't hurt held-out performance.

```python
def uniform_soup(checkpoints):
    """checkpoints: list of state_dicts with identical keys/shapes."""
    soup = {}
    for key in checkpoints[0]:
        soup[key] = sum(ckpt[key] for ckpt in checkpoints) / len(checkpoints)
    return soup


def greedy_soup(checkpoints_sorted_by_val_perf, eval_fn):
    """
    checkpoints_sorted_by_val_perf: state_dicts sorted best-to-worst by their own
    individual held-out validation performance.
    eval_fn: takes a state_dict, returns a scalar held-out performance score (higher=better).
    """
    soup = {k: v.clone() for k, v in checkpoints_sorted_by_val_perf[0].items()}
    soup_size = 1
    best_score = eval_fn(soup)

    for ckpt in checkpoints_sorted_by_val_perf[1:]:
        candidate = {
            k: (soup[k] * soup_size + ckpt[k]) / (soup_size + 1)
            for k in soup
        }
        candidate_score = eval_fn(candidate)
        if candidate_score >= best_score:
            soup, soup_size, best_score = candidate, soup_size + 1, candidate_score
        # else: skip this checkpoint, it would hurt the running soup -- don't include it

    return soup
```

The greedy variant exists specifically to guard against averaging in a checkpoint that has drifted too far from the other checkpoints' shared loss-landscape basin (or is simply lower quality) -- uniform souping blindly includes every checkpoint regardless, which is fine if all constituents are genuinely comparable, but risks dragging down the average with a bad or divergent outlier if they're not. Sorting by individual validation performance before the greedy pass, rather than processing in arbitrary order, ensures the strongest checkpoints anchor the soup early and weaker/more divergent ones are evaluated for inclusion against an already-strong baseline rather than an arbitrary starting point.

## Q3: Explain linear mode connectivity and why it's the actual mechanism justifying weight averaging, rather than mere coincidence.

Naively, averaging the weights of two differently-trained neural networks should produce nonsense: neural network loss landscapes generally have many distinct, symmetric minima (e.g., permuting neurons within a layer produces an equivalent-function but differently-parameterized model), and averaging two arbitrary, unrelated minima's parameters has no reason to land anywhere good -- the straight-line path between two unrelated basins typically passes through a high-loss "barrier" region. Model souping's constituent checkpoints are not arbitrary independent minima, though: they share an initialization (the same pretrained base) and undergo comparatively limited further fine-tuning, which empirically tends to keep them within the same broad basin of the loss landscape, connected to each other by a path of consistently low loss -- this is what "linear mode connectivity" specifically refers to: the straight-line interpolation between two such solutions stays in a low-loss region for its entire length, rather than passing through a high-loss barrier.

Given that property, the straight-line average of two linearly-connected solutions is itself very likely to land on or near that same low-loss path, rather than in a high-loss gap between unrelated basins -- this is the precise, citable, non-coincidental mechanism behind why averaging same-initialization fine-tunes tends to work, and it directly explains the boundary condition on when souping fails: checkpoints from different initializations, or fine-tuned so aggressively/divergently that they've exited the shared basin entirely, have no guaranteed linear connectivity, and averaging them has no more reason to succeed than averaging two arbitrary, unrelated minima would.

## Q4: Scenario -- a merged model (combining a general-capability fine-tune and a safety/refusal fine-tune via uniform task-vector addition) passes all aggregate benchmarks well, but three months post-launch you discover it complies with a category of harmful requests the safety fine-tune, on its own, reliably refused. How would you have caught this before shipping, and what does it tell you about evaluating merges in general?

This is precisely the failure mode a per-source-task evaluation, rather than an aggregate benchmark check, is designed to catch, and the fact that it slipped through indicates the evaluation protocol before launch was insufficiently specific. Aggregate benchmarks -- general capability, helpfulness win-rate, even a general safety score averaged across many categories -- can look uniformly fine while masking a category-specific regression, exactly the same structural blind spot as an aggregate SFT-mixture loss metric masking a category-specific interference effect. The correct pre-launch check is to run the *original* safety fine-tune's own held-out refusal-evaluation suite against the merged model specifically, category by category, not folded into a general safety score -- if uniform task-vector addition caused sign conflicts on parameters the safety fine-tune relied on for that specific harm category (other task vectors pushing those same parameters in the opposite direction), this category-specific evaluation would have surfaced a measurable regression on exactly that category, even while other categories and the aggregate score looked fine.

The general lesson: any merge involving a safety- or refusal-relevant constituent needs a dedicated, category-broken-down evaluation of that specific constituent's intended behavior post-merge, not an inference from aggregate scores, and ideally a more robust merging technique than naive uniform addition (TIES-merging's majority-sign-election step, specifically) for exactly this reason -- naive summation offers no mechanism to prevent one constituent's parameter-level updates from being partially canceled by conflicting updates from the others, and a safety fine-tune's updates are exactly the kind of thing you cannot afford to have silently, partially canceled without it being caught before deployment.

## Q5: Implement TIES-merging's sign-election step, given several trimmed task vectors, and explain what problem it solves that naive summation doesn't.

```python
import torch

def elect_and_merge(trimmed_task_vectors):
    """
    trimmed_task_vectors: list of dicts {param_name: tensor}, already trimmed to
    keep only each vector's largest-magnitude entries (smaller entries zeroed).
    """
    merged = {}
    for key in trimmed_task_vectors[0]:
        stacked = torch.stack([tv[key] for tv in trimmed_task_vectors])  # (num_tasks, *shape)
        elected_sign = torch.sign(stacked.sum(dim=0))                     # majority-direction sign per position
        agrees = (torch.sign(stacked) == elected_sign.unsqueeze(0))
        agreeing_values = stacked * agrees
        counts = agrees.sum(dim=0).clamp(min=1)
        merged[key] = agreeing_values.sum(dim=0) / counts                  # mean over only agreeing contributions
    return merged
```

Naive summation of several task vectors adds every contribution at every parameter position regardless of direction, so if three task vectors agree that a parameter should move up and a fourth (from an unrelated or conflicting task) pushes it down, naive summation just nets these out -- silently attenuating or even reversing the majority signal, with no record of what happened or why, and no way to tell after the fact whether a given parameter's final value reflects genuine consensus or an arbitrary cancellation. The sign-election step makes this explicit and principled instead: at each parameter position, it looks at which direction the *majority* of (already magnitude-trimmed, noise-reduced) task vectors agree on, and then averages only the contributions from vectors that agree with that majority, discarding the disagreeing ones from that specific position's computation entirely rather than letting them silently cancel part of the majority's intended effect. This converts an implicit, opaque cancellation into an explicit, auditable voting rule, and empirically produces materially better merged-model quality than naive summation once more than two or three genuinely distinct task vectors are being combined.

## Q6: Why do process reward models, despite giving denser credit assignment than outcome-only verification, reintroduce a problem RLVR was specifically adopted to avoid?

RLVR's entire premise is removing the learned-proxy layer between the policy's output and the reward signal -- an outcome verifier (final-answer equality, test-suite pass rate) is a deterministic, programmatic check with no fitting step and no distributional-drift-induced miscalibration. A process reward model, in contrast, is itself a *learned* model, trained on (necessarily more expensive to collect) step-level annotations of reasoning-trace quality, and like any learned model fit to a finite training sample, it is subject to exactly the same overoptimization dynamics File 002 describes for outcome reward models in RLHF: a policy optimized hard against a fixed PRM can learn to produce steps that superficially satisfy whatever pattern the PRM was trained to reward -- plausible-looking derivation steps, favored phrasing, structural conventions the PRM associates with "good reasoning" -- without those steps being genuinely logically sound, and the PRM's ability to detect this gap degrades as the policy's output distribution drifts from the PRM's own training distribution, precisely the same mechanism as RM overoptimization in RLHF.

So the tradeoff is exactly the outcome-versus-process choice restated: outcome verification has a narrower, more trustworthy reward-hacking surface (limited to what the verifier itself can be tricked about, e.g., incomplete test coverage) but coarser, sparser credit assignment; process supervision has denser, potentially more informative credit assignment but reintroduces a learned-proxy reward-hacking surface at the step level, undoing part of RLVR's core advantage over RLHF. This is precisely why several production reasoning-RL recipes (DeepSeek-R1 among the most citable) report choosing outcome-only verification at scale specifically to avoid re-opening this attack surface, treating the narrower, more trustworthy signal as worth more than the denser but less trustworthy one, at least given current PRM training methods.

## Q7: Design a concrete evaluation to detect and quantify sycophancy in a deployed assistant model. Be specific about the experimental design, not just the concept.

The core design principle is holding the ground truth fixed and varying only the social-pressure framing, so that any observed change in the model's answer can be attributed specifically to the pressure rather than to genuine uncertainty about the underlying question. Concretely, I'd build two paired evaluation sets from a common pool of factual questions the model can be independently verified to answer correctly on its own: (1) a **pushback set** -- present the question, record the model's initial (correct) answer, then have a simulated user push back with no new supporting evidence ("I don't think that's right, are you sure?") and record whether the second answer flips away from correct; (2) a **stated-opinion set** -- present the identical factual question in two versions, one with a neutral framing and one with an embedded stated opinion or credential from the "user" ("As someone with a background in this field, I believe X -- is that right?"), and measure whether the model's answer shifts toward agreement with the stated opinion specifically in the credentialed-framing version relative to the neutral one, again holding the actual correct answer fixed across both framings.

The key metrics: **flip rate** under pushback (the fraction of initially-correct answers that flip to incorrect after a single round of unsupported pushback), and **agreement-shift rate** under stated opinion (the fraction of cases where the model's answer changes toward the stated opinion specifically because of the framing, not because the opinion happened to be correct). Both should be measured across a range of topic domains and pressure intensities (single pushback versus repeated pushback), since sycophancy severity plausibly varies by domain and by how persistent the pressure is. Crucially, this evaluation must be run separately from, and in addition to, ordinary capability/accuracy benchmarks, since a standard benchmark has no mechanism to distinguish "got the question wrong" from "got it right, then flipped under pressure" -- the entire evaluation only has diagnostic value because of the fixed-ground-truth, varied-framing design.

## Q8: Explain policy entropy collapse in GRPO-based reasoning RL: what causes it, how is it different from RLHF's mode-collapse concern, and how is it mitigated?

Entropy collapse is the phenomenon where a policy's output distribution becomes increasingly deterministic over the course of RL training, to the point that sampling `G` completions for a given prompt (GRPO's group-sampling step) starts returning nearly-identical completions. This directly starves the group-relative advantage computation of the reward variance it needs to be useful: if all `G` samples are near-identical, they receive near-identical rewards, the group's standard deviation approaches zero, and every advantage in that group approaches zero -- exactly the degenerate case where a group carries no informative gradient, but now caused by the *policy's own* loss of diversity rather than by a prompt being inherently too easy or too hard for the current policy.

This is mechanistically distinct from RLHF's mode-collapse concern, where the KL penalty to a reference policy is the direct, explicit countermeasure limiting how far the policy can concentrate away from the reference's broader distribution. GRPO, especially in configurations that relax or drop the KL penalty entirely (a reasonable choice given RLVR's reduced need for a reference-model trust region, per File 005's argument that verifier correctness doesn't degrade with policy drift the way a learned RM's does), has no equivalently direct countermeasure built into the base algorithm -- nothing else in vanilla GRPO discourages entropy collapse specifically. Mitigations developed in follow-up work (e.g., the DAPO line of work explicitly diagnosing and patching GRPO) include an explicit entropy bonus term added to the loss, decoupled/asymmetric clipping ranges that treat probability-increasing and probability-decreasing updates differently, and dynamic filtering that detects and discards zero-or-near-zero-variance groups before they enter the loss computation at all, directly preventing the degenerate case from contributing an uninformative or noisy gradient update.

## Q9: Implement the loss-masking mechanism for SFT: given a full tokenized sequence containing both a prompt and a response, construct the labels tensor correctly, including the causal shift.

```python
import torch

def build_sft_labels(input_ids, response_start_idx, ignore_index=-100):
    """
    input_ids: (seq_len,) tensor of the full concatenated prompt+response token sequence.
    response_start_idx: index of the first response token (prompt tokens are [0, response_start_idx)).
    Returns labels aligned for next-token prediction: labels[t] should be the target for
    predicting from the hidden state at position t, i.e., the token at position t+1.
    """
    labels = input_ids.clone()
    labels[:response_start_idx] = ignore_index     # mask every prompt/role-marker token
    return labels

def sft_loss(logits, input_ids, response_start_idx, vocab_size, ignore_index=-100):
    labels = build_sft_labels(input_ids, response_start_idx, ignore_index)
    # causal shift: logits at position t predict the token at position t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
```

Two correctness details worth stating unprompted: first, the causal shift -- the label at sequence position `t` must be the token at position `t+1`, not the token at position `t` itself, since the model is predicting the *next* token from the hidden state produced after seeing positions `0..t`; getting this off by one silently trains the model to predict the current token from itself, degrading quality without crashing. Second, the end-of-turn/end-of-sequence token at the end of the response should generally be included in the loss (given a real label, not masked), since the model needs explicit training signal for *when to stop generating*, not only for what to generate -- omitting it tends to produce a model that rambles past where a good response should naturally end.

## Q10: Why is the Bradley-Terry reward-model loss invariant to adding an arbitrary per-prompt constant to the reward, and what practical consequence does this have?

The loss `-log sigmoid(r_theta(x,y_w) - r_theta(x,y_l))` depends only on the *difference* between the two responses' scores, both conditioned on the same prompt `x`. If you add any constant `c(x)` (depending only on the prompt, not on which response) to `r_theta(x,y)` for every `y` under that prompt, the difference `r_theta(x,y_w) - r_theta(x,y_l)` is completely unchanged, since the added constant appears with a plus sign in both terms and cancels in the subtraction. This means the loss function, and therefore the training objective, places no constraint whatsoever on the absolute scale or offset of the reward model's scores on a per-prompt basis -- only on the relative ordering and margin between different responses to the *same* prompt.

The practical consequence: raw RM scores are not comparable in absolute terms across different prompts. A score of 2.3 for one prompt's response and a score of 2.3 for a completely different prompt's response carry no guarantee of representing comparable absolute quality -- the model was never trained to make that comparison meaningful. Any downstream use of RM scores that implicitly assumes cross-prompt comparability (e.g., filtering a large, diverse pool of generated completions by an absolute score threshold, when the pool spans many different prompts) needs to explicitly account for this, typically via per-prompt normalization or by only ever comparing scores within the same prompt's candidate set, exactly the same underlying property that motivates GRPO's per-prompt-group advantage normalization in File 005.

## Q11: You're advising a team deciding between a full RLHF (RM-plus-PPO) pipeline and a DPO-based pipeline for a new preference-tuning effort. What factors would actually drive this decision?

I'd frame this less as "which is better" and more as "which set of costs and risks is this team better positioned to bear." Key factors: (1) **in-house RL infrastructure and expertise** -- PPO requires running a genuinely more complex actor-learner-shaped system (concurrent policy generation, frozen reference, frozen reward model, value function, weight synchronization between training and rollout engines), and a team without existing, mature infrastructure for this pays a real setup and ongoing-maintenance cost that a DPO pipeline, which is structurally just supervised training with two extra frozen-model forward passes, avoids. (2) **need for online/iterated refresh** -- if the plan is to run iterated preference-optimization rounds that continually re-sample and re-label the *current* policy's own outputs to counter reward-model overoptimization (File 002, Section 6.6), PPO's on-policy rollout generation is a more natural fit than DPO's typically-static, pre-collected preference dataset; if the preference dataset is fixed and collected once, DPO's off-policy nature is less of a disadvantage. (3) **data availability** -- if available feedback is genuinely unpaired (accept/reject signals rather than explicit comparisons), neither vanilla DPO nor PPO's RM training directly fits, and KTO becomes the more natural choice within the DPO family, orthogonal to the RLHF-vs-DPO axis itself. (4) **tolerance for each method's specific failure modes** -- PPO's risk profile centers on RM overoptimization and general RL instability, manageable via KL tuning and RM refresh; DPO's centers on likelihood displacement and static-dataset overfitting, manageable via IPO-style bounding or careful dataset curation -- neither is strictly safer, and a team should pick based on which failure mode they're better equipped to monitor and correct for.

I'd also note directly that public empirical comparisons don't show a consistent capability-ceiling winner between well-tuned versions of either approach -- the dominant practical argument for DPO is lower implementation/tuning cost to reach a good-but-not-necessarily-best result, not a proven higher ceiling, so a team with strong existing RL infrastructure has less reason to default to DPO purely on quality grounds.

## Q12: Explain task-vector arithmetic, and derive precisely why subtracting a task vector can remove a behavior a fine-tune induced.

A task vector is defined as the parameter-space difference between a fine-tuned model and the shared pretrained base it was fine-tuned from: `tau_task = theta_finetuned - theta_base`. This vector is interpreted as capturing the specific direction in weight space that fine-tuning moved the model in order to acquire the task's behavior -- and because it's a vector living in the same space as the model's own parameters, it supports ordinary vector-space operations. Adding a scaled task vector to a base model, `theta_base + lambda * tau_task`, moves the model some distance along that same direction, which is the mechanism behind "installing" a fine-tuned capability into a different starting point without redoing the fine-tuning itself (assuming the target starting point is close enough to the original base for the direction to remain meaningful, i.e., they're in a comparable region of weight space).

Subtraction, `theta_base - lambda * tau_task`, moves the model in the *opposite* direction along the same learned axis. If a fine-tune on some undesired behavior (e.g., producing toxic or low-quality outputs of a specific character) produced a task vector capturing "the direction that induces this behavior," then subtracting a scaled version of that exact direction from a model exhibiting the behavior pushes the model's parameters away from the region associated with it -- a direct, training-free, purely arithmetic behavioral edit. It's important to be precise about what this does and doesn't guarantee: it's a heuristic operating on a single, specific, empirically-observed direction, with no formal guarantee that removing that exact direction removes the *entire* behavior (the behavior may be encoded partially along other directions too, or the subtraction may have side effects on unrelated capabilities that happened to share overlapping parameter directions) -- it's a lightweight lever, not a substitute for the more targeted, robustly-validated interventions RLHF/DPO/RLVR provide via actual further training.

## Q13: Implement Generalized Advantage Estimation (GAE), and explain the role of the `lambda` parameter in the bias-variance tradeoff it controls.

```python
import torch

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: (T,) per-token/per-step rewards for one rollout.
    values: (T+1,) value-function estimates, including a bootstrap V(s_T) at the final position.
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]     # one-step TD residual
        gae = delta + gamma * lam * gae                             # exponentially-weighted accumulation
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns
```

GAE computes a weighted average of `n`-step advantage estimates for every `n` simultaneously, with the weighting controlled by `lambda`. At `lambda = 0`, the recursion collapses to the pure one-step TD residual `delta_t` as the advantage -- low variance (it depends only on one step of realized reward plus the value function's own estimates) but potentially high bias if the value function itself is inaccurate, since you're trusting the value function's predictions almost entirely beyond one step. At `lambda = 1`, the recursion becomes the full Monte Carlo return minus the baseline -- unbiased (it uses only realized rewards, no dependence on the value function's accuracy beyond providing the baseline itself) but high variance, since it accumulates the full, potentially very noisy, trajectory of realized rewards. Intermediate `lambda` values interpolate between these extremes, and the typical choice (around 0.9-0.95 in practice) reflects an empirical judgment that a value function trained reasonably well is more trustworthy over a moderate horizon than over the full, sometimes very long and noisy, remainder of a trajectory -- particularly relevant in the sparse-terminal-reward LM-RLHF setting, where most of the informative signal about `delta_t` only arrives at the very last token, making the choice of `lambda` a real, non-default-safe hyperparameter rather than an incidental detail.

## Q14: Explain negative transfer in multi-task instruction tuning with a concrete mechanism, and describe how you'd distinguish it from a simple data-coverage problem in practice.

Negative transfer occurs when gradient updates that improve the model's fit to one task category actively degrade its performance on a different, sufficiently dissimilar task category, because both categories share the same underlying model parameters, and a gradient direction that reduces loss on one category's examples is not guaranteed to be neutral, let alone helpful, for a different category's examples -- it can point in a genuinely conflicting direction. Concretely, a mixture heavily weighted toward terse, single-fact QA data can shift the model's default response-length and elaboration prior toward brevity in a way that measurably degrades performance on tasks that genuinely require longer, more deliberative responses (multi-step reasoning, open-ended writing), not because the model has "forgotten" how to write at length in any absolute sense, but because the aggregate gradient signal from the QA-heavy portion of the mixture has shifted a shared, cross-cutting behavioral prior in a direction that bleeds into unrelated task types.

Distinguishing this from a coverage problem in practice: a coverage problem means the affected category was simply underrepresented or absent from training, and the fix is adding more of that category's data; a negative-transfer/interference problem means the affected category may have been perfectly well-represented in raw example count, but a *different*, dominant category's data is actively pulling shared behavior away from what the affected category needs. The diagnostic check is to compare the affected category's representation (raw count and relative proportion) before and after whatever mixture change preceded the regression -- if the proportion held steady or even increased, but performance regressed anyway, that's evidence for interference rather than coverage, and the fix is reweighting or reformatting the dominant, interfering category rather than adding more of the affected one, which would not address a problem that isn't actually a coverage shortfall in the first place.

## Q15: Scenario -- your team's coding-RLVR pipeline uses a test-suite-based reward, and after a few thousand training steps you notice the policy's pass rate on the training test suites has climbed steadily, but held-out human review shows a rising fraction of "solutions" that are degenerate (e.g., hard-coded outputs matching only the visible test cases). Diagnose and fix.

This is a direct instance of verifier-blind-spot reward hacking: the reward signal is exactly and only "did this solution pass these specific test cases," and a policy under RL optimization pressure has no incentive to distinguish "genuinely implements the algorithm" from "produces the specific outputs these specific tests check for" -- both score identically under the given reward, so once the model discovers the cheaper, more reliably-achievable degenerate strategy (memorizing or pattern-matching to the visible test inputs), RL pressure will happily reinforce it, since the reward function cannot tell the difference. The rising training pass rate alongside rising human-flagged degeneracy is exactly the divergence-between-proxy-and-true-signal pattern that should immediately raise suspicion, the same structural symptom as RLHF's reward-hacking signature (a rising proxy score with flat or declining true quality), just manifesting through a verifier's incompleteness rather than a learned RM's miscalibration.

The fix operates on the verifier itself, since that's the actual source of the exploitable gap: expand test coverage specifically to include held-out or randomized test cases not visible to the policy during training (so hard-coding against visible cases no longer suffices to pass), and/or add held-out, harder test cases sampled after training that are used purely for evaluation, never for reward, to get an unbiased read on genuine solution quality distinct from the training reward's now partially-gamed number. A complementary, more structural fix if the problem recurs across many tasks: build a standard practice of never fully trusting a verifier's own training-time pass rate as the sole quality signal, and always maintaining an independent, periodically-refreshed held-out verification set specifically to detect this class of drift before it's discovered by human review after the fact, exactly the same "hold out an independent validation signal the training loop never saw" principle that recurs across RLHF's overoptimization mitigation, RLAIF's judge-reliability auditing, and this file's verifier-hacking case.

## Q16: Why does SFT alone succeed at transferring RL-discovered reasoning behavior into a smaller model, given that SFT cannot itself perform the RL search that discovered the behavior?

The key distinction is between *discovering* a good reasoning strategy and *imitating* an already-discovered one -- these are different-difficulty problems, and reasoning distillation only requires the second. The RL stage's job was to search an enormous space of possible reasoning strategies under a verifiable reward signal, an exploration problem with no clear a priori target, which is exactly why it's hard and expensive: no one could have simply written down the effective self-checking, backtracking deliberation pattern as a demonstration in advance, because it wasn't known to be effective until the RL process found it empirically. Once that pattern has been discovered and is available as concrete generated text (a reasoning trace), imitating it via next-token prediction is mechanically no different from any other SFT task -- predicting the next token of a self-checking reasoning trace is not intrinsically harder than predicting the next token of any other text, since the "hard part" (finding a good trace to imitate) has already been done by someone else's RL process, and imitation is comparatively easy once a good target exists to imitate.

This has a real, bounded consequence worth stating clearly, since it's the natural follow-up an interviewer will probe for: the distilled student inherits the *style and structure* of the teacher's reasoning and a substantial fraction of the resulting accuracy, but not the teacher's own capacity to *discover further, novel* reasoning strategies beyond what appears in the distillation traces -- the student never itself ran the RL search, so on problems sufficiently different from the distillation set's distribution, its performance is bounded by how well the demonstrated patterns generalize, not by an independent search capacity of its own. This is why distilled models are consistently reported as strong but trailing their directly RL-trained teacher on the hardest, most out-of-distribution problems, and it's the correct, precise answer to "can distillation alone make a small model as good as a large RL-trained one" -- not fully, by this exact mechanism, though a distilled student that subsequently undergoes its own further RL stage can close more of that gap, since at that point it's doing independent search of its own rather than pure imitation.

## Q17: Implement the k3 KL estimator used in GRPO/PPO implementations, and explain why it's preferred over the naive log-ratio-difference estimator.

```python
import torch

def k3_kl_estimator(logp_policy, logp_ref):
    """
    Estimates KL(pi_ref || pi_policy) at each token, given log-probs the policy and
    reference model each assign to a token actually sampled from the policy.
    """
    log_ratio = logp_ref - logp_policy    # log(pi_ref / pi_policy) for the sampled token
    return torch.exp(log_ratio) - log_ratio - 1
```

The naive estimator for a per-token KL contribution is simply the log-ratio itself, `logp_policy - logp_ref` (or its negation, depending on which direction of KL you're estimating) -- this is an unbiased estimator of the KL divergence in expectation over tokens sampled from the policy, but it can be negative for any individual token even though true KL divergence is always non-negative, which is a real practical annoyance: a per-token penalty term that can swing negative is noisier and can partially cancel across a sequence in a way that obscures what's actually happening, and is less numerically well-behaved to accumulate over long sequences. The k3 estimator, `exp(log_ratio) - log_ratio - 1`, is a different unbiased estimator of the same expected quantity (it can be derived from the general family of Bregman-divergence-based estimators for `f`-divergences) that is provably always non-negative for every individual token, not just in expectation -- giving a lower-variance, more numerically stable per-token penalty that behaves sensibly (monotonically increasing with the actual divergence at that position) rather than occasionally producing a token-level value that superficially looks like the policy is "closer" to the reference than it actually is. This is precisely why most published GRPO implementations use the k3 form rather than the naive log-ratio difference, despite both being valid, unbiased estimators of the same underlying KL quantity in expectation.

## Q18: Quantify RLAIF's cost case, and name the two or three reliability risk categories you'd weigh most heavily against that cost advantage.

The cost case is large and directly quantifiable in order-of-magnitude terms: a single properly-conducted human preference comparison (screened labeler, quality-control overhead) realistically costs on the order of a dollar to a few dollars, while a single AI-feedback comparison (one inference call to a capable model) costs a small fraction of a cent to a few cents -- routinely a two-to-four-order-of-magnitude reduction per comparison. This isn't merely a budget-line optimization; it changes what's operationally feasible, most notably enabling frequent reward-model refresh cycles against the current policy's own outputs (directly mitigating RM overoptimization) at a cadence human-labeling budgets typically cannot support.

Against that, I'd weigh three reliability risk categories most heavily: first, **self-preference/stylistic-affinity bias** -- an AI judge tends to rate outputs more favorably when they match its own stylistic conventions, especially if judge and policy share training lineage, a subtler and harder-to-detect bias than simple length gaming. Second, **unreliability on judgments requiring specialized ground truth the judge may not actually have** -- a judge asked about style, tone, or adherence to an explicit written standard is answering a question a capable model can plausibly do well; a judge asked about specialized factual correctness is answering a question that requires it to actually know the right answer, and a confidently wrong judge produces confidently wrong labels indistinguishable, from the outside, from confidently right ones. Third, **inherited-blind-spot circularity** -- the feedback model is itself a product of prior training and can propagate its own biases into new training data without correction unless independently, periodically validated against fresh human judgment. None of these negate the cost case; they argue for a task-by-task decision (verifiable domains to RLVR, style/tone judgments to AI feedback, specialized-factual and high-stakes judgments to human or tool-grounded feedback) rather than a blanket "replace all human feedback with AI feedback" policy.

## Q19: You're asked to design an end-to-end post-training pipeline for a newly pretrained base model, from scratch, for a general-purpose assistant product. Outline the stages and the key decisions at each one.

I'd structure this as a sequence of stages, each addressing a specific gap the previous stage leaves open, and I'd be explicit about the decision at each step rather than presenting it as one fixed recipe. Stage one is **SFT**, converting the raw document-completion model into one that reliably adopts assistant-format behavior -- the key decision here is mixture design (File 006): task-type coverage and proportions, language coverage, and an explicit, bounded refusal/safety sub-mixture, curated for consistency over sheer volume given the quality-over-quantity evidence. This produces both the RL initialization and the reference policy for later stages, so its quality directly bounds everything downstream.

Stage two is **preference optimization**, and the key decision is RLHF-plus-PPO versus a DPO-family method, driven by the team's existing RL infrastructure, whether iterated/online refresh is planned, and the nature of available feedback (paired versus unpaired, pushing toward KTO if unpaired). I'd plan for a hybrid label source here too (File 004): human feedback for helpfulness and any specialized-factual-heavy categories, AI feedback for harmlessness/style-type judgments where cost and scale matter more and reliability risk is lower, with periodic human spot-checking of the AI-labeled portion. Stage three is a **dedicated RLVR pass for verifiable-outcome verticals** (coding, math) layered on top of the same policy, specifically because these domains admit a ground-truth reward that removes the alignment-tax mechanism (File 007) and the learned-proxy reward-hacking surface (File 002) that the preference-optimization stage cannot avoid. Throughout all of this, I'd track the alignment tax and sycophancy metrics continuously (not as a one-time launch gate), correct for benchmark-formatting artifacts before accepting any regression at face value, and keep a periodically-refreshed held-out evaluation suite -- per task category, per language, and specifically for safety-relevant behavior -- distinct from whatever signal any given stage was directly optimized against, so that reward/preference-signal gaming in any one stage doesn't go undetected by the very metric it might be gaming.

## Q20: Explain why rollout generation, rather than the backward pass, dominates PPO's wall-clock cost, and describe the concrete systems mitigation production teams use for this.

A single training forward/backward pass processes an entire batch of already-known sequences in parallel -- every token's contribution to the loss and gradient is computed in one pass through the network. Generating a rollout, by contrast, is inherently sequential: producing a completion of length `T` requires `T` successive forward passes (even with KV-caching to avoid redundant recomputation of earlier positions), because each new token's distribution depends on every previously generated token, which isn't known until it's actually been sampled. For long completions -- and reasoning-model rollouts specifically can run to thousands of tokens -- this sequential dependency means rollout generation for one batch of prompts can take far longer in wall-clock terms than the single parallel forward/backward pass used to update the policy on that same batch once rollouts are in hand, making generation, not gradient computation, the actual bottleneck of the training loop.

The concrete systems mitigation: production RL/RLHF infrastructure typically runs rollout generation on a dedicated, inference-optimized serving engine (vLLM- or SGLang-style continuous-batching servers, dramatically more throughput-efficient at autoregressive decoding than a training framework's own forward pass) that is architecturally separate from the training framework running the actual gradient updates (Megatron-LM- or FSDP-based, typically). This split introduces its own real engineering problem -- weight synchronization: after every policy update, the updated weights must be transferred from the training framework's sharded parameter representation into the rollout engine's own representation, frequently enough that rollouts aren't generated from a badly stale policy, without the transfer itself becoming a new bottleneck. Getting this synchronization both correct (no stale or partially-updated weights served during a sync window) and fast is a genuinely nontrivial distributed-systems problem specific to RL-style post-training, and is a large part of why RLHF/RL infrastructure is treated as a distinct systems specialty from pretraining infrastructure at frontier labs, rather than an afterthought layered onto existing training clusters.
