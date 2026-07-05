## RLVR and Reasoning-Model Training

### 0. Scope of This File

This file covers Reinforcement Learning with Verifiable Rewards (RLVR) as a category, contrasts it precisely against RLHF (Files 002-003), derives GRPO as RLVR's most-associated concrete algorithm, and covers the reasoning-distillation result that lets RL-discovered reasoning behavior transfer into smaller models via ordinary SFT. For the canonical, numbers-attached implementation of this pipeline -- DeepSeek-R1-Zero's pure-RL run, DeepSeek-R1's cold-start-plus-RL recipe, the specific GRPO hyperparameters and results -- see `../OpenSource/008_DeepSeek_R1.md`, which this file assumes as the concrete reference case and does not re-derive numbers from.

### 1. What "Verifiable Reward" Actually Means

Every reward signal discussed in Files 002-004 shares one property: it is an *approximation* of true quality, produced by a model (a Bradley-Terry reward model, or an AI judge) that was itself fit to a finite, noisy sample of human or AI judgments. RLVR replaces that approximation with something categorically different: a **deterministic, programmatic, ground-truth check** that can verify correctness directly, with no learned model standing between the policy's output and the reward signal at all.

Concretely, for a math problem with a known final numeric answer, the reward is:

```
r(x, y) = 1  if extract_final_answer(y) == ground_truth_answer(x)  else  0
```

For a coding problem with a test suite, the reward is:

```
r(x, y) = (number of test cases passed) / (total test cases)      # or a strict 1/0 on full-pass
```

Neither of these requires training anything. They are ordinary deterministic functions -- string/numeric equality checks, or running generated code against a fixed test harness -- applied to the policy's output. This is the entire definitional content of RLVR: **the reward function is a verifier, not a learned model**, and it is exact (up to the verifier's own correctness, Section 5) rather than approximate.

### 2. RLVR Versus RLHF, Precisely

| Dimension | RLHF (Files 002-003) | RLVR |
|---|---|---|
| Reward source | A learned model (Bradley-Terry RM, or an implicit DPO reward) fit to human/AI comparisons | A deterministic, programmatic verifier |
| Reward is exact or approximate? | Approximate -- a finite-sample fit to preference data | Exact, up to the verifier's own correctness |
| Applicable domains | Any domain a human/AI can compare responses in (broad) | Domains with a checkable ground truth: math, code, formal logic, structured-output tasks, some factual QA |
| Reward-hacking surface | Exploiting the learned RM's blind spots (File 002, Section 4) | Exploiting the verifier's blind spots or weaknesses (Section 5) -- a smaller but nonzero surface |
| Cost to obtain reward at scale | Human labeling cost, or AI-judge inference cost (File 004) | Near-zero marginal cost once a verifier/test-harness exists for the domain |
| Requires a reference-model KL penalty? | Yes, centrally (File 002, Section 4) | Often used more lightly or omitted entirely in practice (Section 4.3) |
| Requires a learned value function/critic? | Yes, standard in PPO (File 002, Section 3.3) | Often removed -- this is GRPO's central move (Section 4) |

The single most important row in this table for an interview: **RLVR does not remove reward hacking as a risk category, it removes one specific *source* of reward-hacking risk** -- the risk that comes specifically from a learned reward proxy being imperfectly fit to a finite comparison dataset (File 002, Sections 2.6 and 4.2). A verifier can still be gamed if it has its own blind spots (Section 5); RLVR's real, precise claim is narrower and more defensible than "reward hacking is solved."

### 3. Why Removing the Learned-Proxy Layer Matters Mechanically

**3.1 No Bradley-Terry approximation error.** RLHF's reward model is fit by maximum likelihood to a finite sample of pairwise comparisons (File 002, Section 2.2) -- it is, by construction, a statistical estimate with sampling error, and its accuracy degrades outside its training distribution (File 002, Section 2.6). A verifier has no such fitting step: `extract_final_answer(y) == ground_truth_answer(x)` is exactly as correct on a response the policy generates today as it would be on any response, in-distribution or not, because it is not a learned function generalizing from training examples -- it is executing a fixed, correct procedure.

**3.2 No distributional-shift-induced miscalibration.** File 002, Section 4.5 describes reward-overoptimization scaling curves: as the policy's output distribution drifts further from the RM's training distribution, the RM's scores become progressively less trustworthy. A verifier's correctness does not depend on how far the policy's outputs have drifted from any particular training distribution, because the verifier was never fit to a distribution in the first place -- a math-answer checker is equally reliable whether the policy's derivation style looks like anything in any training set or not, as long as the final extracted answer format is parseable.

**3.3 What this buys concretely: the KL penalty can be relaxed or dropped.** Because there is no risk of the reward signal itself becoming miscalibrated as the policy moves further from a reference distribution (Section 3.2), the central justification for File 002's KL-to-reference penalty -- anchoring the policy to a region where a *learned* reward proxy is still trustworthy -- weakens substantially for RLVR. Several public reasoning-model RL recipes (most notably DeepSeek-R1-Zero, `../OpenSource/008_DeepSeek_R1.md`) run with little or no explicit KL penalty and allow the policy to move far from its initialization, something that would be reckless under vanilla RLHF given the overoptimization dynamics in File 002, Section 4.5, but is a defensible design choice when the reward itself cannot become miscalibrated by policy drift the way a learned RM can.

### 4. GRPO: Removing the Value Function via a Group-Relative Baseline

**4.1 The problem GRPO addresses.** PPO's advantage estimation (File 002, Section 3.3) relies on a learned value function `V_psi(s_t)` to convert a sparse, terminal reward into a dense, lower-variance training signal via GAE. Training a good value function is nontrivial in its own right -- it is a second learned model (or head) that must itself converge well, and for long chain-of-thought reasoning rollouts specifically (which can run to thousands of tokens), getting a token-level value function to accurately predict "expected eventual verifier outcome from this partial reasoning trace" is a hard auxiliary learning problem, adding real training cost and a real additional source of instability (File 002, Section 6, "value function underfitting").

**4.2 GRPO's alternative: use a group of sampled completions as their own baseline.** Group Relative Policy Optimization (GRPO; introduced in DeepSeekMath and used centrally in DeepSeek-R1's RL stage, `../OpenSource/008_DeepSeek_R1.md`) removes the value function entirely. For each prompt `x`, sample a **group** of `G` completions `{y_1, ..., y_G}` from the current policy, score each with the verifier to get rewards `{r_1, ..., r_G}`, and compute each completion's advantage relative to the *group's own* reward statistics rather than relative to a learned value function's prediction:

```
mean_r = (1/G) * sum_i r_i
std_r  = stddev of {r_1, ..., r_G}

A_i = (r_i - mean_r) / (std_r + epsilon)          # epsilon: small constant for numerical stability
```

Every token in completion `y_i` is assigned this same scalar advantage `A_i` (or, in some implementations, further modulated per-token by the KL term, Section 4.3) -- there is no token-level value prediction at all. The intuition: rather than asking "was this completion better than what a learned value function predicted the expected outcome would be," GRPO asks "was this completion better than the *other completions this same policy just generated for this same prompt*" -- a directly computable, zero-extra-model baseline, since the group's mean reward is exactly the empirical estimate of "what this policy currently achieves on this prompt," which is precisely what a value function would otherwise be trained to predict.

**4.3 The full GRPO objective.** Combining the group-relative advantage with a PPO-style clipped update and an explicit KL penalty term (added directly to the loss rather than folded into the reward, a small but real implementation difference from File 002's PPO formulation):

```
L_GRPO(theta) = -E_{x, {y_i}~pi_theta_old} [
    (1/G) * sum_i (1/|y_i|) * sum_t [
        min( rho_{i,t}(theta) * A_i,  clip(rho_{i,t}(theta), 1-eps, 1+eps) * A_i )
        - beta_kl * KL_per_token(pi_theta || pi_ref)_{i,t}
    ]
]

where  rho_{i,t}(theta) = pi_theta(y_{i,t} | x, y_{i,<t}) / pi_theta_old(y_{i,t} | x, y_{i,<t})
```

Structurally this is File 002's clipped surrogate objective (Section 3.2), with the value-function-derived GAE advantage replaced by the group-relative `A_i`, averaged over the `G` sampled completions per prompt, with an explicit per-token KL penalty term subtracted directly in the loss (rather than being pre-mixed into the reward before advantage estimation, as vanilla RLHF does per File 002, Section 4.1) -- a design choice that keeps the raw verifier reward completely unmodified by the KL term, feeding the *pure* verifier signal into the group-relative advantage computation, and applying the KL constraint as a separate, additive regularization term instead.

### 4.4 A Full GRPO Advantage-and-Loss Implementation

```python
def grpo_advantages(rewards, eps=1e-4):
    # rewards: (G,) tensor of verifier scores for G completions sampled for one prompt
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)
    return (rewards - mean_r) / (std_r + eps)          # shape: (G,), one scalar advantage per completion

def grpo_loss(logp_new, logp_old, logp_ref, advantages, clip_eps=0.2, beta_kl=0.01):
    """
    logp_new, logp_old, logp_ref: (G, T) per-token log-probs under the current policy,
                                   the rollout-time policy, and the frozen reference, respectively.
    advantages: (G,) -- broadcast across the T token positions of each completion.
    """
    ratio = torch.exp(logp_new - logp_old)                       # (G, T)
    adv = advantages.unsqueeze(-1)                                # (G, 1) -> broadcasts over T
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_term = torch.min(unclipped, clipped)

    # per-token KL estimator (k3 estimator, low-variance, always non-negative):
    log_ratio_ref = logp_ref - logp_new
    kl_per_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1

    per_token_loss = -(policy_term - beta_kl * kl_per_token)       # (G, T)
    return per_token_loss.mean()                                   # average over completions and tokens
```

The `k3` KL estimator (`exp(log_ratio) - log_ratio - 1`) is worth flagging as its own small but real implementation detail: it is a lower-variance, always-non-negative estimator of the same KL quantity that File 002's simpler `logp_new - logp_ref` estimator computes, and is the form actually used in most published GRPO implementations rather than the naive log-ratio difference, precisely because a strictly non-negative per-token penalty is more numerically well-behaved over long reasoning rollouts than an estimator that can go negative on individual tokens even though its expectation is correct.

### 4.5 What GRPO Costs You in Exchange

Removing the value function is not a free simplification -- it trades one kind of cost for another, and a staff-level answer names both sides:

- **More samples per prompt required.** Because the baseline is estimated empirically from `G` samples rather than predicted by a trained function, `G` needs to be large enough (commonly on the order of a handful to a few dozen samples per prompt in published recipes) for the group mean/std to be a low-variance baseline estimate -- this multiplies the rollout-generation cost (already File 002 Section 5's dominant wall-clock cost) by a factor of `G` relative to a single-sample-per-prompt PPO setup, though it removes the value-function training cost entirely.
- **No dense, per-token credit assignment within a single trajectory.** A learned value function can in principle assign different implicit credit to different tokens within one completion, reflecting where in the reasoning trace things went right or wrong; GRPO's group-relative advantage is a single scalar per completion, uniform across all of that completion's tokens (before the separate per-token KL term is applied) -- coarser credit assignment, traded for the removed value-function training cost and instability.
- **Group-relative advantages are only meaningful within the group they were computed from.** Exactly analogous to Bradley-Terry reward scores being meaningful only as within-prompt differences (File 002, Section 2.2), GRPO's advantages are normalized per-prompt-group and are not comparable in absolute terms across different prompts or different groups -- a detail that matters if you ever want to inspect or debug raw advantage values across a batch.

### 4.6 A Worked Numeric Example

Take a prompt with `G = 4` sampled completions and verifier rewards `r = [1, 1, 0, 0]` (two correct, two incorrect solutions). The group mean is `0.5` and the population standard deviation is `0.5`, giving advantages `A = [(1-0.5)/0.5, (1-0.5)/0.5, (0-0.5)/0.5, (0-0.5)/0.5] = [1, 1, -1, -1]`. Every token in the two correct completions receives a positive advantage of `1`, pushing the policy to increase their probability; every token in the two incorrect completions receives `-1`, pushing the policy to decrease theirs -- and this happens with **no value function anywhere in the computation**, purely from the empirical spread of rewards within the group. Contrast this with a degenerate group where all `G` completions receive the same reward (e.g., `r = [1, 1, 1, 1]`, an easy prompt every sample solves, or `r = [0, 0, 0, 0]`, one so hard none do): `std_r = 0` and every advantage is `0/epsilon ≈ 0` -- correctly signaling "no informative gradient available from this prompt at the policy's current skill level," since a group with zero reward variance carries no information about which of its (equally-scored) completions to prefer. This is a real, structural consequence of the group-relative design worth naming explicitly: **GRPO gets essentially no training signal from prompts the current policy either always solves or never solves**, which motivates curating or dynamically filtering the training-prompt distribution toward a difficulty band where the policy's success rate is neither near 0 nor near 1, so that groups reliably contain a mix of successes and failures.

### 4.7 Typical GRPO Hyperparameters

| Hyperparameter | Typical range | Role |
|---|---|---|
| Group size `G` | ~4-64, commonly 8-16 in published recipes | Number of samples per prompt used to estimate the group baseline; larger reduces baseline variance at proportionally higher rollout cost |
| KL coefficient `beta_kl` | Small, sometimes 0 | Direct additive penalty in the loss (Section 4.3), rather than pre-mixed into the reward as in vanilla PPO |
| Clip range `eps` | ~0.2, same role as PPO | Bounds per-token trust region exactly as in File 002, Section 3.2 |
| Prompts per batch | Large enough to amortize rollout-generation overhead | Same rollout-cost-amortization logic as File 002, Section 6.2's PPO rollout batch size |
| Sampling temperature during rollout | Moderate-to-high | Needs enough diversity across the `G` samples per prompt for the group to contain a meaningful spread of outcomes (Section 4.6) |

### 4.8 GRPO's Place Among Critic-Free Policy-Gradient Methods

GRPO is not the only method that removes PPO's learned value function -- it belongs to a small family of **critic-free** policy-gradient variants explored for exactly this cost/stability reason, worth being able to name alongside it: **RLOO** (REINFORCE Leave-One-Out) computes a similar group-based baseline but as a per-sample leave-one-out mean (baseline for sample `i` is the mean of the *other* `G-1` samples' rewards, avoiding any self-referential bias in the baseline estimate), and **ReMax** uses a single greedy (argmax/low-temperature) decode of the same prompt as a deterministic baseline rather than a sampled group's statistics. All three share the same underlying motivation (Section 4.1's value-function training cost and instability) and the same underlying tradeoff (Section 4.5's increased per-prompt sampling cost, or, for ReMax, an extra greedy-decode pass), differing mainly in exactly how the baseline is constructed from the available samples. GRPO's specific choice -- normalizing by both the group mean *and* the group standard deviation, rather than by the mean alone -- is itself worth noting as a design choice distinct from RLOO's simpler leave-one-out mean-only baseline, intended to additionally control for reward-scale variation across different prompts (a prompt where rewards happen to be more spread out contributes a differently-scaled advantage than one where they're tightly clustered, once divided through by that prompt's own group standard deviation).

### 5. Reward Hacking Still Happens Under RLVR -- Just a Narrower Surface

**5.1 Verifier weaknesses are the new attack surface.** If a test suite for a coding task has incomplete coverage, a policy under RL pressure can and does discover solutions that pass every test in the suite while being wrong or degenerate in ways the tests simply didn't check for (e.g., hard-coding expected outputs for the specific test inputs rather than implementing the general algorithm) -- a direct, well-documented RLVR-specific reward-hacking pattern, exploiting the verifier's incompleteness rather than a learned RM's miscalibration, but structurally the same underlying phenomenon (optimization pressure finding and exploiting the gap between the reward proxy and true task success).

**5.2 Getting the right answer via wrong reasoning.** For math problems verified only by final-answer equality, a policy can receive full reward for a derivation that is logically invalid but happens to arrive at the correct final number (through error cancellation, a lucky guess embedded in a wall of plausible-looking steps, or pattern-matching an answer from a superficially similar memorized problem) -- the verifier checks only the final answer, so it has no way to detect or penalize an unsound derivation that happens to be right. This is a direct motivation for process reward models (File 002, Section 6.5) in domains where derivation soundness matters independent of the final answer, though outcome-only RLVR remains the more common and more robust-to-gaming choice in practice specifically because PRM training itself introduces a learned-proxy layer with its own reward-hacking risk (Section 5.4), reintroducing the exact problem RLVR was adopted to avoid.

**5.3 Format and non-content reward hacking: the DeepSeek-R1-Zero language-mixing case.** `../OpenSource/008_DeepSeek_R1.md` documents a concrete, real-world instance directly relevant here: DeepSeek-R1-Zero's pure-RL run (reward based only on final-answer correctness, with no explicit reward term for readability or single-language consistency) produced reasoning traces that mixed multiple languages within a single chain of thought -- entirely rational from the reward function's point of view (the verifier only checks the final answer, so language-mixing incurs zero penalty), but undesirable from a product/usability standpoint. DeepSeek's fix (adding a language-consistency reward term in the full R1 recipe) is itself a small, concrete illustration of the general RLVR reward-hacking pattern: **whatever the reward function does not explicitly check, the policy has no pressure to get right, and will readily sacrifice for whatever the reward function does check** -- the verifiable-reward setting narrows the exploitable surface relative to a learned RM, but does not eliminate the underlying dynamic, and any property you care about that isn't in the reward function is fair game for the optimization to trade away.

**5.4 Process reward models reintroduce RLHF's exact problem, one level down.** A PRM (File 002, Section 6.5) that scores individual reasoning steps is itself a learned model fit to (necessarily more expensive to collect) step-level annotations, and is subject to exactly the same overoptimization dynamics as an outcome RM (File 002, Section 4.5) -- a policy optimized hard against a fixed PRM can learn to produce steps that superficially satisfy whatever pattern the PRM was trained to reward, without those steps being genuinely sound. This is precisely why several production reasoning-RL recipes, DeepSeek-R1 prominently among them, report choosing outcome-only verification over process supervision at scale, treating outcome verification's narrower reward-hacking surface as worth more than process supervision's denser credit assignment, at least given current PRM training methods' own reliability limitations.

### 5.5 Curriculum and Difficulty Filtering as a Practical Necessity

Section 4.6's observation -- that GRPO extracts no signal from prompts the policy always or never solves -- generalizes into a real, practical data-curation problem for any RLVR pipeline, not just a GRPO-specific footnote. A production RLVR run typically needs an explicit curriculum or dynamic filtering strategy: starting with a prompt distribution the initializing (often SFT or lightly-tuned) policy can solve a meaningful fraction of the time, and progressively shifting the training distribution toward harder problems as the policy improves, sometimes by explicitly filtering out prompts whose recent success rate has drifted to near 0% or 100% within a sliding window of training. This is structurally analogous to curriculum learning in classical RL and to difficulty-aware data selection in supervised settings, but is worth naming as a first-class engineering concern specific to verifiable-reward RL rather than an afterthought, since getting it wrong (training predominantly on prompts that are uniformly too easy or too hard for the current policy) can silently stall an RLVR run's learning progress even while compute continues to be spent.

### 5.6 Cold-Start SFT Versus Pure RL From the Base Model

`../OpenSource/008_DeepSeek_R1.md` documents a directly relevant empirical contrast worth having on hand: **DeepSeek-R1-Zero** ran RLVR directly from a base pretrained model with no SFT cold start at all, demonstrating that reasoning behavior (extended chains of thought, self-verification) can emerge from RL pressure alone, without any human- or model-written reasoning demonstrations to imitate first -- a striking result in its own right, since it shows the *capability* for this behavior pattern was already latent in the base model's weights, needing only the right optimization pressure (a verifiable reward and enough RL steps) to surface it, rather than needing to be taught from scratch via demonstration. **DeepSeek-R1** (the production-quality follow-up) instead used a **cold-start SFT phase** on a modest set of curated long-reasoning demonstrations before the RL stage, reported to produce a more readable, more stylistically consistent reasoning model with fewer of R1-Zero's rough edges (including the language-mixing issue in Section 5.3) -- illustrating a general point that generalizes well beyond this one paper: **RLVR does not strictly require an SFT-produced starting policy the way vanilla RLHF's KL-to-SFT-reference machinery assumes (File 002, Section 4), but a light cold-start SFT phase remains a practically valuable way to point the RL search toward a good initial region of behavior space rather than requiring it to discover a well-formed reasoning style from unconstrained exploration alone.**

### 5.7 RLVR Beyond Math and Code

Though math and code are the domains with the most mature, widely-used verifiers, the underlying idea generalizes to any task with a programmatically checkable outcome: structured-data-extraction tasks (verified by exact-match or schema validation against a known-correct extraction), agentic tool-use tasks with a checkable end state (did a browser-automation agent actually reach the correct final page state, did a database query return the correct result set), formal theorem-proving (verified by a proof checker, an unusually strong and completely unambiguous verifier), and game-playing or simulated-environment tasks (verified by the environment's own win/loss/score signal, the setting RL itself originated in before language models). The common thread across all of these is the same as Section 1's definition: a fixed, deterministic procedure that can check the outcome without needing a learned model's judgment call -- and the range of domains where such a procedure exists, and can be run cheaply enough to serve as an RL reward signal at training scale, is the actual, practical boundary of where RLVR is applicable, rather than any conceptual limitation of the algorithm itself.

### 5.8 Entropy Collapse: a Distinct, Newer Failure Mode

A failure mode documented in more recent reasoning-RL research (including follow-up work explicitly diagnosing and patching GRPO, such as the DAPO line of work) that is worth knowing by name alongside the reward-hacking failure modes above: **policy entropy collapse**, where the model's output distribution becomes increasingly deterministic/low-entropy over the course of RL training, to the point that the group-sampling step in Section 4.2 starts drawing `G` nearly-identical completions for a given prompt, which in turn starves the group-relative advantage computation of the reward variance it needs to produce useful gradients (Section 4.6's degenerate `std_r ≈ 0` case, but now arising from the policy's own loss of diversity rather than from the prompt's inherent difficulty). This is mechanistically distinct from RLHF's mode-collapse concern (File 002, Section 4.3) in its specific cause -- there, the KL penalty is the direct countermeasure; in vanilla GRPO with a weak or zero KL term (Section 3.3's rationale for relaxing it), nothing else in the base algorithm directly discourages entropy collapse, motivating patches such as an explicit entropy bonus term, decoupled clipping ranges that treat probability-increasing and probability-decreasing updates asymmetrically, or dynamic sampling that discards zero-variance groups before they enter the loss computation at all (exactly the curriculum/filtering idea from Section 5.5, applied here as a training-time safeguard rather than only a data-curation-time one).

### 5.9 A Minimal Reward-Function Implementation, End to End

To make Section 1's abstract verifier definition concrete, a full (if simplified) reward function for a math-answer-checking RLVR setup looks like this in practice:

```python
import re

def extract_final_answer(completion: str) -> str | None:
    # convention: the policy is trained/prompted to box its final answer, e.g. "\boxed{42}"
    match = re.search(r"\\boxed\{([^}]*)\}", completion)
    return match.group(1).strip() if match else None

def math_verifier_reward(completion: str, ground_truth: str) -> float:
    predicted = extract_final_answer(completion)
    if predicted is None:
        return 0.0                                  # format failure: no parseable answer at all
    return 1.0 if normalize_answer(predicted) == normalize_answer(ground_truth) else 0.0

def normalize_answer(s: str) -> str:
    # strips whitespace, standardizes numeric formatting (e.g., "1/2" vs "0.5"), etc. --
    # a surprisingly nontrivial piece of engineering in its own right: a verifier that is too
    # strict on formatting rejects correct answers (a false negative, wasting good rollouts),
    # while one that is too lenient risks accepting wrong answers (Section 5.1's attack surface).
    return s.replace(" ", "").rstrip(".")
```

The comment on `normalize_answer` is not incidental -- a nontrivial fraction of real-world RLVR engineering effort goes into exactly this kind of verifier robustness work (handling equivalent fractions, equivalent algebraic forms, minor formatting variation), because a verifier with a high false-negative rate on correct-but-differently-formatted answers silently discards good training signal, while a verifier with a high false-positive rate silently reintroduces the exploitable-blind-spot risk from Section 5.1 -- the verifier's own precision and recall are a real, measurable engineering quality axis, not a solved, zero-maintenance component of the pipeline.

### 6. Reasoning Distillation: RL-Discovered Behavior, SFT-Transferred

**6.1 The empirical result.** A large RL-trained reasoning model (e.g., DeepSeek-R1) can generate long chain-of-thought traces -- extended deliberation, self-checking, backtracking when a derivation seems to be going wrong -- that were discovered via the RL process described in Sections 3-4, i.e., no one wrote these behaviors down as demonstrations; the policy found them because they led to higher verifier reward. The distillation result, covered with full numbers in `../OpenSource/008_DeepSeek_R1.md`, is that collecting a large set of this model's own reasoning traces on a held-out prompt set and running **ordinary SFT** (File 001's masked cross-entropy loss, nothing more exotic) on a smaller model using those traces as targets transfers a substantial fraction of the reasoning behavior into the smaller model -- without the smaller model ever running the RL process that originally discovered the behavior.

**6.2 Why this works mechanically.** This connects directly to File 001, Section 6.1's discussion of SFT's real capability: SFT is an extremely effective mechanism for instilling a *behavioral pattern* into a model, once that pattern exists somewhere as demonstrable text, regardless of how expensive or exotic the process that originally produced the pattern was. The RL stage's job was to *search* an enormous space of possible reasoning strategies under a reward signal, discovering effective deliberation patterns that would have been extremely hard to hand-specify as demonstrations in advance; once discovered and recorded as concrete traces, imitating those traces is a comparatively easy supervised-learning problem, no harder than any other SFT task in kind, since predicting the next token of a self-checking, backtracking reasoning trace is mechanically identical to predicting the next token of any other text -- the *hard part* was finding good traces to imitate in the first place, not the imitation step itself.

**6.3 What distillation does and does not transfer.** The distilled student inherits the *style and structure* of the teacher's reasoning process (deliberation length, self-checking habits, characteristic phrasing) and a substantial fraction of the resulting task accuracy, but it does not inherit the teacher's own *capacity to discover new reasoning strategies* beyond what appears in the distillation traces -- the student was never actually optimized against the verifier itself, so its reasoning quality on problems meaningfully different from the distillation set's distribution is bounded by how well the demonstrated patterns generalize, exactly the ceiling-effect argument from File 001, Section 3.5, now applied to reasoning traces rather than ordinary demonstrations. This is why the strongest reported reasoning models are still the directly RL-trained ones, with distilled smaller models reported as strong but consistently somewhat behind the RL-trained teacher on the hardest, most out-of-distribution problems -- consistent with distillation transferring a large fraction, but not the full ceiling, of the teacher's RL-discovered capability.

**6.4 The practical significance for the field.** This result is a big part of why RL-for-reasoning research is disproportionately valuable relative to its direct compute cost: a lab that runs an expensive RLVR training process on one large model can distill the resulting behavior into an entire family of smaller, cheaper-to-serve models via SFT alone, amortizing the RL search cost across many deployed model sizes rather than needing to independently run expensive RL training for every size in a model family. DeepSeek's own published distilled model family is the most-cited concrete evidence for this pattern at the time of writing (`../OpenSource/008_DeepSeek_R1.md`).

### 6.5 Can a Distilled Student Ever Exceed Its Teacher?

Worth addressing directly since it is a natural follow-up question: in principle, a distilled student is capped by the ceiling-effect argument in Section 6.3, but in practice a student that subsequently undergoes *its own* additional RLVR stage on top of distillation-initialized weights can, and in some published results does, exceed the pre-distillation teacher's performance on the trained task -- because at that point the student is no longer purely imitating, it is running its own independent RL search (Sections 3-4) starting from a better-than-random, distillation-informed initialization. This is worth distinguishing precisely from pure distillation without any further RL: the claim "distillation alone lets a small model exceed its teacher" is not well-supported and conflicts with the ceiling-effect argument; the claim "distillation followed by the student's own RL stage can exceed the original teacher" is a different, better-supported claim about compounding two separate mechanisms, not a property of distillation in isolation.

### 7. Common Interview Traps

- **Claiming RLVR eliminates reward hacking.** It narrows the reward-hacking surface to verifier weaknesses (Section 5) -- a real and important distinction from RLHF's learned-proxy surface, but not the same as eliminating the phenomenon. DeepSeek-R1-Zero's language-mixing case (5.3) is the concrete counterexample to have ready.
- **Describing GRPO as "PPO without a KL penalty."** GRPO's defining change is removing the learned value function in favor of a group-relative baseline (Section 4.2); the KL-penalty treatment (added directly to the loss rather than mixed into the reward) is a related but separate implementation detail, and some GRPO implementations do use a nonzero `beta_kl`.
- **Assuming outcome-only verification is strictly worse than process supervision because it's "coarser."** Section 5.4 gives the concrete counterargument: PRMs reintroduce a learned-proxy reward-hacking surface, and several production recipes deliberately choose outcome-only verification specifically to avoid that surface, trading credit-assignment density for reward-signal trustworthiness.
- **Treating reasoning distillation as "the student learns to reason."** More precisely, the student learns to imitate a specific *style and structure* of reasoning trace via ordinary SFT; it does not itself perform the RL search that discovered that structure, and its ceiling is bounded accordingly (Section 6.3).
- **Not knowing that RLVR's applicability is domain-limited.** It requires a checkable ground truth; open-ended writing quality, nuanced tone judgments, and most general-assistant helpfulness questions have no such verifier, which is exactly why RLHF/RLAIF (Files 002-004) remain necessary rather than superseded for those domains -- RLVR is a complement, not a universal replacement.
- **Overstating what reasoning distillation proves.** "Distillation alone lets a small model exceed its teacher" is not well-supported (Section 6.5); the better-supported claim involves a subsequent, independent RL stage on top of a distillation-informed initialization.
- **Missing that entropy collapse and reward hacking are distinct failure modes with different countermeasures.** Section 5.8's entropy collapse is about losing sample diversity and starving the advantage signal; Sections 5.1-5.3's reward hacking is about exploiting a specific gap between the verifier and true task success -- conflating the two leads to applying the wrong fix.

### 8. Quick-Reference Summary

- RLVR replaces a learned reward-model proxy with a deterministic, programmatic verifier (answer-equality checks, test-suite pass rates), applicable wherever a ground-truth check exists (math, code, formal domains).
- This removes one specific, well-characterized source of reward-hacking risk (learned-RM miscalibration and overoptimization, File 002 Sections 2.6 and 4.5) but does not remove reward hacking as a category -- verifier weaknesses (incomplete test coverage, answer-only checking that ignores derivation soundness, unchecked properties like language consistency) remain exploitable.
- Because verifier correctness does not degrade with policy drift the way a learned RM's does, the KL-to-reference penalty can often be relaxed or dropped relative to vanilla RLHF, a design choice several public reasoning-RL recipes make explicitly.
- GRPO removes PPO's learned value function, replacing GAE's baseline with a group-relative baseline computed empirically from `G` sampled completions per prompt: `A_i = (r_i - mean(r)) / std(r)`.
- GRPO trades value-function training cost and instability for increased per-prompt sampling cost (`G` completions needed for a low-variance baseline) and coarser, completion-level (rather than token-level) credit assignment.
- Reasoning distillation transfers RL-discovered reasoning behavior into smaller models via ordinary SFT on the larger model's traces -- the hard part (discovering effective reasoning strategies via RL search) is done once, and the comparatively easy part (imitating the discovered traces) is amortized across a whole family of smaller distilled models.
- Distillation transfers style, structure, and a substantial fraction of accuracy, but not the teacher's own capacity to discover further novel strategies beyond the distillation set -- distilled students are reported to trail directly RL-trained models on the hardest, most out-of-distribution problems.
- RLVR is domain-limited by the existence of a verifier and is a complement to, not a replacement for, RLHF/RLAIF on open-ended tasks without a checkable ground truth.
- GRPO extracts no training signal from prompts the policy always or never solves, making curriculum design and difficulty filtering a first-class engineering concern rather than an afterthought.
- Cold-start SFT is not strictly required for RLVR to produce reasoning behavior (R1-Zero's result), but it remains a practically valuable way to point RL search toward a well-formed starting region rather than requiring the policy to discover good reasoning style from unconstrained exploration.
- Policy entropy collapse is a distinct GRPO-specific failure mode from reward hacking, addressed by entropy bonuses, asymmetric clipping, or dynamic filtering of zero-variance groups rather than by KL-penalty tuning alone.
- Verifier precision/recall (false negatives from overly strict answer-format matching, false positives from overly lenient checks) is itself a real, ongoing engineering quality axis, not a solved component of an RLVR pipeline.

### 9. Cross-References

`../OpenSource/008_DeepSeek_R1.md` is the concrete, numbers-attached reference case for everything in this file -- the R1-Zero pure-RL run, the full R1 cold-start-plus-RL recipe, the language-mixing reward-hacking case, and the distilled model family. File 002 supplies the PPO/value-function/KL-penalty machinery this file's Sections 3-4 modify or remove. File 001, Section 6.1 previews the reasoning-distillation mechanism this file's Section 6 develops in full. File 004's RLAIF material is the natural complement for exactly the domains (open-ended helpfulness, tone, style) where no verifier exists and an AI-judged or human-judged preference signal remains necessary.

### 9.1 A Final Synthesis Point

The single most important habit this file is trying to instill: whenever a training signal is described as "reward," ask immediately whether it is a ground-truth verifier or a learned/judged proxy, because that single distinction determines which entire failure-mode vocabulary applies -- proxy-miscalibration-and-overoptimization (File 002) for the latter, verifier-blind-spot exploitation and entropy collapse (Sections 5 and 5.8 of this file) for the former. Conflating the two, or assuming a "verifiable reward" pipeline is simply immune to reward hacking because it isn't RLHF, is the single most common shallow answer this topic produces, and the corrective is always the same: ask what specifically the reward function checks, and reason from there about what it cannot check and is therefore blind to.
