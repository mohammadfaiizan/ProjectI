## DPO and the Direct Preference Optimization Variant Landscape

### 0. Scope and Motivation

File 002 derived the classical RLHF pipeline: train a reward model on human comparisons via a Bradley-Terry loss, then run PPO -- a genuinely finicky, multi-model, actor-learner-shaped RL system -- to optimize a policy against that reward model under a KL constraint to a reference policy. That pipeline works, but it is expensive and operationally fragile: it requires training and maintaining a separate reward model, running online rollout generation at training time, tuning RL-specific hyperparameters (clip range, GAE lambda, KL coefficient) that have no analogue in supervised training, and managing the systems complexity of 3-4 concurrently resident models (Section 5 of File 002).

Direct Preference Optimization (DPO; Rafailov et al., 2023, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model") asks a sharp question: if the entire point of the reward model is to be a differentiable proxy that PPO then optimizes against under a KL constraint, and if that constrained optimization problem has a closed-form solution, can you skip straight to training the policy on preference data with a single supervised-style loss, never explicitly instantiating a reward model or running RL at all? The answer, derived below, is yes -- and the derivation is the single most important piece of algebra in modern preference-tuning to be able to reproduce from memory in a staff interview.

### 1. The KL-Constrained Reward Maximization Problem

Recall the objective PPO is (approximately) optimizing in File 002, written as a direct maximization over the policy `pi` rather than in RL update-rule form:

```
max_pi  E_{x~D, y~pi(.|x)} [ r(x, y) ]  -  beta * E_{x~D} [ KL( pi(.|x) || pi_ref(.|x) ) ]
```

where `pi_ref` is the frozen reference policy (the SFT model) and `beta` controls how far the optimized policy is allowed to drift from it. This is exactly the "reward objective, KL-penalized" combination introduced in File 002, Section 4 -- just written as a direct functional-optimization problem over `pi` rather than as an RL training procedure.

### 2. The Closed-Form Optimal Policy

**2.1 Setting up the per-prompt optimization.** Because the objective decomposes as an expectation over independent prompts `x`, it suffices to solve the optimization separately for each fixed `x`. Expanding the KL term and writing the per-prompt objective to be maximized over the distribution `pi(.|x)`:

```
J(pi) = sum_y pi(y|x) * r(x,y)  -  beta * sum_y pi(y|x) * log( pi(y|x) / pi_ref(y|x) )
```

**2.2 Solving via calculus of variations / Lagrangian.** We maximize `J(pi)` subject to `sum_y pi(y|x) = 1` (pi must be a valid distribution). Introduce a Lagrange multiplier `lambda` for the normalization constraint and take the functional derivative with respect to `pi(y|x)` for each `y`, setting it to zero:

```
d/d(pi(y|x)) [ pi(y|x) * r(x,y) - beta * pi(y|x) * log(pi(y|x)/pi_ref(y|x)) - lambda * pi(y|x) ] = 0

=> r(x,y) - beta * log(pi(y|x)/pi_ref(y|x)) - beta - lambda = 0

=> log(pi(y|x)/pi_ref(y|x)) = r(x,y)/beta - 1 - lambda/beta

=> pi(y|x) = pi_ref(y|x) * exp( r(x,y)/beta ) * exp( -1 - lambda/beta )
```

The final factor, `exp(-1 - lambda/beta)`, does not depend on `y` -- it is exactly the normalization constant needed to make `pi(.|x)` sum to 1 over `y`. Calling this normalizer `1/Z(x)`:

```
pi*(y|x) = ( 1 / Z(x) ) * pi_ref(y|x) * exp( r(x,y) / beta )

where   Z(x) = sum_y  pi_ref(y|x) * exp( r(x,y) / beta )
```

This is the **closed-form optimal policy** for the KL-constrained reward-maximization problem: it is the reference policy, *reweighted* by an exponential tilt toward high-reward responses, with the tilt's sharpness controlled by `1/beta`. As `beta -> infinity`, the exponential term flattens toward 1 and `pi* -> pi_ref` (no reward-seeking at all, matching the "too-large-beta forfeits RL benefit" intuition from File 002, Section 4.4). As `beta -> 0`, the exponential term becomes a sharp indicator concentrating all mass on the single highest-reward response (unconstrained reward maximization, matching the "too-small-beta overoptimizes the reward" intuition from the same section). This closed form makes File 002's qualitative `beta` tradeoff into an explicit, derivable mathematical statement.

**2.3 `Z(x)` is intractable, and that turns out not to matter.** `Z(x)` requires summing `pi_ref(y|x) * exp(r(x,y)/beta)` over every possible response `y` -- an astronomically large space for any nontrivial sequence length, with no closed form. This is exactly why the classical pipeline resorts to *sampling*-based RL (PPO) rather than directly computing `pi*` from this formula: the formula is mathematically exact but computationally useless as a way to construct the policy directly. DPO's key move, in the next section, is to use this formula in the *other direction* -- not to compute `pi*` from `r`, but to express `r` in terms of `pi*`, in a way that makes `Z(x)` cancel out entirely before you ever need to evaluate it.

### 3. Inverting the Relationship: Reward as a Function of the Policy

Rearranging Section 2.2's result to solve for `r(x,y)`:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y)/beta)

=> exp(r(x,y)/beta) = Z(x) * pi*(y|x) / pi_ref(y|x)

=> r(x,y) = beta * log( pi*(y|x) / pi_ref(y|x) )  +  beta * log Z(x)
```

This is the paper's central identity: **the reward function that any KL-constrained-optimal policy is implicitly optimal for can be recovered, exactly, from the log-ratio between that policy and the reference policy**, up to a per-prompt-constant term `beta * log Z(x)` that does not depend on `y` at all. This single equation is why the paper is titled "your language model is secretly a reward model": any policy you have (or are training) implicitly *defines* a reward function via this formula, whether or not anyone ever trained an explicit reward model.

### 4. Substituting Into the Bradley-Terry Loss: the DPO Objective

Recall the Bradley-Terry preference loss from File 002, Section 2.2, in terms of an explicit reward model:

```
L_RM(theta) = - E_{(x,y_w,y_l)~D} [ log sigmoid( r_theta(x,y_w) - r_theta(x,y_l) ) ]
```

Now substitute Section 3's expression for `r` in terms of a policy `pi_theta` (parameterizing the policy we're directly training, playing the role `pi*` played above) and the fixed reference `pi_ref`:

```
r_theta(x,y_w) - r_theta(x,y_l)
  = [ beta*log(pi_theta(y_w|x)/pi_ref(y_w|x)) + beta*log Z(x) ]
  - [ beta*log(pi_theta(y_l|x)/pi_ref(y_l|x)) + beta*log Z(x) ]
  = beta * [ log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x)) ]
```

**The `beta * log Z(x)` terms cancel exactly**, because they are identical for `y_w` and `y_l` (both conditioned on the same prompt `x`) and appear with opposite sign in the subtraction. This cancellation is the entire trick: it means you never need to compute or even know `Z(x)` to train with this loss, despite `Z(x)` being individually intractable. Substituting this difference back into the Bradley-Terry loss gives the **DPO loss**:

```
L_DPO(theta) = - E_{(x,y_w,y_l)~D} [ log sigmoid( beta * ( log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x)) ) ) ]
```

This is a loss computed entirely from **log-probabilities the policy itself assigns to the preferred and dispreferred responses in a fixed, pre-collected preference dataset**, plus the same log-probabilities under a frozen reference model. There is no reward model anywhere in this expression -- Section 3's identity has let us substitute "the score two responses get from an explicit RM" with "the log-probability ratio the policy itself assigns those two responses relative to the reference policy," and the Bradley-Terry loss structure (a sigmoid of a score difference) carries over unchanged.

### 5. A Full, Runnable DPO Loss Implementation

```python
def dpo_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    """
    policy_logp_w/l: sum of per-token log-probs the *trained* policy assigns to the
                      winning/losing response, given the prompt. Shape: (batch,)
    ref_logp_w/l:    the same, under the frozen reference (SFT) model. Shape: (batch,)
    """
    policy_logratio = policy_logp_w - policy_logp_l
    ref_logratio     = ref_logp_w    - ref_logp_l
    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()          # same numerical-stability reasoning as File 002, Section 2.4
    return loss
```

A crucial, easy-to-get-wrong detail: `policy_logp_w` must be the **sum (not mean) of per-token log-probabilities** over the response, since the derivation in Sections 2-4 treats `pi_theta(y|x)` as the joint probability of the entire response sequence, which under the autoregressive factorization is `sum_t log pi_theta(y_t | x, y_<t)`. Using a mean instead of a sum changes the effective `beta` in a length-dependent way and is a documented source of subtly-wrong DPO implementations.

**What the gradient of this loss is actually doing.** Differentiating `L_DPO` with respect to `theta` gives a gradient that can be written as:

```
grad L_DPO  =  -beta * E[ sigmoid( -beta * (implicit_reward_diff) ) * ( grad log pi_theta(y_w|x) - grad log pi_theta(y_l|x) ) ]
```

The `sigmoid(-beta * implicit_reward_diff)` factor is exactly "how wrong the implicit reward model currently is about this pair" -- close to 1 (a large gradient weight) when the policy currently assigns the *loser* higher implicit reward than the winner (a badly wrong pair, needing a big correction), and close to 0 when the policy already strongly prefers the winner (a pair the policy already gets right, needing little further adjustment). This is a clean, interpretable statement: **DPO's gradient automatically upweights training signal from preference pairs the current policy is getting most wrong**, without ever needing an explicit reward model to compute that weighting -- an elegant, non-obvious consequence of the derivation, not a hand-designed heuristic.

### 5.1 Deriving the Gradient in Full

It is worth deriving the gradient claim from Section 5 rather than taking it on faith, since "the loss automatically reweights by how wrong the model is" is exactly the kind of claim an interviewer will ask you to justify. Write `u = beta * ( log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x)) )` for the scalar the loss depends on, so `L_DPO = -log sigmoid(u)`. Using `d/du [-log sigmoid(u)] = -(1 - sigmoid(u)) = sigmoid(u) - 1 = -sigmoid(-u)`, the chain rule gives:

```
grad_theta L_DPO = -sigmoid(-u) * grad_theta[u]
                 = -sigmoid(-u) * beta * ( grad_theta log pi_theta(y_w|x) - grad_theta log pi_theta(y_l|x) )
```

`sigmoid(-u)` is large (close to 1) exactly when `u` is very negative -- i.e., when the implicit reward difference `r_theta(x,y_w) - r_theta(x,y_l)` (which equals `u`, up to the constant `beta*log Z(x)` terms that canceled in Section 4) is negative, meaning the policy currently, wrongly, favors the loser. `sigmoid(-u)` is small (close to 0) when `u` is already strongly positive, i.e., the policy already strongly favors the winner. This confirms precisely, via the chain rule and nothing else, the claim in Section 5 and the numeric illustration in Section 8.4: the per-example gradient magnitude is scaled by `sigmoid(-u)`, which is large for pairs the model gets wrong and small for pairs it already gets right -- an emergent consequence of differentiating a logistic loss, not a separately engineered weighting scheme.

### 5.2 Practical Sensitivity to Beta

`beta` plays a double role that is worth being explicit about in an interview: it is simultaneously the trust-region strength (as in PPO) and, per Section 5.1, a multiplicative scale factor on the gradient signal itself, since it appears both inside the `sigmoid` (shaping how sharply the loss distinguishes well-fit from badly-fit pairs) and as a direct multiplier on the gradient. In practice this makes DPO's effective learning dynamics fairly sensitive to `beta` in a way that is easy to underappreciate coming from an RLHF background where `beta` only ever appeared as an additive penalty coefficient: a `beta` that is too small flattens `sigmoid(-u)` toward `0.5` for almost all pairs regardless of how right or wrong the model currently is (weak, undifferentiated gradient signal, and a policy that can drift far from `pi_ref` for a given amount of loss reduction, mirroring the too-small-beta reward-hacking risk from RLHF); a `beta` that is too large makes `sigmoid(-u)` saturate toward 0 or 1 very quickly for typical log-probability gaps, effectively making the loss binary/uninformative once the model is even slightly on the correct side of a pair, and can stall learning on pairs where the model needs to move further than one saturated gradient step will take it. Typical published DPO recipes use `beta` in the range of roughly 0.1 to 0.5, but the right value is dataset- and model-scale dependent and is treated as a real hyperparameter to sweep, not a default to set once.

### 6. Why This Eliminates the Reward Model and the RL Loop -- and What It Doesn't Eliminate

**6.1 What's genuinely gone.** There is no separate reward-model training stage, no rollout generation, no value function, no GAE, no PPO clipping, no actor-learner systems complexity (File 002, Section 5). Training is a single, standard supervised-style loss computed on a fixed, pre-collected preference dataset via ordinary forward/backward passes -- the same computational shape as SFT, just with two forward passes per training example (policy on `y_w` and `y_l`) plus two more through the frozen reference model (or, more efficiently, one batched forward pass covering all four).

**6.2 What's still there, just implicit.** DPO does not eliminate the *reward model concept* -- Section 3's identity shows the trained policy always implicitly defines a reward function via its log-ratio to the reference. Nor does it eliminate the reference model or the `beta` hyperparameter: `pi_ref` must still be computed (typically the SFT checkpoint, held frozen, requiring a forward pass through a full second model during DPO training, exactly as the reference model did during PPO) and `beta` still controls the same fundamental reward-seeking-versus-staying-close-to-reference tradeoff as in PPO, just algebraically folded into a single loss rather than realized as a separate penalty term.

**6.3 What's genuinely different in the failure-mode profile.** DPO is off-policy (or, more precisely, trained on a fixed, pre-collected dataset of comparisons that may have been generated by a different policy snapshot than the one currently training), whereas PPO is on-policy (it always trains on rollouts from the current policy). This removes the "generate rollouts, wait for them, then update" bottleneck of Section 5 in File 002, but it also removes RLHF's ability to keep sampling *fresh* comparisons from the model's current output distribution as training progresses (Section 6.6 of File 002's discussion of iterated RLHF) -- DPO trains on however-stale a preference dataset was originally collected, which introduces its own failure modes (Section 7).

### 7. Known DPO Failure Modes

**7.1 Likelihood displacement / the "probability of both responses decreases" phenomenon.** A widely observed empirical pathology: DPO training can cause the *absolute* log-probability of both the winning and losing response to decrease over training, with only the *relative gap* between them (which is the only thing the loss actually optimizes, since the loss is a function of the log-ratio difference) increasing as intended. Mechanically, this happens because the loss has no term anchoring the winning response's absolute probability -- it only ever sees the pairwise contrast, so the optimizer is free to satisfy the loss by decreasing `pi_theta(y_l|x)` faster than `pi_theta(y_w|x)`, or even by decreasing both, as long as the *gap* moves in the right direction. In the worst observed cases, this can push probability mass onto responses that were *never in the training data at all*, since nothing in the objective prevents mass from being redistributed to unseen completions as a side effect of suppressing the specific losing completion in the dataset.

**7.2 Overfitting to a finite, static preference dataset.** Because DPO trains directly on log-probabilities of specific, fixed `(y_w, y_l)` pairs rather than through a smoothing, generalizing reward model intermediary, DPO can be more prone than RLHF to overfitting the idiosyncrasies of the exact pairs in the training set, rather than learning a generalizable notion of quality -- a reward model, being a separate parametric function fit across many comparisons, can act as an implicit regularizer/generalizer that a direct pairwise loss does not provide as strongly. This motivates IPO (Section 8.1).

**7.3 Sensitivity to reference-policy quality and distribution.** Since the entire loss is defined relative to `pi_ref`, a poorly chosen or low-quality reference model (or a reference model whose output distribution is very different from the distribution the preference data was collected against) can distort training in ways that are harder to diagnose than an analogous issue in RLHF, where the reward model's absolute scores at least provide an independent diagnostic signal.

**7.4 Length bias persists, in a different form.** Just as RM-based RLHF can learn spurious length-reward correlations (File 002, Section 2.6), DPO-trained policies are empirically observed to increase response length during training even when length is not the intended signal, because the log-probability-ratio objective can be satisfied in part by changes correlated with length (e.g., a longer response naturally has more tokens over which small per-token log-probability gains can accumulate into a larger sequence-level log-ratio). Mitigations mirror RLHF's: length-normalizing the loss, or curating preference data that decorrelates length from preference labels.

### 8. The Variant Landscape

Each variant below targets one specific, nameable problem with vanilla DPO (or with the pairwise-preference-data assumption itself) rather than being a generic "improvement" -- a staff-level answer distinguishes them on exactly this axis.

**8.1 IPO (Identity Preference Optimization) -- fixing the overfitting/unbounded-optimization failure mode.** DPO's loss, `-log sigmoid(beta * logit_gap)`, is minimized as `logit_gap -> +infinity` -- there is no finite optimum, so as training continues (or as `beta` is set too low), the optimizer is pushed to drive the log-ratio gap between winner and loser arbitrarily large, which is exactly the mechanism behind Section 7.1's likelihood-displacement pathology and general overfitting to the training pairs. IPO (Azar et al., 2023) replaces the logistic loss with a **squared-error loss** targeting a *finite* desired gap, directly capping how far the optimization is incentivized to push:

```
L_IPO(theta) = E_{(x,y_w,y_l)~D} [ ( beta * (log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x)))  -  1/2 )^2 ]
```

Because this is a squared-error loss with a finite target (here written with a target gap of `1/(2*something)`-scale, exact constant depending on the derivation's normalization), the loss is minimized at a *specific finite value* of the log-ratio gap rather than being driven toward infinity -- directly removing the unbounded-optimization pathology that motivates DPO's practical instability, at the cost of a new hyperparameter (the target gap) and a loss shape (squared error) that no longer has DPO's clean "automatically upweight the pairs the policy gets most wrong" gradient interpretation from Section 5, since squared error weights all pairs by their deviation from the target rather than by an implicit-reward-model-error term.

**8.2 KTO (Kahneman-Tversky Optimization) -- removing the paired-comparison requirement.** Both DPO and IPO require *paired* preference data: for the same prompt, one response marked better and one worse. In practice, a large amount of real-world feedback is not naturally paired -- a thumbs up/down on a single response, an accept/reject of a single suggested completion, implicit signals like whether a user kept or discarded a generated edit. Collecting genuinely comparable pairs (two responses to the *same* prompt, judged against each other) is a real data-collection constraint that KTO (Ethayarajh et al., 2024) removes by deriving a loss directly from **unpaired binary desirable/undesirable labels** on individual `(x, y)` examples, motivated explicitly by prospect theory (Kahneman and Tversky's behavioral-economics model of how humans perceive gains and losses asymmetrically relative to a reference point, rather than treating utility as a simple linear function of outcome). The KTO loss constructs an implicit reference point from the *batch* (using the average implicit reward across the current batch as a proxy for "what a comparison point would have been," in the absence of an actual paired comparison), and then applies an asymmetric loss that (per prospect theory) weights losses more heavily than equivalent gains:

```
r_theta(x,y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )        # the same implicit reward as DPO
z_ref = mean over the batch of KL(pi_theta(.|x) || pi_ref(.|x)) # batch-level reference point

L_KTO = E[ lambda_D * (1 - sigmoid(r_theta(x,y) - z_ref)) ]   if y is labeled desirable
      = E[ lambda_U * (1 - sigmoid(z_ref - r_theta(x,y))) ]   if y is labeled undesirable
```

with `lambda_D, lambda_U` separate weighting constants (allowing, e.g., undesirable examples to be weighted more heavily, matching prospect theory's loss-aversion asymmetry). The practical significance: KTO lets a team train a DPO-like objective directly on the kind of feedback that is actually cheap and abundant in production (binary accept/reject signals at the level of individual responses) rather than requiring the more expensive, more deliberately-collected paired-comparison data DPO and IPO need.

**8.3 ORPO (Odds Ratio Preference Optimization) -- collapsing SFT and preference-tuning into a single stage, with no reference model at all.** DPO (and IPO, KTO) all assume you already have an SFT model to use as `pi_ref` -- they are explicitly a *second* stage, run after SFT. ORPO (Hong et al., 2024) asks whether the reference-model requirement can be removed entirely by combining the SFT objective and a preference-based penalty into one loss, trained from the base pretrained model directly. The ORPO loss adds an **odds-ratio-based penalty term** to the ordinary SFT cross-entropy loss:

```
odds(y|x) = pi_theta(y|x) / (1 - pi_theta(y|x))    # per-sequence odds, using the length-normalized sequence probability

L_ORPO = L_SFT(y_w)  -  lambda * log sigmoid( log( odds(y_w|x) / odds(y_l|x) ) )
```

The first term is the ordinary SFT loss on the preferred response `y_w` alone (exactly File 001's masked cross-entropy). The second term is a preference penalty expressed via the **log odds ratio** between the preferred and dispreferred response's sequence-level probabilities under the *same* model currently being trained -- notably, with no separate frozen reference model anywhere in the expression, since the "contrast" is between the model's current odds for `y_w` versus `y_l`, not between the model and a reference. This makes ORPO a genuinely single-stage recipe: train directly from a base or lightly-SFT'd model on preference-labeled data, getting both the imitation signal (via `L_SFT`) and a preference-ranking signal (via the odds-ratio term) from one loss, one dataset, one training run -- at the cost of losing the explicit, tunable KL-to-reference trust-region mechanism that both PPO and DPO rely on to bound how far the policy is allowed to move, replaced instead by whatever implicit regularization the combined SFT-plus-odds-ratio loss provides on its own.

### 8.4 A Worked Numeric Example of the Implicit Gradient Weighting

Take `beta = 0.1` and a preference pair where the policy currently assigns log-probabilities `log pi_theta(y_w|x) = -12.0`, `log pi_theta(y_l|x) = -10.0` (i.e., the policy currently thinks the *loser* is more likely than the winner -- a badly wrong pair), and reference log-probabilities `log pi_ref(y_w|x) = -13.0`, `log pi_ref(y_l|x) = -13.0` (the reference treats them as equally likely). The implicit reward difference is `beta * [ (-12.0 - (-13.0)) - (-10.0 - (-13.0)) ] = 0.1 * [1.0 - 3.0] = -0.2`. Because this is negative (the model's implicit reward currently favors the loser), `sigmoid(-1 * (-0.2)) = sigmoid(0.2) ≈ 0.55`, giving a substantial gradient weight on this pair -- the optimizer will push meaningfully to increase `pi_theta(y_w|x)` relative to `pi_theta(y_l|x)`. Contrast this with a pair the model already gets right by a wide margin, say implicit reward difference `+3.0`: `sigmoid(-3.0) ≈ 0.047`, a small gradient weight, since there is little to correct. This numeric contrast is the concrete cash-value of the "DPO automatically upweights the pairs the model is most wrong about" claim in Section 5 -- it is not a qualitative gesture, it is a specific, computable per-example weight that falls directly out of the loss's derivative.

### 8.5 Empirical DPO-vs-PPO Findings, Reported Honestly

Public empirical comparisons between DPO and PPO-based RLHF do not point to a uniform winner, and a staff-level answer should reflect that rather than asserting DPO has simply superseded RLHF. Several independent studies and practitioner reports (most notably work explicitly titled around the question "is DPO superior to PPO") find that a *well-tuned* PPO pipeline, with a good reward model and careful KL tuning, can match or exceed DPO's performance on some preference benchmarks -- suggesting DPO's main advantage is not a higher achievable ceiling but a dramatically lower implementation and tuning cost to reach a good-but-not-necessarily-best result, plus removal of RL-specific instability risks (Section 6.1). Where DPO is reported to underperform, the gap is often attributable to exactly the failure modes in Section 7 (particularly sensitivity to preference-data quality and distribution, since DPO has no learned, generalizing reward model standing between the raw comparison data and the policy update) rather than to any inherent ceiling in the closed-form derivation itself. The pragmatic, widely-adopted industry conclusion has been that DPO-family methods are an excellent default for teams without deep in-house RL infrastructure and expertise, while frontier labs with mature RL systems continue to use both DPO-family and PPO-family methods depending on the specific stage and objective of a given post-training run, sometimes within the same overall pipeline (e.g., DPO for an early, cheap preference-tuning pass, PPO or a verifiable-reward RL stage for later, higher-stakes refinement).

### 9. Comparative Summary Table

| Method | Data requirement | Reference model needed? | Separate SFT stage needed? | Specific problem targeted |
|---|---|---|---|---|
| RLHF (PPO) | Paired/ranked comparisons, RM trained separately | Yes (KL anchor) | Yes | Baseline; optimizes preferences via RL against a learned RM |
| DPO | Paired comparisons | Yes (frozen, for the log-ratio) | Yes | Removes RM + RL loop via the closed-form policy identity |
| IPO | Paired comparisons | Yes | Yes | Removes DPO's unbounded-optimization / overfitting pathology |
| KTO | Unpaired binary desirable/undesirable labels | Yes | Yes | Removes the paired-comparison data requirement |
| ORPO | Paired comparisons | No | No -- folded into one stage | Removes the reference model and the separate SFT stage entirely |

### 9.1 The Variant Losses, Side by Side in Code

Seeing all four losses implemented next to each other makes the differences concrete rather than purely notational:

```python
def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta=0.1):
    u = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -F.logsigmoid(u).mean()

def ipo_loss(lp_w, lp_l, ref_w, ref_l, beta=0.1, tau=1.0):
    u = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return ((u - tau) ** 2).mean()                      # squared error to a finite target, not a logistic loss

def kto_loss(lp, ref, is_desirable, beta=0.1, lam_d=1.0, lam_u=1.0, z_ref=0.0):
    r = beta * (lp - ref)                                # implicit reward for a single (unpaired) example
    desirable_term   = lam_d * (1 - torch.sigmoid(r - z_ref))
    undesirable_term = lam_u * (1 - torch.sigmoid(z_ref - r))
    return torch.where(is_desirable, desirable_term, undesirable_term).mean()

def orpo_loss(sft_loss, lp_w, lp_l, seq_len_w, seq_len_l, lam=0.1):
    # lp_w/lp_l here are *length-normalized* sequence log-probs (log-prob per token) for the odds computation
    logodds_w = lp_w - torch.log1p(-torch.exp(lp_w))     # log( p / (1-p) ) in a numerically safer form
    logodds_l = lp_l - torch.log1p(-torch.exp(lp_l))
    preference_term = -F.logsigmoid(logodds_w - logodds_l)
    return sft_loss + lam * preference_term.mean()        # sft_loss is the ordinary File 001 masked CE loss
```

Reading down this list: DPO and IPO share the exact same `u` computation (the beta-scaled log-ratio-of-log-ratios) and differ only in the final loss shape (logistic versus squared-error) applied to it -- underscoring that IPO is a minimal, surgical change to DPO's objective, not a different framework. KTO drops the paired subtraction entirely, operating on one example's implicit reward against a batch-derived reference point. ORPO is the odd one out structurally: it is the only one of the four with an explicit additive SFT loss term, and the only one with no `ref` log-probabilities anywhere in the expression.

### 10. Common Interview Traps

- **Reciting the DPO loss without being able to derive it.** The derivation (Sections 2-4) is the actual asset; being able to reproduce "reward equals beta times log-ratio plus a constant, substitute into Bradley-Terry, watch the per-prompt constant cancel" from memory is what distinguishes a staff-level answer from a memorized formula.
- **Claiming DPO removes the need for a reference model.** It does not (Section 6.2) -- only ORPO among the variants above removes it, and it does so by folding SFT and preference-tuning into one stage rather than by finding a way around needing a reference distribution in general.
- **Claiming DPO is strictly more stable than RLHF.** Section 7 gives concrete, real counterexamples (likelihood displacement, length bias, overfitting to static pairs); the honest comparison is "different failure modes, not fewer failure modes."
- **Confusing KTO's motivation with IPO's.** IPO fixes an optimization pathology given the *same kind* of paired data DPO uses; KTO changes what *kind* of data is required altogether. These are orthogonal contributions that happen to both post-date and modify DPO.
- **Describing ORPO as "just DPO without a reference model" without noting it also removes the separate SFT stage.** Both removals (reference model and separate SFT stage) are the point of ORPO's specific loss construction, not just one of them.

### 11. Quick-Reference Summary

- The core derivation: the KL-constrained reward-maximization problem has a closed-form optimal policy `pi*(y|x) = pi_ref(y|x) * exp(r(x,y)/beta) / Z(x)`; inverting gives `r(x,y) = beta*log(pi*(y|x)/pi_ref(y|x)) + beta*log Z(x)`.
- Substituting this into the Bradley-Terry loss makes the intractable, per-prompt `Z(x)` term cancel exactly, yielding a loss computable purely from policy and reference log-probabilities on a fixed preference dataset: the DPO loss.
- DPO's gradient automatically weights each preference pair by how wrong the model's current implicit reward ordering is for that pair -- an emergent, not hand-designed, property of the derivation.
- DPO removes the separate reward model and the RL loop's systems complexity, but keeps the reference model, the `beta` hyperparameter, and the underlying reward-seeking-vs-staying-close-to-reference tradeoff, now folded into one loss.
- Known DPO failure modes: likelihood displacement (absolute probabilities of both responses can fall), overfitting to a static finite preference set, reference-quality sensitivity, and a length bias analogous to RLHF's.
- IPO bounds the optimization target to fix the unbounded-logit-gap pathology; KTO removes the paired-comparison data requirement via a prospect-theory-motivated loss on unpaired binary labels; ORPO removes the reference model and the separate SFT stage by combining SFT and an odds-ratio preference penalty into one loss.
- None of these variants is a strict Pareto improvement on the others -- each trades a specific capability or data assumption for a specific fix, and a good answer names the trade precisely rather than asserting a total ordering.

### 12. Cross-References

File 002 is the necessary prerequisite for this file's Section 4 derivation (the Bradley-Terry loss being substituted into) and Section 1 setup (the KL-constrained objective being solved). File 004 covers RLAIF, which is orthogonal to the RLHF-vs-DPO axis discussed here -- AI-generated preference labels can feed either a reward-model-plus-PPO pipeline or a DPO-style direct loss equally well, since both consume the same underlying `(x, y_w, y_l)` comparison format. File 005 covers RLVR, where the "reward" is a ground-truth verifier rather than a learned or preference-derived signal at all -- worth contrasting against this file's entire premise, which is about how to optimize against *preferences* efficiently, not about replacing a learned reward with a checkable one.
