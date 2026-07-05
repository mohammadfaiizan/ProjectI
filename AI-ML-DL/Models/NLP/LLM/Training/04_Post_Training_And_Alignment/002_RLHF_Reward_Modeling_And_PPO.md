## RLHF: Reward Modeling and PPO

### 0. Scope of This File

This file derives the mechanics of the classical three-stage RLHF pipeline -- SFT, reward modeling, PPO -- from first principles, at a level of generality that applies regardless of which lab or paper you're discussing. For the canonical, citable, numbers-attached implementation of this exact pipeline (InstructGPT: model sizes, dataset sizes, the PPO-ptx mitigation, the specific empirical results), see `../GPT/004_InstructGPT_And_RLHF.md`, which this file assumes you have read or will read alongside it and does not re-derive line by line. What follows here is the general mechanism -- the loss functions, the reasons they take the form they do, and the systems reality of running RLHF in practice -- treated as a technique independent of any one paper's specific numbers.

### 1. The Pipeline, End to End

1. **SFT** (File 001): produce a policy that reliably behaves like an assistant, via supervised imitation of demonstration data. This model is called `pi^SFT` throughout this file, and it plays two roles downstream: initialization for the RL policy, and the frozen reference distribution the KL penalty is measured against.
2. **Reward model (RM) training** (Section 2): produce a scalar scoring function `r_theta(x, y)` that assigns higher scores to responses `y` (to prompt `x`) that humans prefer, trained on human pairwise/ranked comparisons via a Bradley-Terry-style loss.
3. **RL fine-tuning via PPO** (Section 3): use `r_theta` as a reward signal to further train the policy with reinforcement learning, subject to a KL constraint anchoring it to `pi^SFT`.

The entire point of inserting stage 2 between stage 1 and stage 3 is that "what humans prefer" is not a differentiable function you can backpropagate through directly -- a human's judgment of response quality is not a closed-form expression of the model's parameters. The RM is a learned, differentiable *proxy* for that judgment, trained once on a fixed comparison dataset, that can then be queried cheaply and repeatedly (as many times as needed during the RL loop) without needing a human in the loop for every single policy update.

### 2. Reward Model Training: the Bradley-Terry Derivation

**2.1 The model of pairwise choice.** The Bradley-Terry model, originally from 1952 paired-comparison statistics, posits that every item `y` has a latent, real-valued "strength" or "goodness" score `r(y)`, and the probability that a comparison between two items `y_w` and `y_l` results in `y_w` being judged better is a logistic function of the *difference* in their scores:

```
P(y_w preferred over y_l) = sigmoid( r(y_w) - r(y_l) ) = exp(r(y_w)) / (exp(r(y_w)) + exp(r(y_l)))
```

Applied to language-model outputs, the score is made conditional on the prompt: `r_theta(x, y)`, parameterized by a neural network (Section 2.3), representing "how good is response `y` to prompt `x`."

**2.2 Deriving the loss as maximum likelihood.** Given a dataset `D` of comparisons `(x, y_w, y_l)` -- for prompt `x`, response `y_w` was judged preferred over response `y_l` -- the Bradley-Terry model's log-likelihood of the observed preference is `log sigmoid(r_theta(x,y_w) - r_theta(x,y_l))`. Training the RM by maximum likelihood over the whole dataset means minimizing the negative log-likelihood:

```
L_RM(theta) = - E_{(x, y_w, y_l) ~ D} [ log( sigmoid( r_theta(x, y_w) - r_theta(x, y_l) ) ) ]
```

This is mechanically identical to training a logistic regression on score *differences*: the network only ever needs to get the relative ordering, and the relative margin, right -- it never receives a supervised target for the absolute value of `r_theta(x, y)` for any single `y` in isolation. This has an immediate, important consequence worth stating explicitly: **reward model scores are only meaningful up to an additive, per-prompt constant.** Adding an arbitrary constant `c(x)` to `r_theta(x, y)` for every `y` conditioned on a given `x` leaves every difference `r_theta(x,y_w) - r_theta(x,y_l)` (and hence the loss) completely unchanged. This is why raw RM scores are not comparable across different prompts, and why any downstream use of the RM (e.g., filtering completions above some absolute score threshold across a diverse prompt set) needs to normalize or otherwise account for this per-prompt scale-and-shift ambiguity.

**2.3 The reward model's architecture.** Concretely, an RM is built by taking a transformer backbone -- often literally initialized from the SFT checkpoint, since it already understands language and the task distribution -- removing the softmax LM head, and replacing it with a linear projection from the final-token hidden state to a single scalar:

```python
class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone           # e.g., initialized from the SFT checkpoint
        self.value_head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        last_token_idx = attention_mask.sum(dim=1) - 1              # index of last real (non-pad) token
        last_hidden = hidden[torch.arange(hidden.size(0)), last_token_idx]
        return self.value_head(last_hidden).squeeze(-1)             # shape: (batch,)
```

**2.4 A full, runnable pairwise loss implementation:**

```python
def bradley_terry_loss(reward_model, x_ids, y_w_ids, y_l_ids, attn_w, attn_l):
    r_w = reward_model(y_w_ids, attn_w)     # x is concatenated into y_w_ids/y_l_ids already
    r_l = reward_model(y_l_ids, attn_l)
    return -F.logsigmoid(r_w - r_l).mean()  # logsigmoid is numerically stabler than log(sigmoid(.))
```

Using `F.logsigmoid` rather than `torch.log(torch.sigmoid(...))` is not a stylistic nicety -- for large negative margins `sigmoid` underflows to exactly `0.0` in floating point well before the corresponding `logsigmoid` value would, so a naive two-step implementation silently produces `log(0) = -inf` and `NaN` gradients on hard, confidently-wrong examples; `logsigmoid` computes the same mathematical quantity via a numerically stable formulation (`-softplus(-x)`) that never has this failure mode.

**2.5 Handling K-wise rankings.** When a labeler ranks `K` completions for one prompt from best to worst rather than doing a single binary comparison, the standard approach decomposes the full ranking into all `C(K,2)` pairwise comparisons implied by it, and trains on all of them. Two efficiency/statistics details matter here: all `C(K,2)` pairs from one prompt are highly correlated (they share the same handful of underlying completions), so treating each pair as an independent i.i.d. training example both overweights prompts with larger `K` and risks overfitting -- the standard mitigation is to normalize the loss by the number of comparisons per prompt, and to batch all `C(K,2)` comparisons from a single prompt as one unit of computation (compute each of the `K` completions' scalar scores once via `K` forward passes, then form all `C(K,2)` loss terms from those cached scores, rather than re-running the transformer redundantly for every pairing).

**2.6 What the RM is actually learning, and where it breaks.** The RM is a supervised classifier trained on a finite, necessarily incomplete sample of human judgments, and it will generalize the way any supervised model generalizes: well within the training distribution of prompts and response styles, and unreliably outside it. Two concrete, well-documented failure patterns: **length bias** (RMs trained on typical human comparison data reliably learn a spurious positive correlation between response length and predicted quality, because longer responses often *do* correlate with thoroughness within the training distribution, but the RM overgeneralizes this correlation to reward verbosity even when it doesn't track genuine quality); and **out-of-distribution overconfidence** (once the policy being optimized against the RM starts producing text stylistically different from the RM's training distribution -- which is guaranteed to happen as RL optimization proceeds, since RL is explicitly searching for whatever text scores highest -- the RM's scores on that off-distribution text are not validated by any human judgment and can be arbitrarily miscalibrated while still looking confident). This second failure mode is the direct setup for Section 4's reward-hacking discussion.

### 3. PPO Fine-Tuning: Policy-Gradient Mechanics for Language Generation

**3.1 Mapping RL vocabulary onto autoregressive generation.** In the standard RL framing, an agent takes actions in a state, receiving rewards, and the objective is to maximize expected cumulative reward. Mapped onto language generation: the **state** at step `t` is the prompt plus all tokens generated so far; the **action** is choosing the next token; the **policy** `pi_phi(a_t | s_t)` is the language model itself, viewed as a distribution over next tokens; and an **episode** is one full generated completion, from the start of the response to an end-of-sequence token. Reward, in the vanilla RLHF setup, is **sparse and terminal**: the reward model only scores a *completed* response, so the "environment" gives zero reward at every intermediate token and the full `r_theta(x, y)` score only at the final step -- a materially harder credit-assignment problem than RL settings with reward at every environment step (robotics, most Atari games), and part of why the value-function baseline (3.3) matters as much as it does here.

**3.2 The clipped surrogate objective.** PPO (Schulman et al., 2017) is a policy-gradient method designed to take large, efficient steps without the instability of naive policy gradient (which is highly sensitive to step size: too large a step and a single bad update can catastrophically and irreversibly worsen the policy, since all future rollouts are then sampled from the now-worse policy). Let `pi_phi` be the current policy being updated and `pi_phi_old` be the (fixed, for the current update) policy that generated the rollouts being trained on. Define the probability ratio at each token:

```
r_t(phi) = pi_phi(a_t | s_t) / pi_phi_old(a_t | s_t)
```

Given an advantage estimate `A_t` (3.3) for that token, the clipped surrogate objective to *maximize* is:

```
L^CLIP(phi) = E_t[ min( r_t(phi) * A_t, clip(r_t(phi), 1 - eps, 1 + eps) * A_t ) ]
```

with `eps` typically around 0.2. Intuition: if `A_t > 0` (the action was better than the value baseline predicted), the objective wants to increase `r_t(phi)` -- but the `min` with the clipped term caps the *reward* for doing so once `r_t(phi)` exceeds `1 + eps`, removing the incentive to move the policy arbitrarily far from `pi_phi_old` on the strength of a single advantage estimate. Symmetrically, for `A_t < 0`, the penalty for decreasing `r_t(phi)` is capped once it drops below `1 - eps`. This is a first-order, per-token approximation to a trust-region constraint (the same underlying goal as TRPO's explicit KL-divergence constraint, achieved here via simple clipping instead of TRPO's expensive conjugate-gradient machinery).

**3.3 The value function and Generalized Advantage Estimation (GAE).** Because reward is sparse (terminal-only in the vanilla setup), naively using the raw terminal reward as the training signal for every token in the sequence would be extremely high-variance: every token's contribution would be credited with the *entire* sequence-level reward regardless of that specific token's actual contribution to quality. PPO addresses this with a learned **value function** `V_psi(s_t)`, predicting the expected future return from state `s_t`, and uses it to compute a lower-variance **advantage** estimate: `A_t = (actual return from t onward) - V_psi(s_t)`, i.e., "how much better was this trajectory than the value function expected, from this point on." GAE further reduces variance (at the cost of some bias) by exponentially averaging `n`-step advantage estimates for multiple `n`, controlled by a parameter `lambda`:

```
delta_t = r_t + gamma * V_psi(s_{t+1}) - V_psi(s_t)              # one-step TD residual
A_t^GAE = sum_{k=0}^{infinity} (gamma * lambda)^k * delta_{t+k}   # exponentially-weighted sum
```

In the LM-RLHF setting, `r_t` is zero for every non-terminal token except for the per-token KL penalty (Section 4), and equals the RM's terminal score at the final token -- so the value function's job is specifically to learn to *predict*, from a partial completion, what the eventual RM score will be, giving the policy a dense, lower-variance training signal derived from a sparse, terminal one. The value function is typically implemented as an auxiliary scalar head sharing the policy's transformer backbone (cheaper, and the standard production choice) rather than as a fully separate model.

**3.4 A minimal, correct GAE + PPO-loss implementation:**

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    # rewards, values: (T,) tensors for one rollout; values includes V(s_T) as a bootstrap at the end
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns

def ppo_clip_loss(logp_new, logp_old, advantages, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)              # exp(log-ratio) = probability ratio r_t(phi)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.min(unclipped, clipped).mean()          # negate: we minimize loss, PPO objective is a max
```

### 4. The KL Penalty: Why It Exists, Mechanically

**4.1 The reward actually optimized.** PPO does not maximize `r_theta(x,y)` alone. The per-token reward fed into the RL algorithm is:

```
R(x, y_t) = [ r_theta(x, y) if t is the final token else 0 ]  -  beta * ( log pi_phi(y_t | ...) - log pi^SFT(y_t | ...) )
```

The second term, summed over the sequence, is a per-token Monte Carlo estimate of `beta * KL(pi_phi(.|x) || pi^SFT(.|x))` -- an estimator that works because for a token actually sampled from `pi_phi`, the log-ratio `log pi_phi(y_t) - log pi^SFT(y_t)` is an unbiased sample of the local contribution to the KL divergence between the two distributions.

**4.2 Why this term must be there -- Goodhart's Law, made mechanical.** `r_theta` is a supervised model trained on a finite, noisy sample of human comparisons (Section 2.6); it is, by construction, only accurate in the region of output-space that resembles its training distribution. An unconstrained RL process has exactly one objective: climb `r_theta` as high as possible, by whatever means available -- and because `r_theta` is an imperfect proxy, there necessarily exist regions of output-space where `r_theta` is *wrong* (assigns high score to text that is not actually good), and unconstrained optimization pressure will find and exploit those regions, precisely because that is what "maximize a function" means when the function has exploitable blind spots. This is Goodhart's Law -- "when a measure becomes a target, it ceases to be a good measure" -- applied concretely to a learned reward model, and in this literature it's called **reward model overoptimization** or **reward hacking**. The KL penalty anchors the policy near `pi^SFT`, the region of output-space the RM was actually trained and validated against real human judgments, allowing the policy to climb reward *within* a trust region where the RM's judgments are more likely to still track genuine human preference, while bounding how far it can wander into regions where the RM's judgments are unvalidated and untrustworthy.

**4.3 A secondary function: preventing entropy/diversity collapse.** Because divergence away from `pi^SFT`'s distribution is itself penalized, the KL term also discourages the policy from collapsing onto a narrow set of high-scoring but low-diversity outputs purely because that narrow mode happens to score well under `r_theta` -- a real, observed failure mode (mode collapse) in RLHF pipelines with an insufficiently strong KL penalty.

**4.4 The `beta` tradeoff, precisely.** `beta` is not a free parameter to set once and forget; it directly trades off the two failure modes on either side of it. Too small a `beta`, and the policy overoptimizes the RM, drifting into text that scores well under `r_theta` but is verbose, hedgy, or otherwise "gamed" relative to genuine quality -- an early, well-documented manifestation of what the field broadly now calls sycophancy risk. Too large a `beta`, and the policy barely moves from `pi^SFT` at all, forfeiting most of the benefit of the RL stage. In practice, `beta` is tuned empirically, sometimes with an adaptive schedule that increases `beta` if measured KL divergence exceeds a target value and decreases it otherwise -- a PID-controller-like mechanism for holding the *realized* KL divergence near a target rather than fixing the coefficient and hoping the resulting KL lands somewhere reasonable.

**4.5 Reward-overoptimization scaling curves.** Follow-up empirical work (post-dating InstructGPT, studying this exact setup more rigorously) shows that as a policy is optimized further against a *fixed* RM -- equivalently, as measured KL divergence from the reference policy grows -- the gap between the RM's proxy score and an independent, more trustworthy "true reward" signal (e.g., a larger/better RM, or fresh human ratings) tends to grow in a roughly predictable, monotonically worsening way. This gives the qualitative Goodhart argument in 4.2 a quantitative, curve-shaped form: there is empirically a point past which continuing to optimize against a fixed RM makes the *true* quality of the policy's outputs worse even as the *measured* (proxy) reward keeps climbing -- direct, load-bearing evidence for why the KL constraint (or equivalent early-stopping/RM-refresh strategies) is not an optional safety margin but a structural requirement of this training paradigm.

### 5. The Concurrent-Models Engineering Reality

Unlike SFT, which is a pure gradient-descent problem (forward, loss, backward, step, repeat, homogeneously), a single PPO training step requires several models' worth of compute concurrently, with genuinely different computational profiles:

| Model | Role | Compute profile | Trainable? |
|---|---|---|---|
| Policy (`pi_phi`) | Generates rollouts (sampling completions for prompts) | Inference-shaped: sequential, KV-cached autoregressive decode -- often the wall-clock bottleneck | Yes (this is what PPO updates) |
| Reference (`pi^SFT`) | Re-scores the same rollout tokens to compute the KL penalty | Forward-pass-only, but a full second model's worth of compute per step | No -- frozen throughout |
| Reward model (`r_theta`) | Scores completed (prompt, response) pairs | Forward-pass-only, queried once per completed rollout | No -- frozen throughout (trained once, in Stage 2) |
| Value function (`V_psi`) | Baseline for advantage estimation | Forward-pass-only per token if a separate model; folded into the policy's own forward pass if implemented as an auxiliary head | Yes, if implemented as a separate model; typically shares and trains alongside the policy's backbone if implemented as a head |
| Policy (`pi_phi`), again | Recomputes log-probs of already-generated tokens under the current (updating) parameters, for the PPO loss | Standard backward-pass training | Yes |

The net systems shape is much closer to an **actor-learner reinforcement learning system** (in the sense of IMPALA/SEED-RL/RLlib-style architectures) than to a data-parallel pretraining job: a rollout-generation phase that is inference-bound and benefits from efficient batched sampling infrastructure (large KV caches, continuous batching), interleaved with a policy-update phase that is standard backward-pass training, with two additional frozen models that must be kept resident and queryable throughout the loop. At minimum three models' worth of concurrent forward-pass compute (policy generation, frozen reference, frozen reward model) plus the policy's own backward pass is required at every training step -- a fundamentally more heterogeneous, more inference-heavy systems problem than gradient-descent-only pretraining or SFT. See `../GPT/004_InstructGPT_And_RLHF.md` Section 4 for the fuller discussion of how this shows up concretely in a production pipeline, including the finding that reward-model size can be decoupled from policy size (a smaller, cheaper, more stable RM can outperform a larger one as the training-time reward signal).

A second, easily underestimated cost driver: **rollout generation is the dominant wall-clock cost of PPO** in most practical setups, because generating a completion requires as many sequential forward passes as there are output tokens (even with KV caching, decoding is inherently sequential per sequence), whereas a training forward/backward pass processes an entire sequence's tokens in parallel. This is why production PPO implementations invest heavily in optimized batched/continuous-batching generation infrastructure for the rollout phase specifically -- the same infrastructure concerns that motivate dedicated inference-serving stacks (File 08 in this curriculum's Inference and Serving module) apply directly inside the training loop here, not just at deployment time.

### 5.1 Weight Synchronization Between Training and Rollout Engines

A concrete, modern systems detail worth knowing beyond what any 2022-era paper discloses: production RLHF/RL infrastructure today typically runs rollout generation on a dedicated, inference-optimized serving engine (e.g., vLLM- or SGLang-style continuous-batching servers, which are dramatically more throughput-efficient at autoregressive decoding than a training framework's forward pass) that is architecturally *separate* from the training framework doing the PPO gradient updates (e.g., Megatron-LM- or FSDP-based). This split is a direct consequence of Section 5's observation that generation and training have fundamentally different computational shapes and are best served by different, specialized software stacks.

The engineering cost of this split is **weight synchronization**: after every policy update (or every few), the updated policy weights must be pushed from the training framework's sharded parameter representation into the rollout engine's own (typically differently-sharded, sometimes differently-quantized) representation, and this transfer must happen frequently enough that rollouts are not generated from a badly stale policy, while not so frequently that the transfer itself dominates wall-clock time. Common approaches include broadcasting full or delta weight updates over high-bandwidth interconnect between colocated training and inference GPU pools, or, in disaggregated setups, serializing updated weights through a shared, fast storage layer. Getting this synchronization both correct (no stale or partially-updated weights silently served during a sync window) and fast is a genuinely nontrivial distributed-systems problem layered on top of the RL algorithm itself, and is one of the primary reasons RLHF/RL post-training infrastructure at frontier labs is a distinct systems specialty from pretraining infrastructure, warranting dedicated engineering investment rather than being treated as an afterthought bolted onto existing training clusters.

### 6. Failure Modes Worth Being Able to Diagnose Cold

- **Reward hacking / RM overoptimization** (Section 4): the policy finds and exploits a blind spot in `r_theta`. Diagnostic signal: measured RM score keeps climbing while human-rated or independently-scored quality plateaus or declines, and/or measured KL from the reference model grows past where the RM's training distribution can be trusted. Fix: tighten `beta`, refresh/retrain the RM on more recent policy samples, early-stop, or reduce PPO epochs/steps against a fixed RM checkpoint.
- **Length/verbosity gaming**: the policy learns that longer responses score better under the RM's learned length bias (Section 2.6) regardless of genuine informativeness. Diagnostic: RM score correlates strongly with response length on held-out prompts even after controlling for task type. Fix: length-normalize or length-penalize the reward explicitly, or retrain the RM on a comparison dataset that deliberately balances length across preferred/dispreferred pairs.
- **Mode collapse / diversity loss** (Section 4.3): the policy converges to a narrow set of stereotyped high-scoring response patterns. Diagnostic: falling output entropy/diversity metrics across training, or qualitatively repetitive completions across diverse prompts. Fix: increase `beta`, or add an explicit entropy bonus term to the PPO objective.
- **Value function underfitting**: if `V_psi` is a poor predictor of realized return, advantage estimates are high-variance or biased, destabilizing PPO updates. Diagnostic: high variance in the value-function's own loss, or PPO updates that oscillate rather than steadily improving measured reward. Fix: more value-function training epochs per PPO iteration, tuning GAE's `lambda`, or a larger/better-initialized value head.
- **RM training instability at very large scale**: as InstructGPT's own 175B-RM finding shows (`../GPT/004_InstructGPT_And_RLHF.md`, Sections 2-3), bigger is not automatically better or more stable for a reward model specifically, in contrast to the policy-scaling story -- worth naming directly if asked "would you just always use the biggest possible RM."

### 6.1 Why PPO and Not Plain REINFORCE

It is worth being able to answer "why not just use vanilla policy gradient (REINFORCE) instead of all this clipping machinery" precisely, since it is a natural follow-up question. Vanilla REINFORCE's update is `E_t[ log pi_phi(a_t|s_t) * A_t ]`, taking a gradient step directly proportional to the advantage, with no constraint at all on how far a single update moves the policy. Because language-model action spaces are enormous (the vocabulary, at every one of potentially thousands of generated tokens) and advantage estimates from a sparse, terminal, RM-derived reward are noisy, an unconstrained step sized by a noisy advantage estimate can occasionally take a very large, destructive step -- and because every subsequent rollout is sampled from the now-updated policy, a single bad step can compound (the policy degrades, generates worse rollouts, advantage estimates against those worse rollouts become even less reliable, and so on). PPO's clipping directly bounds the per-update change in the policy's action probabilities, converting an unconstrained, noise-sensitive update into a bounded, empirically much more stable one, at a modest implementation cost (tracking `pi_phi_old` and the clip range) relative to TRPO's far more expensive exact trust-region machinery. This robustness-per-unit-of-implementation-complexity tradeoff is exactly why PPO, rather than either simpler REINFORCE or more rigorous-but-expensive TRPO, became the default choice for this setting.

### 6.2 Typical PPO Hyperparameters, as a Sanity-Check Reference

| Hyperparameter | Typical range | Role |
|---|---|---|
| Clip range `eps` | ~0.1 - 0.2 | Bounds the per-token probability-ratio change trusted in one update |
| KL coefficient `beta` | Small, often adaptively tuned toward a target KL | Trust-region anchor to `pi^SFT`; the primary reward-hacking countermeasure |
| GAE `lambda` | ~0.9 - 0.95 | Bias/variance tradeoff in advantage estimation; higher trusts longer-horizon returns more |
| Discount `gamma` | Often close to 1.0 (episodes are short relative to typical RL settings) | Discounting of future reward within one completion |
| PPO epochs per rollout batch | Small (a handful) | Number of gradient passes reusing the same batch of rollouts before generating fresh ones; too many risks the policy drifting too far from `pi_phi_old`, invalidating the importance-sampling ratio's validity |
| Rollout batch size | Large relative to gradient batch size | Amortizes the expensive generation phase (Section 5) across many gradient steps |

### 6.3 A Worked Toy Example of the Reward and Advantage Calculation

Concretely, for a 4-token completion `y = (y_1, y_2, y_3, y_4)` to prompt `x`, with per-token KL contributions `k_1, k_2, k_3, k_4` and terminal RM score `r_theta(x,y) = 2.0`, the per-token reward sequence fed into GAE is `[-beta*k_1, -beta*k_2, -beta*k_3, r_theta(x,y) - beta*k_4]` -- zero "task" reward at every position except the last, where the RM score is added in, with the KL penalty subtracted at *every* position since it is a dense, per-token quantity even though the task reward is sparse. If `beta = 0.02` and the KL contributions are small (say, `0.1` each), the reward sequence might concretely be `[-0.002, -0.002, -0.002, 1.998]` -- illustrating numerically just how dominated the per-token reward signal is by the single terminal RM score relative to the comparatively tiny per-token KL cost under a well-tuned `beta`, and why the value function's job of "learning to predict this terminal score from partial context" is the real work being done by GAE here, not the KL bookkeeping.

### 6.4 Common Interview Traps on This Topic

- **Saying "PPO trains the reward model."** The RM is trained once, in Stage 2, as an ordinary supervised model, and is frozen for the entirety of Stage 3. Conflating the two stages, or describing the RM as continually updated during PPO, is a common and telling error.
- **Forgetting that RM scores are only meaningful as differences.** A candidate who treats a raw RM score as an absolute, cross-prompt-comparable quality measure (Section 2.2) is missing a structural property of Bradley-Terry-style training, not a minor implementation detail.
- **Describing the KL penalty as "just a regularizer."** It is that, mechanically, but the *reason* it's there is specific and load-bearing: it is the direct countermeasure to Goodhart's-Law-style exploitation of a known-imperfect learned proxy, not a generic overfitting concern. An answer that doesn't connect the KL term to reward-model overoptimization is incomplete.
- **Assuming a bigger reward model is always better.** Section 6 and `../GPT/004_InstructGPT_And_RLHF.md` both give a concrete, citable counterexample.
- **Not knowing why rollout generation, not the gradient step, is usually the PPO wall-clock bottleneck.** This is a systems-level fact (Section 5) that a staff candidate should be able to explain from the sequential nature of autoregressive decoding, not just recite.

### 6.5 Outcome Reward Models Versus Process Reward Models

Everything above describes an **outcome reward model (ORM)**: a single scalar judgment of a completed response, with no visibility into or credit assignment across intermediate reasoning steps. An alternative, increasingly relevant design (most prominent in reasoning-model training, covered in depth in File 005) is a **process reward model (PRM)**, which scores the correctness or quality of *each individual step* of a multi-step solution (e.g., each line of a mathematical derivation), providing a dense, step-level reward signal rather than a single sequence-level one. PRMs directly address the sparse-terminal-reward credit-assignment problem discussed in Section 3.1 by construction, at the cost of needing step-level human or automated annotation (which is more expensive to collect than a single end-to-end preference judgment) and needing a well-defined notion of "step" for the task domain, which is natural for structured domains like mathematical proofs or code but less obviously well-defined for open-ended prose. Vanilla RLHF as described in this file uses an ORM; process supervision is a design point worth knowing exists on the same reward-modeling spectrum, and its most successful production use to date has been in mathematical-reasoning-focused training pipelines rather than general-purpose assistant RLHF.

### 6.6 Online, Iterated, and Offline RLHF

A further design axis worth distinguishing precisely: **offline RLHF** trains the RM once on a fixed, pre-collected comparison dataset and then runs PPO to convergence against that fixed RM (the setup described throughout this file). **Iterated RLHF** periodically collects fresh human comparisons on the *current* policy's outputs, retrains or updates the RM on the enlarged/refreshed comparison set, and resumes RL against the updated RM -- directly mitigating reward-model overoptimization (Section 4.5) by ensuring the RM's training distribution is periodically refreshed to include the policy's actual current output distribution, rather than staying permanently anchored to the SFT-era distribution the RM was originally trained on. **Fully online RLHF** would have a human (or AI, see File 004) rate every rollout in real time with no separate RM at all -- rarely used in practice for cost/latency reasons at scale, though it is the conceptual limit that iterated RLHF approximates with decreasing staleness. Most production RLHF pipelines at frontier labs are understood to run some form of iteration (periodic RM refresh against updated policy samples) rather than pure single-shot offline RLHF, precisely because the overoptimization problem in Section 4.5 otherwise worsens the longer a fixed RM is trained against.

### 7. Quick-Reference Summary

- Three stages: SFT (initialization/reference) -> RM training (Bradley-Terry MLE on pairwise comparisons) -> PPO RL fine-tuning against the RM with a KL-to-reference penalty.
- RM loss: `-E[log sigmoid(r_theta(x,y_w) - r_theta(x,y_l))]`, numerically implemented via `logsigmoid` for stability; scores are only meaningful up to a per-prompt additive constant.
- PPO's clipped surrogate objective approximates a trust-region update cheaply, using the probability ratio between current and rollout-time policy and a clip range `eps`.
- Advantage estimation via a value function and GAE turns a sparse, terminal reward signal into a denser, lower-variance training signal.
- The KL penalty to the SFT reference is not a regularization nicety -- it is the direct, mechanical countermeasure to Goodhart's Law applied to a learned, imperfect reward proxy, and its coefficient `beta` is the primary lever trading off reward-hacking risk against benefit forfeited from RL.
- Running PPO requires 3-4 concurrent models (policy generation, frozen reference, frozen reward model, value function), an actor-learner-shaped systems problem dominated in wall-clock terms by sequential rollout generation, not by the backward pass.
- Reward-overoptimization is empirically curve-shaped, not a binary risk: true-vs-proxy reward gap grows roughly monotonically as KL-from-reference grows, giving a quantitative basis for early stopping or RM refresh policies.
- Length bias and other spurious correlations are learned by RMs the same way any supervised model learns spurious correlations present in its training distribution -- this is a data problem as much as an algorithm problem.
- PPO's clip-based trust region is a cheap, first-order stand-in for TRPO's more rigorous but far more expensive exact trust-region update; the choice trades a small amount of theoretical rigor for a large amount of practical simplicity and stability.
- Outcome reward models (whole-response scalar judgment) and process reward models (step-level judgment) sit on the same reward-modeling spectrum; PRMs directly address sparse-reward credit assignment at the cost of needing step-level annotation.
- Most production RLHF is iterated rather than single-shot: the RM is periodically refreshed against the current policy's own output distribution specifically to keep the overoptimization gap from growing unchecked.

### 7.1 The Full Loop, End to End, in Pseudocode

Pulling every mechanism in this file into one place -- the four models, the reward construction, the advantage estimation, and the clipped update -- a single PPO training iteration looks like this:

```python
for iteration in range(num_iterations):
    prompts = sample_prompts(batch_size)

    # --- rollout phase (inference-shaped; the wall-clock bottleneck, Section 5) ---
    with torch.no_grad():
        responses, logp_old = policy.generate_with_logprobs(prompts)     # pi_phi_old
        logp_ref = reference_model.logprobs(prompts, responses)          # frozen pi^SFT
        rm_scores = reward_model(prompts, responses)                     # frozen r_theta, terminal only
        values = value_head(policy.hidden_states(prompts, responses))    # V_psi, per-token

    # --- reward construction (Section 4.1) ---
    kl_per_token = logp_old - logp_ref                       # sampled from pi_phi_old, so this estimates
    rewards = -beta * kl_per_token                            # KL(pi_phi_old || pi^SFT) at each position
    rewards[:, -1] += rm_scores                               # terminal RM score added at the last token

    # --- advantage estimation (Section 3.3-3.4) ---
    advantages, returns = compute_gae(rewards, values, gamma, lam)

    # --- policy update phase (training-shaped; a handful of epochs over this rollout batch) ---
    for ppo_epoch in range(ppo_epochs):
        logp_new = policy.logprobs(prompts, responses)        # pi_phi, re-evaluated each ppo_epoch
        policy_loss = ppo_clip_loss(logp_new, logp_old, advantages, eps)
        value_loss = F.mse_loss(value_head(...), returns)
        loss = policy_loss + vf_coef * value_loss
        loss.backward()
        optimizer.step()
```

The comment structure above is deliberately organized to mirror this file's own section structure: rollout generation (Section 3.1/Section 5), reward construction (Section 4.1), advantage estimation (Section 3.3), and the clipped policy update (Section 3.2) are four genuinely separate concerns that a correct implementation must get right independently, and a bug in any one of them (e.g., applying the KL penalty at the wrong token position, or bootstrapping the GAE recursion incorrectly at sequence boundaries) tends to produce a model that trains without crashing but converges to a subtly or badly wrong policy -- silent correctness bugs are the norm in RL implementations, not the exception, which is part of why extensive unit testing of each of these four pieces in isolation (verifying GAE against a hand-computed toy trajectory, verifying the clip loss reduces to vanilla policy gradient when `eps -> infinity`, etc.) is standard practice on any real RLHF engineering team.

### 8. Where This Goes Next

The single biggest practical complaint about the pipeline in this file is that it requires training and maintaining a separate reward model and running a genuinely finicky, multi-model RL loop -- both of which are expensive, unstable relative to supervised training, and require substantial RL-specific engineering expertise that many teams outside a handful of frontier labs historically lacked. File 003 covers Direct Preference Optimization (DPO) and its variants, which derive a way to skip the reward-model-training-plus-PPO machinery entirely and optimize a policy directly against a preference dataset with a single supervised-style loss -- motivated exactly by the practical and stability costs documented in Sections 3-6 of this file. File 004 covers RLAIF, which keeps this file's RM-plus-RL structure intact but replaces the *source* of the preference labels feeding RM training with AI-generated judgments rather than (only) human ones. File 005 covers RLVR, which replaces the learned RM entirely with a ground-truth-checkable verifier for domains where one exists (math, code), removing the RM-overoptimization failure mode in Section 4 by removing the learned proxy altogether. Each of these should be understood as addressing a specific, nameable limitation of the pipeline derived in this file, not as an unrelated alternative technique.

Holding all of Files 002-005 in your head simultaneously, the throughline is this: every subsequent technique keeps the same underlying goal (optimize the policy toward what's actually preferred or actually correct, beyond what SFT's imitation objective can express) while trading away a specific piece of this file's machinery -- the separate RM, the RL loop, or the "reward is only an approximation" assumption itself -- for a cheaper, more stable, or more accurate alternative, at some other cost that is worth being able to name for each variant rather than treating "DPO/RLAIF/RLVR is strictly better" as a settled fact.

That cost/benefit ledger, made explicit and comparable across techniques, is the single most valuable thing to be able to produce on a whiteboard in a staff interview on this topic -- more valuable than being able to recite any one method's derivation in isolation.
