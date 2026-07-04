# Alignment: RLHF, PPO, DPO, and Related Methods

## The Gap Instruction Tuning Doesn't Close

Pretraining teaches a model to model the statistics of internet-scale text; instruction tuning (supervised fine-tuning, or SFT, covered in the previous chapter) teaches it to map instructions to plausible, on-task responses by imitating curated demonstrations. Neither of these steps, on their own, reliably produces a model that behaves the way you actually want a deployed assistant to behave. A model can be an excellent next-token predictor and a competent instruction-follower and still be unhelpful in subtle ways, confidently wrong, needlessly verbose, evasive when it should answer, compliant when it should refuse, or refusing when it should comply — because none of the training signal so far has ever explicitly told it "this response is better than that one" in the dimensions humans actually care about: helpfulness, honesty, and harmlessness.

The fundamental obstacle is that there is no simple, differentiable loss function for "good response" the way there is for "predict the correct next token." Cross-entropy loss works beautifully when there's a single correct target to match, which is exactly the situation in pretraining and SFT — you have a ground-truth token or a ground-truth demonstration, and you minimize the distance between the model's distribution and that target. But "goodness" of an open-ended response is not a single correct string; it's a comparative, often subjective judgment that varies with context, and it's dramatically easier for a human (or another model) to *compare* two candidate responses and say which is better than it is for anyone to *write down* a loss function that scores absolute quality directly. This asymmetry — comparison is easy, direct scoring is hard — is the entire reason preference-based training methods exist, and it motivates every technique in this chapter. The goal of alignment techniques is to take that comparative human judgment and turn it into a training signal a model can actually be optimized against.

## The Classic Three-Stage RLHF Pipeline

Reinforcement Learning from Human Feedback, as popularized by OpenAI's InstructGPT work and used in some form to train ChatGPT, Claude, and most other production assistants in some part of their training history, is a three-stage pipeline. It's worth walking through all three stages in real depth, because interview questions about RLHF frequently probe whether you understand *why* each stage exists and *why* the pipeline is structured this way rather than some simpler alternative.

### Stage 1 — Supervised Fine-Tuning (Recap)

The pipeline starts from a model that has already been instruction-tuned via SFT, as covered in the previous chapter: human-written or curated demonstration data teaches the base pretrained model to follow instructions and produce assistant-shaped responses at all. This SFT model serves two roles going forward — it is the starting policy that gets further refined by reinforcement learning in stage 3, and it is also kept around unchanged as a fixed reference policy that the RL stage is penalized for drifting too far away from. Both roles matter, and conflating them is a common source of confusion: the "reference model" and the "policy being trained" start out as literally the same weights, but only one of them keeps updating.

### Stage 2 — Reward Modeling

The SFT model can already generate multiple different candidate responses to the same prompt (by sampling with some temperature). The next step is to have humans express a *preference* between pairs of these candidates rather than write a new gold-standard answer from scratch — a labeler is shown a prompt and two (or more) model-generated responses and simply picks which one is better, or ranks a small set of them. This is a much cheaper, faster, and more reliable labeling task than asking annotators to compose an ideal response for every prompt, and it captures something demonstration data cannot: comparative judgment about response quality along the dimensions annotators are instructed to weigh (helpfulness, factual accuracy, safety, tone, and so on).

These pairwise comparisons are used to train a separate model — the reward model (RM) — to predict human preference. The RM is typically initialized from the same pretrained/SFT backbone (often a smaller one, for efficiency) with its unembedding layer replaced by a single scalar output head, so that given a prompt and a candidate response, it outputs one number representing how "good" that response is estimated to be. The training objective comes from the Bradley-Terry model of pairwise comparison, a classical statistical model for ranking data from pairwise outcomes (originally developed for things like ranking chess players from match results). Under the Bradley-Terry model, if response `y_w` ("winner") was preferred by the human over response `y_l` ("loser") for the same prompt `x`, the probability of that preference is modeled as a sigmoid of the difference in the two responses' underlying reward scores:

```
P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))
```

The reward model is trained to maximize the likelihood of the observed human preferences under this model, which in practice means minimizing the negative log-likelihood loss:

```python
import torch
import torch.nn.functional as F

def reward_model_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise preference loss for reward model training.
    reward_chosen and reward_rejected are scalar reward-model outputs (one per
    example in the batch) for the human-preferred and human-rejected response
    to the same prompt, respectively."""
    # -log(sigmoid(r_chosen - r_rejected)), averaged over the batch
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()

# Illustrative batch: reward model's current scalar scores for chosen vs rejected responses
reward_chosen = torch.tensor([2.3, 1.1, 0.4])
reward_rejected = torch.tensor([0.8, 1.4, -0.2])
loss = reward_model_loss(reward_chosen, reward_rejected)
print(loss.item())  # a real number; note example 2 has chosen < rejected, contributing high loss
```

Notice what this buys you: once trained, the reward model can score *any* response to *any* prompt with a single forward pass, without needing a human in the loop for that specific example. This is the entire point of introducing a learned reward model rather than trying to use human judgment directly inside the RL loop — reinforcement learning requires evaluating the reward for potentially millions of sampled rollouts during training, and having a human rate every single one of those rollouts in real time is completely infeasible at that scale, both in cost and in latency. Training one reward model on a comparatively modest number of human comparisons (tens of thousands, not millions) and then using it as a fast, differentiable-in-effect, always-available proxy for human judgment is what makes the RL stage computationally tractable at all.

### Stage 3 — Reinforcement Learning with PPO

With a trained reward model in hand, the SFT model becomes the starting point for an RL fine-tuning stage. Conceptually, the objective is simple to state: adjust the policy (the language model, now framed as an RL agent choosing which token to emit) to maximize the reward model's score on its generations. In practice, optimizing purely for reward-model score is dangerous, because the reward model is an imperfect proxy for actual human preference, and a sufficiently powerful optimizer will find and exploit its blind spots — this failure mode is called reward hacking, and it shows up as the policy learning to produce responses that game the RM's scoring function (padding responses with reward-model-pleasing phrases, degenerately repeating certain flattering patterns, or drifting into strange, overconfident, or repetitive text) rather than genuinely improving quality as a human would judge it.

The standard fix is to add a KL-divergence penalty between the policy currently being trained and the fixed reference policy (the frozen SFT model from stage 1), subtracted from the reward:

```
objective(x, y) = r(x, y) - beta * KL(pi_theta(y|x) || pi_ref(y|x))
```

This term penalizes the policy for straying too far, in a probabilistic sense, from the distribution of outputs the SFT model would have produced. It acts as a regularizer against both reward hacking and mode collapse (the policy converging onto a narrow set of "safe," high-reward-scoring outputs at the expense of the diversity and general competence the SFT model had). The `beta` coefficient controls the strength of this tether — too small and the policy drifts and degenerates chasing reward; too large and the RL stage barely moves the model away from its SFT starting point, defeating the purpose of doing RL at all. Tuning beta is one of the genuinely fiddly, empirically-driven parts of running RLHF in practice.

The specific RL algorithm almost universally used to perform this optimization is Proximal Policy Optimization (PPO), an actor-critic policy-gradient method. In this setting, the "actor" is the language model policy itself, choosing actions (tokens) to maximize expected reward, and the "critic" is a separate learned value function that estimates the expected future reward from a given partial sequence, used to compute an advantage estimate (how much better a given action was than the value function's baseline expectation) that reduces the variance of the policy gradient. PPO's specific contribution, relative to vanilla policy gradient methods, is a clipped surrogate objective that constrains how much the policy is allowed to change in a single update step, which is exactly what makes it "proximal" — it explicitly keeps new-policy behavior proximate to old-policy behavior at each optimization step:

```
r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)

L_CLIP(theta) = E[ min( r_t(theta) * A_t,  clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]
```

Here `r_t(theta)` is the probability ratio between the updated policy and the policy that generated the data (not to be confused with the reward `r(x,y)` from earlier — unfortunately both are conventionally called `r` in the literature), and `A_t` is the advantage estimate from the critic. The clipping term caps how much a single update can shift the probability ratio away from 1 in the direction the advantage is pushing it, which prevents any single batch of experience from producing a destructively large policy update that could collapse the model's behavior. This matters enormously in the LLM setting because a bad update to a multi-billion-parameter language model isn't just a temporary setback the way it might be in a small robotics RL task — it can produce a policy that generates broken, incoherent, or degenerate text, and recovering from a large enough bad update can be very costly or effectively impossible without restarting from an earlier checkpoint.

```python
import torch

def ppo_clipped_objective(log_probs_new, log_probs_old, advantages, eps=0.2):
    """Simplified PPO clipped surrogate objective for a batch of tokens.
    log_probs_new / log_probs_old are log pi(a|s) under the current and
    old policy respectively; advantages come from the critic (value model)."""
    ratio = torch.exp(log_probs_new - log_probs_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    # take the pessimistic (minimum) of the two -- this is what makes it conservative
    surrogate_loss = -torch.min(unclipped, clipped).mean()
    return surrogate_loss

log_probs_new = torch.tensor([-0.5, -1.2, -0.3])
log_probs_old = torch.tensor([-0.6, -1.0, -0.4])
advantages = torch.tensor([1.5, -0.8, 0.3])
print(ppo_clipped_objective(log_probs_new, log_probs_old, advantages).item())
```

Putting the whole stage 3 loop together, the practical reality is that RLHF-with-PPO requires four separate models resident in memory simultaneously during training: the policy model being actively updated, the frozen reference policy used only to compute the KL penalty, the reward model used only to score generated responses, and the critic/value model used to compute advantage estimates for PPO. All four are typically full-size language models (the critic and reward model are sometimes made smaller than the policy, but not always), which means RLHF's compute and memory footprint during this stage is substantially larger than SFT alone — you are effectively running inference on three extra full models plus training a fourth, orchestrating rollout generation, reward scoring, advantage estimation, and policy updates in a loop. Beyond the raw resource cost, RL training is also notoriously unstable and hyperparameter-sensitive compared to supervised learning: reward hacking, KL coefficient tuning, value function miscalibration, and high variance in policy gradient estimates all make PPO-based RLHF a genuinely difficult system to get right, requiring careful engineering and extensive tuning even for teams with substantial ML infrastructure experience. This complexity is exactly the gap that Direct Preference Optimization was designed to close.

## Direct Preference Optimization (DPO)

DPO, introduced by Rafailov et al. in 2023, starts from a sharp mathematical observation about the RLHF objective itself: the KL-constrained reward maximization problem that PPO is trying to solve numerically actually has a closed-form analytical solution for the optimal policy, expressed directly in terms of the reward function. Specifically, for the objective `maximize E[r(x,y)] - beta * KL(pi_theta || pi_ref)`, the optimal policy pi* satisfies:

```
pi*(y|x) = (1 / Z(x)) * pi_ref(y|x) * exp(r(x,y) / beta)
```

where `Z(x)` is a normalizing constant (a partition function) that depends only on the prompt `x`, not on the specific response `y`. This equation can be algebraically rearranged to express the reward as a function of the optimal policy instead of the other way around:

```
r(x, y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x)
```

This is the key move: it says that any reward function has an implicit, equivalent representation purely in terms of a policy's log-probability ratio against the reference policy. If you substitute this expression for the reward back into the Bradley-Terry preference model from stage 2 (recall: `P(y_w > y_l) = sigmoid(r(x,y_w) - r(x,y_l))`), something convenient happens — the `Z(x)` normalizing term is identical for both `y_w` and `y_l` (it only depends on the shared prompt `x`, not on which response you're looking at), so when you compute the difference `r(x,y_w) - r(x,y_l)`, the `log Z(x)` terms cancel out exactly. That cancellation is what makes the whole approach practical: it removes the one term in the reward expression that would otherwise be intractable to compute.

What remains, after substitution and cancellation, is a loss function defined entirely in terms of the policy's own log-probabilities on chosen and rejected responses, compared against the same quantities under the fixed reference policy — no separate reward model, and no RL rollout loop, required anywhere in the process:

```
L_DPO(theta) = -E[ log sigmoid( beta * ( log pi_theta(y_w|x) - log pi_ref(y_w|x)
                                          - log pi_theta(y_l|x) + log pi_ref(y_l|x) ) ) ]
```

This is, structurally, just a binary classification loss (a logistic loss) computed directly on log-probability differences the policy already produces during an ordinary forward pass — no sampling, no reward model forward passes, no advantage estimation, no critic, no clipped surrogate objective. You take the same human preference pairs collected in RLHF's stage 2, and instead of using them to train a reward model that then feeds an RL loop, you use them to directly fine-tune the policy with a loss that looks almost exactly like supervised fine-tuning in terms of engineering complexity.

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
             ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """Direct Preference Optimization loss.
    Each *_logps tensor holds the summed log-probability the given model
    assigns to the full response (chosen or rejected) conditioned on the prompt.
    policy_* comes from the model being trained; ref_* comes from the frozen
    reference (typically the SFT checkpoint), computed once and cached or
    computed with the reference model in eval mode + no_grad."""
    policy_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps

    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()
    return loss

# Illustrative: log-probs the policy and reference model assign to chosen/rejected responses
policy_chosen_logps = torch.tensor([-12.4, -20.1, -8.7])
policy_rejected_logps = torch.tensor([-15.2, -18.9, -9.0])
ref_chosen_logps = torch.tensor([-13.0, -19.8, -9.1])
ref_rejected_logps = torch.tensor([-13.5, -19.0, -9.3])

loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps)
print(loss.item())
```

It's worth reading the loss intuitively: it pushes the policy to increase `log pi_theta(y_w|x)` relative to the reference model's assessment of `y_w`, and to decrease `log pi_theta(y_l|x)` relative to the reference model's assessment of `y_l`, with the sensitivity of that push controlled by beta. The reference-model terms act as a built-in normalizer that prevents the loss from being satisfied just by generically increasing all probabilities uniformly, or by drifting the policy arbitrarily far from the reference distribution — the same anti-reward-hacking, anti-mode-collapse role that the explicit KL penalty term played in PPO, except here it falls out algebraically from the derivation rather than being bolted on as a separate loss term.

### Why DPO Is Compelling in Production, and Where It Falls Short

DPO's practical appeal follows directly from what it eliminates. It requires only two models in memory — the policy being trained and the frozen reference policy — compared to PPO's four, cutting compute and engineering complexity substantially. It has no RL rollout loop: no on-policy sampling from the model during training, no reward-model scoring pass, no advantage estimation, no PPO-specific instability from noisy policy-gradient variance. Training reduces to a supervised-learning-shaped loop over a fixed dataset of preference pairs, which is dramatically more stable, easier to debug, easier to reproduce, and easier to fit into existing supervised training infrastructure than an RL pipeline. These are not minor conveniences — they're the difference between alignment training being accessible to a broad range of engineering teams versus requiring specialized RL expertise, and DPO's rapid, widespread adoption after publication (showing up in Llama's later post-training recipes, Mistral, Zephyr, and many other open post-training pipelines) is a direct consequence of this simplicity.

The limitations are real, though, and worth stating precisely rather than glossing over. Because DPO trains entirely offline on a fixed, pre-collected preference dataset, its quality is more sensitive to the coverage and distribution of that dataset than PPO's, which continually samples fresh, on-policy generations from the current policy during training and scores them with the reward model. If the preference dataset doesn't well represent the kinds of outputs the policy will actually produce after several steps of optimization, DPO has no built-in mechanism to correct for that distribution shift the way an on-policy RL loop naturally does — it's simply optimizing the fixed offline objective, and the reference model normalizer only controls divergence, not whether the training distribution matches the deployment distribution. There is also credible published and practitioner evidence that a well-tuned PPO-based RLHF pipeline can outperform DPO on some tasks and some quality dimensions, particularly on harder reasoning and instruction-following benchmarks, at the cost of the much greater engineering and tuning effort PPO requires — so DPO's simplicity is a genuine trade against a ceiling on final quality that a fully-invested RL pipeline can sometimes exceed. In practice, many production alignment recipes today use DPO (or its variants, such as IPO or KTO, which adjust the loss to address some of DPO's known failure modes around overfitting to preference strength) as the default, reserving full PPO-based RLHF for cases where the additional quality is worth the substantially higher cost and complexity.

## RLAIF: Reinforcement Learning from AI Feedback

Both the reward-modeling stage of RLHF and the preference-pair collection that DPO consumes depend on human-labeled preference comparisons, and human labeling is slow, expensive, and hard to scale — especially for the very large volumes of preference data that produce the strongest reward models or the most robust DPO training sets. RLAIF (Reinforcement Learning from AI Feedback) replaces the human labeler in this step with another LLM, prompted to act as a judge: given a prompt and two candidate responses, the judge model is asked which one better satisfies some stated criteria (helpfulness, harmlessness, accuracy, adherence to a particular style guide), and its judgment is used exactly where a human preference label would have been used — either to train a reward model, or to construct a preference dataset for DPO-style training directly.

The motivation is straightforward scalability: an LLM judge can generate preference labels far faster and at far lower marginal cost than human annotators, enabling preference datasets orders of magnitude larger than would be practical with human-only labeling. The obvious concern is quality and bias — an AI judge inherits whatever blind spots, stylistic preferences, or systematic errors its own training produced, so RLAIF-derived preference data can encode and even amplify those biases rather than correcting them. In practice, many production alignment pipelines use a hybrid: a smaller, carefully curated set of human preference labels for the highest-stakes categories (safety-critical refusals, for instance), supplemented or scaled up with AI-generated preference labels for larger-volume, lower-stakes categories, sometimes with the AI judge itself calibrated against a human-labeled validation set to check that its judgments track human judgment well enough to trust at scale.

## Constitutional AI

Constitutional AI, developed at Anthropic and used in training Claude, tackles a closely related problem from a different angle: rather than (or in addition to) collecting human preference labels to teach a model what "harmless" or "helpful" looks like, it uses a written constitution — an explicit, human-authored set of principles and guidelines the model is instructed to apply to itself. The process has two phases that mirror the SFT-then-preference-training structure elsewhere in this chapter, but with AI-generated self-critique substituting for a chunk of the human-labeling burden.

In the first phase, the model is prompted to produce an initial response to a query, then prompted again — using a principle drawn from the constitution — to critique its own response for ways it violates that principle, and then to revise the response to address the critique. This critique-then-revise loop can be repeated across multiple constitutional principles, producing a final, self-improved response. These revised responses become supervised fine-tuning data: the model is trained to produce its already-self-corrected output directly, without needing the multi-step critique scaffold at inference time.

In the second phase, the model generates pairs of responses to a prompt and is asked to use the constitution to judge which of the pair better satisfies the stated principles — an AI-generated preference judgment, structurally identical to RLAIF's judge step, except explicitly grounded in a written, auditable set of principles rather than an unconstrained "which is better" prompt. This constitution-grounded preference data is then used to train a reward model or run DPO-style preference training, exactly as in the RLHF or DPO pipelines described above.

The strategic advantage Constitutional AI is aiming for is reducing dependence on having human annotators directly view, judge, and label large volumes of harmful or borderline content in order to teach a model to avoid producing it — a task that is not only expensive to scale but also has a real human cost for the annotators doing that labeling. By encoding the desired behavior as an explicit, inspectable, editable set of written principles that the model applies to itself, the approach also gains a degree of transparency and auditability that isn't as naturally present when "good behavior" is defined implicitly through thousands of individual human preference judgments whose underlying reasoning is never written down anywhere. It doesn't eliminate human judgment from the pipeline entirely — humans still author the constitution and typically still validate outcomes — but it shifts the point of human involvement from "label every example" to "design and refine the principles the model uses to label itself," which scales considerably better.
