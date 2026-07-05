## Speculative Decoding and Inference Acceleration

### 1. The problem: decode wastes the GPU's compute capacity

As established in files 001 and 003, decode generates exactly one token per forward pass per sequence, and that forward pass is memory-bandwidth-bound: the GPU spends most of its time moving weights and KV cache from HBM rather than computing, because the amount of arithmetic per token is tiny relative to the amount of data touched. Continuous batching amortizes this by processing many *sequences'* single tokens in one pass. Speculative decoding attacks the same underutilization from a different angle: instead of adding more sequences to fill the idle compute, it gets **one sequence** to produce more than one verified token per expensive forward pass through the large model.

The enabling fact is that a transformer forward pass over `k` tokens (all their positions computed in parallel, as in prefill) costs only marginally more wall-clock time than a forward pass over 1 token, as long as you're still memory-bandwidth-bound rather than compute-bound — because the dominant cost (loading weights from HBM) is paid once per layer regardless of how many token positions you compute against those loaded weights. This is the exact same amortization principle that makes prefill fast per-token relative to decode (file 005): parallel computation over many positions is "free-ish" up to the point where you become compute-bound; sequential computation over many *steps* is not, because each step re-pays the full memory-bandwidth cost with almost no arithmetic to amortize it against.

Speculative decoding's insight: if you could somehow know, or cheaply guess, several future tokens *before* running the expensive model, you could verify a whole run of guesses in a single parallel forward pass through the large model — converting what would have been `k` sequential expensive-model calls into 1 expensive-model call (verifying `k` guesses in parallel) plus `k` cheap calls (producing the guesses in the first place).

### 2. Mechanics: draft, then verify

Two models are involved:

- The **target model** `M_p` — the large model whose output distribution you actually want to sample from, with per-step distribution `p(x_t | x_{<t})`.
- The **draft model** `M_q` — a much smaller, much faster model (could be a distilled/smaller version of the same model family, a smaller model from the same lab, or in some designs a lightweight auxiliary head attached to the target model itself — see Section 5) that approximates the target's distribution, `q(x_t | x_{<t}) ≈ p(x_t | x_{<t})`, and can run many steps for the cost of one target-model step.

One round of speculative decoding, given the already-accepted context `x_{<t}`:

1. **Draft phase.** Run the small draft model autoregressively for `k` steps, sampling (or greedily picking) a candidate continuation `x_t, x_{t+1}, ..., x_{t+k-1}`, recording the draft model's own probability `q(x_i | x_{<i})` at each of these `k` positions. This is cheap: `k` sequential small-model steps.
2. **Verify phase.** Run the target model **once**, in parallel, over all `k` drafted positions simultaneously (feed the drafted tokens as if they were already-accepted context, exactly like a prefill over `k` tokens) — this yields the target model's own distribution `p(x_i | x_{<i})` at every one of the `k` positions, in a single forward pass, because the target model's per-position distributions only depend on the (now fully known, drafted) prefix up to that position, not on the target model's own not-yet-computed later outputs.
3. **Accept/reject, left to right.** Walk through the `k` drafted tokens in order. At each position `i`, decide whether to *accept* the draft token `x_i` or *reject* it (using the accept/reject rule in Section 3, which is where the whole scheme's correctness lives). Accept as many tokens as pass, in order, **stopping at the first rejection** — everything after a rejection is discarded regardless of whether it would have separately "passed," because a later drafted token was conditioned on the (now-discarded) rejected token and is therefore not a valid continuation of the accepted prefix.
4. **Resample at the rejection point.** At the first rejected position, don't just fall back to the target model's own greedy/sampled token — sample a **corrected** token using the residual distribution described in Section 3, specifically constructed so that this one token is distributed exactly as the target model's true conditional distribution at that position, *despite* the fact that a rejection just occurred. If nothing was rejected (all `k` drafted tokens were accepted), sample one bonus extra token from the target model's distribution at position `t+k` — you already have that distribution for free, since the verify pass's forward computation at position `t+k` conditions only on positions `<t+k`, all of which are now confirmed.
5. Advance `x_{<t}` to include everything accepted plus the one resampled/bonus token, and start the next round.

```python
import numpy as np

def speculative_round(target_model, draft_model, prefix, k, rng):
    """One round of speculative decoding. Returns the extended token sequence.
    target_model / draft_model expose .next_token_probs(prefix) -> prob vector over vocab.
    """
    draft_tokens = []
    draft_probs = []   # q(x_i | x_<i}) at each drafted position, under the DRAFT model
    cur = list(prefix)
    for _ in range(k):
        q = draft_model.next_token_probs(cur)
        x = rng.choice(len(q), p=q)
        draft_tokens.append(x)
        draft_probs.append(q)
        cur = cur + [x]

    # Single parallel forward pass through the target model over all k drafted
    # positions -- in a real implementation this is one batched call; here we model
    # it as target_probs[i] = p(. | prefix + draft_tokens[:i]).
    target_probs = []
    cur = list(prefix)
    for i in range(k):
        p = target_model.next_token_probs(cur)   # conditioned only on ALREADY-DRAFTED
        target_probs.append(p)                    # tokens, hence parallelizable over i
        cur = cur + [draft_tokens[i]]

    accepted = []
    for i in range(k):
        x_i = draft_tokens[i]
        p_i, q_i = target_probs[i], draft_probs[i]
        accept_prob = min(1.0, p_i[x_i] / q_i[x_i])
        if rng.random() <= accept_prob:
            accepted.append(x_i)
            continue
        # First rejection: resample from the residual distribution and stop.
        residual = np.clip(p_i - q_i, a_min=0, a_max=None)
        residual = residual / residual.sum()
        x_corrected = rng.choice(len(residual), p=residual)
        accepted.append(x_corrected)
        return prefix + accepted   # discard everything drafted after the rejection

    # All k drafted tokens accepted -- sample one bonus token from the target
    # model's distribution at position t+k, which the verify pass already computed.
    bonus_probs = target_model.next_token_probs(prefix + accepted)
    bonus = rng.choice(len(bonus_probs), p=bonus_probs)
    accepted.append(bonus)
    return prefix + accepted
```

### 3. Why this preserves the target model's exact output distribution

This is the subtle, load-bearing part of the algorithm, originally due to Leviathan et al. (2023, "Fast Inference from Transformers via Speculative Decoding") and, independently and concurrently, Chen et al. (2023, DeepMind). The claim to prove: **the marginal distribution of the token actually emitted at each position, after running this accept/reject procedure, is identical to `p`, the target model's own distribution — not merely close to it.** Speculative decoding is therefore not an approximation technique; it is an exact sampling technique that happens to be faster, and this distinction matters enormously in an interview context because it is the whole reason labs are comfortable shipping it in production without treating it as a quality trade-off.

**The acceptance rule.** For a drafted token `x` at a given position, with target probability `p(x)` and draft probability `q(x)`:

```
accept with probability  min(1, p(x) / q(x))
```

**The residual/resampling distribution**, used only when a token is rejected:

```
p_residual(x) = max(0, p(x) - q(x))  /  sum_x' max(0, p(x') - q(x'))
```

The denominator normalizes the residual (which is guaranteed non-negative pointwise after the `max(0, .)` clip, and sums to a positive quantity whenever `p != q`) into a valid probability distribution.

**Proof sketch that the emitted token's marginal distribution equals `p`.** Fix a position and let `x` be the drafted token, drawn from `q`. Two ways the algorithm can *emit* any particular value `v`:

- **Path A — draft `v`, and accept it.** Probability: `q(v) * min(1, p(v)/q(v))`. Note `q(v) * min(1, p(v)/q(v)) = min(q(v), p(v))` — this is just algebra (multiply through the `min`). Call this `min(p(v), q(v))`.
- **Path B — draft something else, reject it, and then resample `v` from the residual.** This requires: (i) draft some `x != v` (or even `x = v`, if `v` is itself rejected — but note if `x=v` is rejected, `p(v)/q(v) < 1` meaning `p(v) < q(v)`, so `v` cannot appear in the residual since the residual is `max(0, p(v)-q(v)) = 0` in that case; so effectively this path only contributes via the reject-then-resample step landing on `v` when `v` wasn't the (rejected) draft). The probability of *some* rejection happening at this step, summed over all possible drafted tokens, is `sum_x q(x) * (1 - min(1, p(x)/q(x))) = sum_x max(0, q(x) - p(x))`. Given that a rejection happened, the resampled token is drawn from `p_residual`, independent of which specific `x` was rejected (the residual distribution used is a fixed distribution over the whole vocabulary, not conditioned on which particular draft was rejected) — so the probability of resampling `v` specifically, via this path, is `[sum_x max(0, q(x) - p(x))] * p_residual(v) = [sum_x max(0, q(x) - p(x))] * max(0, p(v)-q(v)) / [sum_x' max(0, p(x')-q(x'))]`.

  A useful identity: `sum_x max(0, q(x) - p(x)) = sum_x max(0, p(x) - q(x))` — both equal the total probability mass by which `p` and `q` disagree, split into the part where `q` exceeds `p` and the part where `p` exceeds `q`, and since both `p` and `q` are normalized distributions summing to 1, these two "excess" masses must be numerically equal (the amount `q` has "too much" of somewhere must exactly equal the amount it has "too little" of elsewhere, relative to `p`, since both sum to 1). Call this shared quantity `Z` (it is exactly the total-variation-related mass, `Z = 1 - sum_x min(p(x), q(x))`).

  So Path B's contribution to emitting `v` is `Z * [max(0, p(v)-q(v)) / Z] = max(0, p(v) - q(v))`.

**Total probability of emitting `v`** = Path A + Path B = `min(p(v), q(v)) + max(0, p(v) - q(v))`.

Now case-split on whether `p(v) >= q(v)` or `p(v) < q(v)`:

- If `p(v) >= q(v)`: `min(p(v),q(v)) = q(v)` and `max(0, p(v)-q(v)) = p(v)-q(v)`. Sum: `q(v) + p(v) - q(v) = p(v)`. ✓.
- If `p(v) < q(v)`: `min(p(v),q(v)) = p(v)` and `max(0, p(v)-q(v)) = 0`. Sum: `p(v) + 0 = p(v)`. ✓.

Either way, the total probability of emitting `v` at this step is exactly `p(v)` — the target model's own distribution, regardless of what the draft model's distribution `q` was. **This is the key result**: the draft model can be arbitrarily bad (as long as `q(x) > 0` wherever `p(x) > 0`, so the residual is well-defined) and the *correctness* of the output distribution is completely unaffected — a bad draft model only hurts *speed* (more rejections, fewer tokens accepted per round), never *quality*. This decoupling of "does this scheme produce exactly the right distribution" from "how good is the draft model" is precisely why speculative decoding is deployed in production as a pure speedup with no quality asterisk, unlike, say, quantization (file 002) or a smaller model used directly (which do trade quality for speed).

### 4. What determines the acceptance rate, and the resulting speedup

The acceptance probability at a given position is, by construction, `min(1, p(x)/q(x))` — so, intuitively, **the more closely the draft model's distribution matches the target model's**, the higher the acceptance rate, because `p(x)/q(x)` stays close to 1 across the tokens the draft model actually proposes. Two practically important levers determine this:

- **Draft-model quality / alignment with the target.** A draft model trained on similar data with a similar tokenizer, or specifically distilled from the target model, will place probability mass on similar tokens in similar situations, giving high acceptance. An unrelated, generic small model will diverge more often on the specific stylistic and factual choices the target model tends to make, giving lower acceptance. This is why production speculative-decoding deployments generally use a draft model from the *same model family/lab* (or explicitly distilled from the target) rather than an arbitrary small off-the-shelf model.
- **Task predictability / output entropy.** Highly predictable spans of text — boilerplate code (`import numpy as np`, closing brackets, repeated variable names), formulaic phrases, structured output following a rigid schema — have low true entropy in `p` itself, meaning almost *any* reasonable model (draft or target) puts most of its probability mass on the same one or two tokens, so acceptance rates are high regardless of draft-model sophistication. High-entropy spans — creative writing, the specific numeral chosen in an arithmetic result, the first content word after a genuinely open-ended prompt — have the target distribution itself spread across many plausible tokens, and a smaller draft model is much more likely to guess a token that, while individually plausible, isn't the specific one the larger target model would have preferred, driving rejections up. This is why reported speedups for speculative decoding vary substantially by workload: code generation and structured/templated output regimes tend to see the largest gains, and open-ended creative or reasoning-heavy generation tends to see smaller gains — a fact worth stating qualitatively rather than attaching a specific multiplier, since the exact number depends heavily on the specific draft/target pair and workload and is easy to over-claim from a single benchmark.

**Speedup model.** Let `alpha` be the average per-token acceptance probability (assume roughly constant across positions, a simplifying approximation) and `k` the number of tokens drafted per round. The expected number of tokens accepted per round (including the guaranteed bonus/correction token) is:

```
E[tokens emitted per round] = (1 - alpha^(k+1)) / (1 - alpha)      for alpha < 1
```

(this is the expectation of a geometric-like process: you get 1 token for free if the first is accepted and so on, terminating at the first rejection, plus one final resampled/bonus token). The cost per round is `k` cheap draft-model steps plus 1 expensive target-model step (verifying `k` positions in parallel, at roughly the same wall-clock cost as verifying 1 position, per Section 1's memory-bandwidth argument). If a draft step costs a fraction `c` of a target step (`c << 1` for a much smaller draft model), the wall-clock cost per round is roughly `k*c + 1` target-model-equivalents, and the effective speedup relative to plain autoregressive decoding (which emits exactly 1 token per target-model-equivalent) is:

```
speedup ≈ E[tokens emitted per round] / (k*c + 1)
```

Both the numerator (higher with better `alpha`) and the denominator (lower with a cheaper draft model, smaller `c`) matter, and there's a real tuning trade-off in choosing `k`: too small and you leave acceptance-rate upside on the table; too large and most rounds terminate on an early rejection anyway (since rejection probability compounds as `1 - alpha^k`), wasting draft-model compute on positions that will just get discarded. Production systems typically tune `k` empirically per draft/target pair and per workload, commonly landing somewhere in the range of a handful of tokens per round rather than trying to draft very long runs.

### 5. Variants worth knowing

- **Medusa / lookahead-style approaches** attach multiple lightweight prediction *heads* directly onto the target model itself (rather than using a wholly separate draft model), each head predicting a token some fixed number of steps ahead using the target model's own hidden states from the current step. This avoids needing to maintain, serve, and align a separate draft model at all, at the cost of the extra heads themselves needing to be trained (typically via a comparatively cheap fine-tuning pass on top of an already-trained target model) and generally producing less accurate multi-step guesses than a genuinely separate, sequentially-run small model would, since each head predicts independently rather than conditioning on the previously drafted tokens the way an autoregressive draft model naturally does.
- **Self-speculative decoding** uses a subset of the target model's own layers (e.g., an early-exit shortcut through the same weights) as the "draft," avoiding a second model's weights entirely, trading some additional engineering complexity (needing an early-exit-capable forward pass) for zero additional model-serving/memory overhead.
- **Tree-based / multi-candidate speculation** (e.g., SpecInfer and related work) drafts a *tree* of candidate continuations rather than a single linear sequence, verifying multiple branches in one target-model pass and accepting whichever branch matches longest — this increases the chance that *some* branch survives longer under the target model's true distribution, at the cost of a larger, more complex single verification pass.
- **N-gram / retrieval-based drafting**, a lighter-weight alternative to a learned draft model entirely: for tasks with heavy repetition against a known context (e.g., editing a document where much of the output echoes input text nearly verbatim), a simple n-gram lookup against the prompt itself can generate draft tokens with no model inference at all, trading generality (this only works well when the generation genuinely echoes prior context) for near-zero draft cost (`c` in Section 4's formula approaches zero, since there's no draft-model forward pass at all).

### 6. A worked numeric example of the speedup formula

Section 4 gave the speedup formula abstractly; walking through actual numbers makes the acceptance-rate sensitivity concrete rather than asserted.

```python
def expected_tokens_per_round(alpha: float, k: int) -> float:
    if alpha >= 1.0:
        return k + 1
    return (1 - alpha ** (k + 1)) / (1 - alpha)

def expected_speedup(alpha: float, k: int, cost_ratio: float) -> float:
    return expected_tokens_per_round(alpha, k) / (k * cost_ratio + 1)

if __name__ == "__main__":
    k, cost_ratio = 5, 0.08   # draft ~8% the cost of one target-model step
    for alpha in (0.3, 0.5, 0.7, 0.85, 0.95):
        tokens = expected_tokens_per_round(alpha, k)
        speedup = expected_speedup(alpha, k, cost_ratio)
        print(f"alpha={alpha:.2f} -> E[tokens/round]={tokens:.2f}, speedup={speedup:.2f}x")
```

Running this for `k=5` and a draft model costing roughly 8% of a target step: at `alpha=0.3` (a poorly-aligned draft, or highly unpredictable generation) expected tokens per round is barely above 1.4, and speedup is modest — you're paying for 5 draft steps' worth of `cost_ratio` overhead and mostly not benefiting from them. At `alpha=0.85` (a well-aligned draft on reasonably predictable text) expected tokens per round approaches 4, and the speedup climbs sharply, because the geometric term `(1-alpha^(k+1))/(1-alpha)` is highly convex in `alpha` near 1 — small improvements in acceptance rate near the high end buy disproportionately large speedup gains, which is exactly why draft-model *alignment* with the target (Section 4, Section 8) matters more to real-world speedup than almost any other single lever, including `k` itself.

### 7. Other inference-acceleration techniques: prompt/prefix caching

Distinct from speculative decoding (which speeds up *generation of new tokens*) is **prompt caching** (also called prefix caching), which speeds up (or entirely skips) the *prefill* of tokens the server has already processed before, for a different request.

The mechanism, mechanically underpinned by exactly the PagedAttention block-sharing/copy-on-write machinery covered in file 003 Section 5: if a system prompt, a shared instruction template, or a long shared document is common across many requests, the KV cache for that shared prefix needs to be computed by the (expensive, compute-bound) prefill pass only **once** — subsequent requests sharing the same prefix can reuse the already-computed KV-cache blocks directly (via shared block-table entries) rather than re-running prefill over that shared span at all. Commercial LLM APIs that advertise a "prompt caching" discount (a substantially reduced per-token price, and reduced time-to-first-token, for the cached portion of a prompt on a cache hit) are exposing exactly this server-side mechanism as a product feature, typically with a time-based eviction policy (a cache entry is retained for some window — commonly on the order of minutes to a small number of hours — after last use, then evicted to free the memory for other traffic) and requiring the shared prefix to match exactly (a single differing token anywhere in the prefix invalidates the cache hit for everything at or after that point, since attention is causal and every later position's computation genuinely depends on everything before it).

This composes directly with everything else in this module: it reduces prefill compute (helping TTFT and overall GPU utilization, file 005), and it reduces the *effective* memory cost of serving many requests that share structure (helping the batch-size economics of file 001 and 003), for exactly the multi-tenant, shared-system-prompt-heavy traffic pattern that dominates real production LLM products (a chat product, an agentic tool-use product, or an API serving many customers against the same fixed instruction template all produce exactly this kind of shared-prefix traffic).

### 7b. Prompt caching as an exposed API primitive, and the granularity question

Section 7 described prompt caching as a server-side mechanism; from the caller's side, commercial APIs generally expose it either implicitly (the server automatically detects and reuses a matching prefix from recent traffic, with no explicit action required from the caller) or explicitly, via a marker the caller inserts into the request indicating "everything up to this point is a stable, reusable prefix worth caching" — a design that shifts the responsibility for identifying cacheable structure from the server (which would otherwise have to guess) to the caller (who genuinely knows which parts of a given request are stable across many calls, e.g. a fixed system prompt or a fixed long reference document, versus which parts are unique per call, e.g. the user's specific question). This caller-side marking is a meaningfully different design point from fully automatic detection: it trades a small amount of integration effort (the caller has to annotate its own request structure) for a stronger guarantee that the *intended* cache boundary is the one actually used, rather than relying on the server to infer where a "meaningful" prefix boundary sits in an arbitrary request.

The granularity of caching matters operationally too: a cache hit is only as valuable as how much of the *expensive* portion of the request it actually covers. Caching a short, cheap prefix while the bulk of the prompt's tokens (and therefore the bulk of its prefill cost) sit *after* the cache boundary captures little of the available benefit; the highest-value caching opportunities are exactly the ones this file has emphasized throughout — a long, fixed system prompt, a large shared reference document, or a long few-shot template — where the cached span is the dominant share of the total prompt's token count. A caller integrating against a prompt-caching API is, in effect, making the same "where is the actual bottleneck" judgment call file 001's crossover analysis makes for KV-cache sizing, just applied to picking a cache boundary rather than to picking an attention architecture.

### 7c. Why the target model's verify pass can't just be "run decode k times faster"

It's worth explicitly ruling out a tempting but wrong simplification: one might ask why the target model can't simply run its own decode loop `k` times per round instead of relying on a separate draft model at all, skipping the whole draft/verify apparatus. The answer is the same sequential-dependency fact from Section 1: the target model's *own* token at position `t+1` is not known until its distribution at position `t` has actually been sampled from, so the target model cannot parallelize its own multi-step generation across positions it hasn't yet decided — there is nothing to feed into position `t+1`'s computation until position `t`'s sample is drawn. The draft model's role is specifically to supply a *plausible guess* at those not-yet-decided future positions cheaply, so the target model's forward pass can be restructured from "one sequential decision, wait, then the next" into "verify several already-guessed positions in parallel" — a fundamentally different computational shape than the target model running its own decode loop faster ever could be, since the sequential dependency the draft model works around is intrinsic to autoregressive generation itself, not merely a speed limitation of the target model specifically.

### 8. Speculative decoding inside a continuous-batching server

Files 003 and 004 are easy to imagine as separate concerns, but a real serving stack has to make them coexist, and the interaction is worth spelling out. Continuous batching's iteration loop (file 003 Section 3) assumes every active request advances by exactly one token per iteration; speculative decoding wants a request to advance by *up to* `k+1` tokens in one round, using a variable amount of "extra" verification work depending on how many drafted tokens survive. A serving engine supporting both simultaneously has to generalize the iteration loop so that, per active request, the number of new KV-cache slots consumed by a single scheduling round is variable (anywhere from 1 to `k+1` tokens), rather than the fixed 1-token-per-iteration accounting file 003's simplified scheduler used — which means the admission-control and KV-cache-budget bookkeeping (file 003 Section 7, file 008 Part 2 Q6) has to reserve for the *worst case* (every draft accepted) even though the *expected* case consumes less, exactly the same conservative-reservation-versus-optimistic-admission trade-off discussed for prompt-length-based admission control, now applied per-round rather than per-request.

There's a second, more subtle interaction: verifying `k` drafted tokens for one request in parallel is itself a small batched forward pass over `k` positions for that one sequence — structurally similar to a mini-prefill. If many sequences in the active batch are each running their own speculative round simultaneously, the verify step's shape (each sequence contributing `k` positions instead of 1) changes the batch's effective compute profile away from "many sequences, each contributing exactly one token" toward something with higher arithmetic intensity per sequence — which is, in a small way, exactly the kind of prefill-shaped burden that chunked prefill exists to smooth out (Section 9 of file 003), and real implementations have to account for this when sizing chunk budgets and batch composition rather than treating speculative verification as a free addition to an otherwise-unchanged decode iteration.

### 9. Draft model sourcing and lifecycle management

A practical question Sections 2-5 didn't address directly: where does the draft model actually come from, and what happens when the target model changes? Three common sourcing strategies, each with a different maintenance burden:

- **A smaller model from the same family**, if one already exists (e.g., a lab's own smaller-tier model serving as a draft for its own largest model) — convenient because it's already trained and maintained for other reasons (as a standalone cheap-tier offering, file 006 Section 1), but its distribution alignment with the target is whatever it happens to be, not something specifically optimized for high acceptance.
- **A purpose-built, distilled draft model**, trained specifically to mimic the target model's output distribution as closely as possible at much lower cost — generally yields the highest acceptance rates of the three options, at the cost of an explicit additional training investment that has to be redone (or at least re-validated) whenever the target model changes materially.
- **Auxiliary heads on the target model itself** (Medusa-style, Section 5) — avoids maintaining a separate model/checkpoint lifecycle entirely, since the heads are trained together with (or fine-tuned on top of) the target model and travel with it, at the cost of the heads' own reduced multi-step accuracy relative to a genuinely separate autoregressive draft model.

Whichever sourcing strategy is used, the **lifecycle management problem is the same**: any update to the target model (a new fine-tune, a new quantized version, a version bump) risks silently degrading the draft/target alignment that the deployed acceptance rate depends on, exactly the failure mode diagnosed in Part 2 Q12 of this module's interview questions — a production speculative-decoding deployment needs an explicit re-validation step wired into its model-update process, not an assumption that "the draft model still works" survives a target-model change unexamined.

A practical consequence worth naming: this re-validation obligation is exactly the kind of hidden coupling a canary-deployment process (file 006 Section 3) needs to know about explicitly — a canary evaluating a new target-model version purely on its own standalone quality/latency metrics can pass cleanly while silently degrading the *paired* acceptance rate of a speculative-decoding setup riding alongside it, a regression that would show up only as an unexplained throughput or cost change (file 005 Section 4's cost-per-token tracking) rather than as any quality signal at all, since the *output* distribution remains exactly correct per Section 3's proof regardless of acceptance rate — it is purely a hidden efficiency regression, easy to miss unless acceptance rate itself is tracked as a first-class monitored signal.

### 10. Tree-based speculation, sketched

Section 5 mentioned tree-based/multi-candidate speculation only briefly; the core idea is worth one concrete pass since it generalizes the linear accept/reject walk in a way that's easy to reason about incrementally. Instead of drafting one linear sequence of `k` tokens, draft a small *tree* of candidates — e.g., at the first position, propose two or three plausible tokens rather than committing to one; for each of those, propose a further one or two continuations; and so on, to some bounded depth and branching factor. The target model then verifies the *entire tree* in one parallel forward pass (every node in the tree is a distinct position with a fully determined prefix — its path from the root — so all nodes' target distributions are computable simultaneously, exactly like the linear case's parallel verify step, just with more positions to verify per round).

Acceptance now walks whichever root-to-leaf path survives longest under the same accept/reject rule applied along that path, rather than a single fixed linear sequence — and because the tree offers several candidate continuations at each branching point rather than one, the probability that *some* path survives further than a single linear draft would have is higher, at the direct cost of verifying more total positions per round (a tree with branching factor `b` and depth `d` has up to `b^d` leaf paths, each needing its own verified position). The engineering trade-off is therefore between wider/deeper trees (higher chance of a longer accepted path, Section 4's `alpha^k` survival-probability argument softened by having multiple chances) against the cost of verifying a larger tree in the single parallel pass (more positions per round, eating into the wall-clock savings the same way a larger `k` does in the linear case) — conceptually the same `k`-tuning trade-off from Section 4, just with an extra branching-factor dimension added to the search space.

### 10b. Speculative decoding's benefit shrinks at large batch size — a subtlety worth having ready

Section 1 grounded the entire technique in decode being memory-bandwidth-bound with idle compute capacity to spare — but file 003 established that continuous batching's whole purpose is to *fill* that idle compute by running many sequences' decode steps together, and a sufficiently large batch eventually becomes compute-bound in its own right (file 005 Section 1's prefill/decode framing applies here too: a big enough decode batch starts behaving like a prefill-shaped matmul). This creates a real tension: speculative decoding's speedup comes from spending otherwise-idle compute to verify multiple drafted tokens per pass "for free," but at a batch size large enough that the GPU is already compute-bound from ordinary batched decode alone, that spare compute capacity has already been consumed by batching itself, and speculative decoding's parallel-verification step now competes for genuinely scarce compute rather than mopping up idle cycles — its marginal benefit shrinks, and past some batch size it can plausibly hurt throughput (spending real compute on drafted tokens that get rejected, when that compute could have gone toward more concurrent sequences' actual decode steps instead).

This is why speculative decoding is generally reported as most beneficial specifically in **low-to-moderate batch size regimes** — a lightly-loaded server, a latency-sensitive deployment intentionally running smaller batches to protect TPOT (file 005 Section 3), or a single-user/low-concurrency deployment — rather than as a universal throughput multiplier applicable identically at every operating point on the batch-size curve. A staff-level answer to "should we turn on speculative decoding" therefore has to ask "at what batch size does this server actually operate" before answering, not treat the technique as unconditionally beneficial; file 005's own cost-per-token framing (its Section 4) is the right lens for actually quantifying whether the trade nets positive at your specific deployment's typical batch size.

### 10c. What's actually disclosed about production use, versus reasonable inference

It is worth being explicit about the boundary between what frontier labs have publicly confirmed and what is reasonable inference, in the same spirit as this document series' treatment of undisclosed architectural facts elsewhere (e.g. `..\GPT\010_GPT5_Series.md` Section 11's confirmed-versus-speculative split). Speculative decoding as an algorithm is public research (the Leviathan et al. and Chen et al. papers cited in Section 3), and several serving frameworks (vLLM, TensorRT-LLM, and others) ship open, documented implementations of it. Whether any specific frontier lab's production API traffic for any specific named model is currently served with speculative decoding switched on, which draft model backs which target model in production, and what acceptance rates are achieved on real traffic are, as a rule, **not publicly disclosed** by any major lab as a specific operational fact — labs occasionally acknowledge using "inference optimization techniques" in broad terms without confirming which techniques apply to which model or endpoint. Treat any specific claim of the form "Model X is served with speculative decoding using draft model Y" as either sourced to a specific public statement you can point to, or as informed inference from the public architecture/serving literature generally (e.g., "labs operating at this scale have strong economic incentive to use some form of this, given the batch-size caveat in Section 10b" is a defensible inference, not a confirmed fact) — and say so explicitly rather than stating it with unwarranted confidence, exactly the discipline this document series applies to undisclosed architectural details throughout.

### 10c-ii. One more implementation detail worth flagging

A subtlety that trips up otherwise-correct implementations: the draft model's recorded probability `q_i` at each drafted position must be the probability *actually used to sample* that token, including whatever temperature, top-p, or top-k truncation was applied at sampling time — not the draft model's raw, untruncated softmax output. If the draft model's sampling distribution is truncated (e.g., top-p sampling), the accept/reject rule's `q(x)` must reflect that truncated, renormalized distribution, and the target model's `p(x)` used in the ratio must be prepared consistently (with whatever truncation, if any, the target model's own sampling policy applies) — mismatching this (comparing a truncated draft distribution against an untruncated target distribution, or vice versa) breaks the exact-distribution guarantee from Section 3, since the proof there assumes `p` and `q` are the actual distributions each model's tokens were drawn from, not some other distribution computed for convenience.

### 10d. A quick self-test

Before moving on, a candidate should be able to answer each of the following without notes, since they are the questions most likely to actually get asked, in some form, once this topic comes up: (1) write the acceptance probability and residual-distribution formulas from memory; (2) explain in one sentence why the residual is `max(0, p-q)` and not `p` alone; (3) state the two levers that determine acceptance rate and give one concrete example of each; (4) explain why the benefit shrinks at large batch size, tying it back to the roofline-style compute-bound/memory-bound argument (file 005 Section 1b); and (5) state precisely what part of this technique is proven-exact versus what part is empirically-tuned (the accept/reject rule is exact by construction; `k`, the draft model choice, and the cost ratio are all empirically-tuned engineering knobs that affect speed but never correctness). Being able to answer all five fluently, in under a minute each, is a reasonable bar for "understands this at the depth this file was written to reach."

### 11. Summary: where these techniques sit relative to each other

Pulling the whole file together into one final statement: the reason this topic recurs so heavily in staff-level interviews is that it sits exactly at the intersection of an elegant piece of probability theory (Section 3's exact-distribution proof) and a genuinely load-bearing systems insight (Section 1's memory-bandwidth argument, sharpened by Section 10b's batch-size caveat) — a candidate who can derive both halves, and who can also place the technique correctly alongside prefix caching, continuous batching, and quantization rather than in isolation, is demonstrating exactly the kind of cross-cutting systems fluency this entire module is built around.

Speculative decoding and prefix caching solve different halves of the same overall problem — "get more useful output per unit of expensive target-model compute" — but at different points in a request's lifecycle: prefix caching eliminates *redundant* prefill work across requests that share history; speculative decoding gets *more than one token* of genuinely new, previously-uncomputed output per expensive verification pass, for the *non-shared* portion of generation that no caching trick can shortcut (nobody has computed *this specific continuation* before). A production serving stack aiming for the best achievable latency/cost profile typically wants both simultaneously, layered on top of continuous batching and PagedAttention (file 003) and running on adequately quantized (file 002), appropriately provisioned (file 005) hardware — each technique attacks a genuinely distinct source of waste, and none of them substitutes for the others.

**A short checklist of pitfalls worth avoiding in an interview answer on this topic:**

- Do not describe speculative decoding as "approximate" or "a quality trade-off" — the entire point, and the subtle part worth demonstrating understanding of, is that it is an exact sampling method (Section 3's proof), and conflating it with genuinely lossy techniques like quantization or model distillation is a common, easily-avoided mistake.
- Do not assume the residual distribution is just `p_target` renormalized after excluding the rejected token — the correct residual is `max(0, p - q)` normalized, which depends on *both* distributions, not merely on masking out one rejected value from `p` (Section 3, and file 007/008's coding questions make this an explicit implementation detail to get right).
- Do not treat draft-model quality as solely a *speed* lever without noting the batch-size caveat in Section 10b — an interviewer probing for depth on this topic will often specifically ask "does this help at high concurrency," and the batch-size-dependent answer is the one that distinguishes a genuinely systems-level understanding from a purely algorithmic one.
- Do not present a single fixed "typical speedup" number as a universal fact — always tie any concrete multiplier to the acceptance rate and workload it was measured on, per Section 4 and Section 6's worked example.
- Do not forget the guaranteed bonus/corrected token when computing expected tokens per round — a common off-by-one is to count only the accepted drafted tokens and omit the final resampled-or-bonus token, understating `E[tokens per round]` by exactly one in every case.
