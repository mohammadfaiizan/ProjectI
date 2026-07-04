# Decoding Strategies and Sampling

## The Fundamental Setup: From Logits to a Token

Every autoregressive language model, no matter how large or how it was trained, ultimately does the same narrow thing at inference time: given a sequence of tokens so far, it produces a single vector of raw scores — logits — one per entry in the vocabulary, typically 32k to 200k+ dimensions depending on the tokenizer. Those logits are unnormalized and unbounded; they only carry meaning relative to each other. To turn them into something interpretable as "how likely is each next token," we pass them through a softmax:

```
P(token_i | context) = exp(logit_i) / sum(exp(logit_j) for j in vocabulary)
```

This gives a full probability distribution over the entire vocabulary at every single generation step. Crucially, the model itself does not choose a token — the forward pass's job ends at producing this distribution. The choice of which token actually gets emitted, appended to the context, and fed back in for the next forward pass is a separate, deliberately pluggable algorithm called the decoding (or sampling) strategy. This separation is easy to gloss over but matters enormously in practice: the exact same trained weights, the exact same logits, can produce dramatically different generation quality, diversity, and failure modes purely as a function of which decoding algorithm sits on top of them. This is why frameworks like Hugging Face's `generate()`, vLLM, and every major provider's API expose a whole family of decoding parameters (`temperature`, `top_p`, `top_k`, `min_p`, penalties) — you are not just calling the model, you are configuring a search/sampling procedure over a distribution the model provides.

It's worth internalizing the autoregressive loop explicitly because everything in this chapter operates inside it:

```python
def generate(model, tokenizer, prompt, max_new_tokens, decode_fn):
    input_ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        logits = model.forward(input_ids)[-1]      # logits for the next position only
        next_token = decode_fn(logits, input_ids)   # the decoding strategy lives here
        input_ids = input_ids + [next_token]
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids)
```

Every strategy discussed below is simply a different implementation of `decode_fn`. Some are deterministic functions of the logits alone (greedy, beam search); some introduce randomness by sampling from a (possibly reshaped) distribution (temperature, top-k, top-p, min-p); some modify the logits based on what has already been generated (repetition and frequency penalties); and some mask out entire regions of the vocabulary based on external structural constraints (grammar-constrained decoding). Production systems typically compose several of these together in a single call — for example, applying a repetition penalty, then temperature, then top-p, then sampling — and understanding each piece in isolation is what lets you reason about that composition instead of treating `temperature=0.7, top_p=0.9` as magic incantations.

## Greedy Decoding

The simplest possible decoding rule is to always take the single highest-probability token at each step: `next_token = argmax(P(token | context))`. This is deterministic — the same prompt with the same model always produces the same output — and it feels intuitively "optimal," since at every individual step you are making the locally best choice available.

That local optimality is exactly the problem. Greedy decoding is a purely myopic, one-step-lookahead procedure; it has no mechanism for realizing that committing to the single most likely next word might back the generation into a corner where every subsequent continuation is mediocre, while a slightly less likely next word would have opened up a much better overall sequence. In practice this myopia manifests as a specific, well-documented failure mode: repetition. Once a language model emits a phrase, the context now contains that phrase, and conditioned on "I have already said X," the model frequently assigns very high probability to saying something like X again — especially in open-ended generation where there's no external signal forcing forward progress the way there is in translation (where the source sentence keeps supplying new content). The result is generation that gets stuck in loops ("the the the" is an extreme, easy-to-parody example, but subtler forms — repeating whole clauses, restating the same idea in a paraphrase, circling the same three or four ideas in an essay — are extremely common with greedy decoding on any modern LLM). This is often called degenerate text: technically coherent, grammatically fine, locally probable at every step, yet globally boring and repetitive in a way that's immediately recognizable as non-human.

Greedy decoding also has essentially zero output diversity — every call with a given prompt returns the identical response — which is fine for some use cases (deterministic tool-calling, code completion where you want the single best guess) but disqualifying for anything where variety across samples matters (creative writing, brainstorming, generating multiple candidate solutions to rerank).

```python
import numpy as np

def greedy_decode(logits: np.ndarray) -> int:
    """Deterministic: always the single highest-probability token."""
    return int(np.argmax(logits))
```

## Beam Search

Beam search generalizes greedy decoding by keeping track of the `k` most promising sequences (called the beam, with `k` the beam width) simultaneously, rather than committing irrevocably to a single best token at each step. At every step, for each of the `k` candidate sequences currently in the beam, the algorithm computes the probability distribution over the next token, forms all `k * vocab_size` possible one-token extensions, scores each extension by its cumulative sequence log-probability (log-probabilities are used instead of raw products so scores don't underflow over long sequences), and keeps only the top `k` extended sequences overall going into the next step. This means beam search is doing an approximate search for the single sequence with the highest total probability under the model, exploring `k` parallel hypotheses instead of the one hypothesis greedy decoding tracks — importantly, it is still not exhaustive (that would require enumerating an exponential number of sequences), it is a pruned, beam-width-limited approximation to the true highest-probability sequence.

A small worked example makes the mechanics concrete. Suppose beam width `k=2`, and after generating "The cat", the model's next-token distribution puts most mass on "sat" (0.5) and "ran" (0.3), with the remainder spread thin. Both "The cat sat" (cumulative log-prob from 0.5) and "The cat ran" (cumulative log-prob from 0.3) enter the beam. Now suppose that conditioned on "The cat sat", the model strongly favors "down" (0.9) but conditioned on "The cat ran", the model is uncertain, spreading probability across "away" (0.4), "quickly" (0.3), and other options. Greedy decoding, having already discarded "The cat ran" at the first step, would only ever explore continuations of "The cat sat". Beam search keeps both alive and compares the actual cumulative scores: `log(0.5) + log(0.9)` for "The cat sat down" versus `log(0.3) + log(0.4)` for "The cat ran away", and so on for every candidate pair, keeping the top 2 overall after this second step. The key structural advantage over greedy is exactly this: a token that looks slightly suboptimal in isolation ("ran" at 0.3 versus "sat" at 0.5) can still end up part of the globally best-scoring sequence once you look one or more steps ahead, and beam search is a mechanism for not throwing that possibility away prematurely.

```python
import numpy as np

def beam_search(model_step_fn, start_tokens, beam_width, max_len, eos_id):
    """model_step_fn(seq) -> log-probabilities over vocab for the next token."""
    beams = [(start_tokens, 0.0)]  # (token sequence, cumulative log-prob)
    completed = []

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq[-1] == eos_id:
                completed.append((seq, score))
                continue
            log_probs = model_step_fn(seq)                 # shape: [vocab_size]
            top_k_ids = np.argsort(log_probs)[-beam_width:]  # cheap top-k for the example
            for token_id in top_k_ids:
                candidates.append((seq + [int(token_id)], score + log_probs[token_id]))

        if not candidates:
            break
        # Keep only the best `beam_width` sequences overall, across all beams
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    completed.extend(beams)
    # Length-normalize, since raw cumulative log-prob unfairly penalizes longer sequences
    completed.sort(key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)
    return completed[0][0]
```

That length-normalization line is not incidental. Cumulative log-probability is a sum of negative numbers (log of a probability less than 1 is negative), so every additional token can only ever decrease the total score. Left uncorrected, plain beam search systematically favors shorter sequences, which is why virtually every real implementation divides by sequence length (or a length-penalty exponent, as in the original Transformer/GNMT beam search formulations) before comparing finished hypotheses.

Beam search is genuinely excellent for tasks where there is a comparatively narrow space of "correct" or "good" outputs and the goal is closer to finding the single best answer than to producing diverse, natural-sounding variety. Machine translation is the canonical example: for "The cat is on the mat," there's a fairly constrained set of acceptable French translations, and finding the sentence the model assigns the highest joint probability to is a reasonable proxy for finding the best translation. The same logic applies to summarization and constrained code generation to a degree.

Beam search performs surprisingly poorly, however, for open-ended generation — chat responses, stories, essays, dialogue — and understanding why is one of the more important pieces of decoding theory to have internalized, since it directly motivates the entire sampling-based approach covered next. This is the "neural text degeneration" problem, named and empirically characterized in Holtzman et al.'s 2019 paper "The Curious Case of Neural Text Degeneration." The core finding is that maximizing sequence probability is simply the wrong objective for open-ended text: high-probability text under a language model is not the same thing as high-quality, human-like text. Beam search, by construction, hunts for exactly the highest-probability sequence it can find (within its beam-width approximation), and that hunt systematically converges on text that is bland, generic, and — just like greedy decoding but often worse, since beam search searches harder for the probability-maximizing path — highly repetitive, because repeating a previously-high-probability phrase is frequently the locally and cumulatively "safe" high-probability move at every step.

The paper's most striking piece of evidence is a comparison of the per-token probability (or "surprisal," the negative log-probability the model assigned to the token that was actually chosen) of human-written text versus beam-search-generated text. Human writing has substantial variance in per-token surprisal: humans routinely choose a moderately-improbable-but-interesting word, then a very predictable function word, then another surprising word — the probability of the "true" next token, as judged by the model, swings up and down constantly throughout genuinely good human writing. Beam search text, by contrast, has almost none of that variance — it consists almost entirely of tokens the model considers highly probable, step after step, because that is literally what the algorithm is optimizing for. That flatness is what a human reader perceives as "boring," "safe," "generic," or "repetitive" even when every individual sentence is grammatically perfect. This is the central insight that motivated the field's move toward sampling-based decoding for open-ended generation: instead of asking "what is the most probable continuation," you want to ask "what is a continuation that looks like it was drawn from the same distribution human text was drawn from" — and those are different questions with different answers.

## Temperature Scaling

Temperature is the simplest lever for controlling how "confident" versus "random" sampling behaves, and it works by rescaling the logits before the softmax rather than after:

```
P(token_i) = exp(logit_i / T) / sum(exp(logit_j / T) for j in vocabulary)
```

The intuition for why dividing by `T` reshapes the distribution follows directly from the exponential in softmax. When `T < 1`, dividing logits by a number less than one makes them larger in magnitude, which — because softmax is an exponential — disproportionately amplifies the gap between the highest logit and the rest. As `T` approaches 0, the distribution approaches a one-hot spike on the single argmax token, which is precisely why greedy decoding is correctly described as the `T=0` limiting case of temperature sampling (and why implementations special-case `T=0` to just call argmax directly, since dividing by zero is undefined). When `T > 1`, dividing by a number greater than one shrinks the logits toward zero, which compresses the differences between them; since a softmax over near-identical logits approaches a uniform distribution, high temperatures push sampling toward "pick almost any token roughly uniformly at random," increasing randomness and, past a certain point, incoherence. `T = 1` leaves the distribution exactly as the model produced it — no reshaping at all.

```python
def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 0:
        # Greedy limit: return a one-hot distribution on the argmax
        one_hot = np.zeros_like(logits)
        one_hot[np.argmax(logits)] = 1.0
        return one_hot
    scaled = logits / temperature
    scaled -= np.max(scaled)  # numerical stability, does not change the softmax result
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled)
```

It's worth being precise that temperature does not change the *ranking* of tokens by probability — it only changes how *peaked or flat* the distribution is. The token that was most likely at `T=1` is still the most likely at any other `T`. This is exactly why temperature is almost always combined with a truncation method like top-k or top-p rather than used alone: at moderate-to-high temperatures, flattening the whole distribution means low-probability, borderline-nonsensical tokens that used to have negligible probability now have a non-trivial chance of being sampled, and without some form of truncation to cut off that low-quality tail, higher temperatures start to noticeably increase the rate of incoherent or off-topic tokens slipping through, not just increase creative diversity among the reasonable candidates.

## Top-k Sampling

Top-k sampling addresses that tail-risk problem directly: instead of sampling from the full vocabulary, restrict the candidate pool to only the `k` tokens with the highest probability, renormalize their probabilities so they sum to 1, and sample from that truncated distribution. This guarantees, by construction, that no token from the long, unreliable tail of the distribution can ever be selected, no matter how high the temperature is set.

```python
def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Zero out (set to -inf) every logit except the top k."""
    if k >= len(logits):
        return logits
    threshold = np.partition(logits, -k)[-k]  # k-th largest value
    filtered = np.where(logits >= threshold, logits, -np.inf)
    return filtered
```

Top-k's weakness is that a single fixed `k` is the wrong choice across the wide range of distribution shapes a model actually produces at different generation steps. Consider a step where the model is extremely confident — say, completing "Barack Oba—" where the correct continuation "ma" has 99.9% probability and every other token is essentially noise. With `k=40` (a common default), you're forcibly including 39 near-zero-probability garbage tokens in the sampling pool; they'll rarely get sampled, but the "rarely" is not "never," and over millions of generation steps in production, that's a nonzero rate of injecting nonsense into otherwise-confident completions. Now consider the opposite case: a genuinely open-ended step, like the very first word of a creative story, where the model spreads reasonable probability mass across hundreds of plausible opening words. Here `k=40` is too restrictive — it arbitrarily cuts off many perfectly reasonable candidates purely because of where they happened to rank, discarding legitimate diversity the model itself considered worthwhile. Top-k has no way to distinguish these two situations because it only looks at rank, never at how the probability mass is actually distributed.

## Top-p (Nucleus) Sampling

Nucleus sampling, introduced by Holtzman et al. in the same paper that diagnosed the neural text degeneration problem, fixes exactly this shortcoming by truncating based on cumulative probability mass rather than a fixed count. Given a threshold `p` (commonly 0.9 to 0.95), sort tokens by probability descending, and keep adding tokens to the candidate set until their cumulative probability first exceeds `p`. That variable-size set — the "nucleus" — is then renormalized and sampled from.

The adaptiveness this produces is exactly what top-k lacks. In the confident "Barack Oba—" example, "ma" alone might already account for 99.9% of the mass, so the nucleus at `p=0.9` contains only that one token (or a small handful) — nucleus sampling automatically behaves almost like greedy decoding when the model is confident, without you having to detect that confidence and switch strategies manually. In the flat, open-ended story-opening example, no single token or small set dominates, so many more tokens are required before cumulative probability crosses `p`, and the nucleus naturally grows to include dozens of plausible candidates. The size of the sampling pool is therefore a direct function of the actual shape of the distribution at that specific step, which is precisely the property fixed-`k` top-k sampling cannot provide.

```python
def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set of top tokens whose
    cumulative probability exceeds p."""
    probs = softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)

    # Find the cutoff: smallest prefix whose cumulative probability exceeds p
    cutoff = np.searchsorted(cumulative, p) + 1
    keep_idx = sorted_idx[:cutoff]

    filtered = np.full_like(logits, -np.inf)
    filtered[keep_idx] = logits[keep_idx]
    return filtered

def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
```

Top-p is the closest thing the field has to a default recommendation for open-ended, chat-style generation, and it's what most production chat APIs ship as the primary truncation parameter (often paired with a moderate temperature, e.g. `temperature=0.7, top_p=0.9`, or with instructions to tune one or the other but not both aggressively at once).

## Min-p Sampling

Min-p sampling is a more recent alternative (popularized in open-source LLM serving communities and formalized in a 2024 paper) that targets a specific awkwardness in top-p: at high temperatures, flattening the distribution can inflate the *count* of tokens needed to reach the cumulative-probability threshold `p` to an uncomfortably large number, since flattening pushes probability mass into the tail precisely where top-p's cumulative-sum criterion is most sensitive to it — you can end up with a large nucleus full of tokens that are individually still quite weak relative to the best option, simply because temperature smeared the distribution out.

Min-p sidesteps the cumulative-sum computation entirely and instead sets a floor directly relative to the top token's probability. Given the top token's probability `p_max` in the (pre-truncation) distribution and a chosen fraction `min_p` (commonly something like 0.05 to 0.1), the algorithm keeps every token whose probability is at least `min_p * p_max`, discarding everything else:

```
threshold = min_p * p_max
keep token_i if P(token_i) >= threshold
```

The appeal is twofold. First, it's cheaper and simpler — no sorting-and-cumulative-sum pass is strictly required, just a single comparison against a scalar threshold. Second, and more importantly, it scales its aggressiveness naturally with model confidence in a way that composes better with high temperature specifically: when the model is very confident (`p_max` close to 1), the threshold is high in absolute terms, so only genuinely competitive tokens survive; when the model is uncertain (`p_max` more modest), the threshold is proportionally lower, allowing a wider set of candidates through, but always defined relative to the best option rather than an absolute cumulative target that temperature can distort. Practitioners who like to run at higher temperatures for more creative or less repetitive output (`T` in the 1.0-1.5 range) have reported that min-p keeps generations more coherent than top-p does at the same temperature, precisely because it doesn't let temperature-induced tail-flattening balloon the candidate pool the way cumulative-mass truncation can.

```python
def min_p_filter(logits: np.ndarray, min_p: float) -> np.ndarray:
    probs = softmax(logits)
    p_max = np.max(probs)
    threshold = min_p * p_max
    filtered = np.where(probs >= threshold, logits, -np.inf)
    return filtered
```

## Putting Sampling Together: A From-Scratch Pipeline

Production decoding almost always composes these primitives in a fixed order: apply any history-based penalties to the raw logits first, apply temperature scaling, apply a truncation method (top-k, top-p, or min-p, sometimes more than one chained together), then sample from what remains.

```python
import numpy as np

def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> int:
    logits = logits.copy()

    if temperature == 0:
        return int(np.argmax(logits))

    logits = logits / temperature

    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    if min_p is not None:
        logits = min_p_filter(logits, min_p)

    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))
```

An important implementation subtlety: top-k/top-p/min-p filtering is typically applied to the *temperature-scaled* logits (as above), because the whole point of min-p in particular is to counteract distortions temperature introduces — filtering before scaling would defeat that purpose. Different libraries make slightly different ordering choices here, so when debugging unexpected generation behavior in a real framework, it's worth checking the actual order of operations rather than assuming.

## Repetition, Frequency, and Presence Penalties

Sampling-based methods reduce repetition relative to greedy/beam search but don't eliminate it, since a model can still assign genuinely high probability to a phrase it has already produced (this is common with smaller or less well-tuned models, and with long generations where the model's effective context handling degrades). Repetition penalties address this directly by modifying the logits based on token history before sampling, rather than relying purely on randomness to avoid loops.

The classic repetition penalty (from Keskar et al.'s CTRL paper) works multiplicatively: for every token that has already appeared in the generated sequence, divide its logit by a penalty factor (if the logit is positive) or multiply by the factor (if negative), which uniformly discourages re-selecting anything already said, regardless of how many times it has appeared.

OpenAI's API popularized a slightly more graduated pair of penalties that are additive on the logits rather than multiplicative, and which are worth knowing precisely since they show up by name in real API parameters:

- **Presence penalty**: subtracts a fixed amount from the logit of any token that has appeared at least once in the generation so far, regardless of how many times. This nudges the model toward introducing new topics/tokens rather than staying only on ones already used, but doesn't further punish something for appearing a third or fourth time versus a second time.
- **Frequency penalty**: subtracts an amount proportional to how many times the token has already appeared — `penalty = frequency_penalty_coefficient * count_so_far`. This specifically targets the "gets stuck saying the same word over and over" loop failure mode, since the penalty compounds with each repeat.

```python
def apply_repetition_penalties(
    logits: np.ndarray,
    generated_token_ids: list[int],
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> np.ndarray:
    logits = logits.copy()
    counts: dict[int, int] = {}
    for tok in generated_token_ids:
        counts[tok] = counts.get(tok, 0) + 1

    for tok_id, count in counts.items():
        logits[tok_id] -= presence_penalty            # flat penalty, applied once
        logits[tok_id] -= frequency_penalty * count    # scales with repeat count
    return logits
```

These penalties are effective but not free — they trade off against tasks where legitimate repetition is required. Structured or factual output frequently needs to repeat specific tokens on purpose: a JSON object with the same key name repeated across an array of objects, a legal or technical document that must reuse exact terminology for precision rather than varying it stylistically, code that legitimately repeats variable names and syntax tokens constantly. Applying an aggressive frequency penalty to such generations can degrade correctness — the model gets pushed away from the exact token it needs to reuse and may substitute a near-synonym or malformed variant instead, purely to satisfy the penalty. This is why these penalties default to 0 (off) in most APIs and should be tuned per use case rather than applied as a blanket "improve quality" setting — they help conversational and creative generation and can actively hurt structured, technical, or code generation.

## Constrained and Grammar-Based Decoding for Structured Output

Every method discussed so far operates on the assumption that the *only* thing that matters is picking a good token from the distribution the model produced — the model is trusted to eventually produce valid structure (correct JSON syntax, a well-formed function call, output matching a specific format) purely through instruction-following and its own learned habits from training on structured examples. That trust is well-placed most of the time with modern instruction-tuned models, but "most of the time" is not good enough for production systems where a downstream parser will throw an exception on a missing closing brace, a trailing comma, or a hallucinated field name — this is exactly the reliability gap that motivated an entirely different category of decoding technique: constraining generation so that invalid output is not merely discouraged but architecturally impossible.

The core idea is to derive, from a specification of the desired output format (most commonly a JSON Schema, a regular expression, or a general context-free grammar in a format like Lark or GBNF), a finite state machine (for regex-like or JSON-Schema-like constraints) or pushdown automaton (for full context-free grammars, needed when constraints require matching nested/recursive structure that a plain FSM can't track, like arbitrarily nested JSON objects). At every decoding step, before sampling, the current automaton state defines the exact set of tokens that would keep the output syntactically valid if chosen next. Every other token in the vocabulary gets its logit set to negative infinity (or is simply excluded from the candidate set) before any of the sampling logic above ever runs. This is often described as "logit masking" and it composes cleanly with everything covered earlier in this chapter — you can still apply temperature, top-p, penalties, and so on, but only within the subset of tokens the grammar permits at that position. After a valid token is chosen, the automaton transitions to a new state reflecting the updated set of valid next tokens, and the process repeats until the automaton reaches an accepting state.

This is the mechanism underlying tools like Outlines, Guidance, and the structured-output modes shipped by OpenAI's API (`response_format={"type": "json_schema", ...}`) and inference servers like vLLM. All of them ultimately compile a JSON Schema (or regex, or grammar) into some form of state machine over the *tokenizer's* vocabulary — a nontrivial engineering step in its own right, since tokens don't align neatly with grammar symbols (a single BPE token might be `{"na`, spanning a brace, a quote, and part of a key name, all at once), so the compiled automaton has to reason about validity at the token level, not the character level, typically by precomputing, for every automaton state, exactly which vocabulary tokens are compatible with the regular/context-free language from that state onward. Because this masking happens *before* sampling and eliminates invalid tokens with zero probability rather than merely low probability, validity becomes a guarantee derived from the automaton's structure rather than a statistical hope resting on how well the model was prompted or fine-tuned — a categorically stronger guarantee than "the model usually gets JSON right" or "we retry until `json.loads()` succeeds."

A minimal illustration of the mechanism, using a toy example of forcing the model to emit one of a fixed set of valid JSON keys at a given position (a simplified stand-in for what a real JSON-Schema-to-FSM compiler does at each step):

```python
import numpy as np

def mask_to_allowed_tokens(logits: np.ndarray, allowed_token_ids: set[int]) -> np.ndarray:
    """Zero out probability mass for every token not currently valid
    under the grammar/schema's finite state machine."""
    mask = np.full_like(logits, -np.inf)
    for tok_id in allowed_token_ids:
        mask[tok_id] = logits[tok_id]
    return mask


class JSONKeyStateMachine:
    """Toy FSM: at the 'expecting a key' state, only tokens that begin
    one of the schema's known key strings (plus the opening quote) are legal.
    A real implementation (e.g. Outlines) builds this over full JSON grammar,
    not just key names, and does it for the actual sub-word tokenizer vocab."""

    def __init__(self, valid_keys: list[str], tokenizer):
        self.valid_keys = valid_keys
        self.tokenizer = tokenizer
        # Precompute, per state, which token ids are legal continuations.
        # Here: only the token ids that correspond to a valid opening quote
        # followed by the start of one of the allowed key names.
        self.key_start_tokens = {
            tokenizer.encode(f'"{key}')[0] for key in valid_keys
        }

    def allowed_tokens(self, state: str) -> set[int]:
        if state == "expect_key":
            return self.key_start_tokens
        raise NotImplementedError("toy example only implements one state")


def decode_step_with_grammar(logits, fsm: JSONKeyStateMachine, state: str, temperature=1.0):
    allowed = fsm.allowed_tokens(state)
    masked_logits = mask_to_allowed_tokens(logits, allowed)
    # Normal sampling machinery still applies, just restricted to the
    # grammar-legal subset of the vocabulary.
    return sample_next_token(masked_logits, temperature=temperature)
```

The practical trade-off to be aware of, and worth raising in an interview to show you understand this isn't a free lunch, is that grammar-constrained decoding adds computational overhead — computing the allowed-token set at every step requires either walking the automaton live (cheap if precomputed transition tables exist) or, for very large or dynamically-constructed schemas, potentially expensive per-step computation — and it strictly bounds what the model *can* say, which occasionally produces awkward or lower-quality content if the schema is more restrictive than what would let the model express its actual "intended" answer naturally (for example, forcing a very specific enum value when the model's genuinely best answer doesn't cleanly fit any listed option). The general industry consensus, however, is that for anything feeding a downstream parser or API — function calling, tool-use argument generation, structured extraction — guaranteed-valid-by-construction output is worth that overhead, since the alternative (prompting for JSON and hoping, then retrying on parse failure) is both less reliable and, across many retries, often slower and more expensive in aggregate than doing it right the first time via constrained decoding.

## Summary: Choosing a Strategy in Practice

The practical decision tree senior engineers should carry into an interview or a real system design: use greedy decoding (or temperature near 0) for tasks with a single well-defined correct answer where determinism and reproducibility matter more than diversity — classification-style outputs, deterministic tool-call argument generation, code completion where variance is undesirable. Use beam search for tasks like translation or constrained summarization where you genuinely want the model's single best-scoring hypothesis and the domain doesn't punish beam search's known bias toward safe, high-probability phrasing. Use temperature plus top-p (optionally with a light frequency penalty) as the default for open-ended chat, creative writing, and brainstorming, tuning temperature down for more focused/deterministic-feeling responses and up for more variety, and reach for min-p specifically if you're running at higher temperatures and finding top-p's cumulative-mass criterion is letting through too much incoherence. And whenever the output must be machine-parseable — JSON, function calls, any fixed grammar — reach for constrained/grammar-based decoding rather than relying on prompting and hoping, since it converts a probabilistic reliability problem into a structural guarantee.
