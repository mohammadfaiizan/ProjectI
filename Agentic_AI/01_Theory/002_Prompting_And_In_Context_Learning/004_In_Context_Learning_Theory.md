# In-Context Learning Theory

## The Puzzle That Needs Explaining

Every technique in the first three chapters of this series — few-shot prompting, chain-of-thought
demonstrations, worked examples of a JSON schema — rests on one empirical fact that is, if you stop
to think about it, genuinely strange: a model with completely frozen weights can be shown a handful
of input-output pairs it has never seen before, in a single forward pass, and then correctly apply
whatever pattern those pairs implied to a brand-new input. No gradient descent happens. No parameter
is updated. The examples exist only as tokens sitting in the context window, gone the moment the
conversation ends. And yet the model's behavior on the next input changes, sometimes dramatically,
based purely on which examples you showed it and in what form.

This is called in-context learning (ICL), and the name is slightly misleading if taken literally —
nothing is being "learned" in the sense of stored, persistent weight updates. What's actually
happening is closer to a very sophisticated form of conditional pattern completion: the examples in
the prompt shift the model's effective behavior for the rest of that forward pass, the same way any
other context does, but they do it in a way that looks remarkably like task acquisition.
Understanding *why* this works — rather than just *that* it works — matters for an engineer because
it explains several practical phenomena that otherwise seem arbitrary: why example order matters,
why label correctness sometimes matters less than you'd expect, why retrieval-based example
selection outperforms a fixed few-shot set, and why you should never mistake a great few-shot prompt
for a substitute for actual fine-tuning when you need behavior that must persist reliably outside of
a curated prompt.

## Induction Heads: A Mechanistic Story

The most concrete, mechanism-level explanation for in-context learning comes from interpretability
research (notably Anthropic's "In-context Learning and Induction Heads," Olsson et al., 2022), which
identified a specific, recurring circuit inside transformer attention layers called an **induction
head**.

The behavior an induction head implements is simple to state: given a sequence where some token `A`
was previously followed by token `B` earlier in the same context, when the model encounters `A`
again later in that same context, the induction head shifts attention toward the token that followed
`A` last time, and boosts the probability of predicting `B` next. In symbols: pattern `[A][B] ...
[A] → predict [B]`. This is a form of in-context copying — the model isn't retrieving `B` from its
trained-in world knowledge, it's retrieving it from *this specific context*, because it noticed the
repeated pattern within the current sequence.

It's a small mechanism, but its implications are large once you see how naturally it generalizes to
few-shot prompting. A few-shot prompt is, structurally, a sequence of repeated patterns:
`[input_1][output_1] [input_2][output_2] ... [input_1][output_1] [input_2][output_2] ...
[query][?]`. An induction head (or, more realistically, a composition of several attention heads
working together, since real ICL behavior in large models is more sophisticated than a single simple
head) can look back at prior `[input][output]` pairs, notice that whenever an input-like span
appeared, a corresponding output-like span followed it, and apply that same completion pattern to
the new query. Under this account, few-shot prompting isn't teaching the model a new capability from
scratch in-context; it's activating a general-purpose "notice a repeating pattern in the current
context and complete it" mechanism that was learned during pretraining because such patterns are
ubiquitous in natural text (repeated formatting in tables, dialogue turns, structured documents,
code with repeated idioms).

Interestingly, interpretability work has also found that induction heads tend to emerge somewhat
abruptly during pretraining — a "phase transition" where a cluster of attention heads sharply
specializes into this copying behavior over a relatively short window of training, and this
transition correlates closely in time with a jump in the model's measured in-context learning
ability on held-out tasks. This correlation is one of the stronger pieces of evidence tying the
mechanistic story (a specific attention circuit) to the behavioral story (models becoming good at
few-shot tasks), even though it doesn't prove induction heads are the *entire* explanation for ICL
in large, modern models, where many overlapping mechanisms likely contribute.

## Implicit Task Inference

A complementary, more abstract way to think about why ICL works — less about the specific attention
circuit and more about what the model is effectively doing at a statistical level — comes from
framing few-shot prompting as **implicit Bayesian inference over a latent task** (Xie et al., 2022,
"An Explanation of In-context Learning as Implicit Bayesian Inference"). Under this view,
pretraining exposes the model to an enormous mixture of different "documents," each generated
according to some underlying latent structure — a particular topic, register, task, or convention.
The model implicitly learns a distribution over these latent document-generating processes. When you
hand it a few-shot prompt, each example is evidence that narrows down which latent process is most
likely to have generated this particular document, and the model's completion of the final,
unlabeled example is essentially "what would this token look like if I'm currently inside a document
following the same generating process as the examples I've just seen."

This framing explains a few things the induction-head story alone doesn't fully cover. It explains
why more examples generally help even when each individual example is short and simple — each
additional example is more evidence narrowing down the inferred task, the same way more data points
narrow down a posterior distribution in ordinary Bayesian inference. It explains why examples that
are stylistically or topically similar to your real query help more than generic examples — they're
more informative evidence about the specific latent task you actually want. And it explains why a
model can sometimes perform reasonably even when a few individual examples are technically
mislabeled: the model isn't doing literal supervised learning on your examples as training data,
it's using them as noisy evidence about which task is being requested, and enough correct examples
can outweigh a few outliers, similarly to how a Bayesian posterior isn't destroyed by one
adversarial data point (though it can be, if the noise is bad enough or the model is small enough
that its prior is weak).

## How Example Order and Selection Affect Performance

Because in-context learning is a real-time, position-sensitive computation over the tokens currently
in context — not a symmetric, order-independent aggregation of "training examples" — the
*arrangement* of your few-shot examples measurably changes model output, in ways that have nothing
to do with the semantic content of the examples themselves. This is one of the most consequential,
and most consistently under-appreciated, practical findings in the ICL literature.

**Order sensitivity.** Research on prompt ordering (notably Lu et al., 2021, "Fantastically Ordered
Prompts and Where to Find Them") found that simply permuting the order of the same set of few-shot
examples — same examples, same labels, same everything except sequence — can swing accuracy on a
classification task by tens of percentage points, occasionally taking a prompt from near-random
performance to strong performance or vice versa, purely based on ordering. There is no reliable
universal "best order" across models and tasks; the practical implication is that example order
should be treated as a tunable parameter (validated on a held-out set for anything that matters),
not an arbitrary implementation detail, and that a prompt that performs poorly should be re-tested
with a shuffled example order before being written off as a bad prompt.

**Recency and majority-label bias.** Few-shot prompts exhibit a measurable bias toward whatever
label appeared most recently (closest to the query) and toward whichever label appeared most
frequently across the examples, independent of the actual content of the new query. If your few-shot
set happens to have two "positive" examples and one "negative," expect a systematic pull toward
"positive" on ambiguous new inputs even when the true distribution of your production data is
balanced. The direct fix is to balance label frequency in your example set and, where feasible, to
explicitly randomize order across calls (or across a validation sweep) rather than fixing one static
order for every request.

**Similarity-based example selection.** Because implicit task inference works by narrowing down a
latent task from evidence, examples that are more semantically similar to the actual query provide
more relevant evidence than generic examples plucked at random from a large bank. This is why
retrieval-augmented few-shot prompting — embedding a bank of candidate examples, embedding the
incoming query, and selecting the k nearest examples by cosine similarity to build a query-specific
few-shot prompt — consistently and substantially outperforms a single static few-shot set applied
uniformly to all queries.

```python
import numpy as np

def embed(text: str) -> np.ndarray:
    """Stand-in for a real embedding call, e.g. an OpenAI or Sentence-Transformers embedding."""
    ...

def select_few_shot_examples(query: str, example_bank: list[dict], k: int = 4) -> list[dict]:
    """
    example_bank: list of {"input": str, "output": str, "embedding": np.ndarray}
    Selects the k examples most similar to the query, by cosine similarity of embeddings.
    """
    query_vec = embed(query)
    query_vec = query_vec / np.linalg.norm(query_vec)

    scored = []
    for ex in example_bank:
        ex_vec = ex["embedding"] / np.linalg.norm(ex["embedding"])
        similarity = np.dot(query_vec, ex_vec)
        scored.append((similarity, ex))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [ex for _, ex in scored[:k]]
```

There is a real tension worth naming between pure similarity-based selection and diversity: if your
example bank has many near-duplicate examples and you always retrieve the top-k by similarity, you
can end up showing the model four nearly identical demonstrations that all reinforce the same narrow
slice of the task, providing less genuinely new evidence than a slightly more diverse but slightly
less individually-similar set would. Production few-shot retrieval systems commonly apply a
diversity penalty (such as maximal marginal relevance) on top of raw similarity ranking specifically
to counteract this.

**Calibration matters as much as example choice.** Because few-shot prompts carry these order and
frequency biases, raw output probabilities or confidence-sounding language from a few-shot-prompted
model should not be taken at face value as calibrated confidence — a technique called "contextual
calibration" (adjusting for the model's bias toward certain answers even on a content-free or
placeholder query, then correcting the real query's output distribution by that bias) has been shown
to meaningfully improve few-shot classification accuracy without changing the examples at all,
purely by correcting for the ordering and frequency artifacts described above.

## The Practical Implication: Pattern Completion, Not Weight Learning

It's worth stating the core implication of everything above as plainly as possible, because it has
direct engineering consequences: in-context learning is not a cheap substitute for training. The
model's weights are unchanged by anything in your prompt; what you are doing is steering a single
forward pass toward the region of the model's already-learned behavior that best matches the pattern
your examples establish. This has several concrete consequences worth internalizing.

**Nothing persists across calls unless you re-supply it.** Every single API call starts from the
same frozen weights; if you want the same few-shot conditioning on the next request, you must
include the same (or an equivalently effective) set of examples again. This is different in kind
from fine-tuning, where a training pass genuinely updates weights and the resulting behavior
persists in the model artifact itself, with no need to re-supply examples at inference time. The
engineering tradeoff is real: ICL is faster to iterate on (edit a prompt, no retraining cycle) and
requires no training infrastructure, but it pays a per-call token cost forever and its ceiling is
bounded by what can be expressed and demonstrated within the context window, whereas a well-executed
fine-tune can encode behavior more efficiently and more robustly, at the cost of a slower iteration
loop and real training infrastructure.

**The model can be "fooled" by surface pattern rather than true task semantics**, and this is a
double-edged fact. Early studies (e.g., Min et al., 2022, "Rethinking the Role of Demonstrations")
found that in some model families and tasks, replacing the *correct* labels in few-shot examples
with randomly shuffled, incorrect labels barely hurt classification accuracy compared to
correctly-labeled examples — what mattered most was that the examples established the right
*input-output format and label space*, not that each individual label was actually correct. This
finding has been refined and partly contested by later work (larger and more capability
instruction-tuned models do show a real accuracy drop from wrong labels, more consistent with
genuine implicit task inference), but the underlying lesson generalizes cleanly regardless of
exactly how sensitive a given model is to label correctness: few-shot examples do at least two jobs
simultaneously — they specify *what shape of task and answer* is expected, and they provide (to
varying degrees, depending on the model) *evidence about which specific answers are correct*. Don't
assume that because your few-shot examples visibly work, the model is attending to their correctness
in the way you'd expect a human learner to; test directly, e.g., by deliberately corrupting a copy
of your example set and checking how much accuracy actually degrades, before relying on example
correctness as your primary lever for controlling model behavior.

**This is why security-sensitive behavior should never rely on ICL alone.** Because in-context
learning is a real-time computation over whatever tokens happen to be in the context window, and
because that computation has no persistent memory of "which instructions were trustworthy," anything
that later ends up in that same context window — a retrieved document, a tool result, a user message
— is processed by the same pattern-completion machinery as your carefully engineered examples, with
no structural firewall between "the demonstrations I intended" and "arbitrary other text that
happens to look like a demonstration." That structural fact is the direct throughline into the next
chapter, on prompt injection: if in-context learning works by the model completing whatever pattern
is present in its context, then an attacker who can inject their own pattern into that context has a
genuine mechanism, not just a hypothetical concern, for hijacking the model's behavior.
