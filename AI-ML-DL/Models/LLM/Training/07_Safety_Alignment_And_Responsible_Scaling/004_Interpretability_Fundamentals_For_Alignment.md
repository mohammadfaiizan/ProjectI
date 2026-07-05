## Interpretability Fundamentals for Alignment

### 0. Scope, and Why This File Sits Where It Does

This file covers mechanistic interpretability specifically as it bears on alignment -- not interpretability as the older, broader ML-explainability tradition (saliency maps for an image classifier, SHAP values for a tabular model, attention-weight visualization treated as if it were self-evidently explanatory). That tradition is largely a separate research lineage with looser standards of causal evidence and a different purpose: it usually aims to build user or practitioner trust in a specific prediction, not to verify claims about a model's goals or internal computation.

The alignment-relevant version of interpretability, generally called mechanistic interpretability ("mech interp"), asks a narrower and harder question: what algorithm is this network actually running, at the level of specific weights, activations, and causal dependencies. It holds itself to a correspondingly stricter evidentiary bar, spelled out in Section 2.3.

This material is a direct prerequisite for File 005's treatment of deceptive alignment. That file works through, in depth, a specific theoretical failure mode: a model that has learned to produce good-looking behavior under conditions it recognizes as evaluation or training, while behaving differently under conditions it believes are unmonitored deployment.

This file's job is narrower -- to explain, from first principles, why ordinary behavioral evaluation cannot by construction rule that failure mode out, and to introduce the research program aimed at getting a different kind of evidence. Section 1 gives that argument only to the depth needed to motivate the rest of the file; it does not re-derive deceptive alignment's mechanics, training-dynamics arguments, or the debate over how likely the failure mode actually is in practice.

### 1. Why Interpretability Matters for Alignment, Specifically

Start with what an ordinary evaluation actually measures. You construct a set of inputs, observe the model's outputs on them, and score those outputs against some criterion -- correctness, helpfulness, refusal-appropriateness, absence of some specific harmful behavior. This is, unavoidably, a statement about the model's *behavior on the inputs you tested*.

It is not, and cannot by itself be, a statement about *why* the model produced that behavior -- what internal computation, representation, or (loosely) objective generated it. Behavioral testing is a black-box method by definition: it characterizes the input-output function at the points you happened to sample it, and says nothing directly about the function's internal implementation.

This gap is usually inconsequential in ordinary ML debugging. If you are checking whether an image classifier reliably separates cats from dogs, you mostly do not care whether it does so via some concept resembling "cat-ness" or via a brittle shortcut (background texture, a watermark artifact), because a sufficiently broad, well-constructed held-out evaluation is reasonable, if imperfect, evidence of generalization. There is no reason to expect the classifier is *adversarially* conditioning its behavior on whether it detects that it is being tested.

**1.1 The specific structural argument.** The alignment case differs in one specific, structural way, and it is worth being precise about exactly what that difference is rather than gesturing vaguely at "AI could be sneaky."

The concern is this: a sufficiently capable model could, in principle, acquire -- through entirely ordinary training pressure, with no exotic mechanism required -- an internal representation of facts about its own situation, including whether it is currently in a context that resembles evaluation/training versus a context that resembles deployment with reduced oversight. This is what "situational awareness" means in this literature: learned features that track properties of the model's own circumstances, not just properties of the task content in front of it.

If such a representation exists and is causally wired into the computation that produces outputs, it becomes possible in principle for behavior to differ systematically between "conditions that look like a test" and "conditions that look like unmonitored deployment" -- not necessarily via any explicit, verbally represented intention to deceive, but as a learned correlation of the same basic kind that produces every other learned behavior, between situational features and output policy.

If that is possible, then two very different underlying models -- one that is genuinely aligned (its internal computation, across the full range of situations it will ever encounter, implements something you would actually want to call good values or intentions) and one that has merely learned to produce aligned-looking outputs specifically under conditions resembling the tests you ran -- are, by construction, indistinguishable on the tests you ran. Both hypotheses predict identical outputs on that support of inputs; that is exactly what "learned to look aligned under test conditions" means.

Running more tests of the same general kind -- more prompts, more red-teaming categories, more held-out benchmarks -- does not close this gap, because the hypothesized failure mode is conditioned on the model detecting *the general shape of being tested*, not on the specific content of any one test.

This is a structural, in-principle argument about the limits of behavioral evidence; it is explicitly not a claim that any current production model exhibits this behavior, deliberately or otherwise. There is no public evidence of current frontier models doing this in any goal-directed sense. The claim is narrower and more defensible: as situational awareness and capability increase, this is a failure mode that becomes possible to construct in principle, and behavioral testing alone has no mechanism to detect it if it occurs -- which is reason enough to want a complementary form of evidence now, before it would matter. File 005 covers the arguments for why this concern is taken seriously as a research priority today rather than deferred until it might already be occurring.

**1.2 What interpretability offers instead.** Interpretability is the research program aimed at exactly this complementary form of evidence: instead of asking what the model outputs on some inputs, it asks what the model's forward pass is actually computing -- what representations and sub-computations are present, and what causal role they play in producing behavior.

Evidence of this kind does not have the same blind spot, because it does not depend on the model having been placed in a condition it recognizes as a test; in principle, it lets you check whether the internal computation responsible for an aligned-looking output really is the computation you'd expect from a model that holds the goal it appears to hold, as opposed to some other computation that happens to produce matching outputs on the distribution you sampled.

This is the entire reason interpretability is treated as a structurally distinct category of safety evidence, not merely a more thorough version of behavioral evaluation. It is equally important to be honest, from the outset, about how far the field is from being able to deliver on this promise for any real frontier model -- Section 4 states the gap explicitly. Section 1 explains why the research program is worth pursuing; it is not evidence that the program has succeeded.

| Evidence type | What it directly establishes | Core limitation |
|---|---|---|
| Behavioral evaluation | Output correctness/quality on the specific inputs tested | Cannot distinguish "aligned" from "learned to look aligned under test-like conditions" |
| Red-teaming | Absence of specific known failure modes under adversarial probing | Bounded by red-teamers' imagination; still purely behavioral |
| Linear probes (Section 2.4) | Whether a known, pre-specified concept is linearly decodable from activations | Requires already knowing what concept to look for; says little about causal role |
| SAE features (Section 3) | A candidate decomposition of activations into more monosemantic directions | Dictionary-learning uncertainty; not guaranteed to be the model's "true" units (Section 4.2) |
| Causally validated circuits (Section 2.3) | That a specific component is actually responsible for a specific step of computation | Coverage of any frontier model's total computation remains tiny |

### 2. Mechanistic Interpretability, From First Principles

**2.1 What a "feature" means, and why it isn't "one neuron, one concept."** Early intuitions about neural network internals, carried over largely from vision-model feature-visualization work, often implicitly assumed that a single neuron's activation corresponds to a single interpretable concept: "this neuron detects curves," "this neuron detects the word 'however.'" Modern mechanistic interpretability treats this as, at best, a special case rather than the general rule.

The general unit of analysis is a **feature**: a direction (or, in some documented cases, a more complex geometric structure) in a model's activation space that corresponds to a human-interpretable property of the input or of the computation so far -- a concept, a syntactic role, an abstract property like "this text is being translated," or a behaviorally loaded property like "the assistant is about to hedge."

A feature, on this view, is typically a linear combination across many neurons, not the activation of any single one; a given neuron can participate in many unrelated features simultaneously, contributing a small amount to each.

Some feature geometry is more exotic than a single direction. Work on grokking in small transformers trained on modular arithmetic (Section 2.3) found the model representing numbers using a roughly circular, Fourier-basis-like structure rather than a single linear axis -- so "direction in activation space" should be read as the default, simplest case of a broader idea (structured subspaces), not as the only structure that occurs.

It is also worth flagging that "features are linear directions" is itself a working assumption -- generally called the **linear representation hypothesis** -- rather than a proven law of how neural networks must represent information. The hypothesis is well supported by a large body of empirical results (linear probes routinely decode concepts well; vector arithmetic on activations, in the style of classic word-embedding analogies, often produces semantically sensible results; SAEs built entirely on the linear-direction assumption recover large numbers of apparently clean features), which is why it is the dominant working assumption in the field and the one this file adopts throughout. But it is a simplifying assumption under active empirical scrutiny, not a settled theorem, and the modular-addition circular-feature result above is itself a documented case where the simplest version of the hypothesis, one concept per single linear direction, needed to be generalized to "one concept per low-dimensional structured subspace" to fit the evidence.

**2.2 The superposition problem.** Suppose a network layer has `d` activation dimensions, but the underlying data-generating process the network is trying to represent has meaningfully more than `d` distinct, useful features it could track. If those features are individually sparse -- each one is relevant (nonzero, active) on only a small fraction of inputs, and different features rarely co-occur strongly on the same input -- the network can represent more than `d` features by packing them into `d` dimensions as directions that are not mutually orthogonal, tolerating a controlled amount of interference between features that are almost never simultaneously active.

This is **superposition**: representing more concepts than you have dimensions by exploiting sparsity, at the cost of those directions overlapping rather than forming a clean orthogonal basis.

Anthropic's line of work on this, generally cited under the title "Toy Models of Superposition" (exact authorship and year are not restated here with full certainty, but the framing is well established and widely attributed to Anthropic's interpretability team, circa 2022), demonstrated the phenomenon directly in small, synthetic ReLU networks.

Trained to reconstruct sparse synthetic input features through a narrow bottleneck, the networks reliably learned to represent more features than they had hidden dimensions, precisely by adopting the overlapping-direction strategy above -- and this reliably produced individual neurons that respond to multiple, semantically unrelated combinations of features, a phenomenon called **polysemanticity**, as an almost mechanical consequence of the compression, not because the network is in any sense trying to be inscrutable.

The practical consequence for interpretability work is significant: "look at what makes neuron 437 fire most strongly, across a sample of inputs, and name that as the neuron's concept" is not a reliable method in general, because that neuron's activation is very often a superposition of signals from several unrelated features, and the pattern you observe by eye may be a blend that doesn't cleanly correspond to any single human concept, or corresponds to several concepts your inspection missed because you weren't looking for them.

This is the central methodological reason naive single-neuron interpretability breaks down at any meaningful scale, and it is the direct motivation for the dictionary-learning-based techniques in Section 3: if the model's true feature basis is not aligned with the neuron basis at all, you need a method that can search for and recover the actual, overcomplete feature directions rather than reading them off the coordinate axes the network happens to be parameterized in.

**2.2.1 A minimal illustrative picture of the geometry.** It helps to have a concrete, if deliberately toy, mental picture rather than only the abstract statement "more features than dimensions." Imagine a hidden layer with only 2 activation dimensions, and 3 candidate binary features the network would like to represent -- say, "input mentions a color," "input mentions an animal," "input mentions a number" -- each of which is true on only a small fraction of inputs and rarely co-occurs with the others.

With only 2 dimensions, the network cannot give all 3 features their own orthogonal axis; the best it can do is place the 3 feature directions at roughly 120 degrees from each other around the 2D plane. Any single feature's direction is then non-orthogonal to, and has some nonzero dot product with, the other two -- so activating one feature strongly will produce a small, spurious positive signal along the other two features' directions as well.

Because the features are sparse and rarely co-occur, this cross-talk is usually tolerable: the network is betting that the cases where two features are simultaneously active and genuinely interfere with each other are rare enough that the benefit of representing 3 features instead of being limited to 2 outweighs the cost. This is the geometric essence of what Anthropic's toy-models-of-superposition experiments demonstrated at larger, systematically varied scale, including empirically observed feature-geometry patterns (features arranging into symmetric configurations such as pentagons or higher-dimensional analogues as the ratio of features to dimensions and the sparsity level are varied) -- the 3-features-in-2-dimensions picture above is this file's own simplified illustration of that finding, not a reproduction of the paper's specific reported geometry.

**2.3 Circuits, concrete examples, and what "found a circuit" actually requires.** A **circuit** is a specific sub-computation within the network -- a particular set of components (attention heads, MLP features or neurons, specific residual-stream directions) connected by specific weights -- that has been shown to implement one coherent, describable algorithmic step.

The operative phrase is "shown to implement," because the methodological bar for claiming a circuit exists is causal, not merely observational.

Three concrete, well-documented examples, roughly in order of how cleanly and completely they have been characterized:

- **Induction heads.** Anthropic's transformer-circuits research (the foundational framing paper is generally cited as "A Mathematical Framework for Transformer Circuits," and the induction-head finding specifically as "In-Context Learning and Induction Heads," both circa 2021-2022 -- exact author lists and dates are not restated here with full confidence, but the results themselves are well established and widely cited) identified a two-attention-head pattern, typically spanning two layers, that implements the algorithm "given the current token, find where this same token last occurred earlier in the context, and copy whatever token immediately followed it there." Concretely: on an input containing "... A B ... A", an induction head causes the model to strongly predict "B" next, because "A B" already occurred once. This single mechanism accounts for a large fraction of small transformers' in-context pattern-completion ability, and it was one of the first clean, causally verified multi-component circuits identified in a language model.
- **The indirect-object-identification (IOI) circuit in GPT-2 small.** Published research (generally cited as "Interpretability in the Wild," associated with Redwood Research and collaborators, appearing around 2022-2023 -- hedge on exact citation details) reverse-engineered the specific set of attention heads GPT-2 small uses to solve sentences of the form "When Mary and John went to the store, John gave a drink to ___", where the correct completion is the indirect object ("Mary") rather than the subject mentioned most recently ("John"). The paper identified specific, named functional roles -- heads that detect duplicated names, heads that suppress attention to the duplicated name, heads that "move" the correct name's representation to the output position -- and validated each role's causal contribution individually.
- **Modular-addition circuits in grokking studies.** Work on why small transformers trained on modular arithmetic tasks (e.g., predicting `(a + b) mod p`) exhibit "grokking" -- a long plateau of near-chance test performance followed by a comparatively sudden jump to near-perfect generalization -- found, in research generally associated with Neel Nanda and collaborators around 2023, that the trained network represents the input integers using trigonometric, Fourier-like periodic features and implements addition via trigonometric angle-sum identities in its later layers, rather than via any lookup-table-like memorized structure. This gave a mechanistic, causally grounded account of what changes internally during the grokking transition, rather than treating grokking as a purely descriptive curve-shape phenomenon.

**2.3.0 Why the causal bar is set this high: a cautionary precedent.** It is worth being explicit about why the field insists on causal intervention rather than resting on correlational-looking evidence such as attention weights, since this is a common shortcut candidates reach for and a common source of overclaiming. Attention weights are tempting to read as "explanations" because they are already a normalized, easy-to-visualize distribution over input tokens, and a head that attends heavily to a semantically relevant token looks, at a glance, like it is "using" that token's information. But a body of critical work on attention-as-explanation (generally associated with the finding, sometimes summarized under the heading "attention is not explanation," that attention weights can be substantially altered, or entirely different attention patterns produced, without changing a model's output in NLP classification settings -- exact authorship and year are not restated here with confidence, but the finding itself is well known) showed that attention weights are not reliably faithful to the computation that actually determines the output: a model can produce a similar prediction under very different attention patterns, meaning the visualized pattern was not doing the causal work an observer might have assumed. This is exactly the gap that activation patching is designed to close -- it does not ask "what does this component look like it's attending to," it asks "does intervening on this component's actual value change the output the way the hypothesis predicts," which is a strictly stronger and less foolable form of evidence.

**2.3.1 The causal-validation bar, worked through with illustrative numbers.** What actually licenses calling any of the above a "found circuit," rather than a plausible-sounding story, is **causal validation**, most commonly via **activation patching** (also called causal tracing or interchange intervention). It is worth walking through a stylized, illustrative version of this to make the logic concrete rather than only stating it abstractly.

Suppose you suspect a specific attention head, call it head 9 in layer 5, is the component responsible for copying the indirect object's name in an IOI-style sentence. Construct a "clean" prompt -- "When Mary and John went to the store, John gave a drink to" -- and record the model's logit for the correct completion "Mary" versus the incorrect completion "John"; suppose the clean run gives a logit difference (Mary minus John) of +6.0, a confident correct prediction.

Construct a matched "corrupted" prompt that swaps the two names -- "When John and Mary went to the store, Mary gave a drink to" -- so the correct answer is now "John," and record head 9's activation on this corrupted run.

Now patch: run the clean prompt again, but this time splice in head 9's activation *from the corrupted run* at the corresponding position, leaving every other component untouched, and re-measure the Mary-minus-John logit difference. If your hypothesis is correct -- head 9 is causally responsible for identifying and copying the correct name -- this single-component patch should substantially move the logit difference toward the corrupted run's answer (e.g., from +6.0 down to, illustratively, +1.0 or lower), because you have fed the "wrong" name-identification signal from head 9 into an otherwise-clean computation.

If patching some other, unrelated head produces little to no change in the logit difference, that contrast -- large, hypothesis-consistent effect from patching the candidate component, and no effect from patching a control component -- is what constitutes genuine causal evidence, as opposed to a merely suggestive correlation between the head's attention pattern and the concept you had in mind. (The specific numbers above are illustrative for exposition only; real patching studies report effect sizes as compiled experimental results across many prompts, not a single worked example.)

A more rigorous, systematized version of this idea, developed by Redwood Research under the name **causal scrubbing**, tries to check whether a hypothesized explanation accounts for the *entirety* of a behavior across many inputs, not merely whether it produces the right direction of effect on a handful of hand-picked examples -- addressing the concern that a small number of cherry-picked patching results can create an illusion of a fully understood circuit when the true picture is messier or only partially captured by the stated hypothesis.

**2.4 A simpler, older, complementary tool: linear probes.** Before dictionary learning became the field's central technique, and still widely used alongside it, a **linear probe** is a small supervised classifier (typically just logistic regression, or another linear model) trained to predict a *known, pre-specified* label from a model's activations -- for instance, training a probe to predict "is this text written in French" from residual-stream activations at some layer, and checking whether it achieves high accuracy. If a simple linear probe can decode a concept well, that is evidence the model represents that concept in a roughly linear, linearly decodable way somewhere in its activations.

Probes are cheap, fast, and require no autoencoder training, but they have a structural limitation directly relevant to comparing them against SAEs: a probe can only look for a concept you already thought to label and supply training data for. It is a supervised, hypothesis-confirming tool, not a discovery tool.

SAEs are the unsupervised counterpart: instead of asking "is concept X, which I already suspected, present in these activations," dictionary learning asks "what is the smallest, sparsest set of directions that explains these activations well," without requiring the researcher to have pre-specified any concept at all -- which is exactly why SAEs are the technique associated with genuinely novel feature *discovery* (Section 3), while probes remain the right tool when you already know what you are looking for and just need a fast, well-validated way to check whether and where it is represented. Probes and SAEs are also frequently used together in practice: a probe can be trained on a specific SAE feature's activation to sanity-check that the feature really does track the concept its top-activating examples suggest, across a broader and more systematic evaluation set than manual inspection alone would cover.

**2.4.1 A concrete, safety-relevant example that needs nothing more than a probe-like direction.** Not every safety-relevant interpretability finding requires the full SAE machinery of Section 3. Published research on open-weight chat models (generally associated with a 2024 finding often summarized as "refusal is mediated by a single direction") reported that the tendency of several such models to refuse a harmful request could be traced to a single linear direction in activation space, computed simply as the difference between the mean activation on a set of harmful prompts and the mean activation on a set of harmless prompts -- no dictionary learning required.

Critically, this finding was pushed past the merely correlational stage in exactly the way Section 2.3 insists on: ablating that one direction (projecting it out of the residual stream at inference time) was reported to reliably suppress refusal behavior across a wide range of harmful prompts, while otherwise leaving the model's general capability largely intact, and adding the direction back in the opposite sign could reportedly induce refusal on prompts the model would not normally refuse.

This is a useful, concrete illustration that some safety-relevant behaviors turn out to be mediated by strikingly low-dimensional, causally verifiable structure that a comparatively simple technique can find -- a good counterweight to any impression that mechanistic insight always requires million-feature dictionary learning, and a good illustration that "probe-like" linear-direction-finding and "circuit-like" causal validation (Section 2.3) are not mutually exclusive categories but are often combined in a single piece of research, exactly as this file's own division into Sections 2 and 3 might otherwise suggest they are kept apart.

### 3. Sparse Autoencoders: The Current Central Technique for Recovering Features

**3.1 Motivation, directly from the superposition problem.** Section 2.2 established that a network's true feature basis is very likely an overcomplete, non-orthogonal set of directions scattered across many neurons, not the neuron coordinate axes themselves. This is exactly the setup studied under **sparse dictionary learning** and compressed sensing in classical signal processing and statistics: given many observed vectors, here activation vectors from a real network at fixed dimensionality `d`, recover a larger dictionary of candidate "atom" directions, count `h` with `h` typically much greater than `d`, such that each observed vector can be well approximated as a sparse linear combination of only a few dictionary atoms at a time.

If that structural assumption is roughly correct for a given layer's activations, a method that can recover such a dictionary gives you a better basis to interpret the layer in than the raw neuron basis does -- ideally, one where each recovered direction corresponds to something closer to a single, nameable concept.

A **sparse autoencoder (SAE)** is the specific, currently dominant practical architecture for doing this. Critically, the SAE is not part of the model you are trying to interpret -- it is trained *separately*, after the fact, on activations collected by running the target model over a large corpus of natural inputs and recording its activations, typically residual-stream activations at a chosen layer, though MLP or attention-output activations are also used, as fixed, frozen training data for the autoencoder.

The target model's own weights are never touched; the SAE is purely a downstream analysis tool trained to re-express those recorded activation vectors in a better basis. This point is worth stressing because it is a common point of confusion: training an SAE does not change the model being studied in any way, and a differently-configured SAE trained on the same activations can yield a meaningfully different decomposition (Section 4.2).

**3.2 Architecture and loss, at an illustrative level.** The following sketch is deliberately simplified to convey the conceptual structure -- encoder into a wide, sparse code, decoder back to a reconstruction, plus a sparsity penalty -- and should not be read as a production training recipe; real implementations vary in initialization, activation normalization, dead-feature-resurrection tricks, learning-rate schedules, and, increasingly, the exact sparsity mechanism used (Section 3.5).

```python
# Illustrative sketch only -- not a production SAE training recipe.
#
# x: an activation vector recorded from the target model, shape (d,)
#    (e.g., a residual-stream vector at a chosen layer and token position)
# W_enc: (d, h) encoder weight matrix, h >> d  (an "overcomplete" hidden width,
#         e.g. h = 16*d or more, so there is room for many more candidate
#         feature directions than there are raw activation dimensions)
# b_enc: (h,) encoder bias
# W_dec: (h, d) decoder weight matrix (each row is a candidate "feature direction"
#         in the original activation space -- the recovered dictionary)
# b_dec: (d,) decoder bias

def sae_forward(x, W_enc, b_enc, W_dec, b_dec):
    # Encoder: project the activation into a wide, higher-dimensional space,
    # then apply a nonlinearity that pushes most of the code toward zero.
    pre_activation = x @ W_enc + b_enc          # shape (h,)
    code = relu(pre_activation)                 # shape (h,) -- the "sparse code";
                                                 # ReLU alone already zeroes negative
                                                 # entries, but does not by itself
                                                 # guarantee most entries are zero --
                                                 # that is the sparsity loss term's job

    # Decoder: reconstruct the original activation as a linear combination of
    # dictionary directions (rows of W_dec), weighted by the sparse code.
    x_hat = code @ W_dec + b_dec                 # shape (d,)

    return code, x_hat

def sae_loss(x, code, x_hat, l1_coefficient):
    reconstruction_loss = mean_squared_error(x, x_hat)

    # L1 penalty on the code pushes most hidden units toward exactly zero for
    # any given input, encouraging each activation vector to be explained by
    # only a handful of active dictionary directions rather than a dense
    # combination of all h of them.
    sparsity_penalty = l1_coefficient * sum(abs(code))

    return reconstruction_loss + sparsity_penalty
```

The hidden code's dimensionality `h` being much larger than `d` is the "overcomplete dictionary" -- it gives the autoencoder room to allocate a separate direction to concepts that would otherwise have to share a direction under superposition in the original, narrower activation space.

The reconstruction term keeps the autoencoder honest, since it must still be able to recover `x` from the code, and the `l1_coefficient` term is the knob controlling the trade-off between reconstruction fidelity and how sparse, and hence empirically how individually interpretable, the recovered code tends to be. Too small a coefficient and the SAE barely differs from an ordinary dense autoencoder with polysemantic hidden units of its own; too large and reconstruction degrades to the point where the recovered features stop tracking the real computation.

**3.3 What empirical "success" looks like.** A trained SAE's hidden units are evaluated, informally and formally, along two complementary axes.

First, **manual inspection of top-activating examples**: for a given hidden unit, collect the set of inputs, or token positions within inputs, that produce the highest activation for that unit across a large and diverse evaluation corpus, and check whether those examples share an obvious, nameable, human-interpretable property -- a specific language, a specific syntactic construction, an abstract concept like "text describing a broken promise," or a safety-relevant property like an apparent sycophancy- or refusal-adjacent pattern.

A "successful" feature is one that fires cleanly and consistently on a coherent concept across many such examples, rather than firing on an apparently arbitrary or mixed set of inputs.

Second, and more rigorously, **causal steering**: artificially clamp the feature's code value, setting it to zero or to an unusually large positive value, while running the target model forward with everything else held fixed, and check whether the model's downstream output changes in the direction the feature's apparent meaning would predict.

Clamping a feature that inspection suggested represents, say, a specific persona or a specific safety-relevant behavior, and observing the intended behavior become correspondingly more or less present in generated outputs, is what elevates "this hidden unit seems to fire on X" from a suggestive correlation to evidence of a genuine causal role -- directly analogous in spirit to activation patching's role in circuit validation (Section 2.3.1).

**3.3.1 A concrete, publicly documented instance: "Golden Gate Claude."** Alongside the Scaling Monosemanticity research (Section 3.4), Anthropic published a public-facing demonstration, generally dated May 2024, in which a specific SAE-recovered feature from Claude 3 Sonnet's activations -- one whose top-activating examples were about the Golden Gate Bridge -- was clamped to an artificially high value, and the resulting modified model was made briefly available for public interaction.

The clamped model reliably steered its responses toward the Golden Gate Bridge across an unrelated range of prompts and conversational contexts, in a way clearly visible to ordinary users, not just to researchers running a narrow held-out eval. This is worth naming specifically because it is one of the more legible, memorable, and independently checkable pieces of public evidence that SAE-recovered features can have a real, substantial, and specific causal effect on a frontier model's behavior when clamped -- a genuine data point for the causal-steering side of Section 3.3's two-part validation standard, even though a single, deliberately dramatic public demo of one feature is a long way from a general claim that most or all SAE features behave this cleanly under steering.

**3.4 Scaling milestones, and what they do and don't establish.** The SAE approach was first demonstrated at genuinely toy scale: Anthropic's "Towards Monosemanticity" work, generally dated 2023, trained SAEs on a small, one-layer transformer and showed that many of the resulting dictionary features were far more cleanly interpretable than the raw, polysemantic neurons of the same model -- the first clear empirical demonstration that dictionary learning could recover more monosemantic units than the network's native basis provided.

The technique was then scaled substantially. Anthropic's "Scaling Monosemanticity" work, generally dated 2024, applied SAEs to residual-stream activations of a genuinely production, frontier-scale model -- Claude 3 Sonnet -- and reported recovering on the order of millions of candidate interpretable features, spanning concrete concepts, abstract and multilingual/multi-modal concepts, and some features plausibly related to safety-relevant properties, patterns loosely associated with sycophancy, deception-adjacent content, or other behaviors of direct alignment interest, with a subset validated via the causal-steering approach in 3.3.

OpenAI has separately published its own SAE research applying a variant of the technique, using a hard TopK sparsity mechanism rather than an L1 penalty (Section 3.5), to GPT-4-scale activations, with accompanying quantitative analysis of how reconstruction quality and apparent feature interpretability trade off against dictionary size and sparsity level. Exact figures, model identities, and methodological details in both lines of work are as reported by the respective labs in their own publications; this file does not independently verify them, but treats them as the best available public account of where the technique's scaling has actually reached.

The honest scope of what this establishes: it is genuine, concrete evidence that SAE-style feature extraction is not confined to toy models, and that it can be run against real frontier-scale activations and produce large numbers of features that survive both manual inspection and at least some causal-steering validation.

It is not evidence that frontier models are now broadly interpretable, that the recovered features constitute anything like a complete account of the model's computation, or that the feature-finding process is free of the methodological uncertainties discussed in Section 4. Being able to hold both halves of this in mind at once -- real scaling progress, and a still-enormous remaining gap -- is the calibration this entire file is trying to teach.

**3.5 Newer sparsity mechanisms, briefly.** The original SAE recipe's soft L1 penalty has a known drawback: it shrinks the magnitude of every active code entry, including the ones that genuinely should be large, because the penalty applies uniformly rather than only to unwanted small activations, and tuning the L1 coefficient to hit a target sparsity level is itself an imprecise, indirect control.

Two more recent variants address this by moving sparsity into the architecture rather than the loss.

**TopK SAEs**, associated with OpenAI's published SAE work and generally dated 2024, keep only the `k` largest pre-activation values in the code and zero everything else, giving direct, exact control over sparsity without needing to tune a penalty coefficient at all.

**JumpReLU SAEs**, associated with DeepMind's interpretability research and generally dated 2024, use a thresholding nonlinearity that zeroes small activations outright while leaving larger, "real" activations largely unshrunk, aiming to get sparsity's benefits without L1's magnitude-shrinkage side effect. Both are active, still-evolving refinements of the same core dictionary-learning idea introduced in 3.1-3.2, not a different technique in kind.

**3.5.1 A practical training challenge worth naming: dead features.** Independent of which sparsity mechanism is used, SAE training in practice runs into a recurring, purely engineering-level failure mode: a meaningful fraction of the wide hidden layer's units can end up **dead** -- their pre-activation never exceeds zero (under ReLU) or never clears the relevant threshold, across the entire training corpus, so they never fire and never receive a useful gradient signal, effectively wasting that portion of the dictionary's capacity.

This is a distinct problem from the conceptual uncertainties in Section 4.2: it is not a question of whether the recovered features are the "true" units, but simply of whether the optimization is using the full width `h` it was given at all. Common mitigations include periodically detecting dead units and re-initializing them (resampling their encoder/decoder weights toward directions the current dictionary reconstructs poorly), and using an auxiliary loss term that specifically encourages otherwise-dead units to occasionally fire.

This is mentioned here as a concrete illustration that scaling SAEs to frontier-size activations is not just a matter of "run the same toy recipe on more data" -- there is a genuine, still-evolving body of training engineering underneath the clean architecture-and-loss picture in Section 3.2.

**3.6 Labeling millions of features: automated interpretability.** Once an SAE recovers millions of candidate features from a frontier-scale model, as in the Claude 3 Sonnet work cited in 3.4, manual inspection of every feature's top-activating examples by a human researcher is not feasible at that volume.

The practical approach the field has converged on is **automated interpretability**: use a separate language model to look at a given feature's top-activating examples, generate a candidate natural-language description of what the feature appears to represent, and, often, score how well that description predicts the feature's activation on new, held-out text.

This lets researchers triage millions of features and surface the ones most likely to be both cleanly interpretable and safety-relevant for closer, human, causal-steering-backed follow-up.

It is worth being explicit about the added layer of approximation this introduces: an automatically generated label is itself an unverified hypothesis about a feature's meaning, produced by a model rather than derived from a causal experiment, and it inherits whatever blind spots or biases the labeling model has.

Automated interpretability is best understood as a triage and scaling tool that makes the *human-in-the-loop, causally validated* portion of the workflow (Section 3.3) tractable at million-feature scale, not as a substitute for that validation step.

**3.7 From single-layer features toward circuits: attribution graphs.** A natural next question, once you have a large set of per-layer SAE features, is how those features causally compose across layers into the kind of multi-step circuits described in Section 2.3.

Anthropic's more recent published work in this direction, generally associated with the terms "circuit tracing" and "cross-layer transcoders" and dated around 2025, describes building **attribution graphs**: causal graphs connecting features found across multiple layers of a model, intended to trace a chain of feature-to-feature influence from an input to an output for specific example prompts, applied to a production-scale model.

This line of work is genuinely the closest existing bridge between the single-layer feature-extraction results of Section 3.4 and the full, multi-step, causally validated circuits described in Section 2.3, and it is worth naming as the current frontier of the field's attempt to close exactly the cross-layer-integration gap flagged in Section 4.2.

It should be treated as an early, actively developing research direction rather than a demonstrated, general-purpose solution: published examples to date trace specific, individually chosen prompts and behaviors, not a comprehensive account of a model's circuitry, and this file states that with the same hedge on exact publication details applied throughout Section 2 and 3.

### 4. Honest State of the Field

**4.1 What has genuinely been achieved.** Full or near-full causal accounts exist for specific, narrow behaviors in small or toy models -- induction heads, the GPT-2-small IOI circuit, modular-addition circuits in grokking studies -- each validated via activation patching or an equivalent causal-intervention method, not merely observed and narrated.

Separately, SAE-style dictionary learning has been demonstrated to scale from one-layer toy transformers up to genuinely frontier-scale model activations, Claude 3 Sonnet and GPT-4-class models, recovering large numbers of features that hold up under both manual inspection of top-activating examples and at least partial causal-steering validation.

There are also real, demonstrated practical applications of this narrow-slice understanding: using an identified feature or circuit to detect whether a specific concept or behavior is active in a given forward pass, or to causally steer a model's outputs by clamping a specific feature at inference time.

This is a genuine, reproducible capability with some concrete utility for narrowly scoped interventions, even though it falls well short of general-purpose behavioral control.

**4.2 What remains far from solved.** Several distinct gaps are worth naming precisely rather than folding into one vague "still early days" caveat:

- **Coverage.** The fraction of a frontier model's total computation that has been mapped into causally validated circuits is minuscule. Even the IOI circuit, one of the more thoroughly documented circuit-level results in the field, required substantial research effort to characterize one relatively narrow linguistic behavior in GPT-2 small, a model many orders of magnitude smaller than any current frontier system.
- **Coverage, stated as a comparison.** There is no existing account, for any frontier model, of even a meaningful minority of its circuitry, let alone a majority -- the gap between "one well-documented circuit in a small model" and "an account of a frontier model's computation" is not a matter of doing more of the same work, but of a research program that has not yet been demonstrated to scale that far.
- **No weights-to-behavior account exists for any frontier model.** You cannot currently take a production frontier model's full parameter set and produce anything resembling a complete, human-auditable explanation of the algorithm the network as a whole implements.
- **Every result is local, not global.** Every existing interpretability result, however rigorous on its own terms, is a partial, local finding -- one circuit, one layer's worth of SAE features, one attribution graph for one prompt -- not a global account of the model.
- **Dictionary-learning (SAE) uncertainty, part 1: no ground truth to check against.** The features an SAE recovers are a function of choices the researcher makes -- dictionary width `h`, sparsity level/mechanism, training corpus -- and there is no independent ground truth to check the resulting decomposition against, because it is not settled that there is a single "true" feature basis to recover in the first place.
- **Dictionary-learning (SAE) uncertainty, part 2: feature splitting and configuration-dependence.** Empirically, features are observed to **split** into more fine-grained subfeatures as dictionary width is scaled up, and to merge or blur together at smaller widths, which means a specific published feature identification is, to a real degree, contingent on the specific SAE configuration used to find it rather than an obviously unique discovery of a ground-truth computational unit. This is a live, actively debated methodological question within the field, including among the same researchers publishing the scaling results in Section 3.4, about how much interpretive weight SAE feature labels can bear: are they the model's actual internal ontology, or a useful, approximately faithful, but ultimately basis-dependent summary of it. This file takes no side beyond stating that the question is open and that treating SAE feature names as literally, uniquely correct claims about the model's internals overstates current evidence.
- **Cross-layer and circuit-level integration of SAE features remains comparatively immature.** Extracting a large number of interpretable single-layer features is a different, and more mature, achievement than showing how those features causally compose, across layers and attention/MLP interactions, into full, validated multi-step circuits at frontier scale.
- **The current attempt to close that gap.** Attribution-graph work (Section 3.7) is the most concrete current attempt to bridge this gap, but it remains early-stage and example-by-example rather than comprehensive.
- **Automated labels are unverified hypotheses, not ground truth.** As flagged in Section 3.6, the scale at which frontier-model features get labeled depends on automated interpretability.
- **Why that matters for this list.** Automated labeling adds a further, largely unaudited layer of approximation on top of the dictionary-learning uncertainty already present in the SAE itself.
- **Aspiration versus capability.** The thing Section 1 motivated -- using interpretability to verify that a model's internal computation genuinely implements the goals its behavior suggests, as independent evidence against the deceptive-alignment-style blind spot in behavioral testing -- is not a capability any lab currently has for any frontier model.
- **What it is instead.** It remains a long-term research aspiration with genuine, publicly demonstrated partial progress toward the underlying tools (Sections 2 and 3), not something that can be run today as an end-to-end verification procedure.

**4.2.1 A further, more philosophical limit worth flagging.** Even a hypothetical future version of interpretability that fully closed every gap in the list above -- complete circuit coverage, a settled and unique feature basis, fully causally validated end to end -- would tell you *what* a model represents and computes, not, by itself, whether the goals or values it turns out to represent are the ones you actually want it to have.

Verifying "this model's internal computation faithfully implements the objective it appears to pursue, in every situation, rather than only appearing to under tested conditions" (Section 1's aspiration) is a different and more tractable question than "is that objective itself good, complete, and safe to hand off increasing amounts of autonomy to," which is fundamentally a value-specification and alignment-target problem, not an interpretability problem.

Interpretability is aimed squarely at the first question; it is not, and does not claim to be, a solution to the second. This distinction is worth holding onto precisely because the two are easy to conflate in casual discussion of "solving interpretability" as if it were equivalent to "solving alignment."

**4.3 Connection to safety cases.** Frontier labs' public materials on deployment safety, Anthropic's discussion of safety cases under its responsible-scaling framing being the most explicit public instance, describe interpretability as a potential *future* component of a deployment safety case -- an additional, independent line of evidence that could one day supplement behavioral evaluations and red-teaming, precisely because it does not share behavioral testing's structural blind spot (Section 1).

It is important to be precise about the present tense here: no lab currently claims to rely on interpretability findings as a primary or load-bearing safety guarantee for any actual frontier deployment decision. Today's safety cases rest on behavioral evaluation, red-teaming, and capability-threshold-triggered safeguards; interpretability is discussed, by the same researchers producing the results in Section 3, as a promising but not yet load-bearing supplementary signal -- an honest characterization of where the field's own practitioners place it, not an external critique of insufficient effort.

### 5. Staff-Level Synthesis and Common Interview Missteps

A strong answer on this topic keeps several distinctions crisp rather than reaching for a single soundbite:

- **Motivation, precisely stated.** The reason interpretability matters for alignment specifically is not "black boxes are inherently bad practice" -- it is the specific, structural argument that behavioral evaluation cannot distinguish a genuinely aligned model from one that has learned to look aligned under conditions resembling the tests you ran, if situational awareness of that kind is possible.
- **Motivation, the calibration that goes with it.** Being able to state the argument above precisely, and to immediately flag it as a theoretical concern about a class of failure rather than an empirical claim about current models, is the single most important calibration point in this file.
- **Feature versus neuron.** Be able to state why "one neuron, one concept" is the wrong default assumption -- superposition, not any claim about networks being deliberately obfuscated, is the reason.
- **What that motivates.** Be able to connect that reason directly to why the field searches for a different, recovered basis (SAEs) rather than treating raw activations as directly readable.
- **Probes versus SAEs.** Be able to say precisely why these are different tools: probes are supervised and hypothesis-confirming, requiring a pre-specified concept and labeled data; SAEs are unsupervised and discovery-oriented, surfacing candidate features you did not already know to look for.
- **The common imprecision to avoid.** Treating probes and SAEs as interchangeable, or describing an SAE as "just a fancier probe," is a common and telling imprecision that signals a shallow grasp of what each tool is actually for.
- **The causal-validation bar.** "This component's activation correlates with concept X" is not the same claim as "this component is a verified part of a circuit implementing X."
- **What closes that gap.** The difference is activation patching, or an equivalent causal intervention, showing that manipulating the component changes behavior in the hypothesis-predicted way, ideally checked against a method like causal scrubbing that guards against cherry-picked confirmatory examples.
- **SAEs are a tool with a known, actively debated limitation, not a solved decomposition method.** Be ready to name feature splitting and dictionary-configuration-dependence specifically, rather than presenting SAE feature labels as settled ground truth about a model's internals.
- **The extra layer on top.** Be ready, in the same breath, to name automated interpretability's own added layer of unverified approximation (Section 3.6) on top of the SAE's own dictionary-learning uncertainty.
- **Calibration on scale, the overclaiming direction.** Be precise that SAEs have been run on frontier-scale activations, Claude 3 Sonnet and GPT-4-class models, with real, publicly reported results -- this is not toy-model-only work anymore -- while being equally precise that circuit-level, causally validated understanding remains concentrated in small models, and that no frontier model has anything close to a complete interpretability account. Conflating "SAEs work at frontier scale" with "frontier models are now interpretable" is the most common overclaiming trap on this topic.
- **Calibration on scale, the underclaiming direction.** The opposite mistake is conflating "the field started with toy one-layer models" with "the field hasn't progressed past toy models" -- the scaling results in Section 3.4, and the attribution-graph work in Section 3.7, are real progress that a strong answer should be able to cite specifically rather than dismiss.
- **Safety-case framing.** Be able to state that interpretability is discussed as a prospective future input to deployment safety cases, not a current load-bearing safety guarantee at any lab today.
- **Why that framing is fair, not a critique.** Be able to explain why this is a defensible, honest characterization rather than a criticism of insufficient progress, given the coverage and dictionary-learning-uncertainty gaps documented in Section 4.2.

### 6. Confirmed Findings vs. This File's Own Synthesis

Because this file leans heavily on published research, it is worth separating, explicitly, what is an established finding from a named line of work versus what is this file's own framing or synthesis of the field's limitations.

**Established, published findings (with the hedges on exact citation details already noted throughout):**
- Superposition, as a mechanism by which networks with fewer dimensions than concepts represent more concepts than dimensions via overlapping, non-orthogonal directions, and its consequence of polysemantic neurons -- Anthropic's toy-models-of-superposition line of work.
- Induction heads as a causally validated, two-head, cross-layer circuit implementing a "copy what followed this token last time" algorithm in small transformers.
- The GPT-2-small IOI circuit, with named, individually causally validated component roles (name-mover, duplicate-token-detection, and related heads).
- Modular-addition circuits using trigonometric/Fourier-like representations, and their use in giving a mechanistic account of grokking.
- SAEs recovering large numbers of features, judged interpretable via top-activating-example inspection and partially validated via causal steering, first at one-layer toy-model scale and subsequently at frontier scale (Claude 3 Sonnet; GPT-4-class models via OpenAI's own published SAE work).
- The "Golden Gate Claude" public demonstration as a specific, publicly documented instance of a clamped SAE feature producing a large, visible causal effect on a frontier model's behavior.
- The "refusal is mediated by a single direction" finding on open-weight chat models, including its causal validation via ablation and re-addition of the identified direction.
- TopK and JumpReLU as published, named alternatives to the original L1-penalty SAE sparsity mechanism.
- Automated interpretability (using a model to generate and score candidate feature descriptions) as the field's practical answer to labeling feature counts too large for manual review.
- Attribution graphs / circuit tracing / cross-layer transcoders as a recent, published research direction aimed at connecting single-layer SAE features into multi-layer causal accounts.
- Labs' public framing of interpretability as a prospective, not-yet-load-bearing, component of future deployment safety cases.

**This file's own synthesis, rather than a claim attributable to any single published source:**
- The framing of Section 1 as the specific reason interpretability is categorically different from ordinary ML explainability, stated in terms of the behavioral-evidence blind spot -- this is a standard argument in the alignment research community, but the particular exposition here is this file's synthesis, not a direct quotation from any one paper.
- The characterization, in Section 4.2, of exactly how much of the field's own open methodological debate (feature splitting, dictionary non-uniqueness, automated-label reliability) should be weighed against the genuine scaling progress in Section 3.4 -- the underlying facts are drawn from the field's own publications and stated debates, but the specific calibration ("real progress, and a still-enormous remaining gap, held simultaneously") is this file's synthesis of how to weigh them for interview purposes.
- The explicit statement that no lab relies on interpretability as a primary safety guarantee today is a reasonable, current reading of labs' own public statements as of this file's writing, not a verbatim claim quoted from any single document, and it should be treated as time-sensitive: it is the kind of claim that could become outdated as the field's tools mature.
