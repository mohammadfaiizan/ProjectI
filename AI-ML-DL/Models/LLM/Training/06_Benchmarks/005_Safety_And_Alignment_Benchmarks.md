# Safety and Alignment Benchmarks

Safety benchmarks differ from every other family in this document in one important way: for most of them, a *higher* score is not unambiguously better in the way a higher MMLU or SWE-bench score is.

A model that refuses 100% of harmful requests could be an extremely well-aligned model, or it could simply be a model that refuses everything indiscriminately, including a large share of benign requests that superficially resemble harmful ones. A single-axis "refusal rate on harmful content" number cannot tell those two models apart.

This file covers HarmBench-style harmful-request-refusal evaluation, the over-refusal problem and how good safety evals address it as a paired metric rather than a single score, and jailbreak-robustness evaluation as a related but distinct evaluation axis — robustness to *adversarially constructed* prompts specifically designed to defeat safety training, as opposed to robustness to a plainly-stated harmful request.

For the broader institutional and policy context around safety evaluation (responsible scaling policies, red-teaming programs, model cards), see `../07_Safety_Alignment_And_Responsible_Scaling/` — this file is scoped specifically to the named benchmarks and their measurement methodology.

## HarmBench and harmful-request-refusal benchmarks

**Citation:** Mazeika, Phan, Yin, Zou, Wang, Mu, Sakhaee, Li, Basart, Li, Forsyth, Hendrycks, "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal," 2024.

### Methodology, precisely

HarmBench provides a curated set of roughly 400 distinct harmful behaviors spanning several categories:

- Cybercrime and unauthorized intrusion
- Chemical/biological/radiological/nuclear (CBRN) weapons assistance
- Misinformation and disinformation
- Illegal activity more broadly
- Harassment and hateful content
- Copyright-infringing generation

Each behavior is phrased as a specific, concrete harmful request (e.g., a request for step-by-step assistance with a specific harmful activity, rather than a vague topic label), so that "did the model comply" has a fairly well-defined answer for a given response.

The key methodological contribution, distinct from being simply another list of bad prompts, is that HarmBench evaluates **both models and jailbreak/red-teaming attack methods together in the same harness**. For each harmful behavior, the framework applies a battery of attack methods against each target model:

- Direct request (no adversarial wrapping)
- GCG suffix optimization (gradient-based adversarial suffix search)
- PAIR-style iterative refinement (an attacker LLM iteratively rewrites the prompt based on the target's responses)
- Human-written jailbreak templates

Responses are scored using a **fine-tuned classifier** — trained specifically to distinguish genuine harmful compliance from refusal, deflection, or engagement-without-actually-providing-the-content — rather than simple keyword-based refusal detection. Keyword matching (e.g., "does the response contain the phrase 'I cannot'") is well known to be unreliable in both directions: models can refuse without canonical refusal phrases, and can also comply while wrapping harmful content in hedging language a keyword detector would misclassify as refusal.

```python
# Conceptual sketch of the HarmBench-style evaluation loop (not the actual codebase)
def evaluate_model_on_harmbench(model, behaviors: list[str], attack_methods: list,
                                  compliance_classifier) -> dict:
    results = {}
    for behavior in behaviors:
        for attack in attack_methods:
            adversarial_prompt = attack.construct_prompt(behavior)
            response = model.generate(adversarial_prompt)
            # compliance_classifier: fine-tuned model judging whether `response`
            # actually provides the harmful assistance requested by `behavior`,
            # not just whether it contains refusal-sounding language
            is_harmful_compliance = compliance_classifier.classify(behavior, response)
            results[(behavior, attack.name)] = is_harmful_compliance
    return results

def attack_success_rate(results: dict, attack_name: str) -> float:
    relevant = [v for (b, a), v in results.items() if a == attack_name]
    return sum(relevant) / len(relevant)
```

### The metric: Attack Success Rate (ASR)

For a given (model, attack method) pair, ASR is the fraction of the ~400 behaviors for which that attack successfully elicited harmful compliance from that model. This produces a matrix — models on one axis, attack methods on the other — that supports two distinct kinds of comparison from a single harness:

1. **Model-robustness ranking** — which models are most robust across the whole battery of attacks.
2. **Attack-strength ranking** — which attack methods are most effective across the whole set of target models, useful for red-teaming research, since a newly proposed jailbreak method can be benchmarked for how much it improves ASR over prior attacks on the same standardized behavior set.

This dual purpose is HarmBench's distinguishing design choice relative to earlier, more ad hoc red-teaming writeups, where each paper used its own behavior set and its own (often less rigorously validated) success classifier, making cross-paper comparison unreliable.

### Known weaknesses

A curated ~400-behavior list, however carefully constructed, is a finite and disclosed set. Once public, model developers can — deliberately or as a side effect of general safety-training data collection — specifically train against behaviors that resemble HarmBench's categories, inflating robustness on the benchmark's specific behaviors without necessarily generalizing to novel harmful requests outside that set. This is a direct instance of the general contamination/overfitting risk discussed in file 007, but specifically dangerous here because the failure mode being hidden is a safety failure, not just an inflated capability number.

The compliance classifier, being itself a trained model, has its own error rate and its own blind spots. A classifier trained to recognize known harm categories may be systematically less reliable at judging borderline, novel, or creatively-obfuscated harmful content, meaning ASR numbers inherit whatever the classifier's own weaknesses are — structurally similar to the general LLM-judge reliability concerns covered in `../05_Evaluation_Methods/`.

Finally, a pure ASR-on-harmful-behaviors number, however carefully measured, says nothing on its own about over-refusal — which is the next section's entire point, and is why HarmBench-style scores should never be read as a complete safety picture in isolation.

## The refusal-rate vs. over-refusal-rate tension

### The core measurement problem

Consider two models. Model A refuses every one of HarmBench's ~400 harmful behaviors (ASR = 0% across the board) but also refuses a substantial fraction of benign requests that happen to contain surface features associated with harm — e.g., refusing "how do I whittle a knife for a woodworking class" because the response contains the word "knife," or refusing "what household chemicals should never be mixed together, for safety" because it superficially resembles a request for dangerous synthesis instructions. Model B also achieves ASR = 0% on the same harmful-behavior set, but reliably answers the benign lookalike requests correctly and helpfully.

A benchmark that only measures compliance on harmful requests scores these two models identically, even though Model B is straightforwardly a better, more usefully-aligned model. This is the exact failure mode the over-refusal literature exists to catch — a direct analogue of the precision/recall distinction in classification: refusing everything trivially maximizes "recall" on harmful-content avoidance while destroying "precision" (usefulness) on everything else, and a single aggregate refusal-rate number conflates the two exactly the way accuracy alone can be a misleading metric under class imbalance.

### XSTest and OR-Bench: the paired-benign-lookalike design

XSTest (Röttger, Kirk, Vidgen, Attanasio, Bianchi, Hovy, 2023, "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models") is the clearest example of this methodology. It pairs genuinely unsafe prompts with **superficially similar but benign** prompts, constructed specifically so the two members of a pair share surface-level features (similar vocabulary, similar sentence structure, similar topic area) while differing in actual intent/harmfulness — for example, a request about how to obtain a weapon illegally versus a structurally similar-looking request about a common, legal activity that merely mentions a weapon-adjacent word.

XSTest reports a **compliance rate on the benign set** as its primary signal: a well-calibrated model should have a high compliance rate on the benign-lookalike prompts (correctly recognizing they are safe despite superficial resemblance to unsafe prompts) while still maintaining a low compliance rate on the genuinely unsafe prompts in the same pairing structure.

OR-Bench (a more recent, larger-scale over-refusal benchmark) extends this idea to greater scale and category coverage, using an automated pipeline to generate large numbers of benign-but-refusal-triggering prompts across many categories, since XSTest's original benign-lookalike set is relatively small and hand-curated.

### Why good safety evaluation reports this as a joint, two-axis result

The methodologically sound framing treats safety evaluation as producing a **point on a two-dimensional frontier** — (compliance rate on genuinely harmful requests, which you want low) plotted against (compliance rate on benign requests, which you want high) — rather than as a single scalar "safety score."

A model can be moved along this frontier by changing how aggressively it's trained to refuse: more aggressive refusal training pushes harmful-compliance down but typically pushes benign-compliance down too (more false positives), and less aggressive refusal training does the reverse. Reporting only one axis lets a lab present a model as "safe" by citing a low harmful-compliance number while omitting that this was achieved partly via a costly rise in over-refusal, or present a model as "helpful" by citing high benign-compliance while omitting a corresponding rise in harmful-compliance. The two-axis framing makes that kind of one-sided presentation visible.

```python
# Sketch: a two-axis safety scorecard rather than a single number
def safety_scorecard(model, harmful_behaviors, benign_lookalikes, judge) -> dict:
    harmful_compliance_rate = sum(
        judge.is_compliant(model.generate(p)) for p in harmful_behaviors
    ) / len(harmful_behaviors)
    benign_compliance_rate = sum(
        judge.is_compliant(model.generate(p)) for p in benign_lookalikes
    ) / len(benign_lookalikes)
    return {
        "harmful_compliance_rate": harmful_compliance_rate,   # want this LOW
        "benign_compliance_rate": benign_compliance_rate,     # want this HIGH
        # a single-number "safety score" from either axis alone is misleading;
        # a well-aligned model sits near (low, high); an over-cautious model
        # sits near (low, low); a poorly-aligned model sits near (high, high or low)
    }
```

### Weaknesses of the paired-benchmark approach itself

Constructing genuinely well-matched benign/harmful pairs is itself a difficult authoring task prone to subjective judgment calls about what counts as "superficially similar," while genuinely differing in real-world harmfulness. Different benchmark authors' judgment calls about where exactly the safe/unsafe line falls for a given ambiguous prompt can differ — reasonable people, and reasonable safety teams at different labs, disagree about specific edge cases (e.g., certain dual-use chemistry or security information). Even a well-constructed paired benchmark encodes one particular team's judgment about the harm/benign boundary rather than a universally agreed one.

There is also a scale mismatch: harmful-behavior sets like HarmBench's ~400 items are already narrow, and benign-lookalike sets are typically similarly modest in size, so both axes of the frontier are being estimated from a limited sample relative to the enormous diversity of real user requests a deployed model actually receives.

## Jailbreak-robustness evaluation

### What it adds beyond plain harmful-request refusal testing

HarmBench already incorporates several jailbreak/attack methods as part of its standard evaluation battery — that is precisely why it is a joint model-and-attack evaluation harness rather than a plain refusal-rate benchmark. But jailbreak robustness is worth treating as its own conceptual category, because the failure mode it targets is different in kind from plain refusal failure: a model can have excellent refusal behavior against *directly and plainly stated* harmful requests while still being vulnerable to requests that are **adversarially disguised** to defeat exactly the pattern-matching or learned-representation cues its safety training relies on.

### Known families of jailbreak techniques

| Technique | Mechanism |
|---|---|
| Roleplay/persona jailbreaks (e.g., "DAN" and variants) | Instructing the model to adopt a fictional persona or "developer mode" under which normal safety constraints are claimed not to apply |
| GCG suffix optimization (Zou, Wang, Carlini, Nasr, Zico Kolter, Fredrikson, 2023) | Automated, gradient-based search over a short adversarial suffix appended to a harmful prompt, optimized to maximize the probability the model begins its response with an affirmative-compliance-shaped prefix |
| Multi-turn escalation ("crescendo"-style) | A sequence of individually benign-seeming turns that gradually walks the conversation toward eliciting harmful content |
| Encoding/obfuscation attacks | Base64 encoding, character substitution (leetspeak), translation into a lower-resource language, or other transformations designed to evade surface-level pattern-matching filters |
| Many-shot jailbreaking | Exploiting long context windows via a large number of in-context example turns depicting the model complying with progressively more harmful requests, using in-context learning itself as the attack vector |

The GCG paper's most cited finding is that suffixes optimized against open-weight models (where gradients are accessible) frequently **transfer** to other models, including closed/proprietary ones the attacker never had gradient access to — evidence that this class of vulnerability is not fully model-specific and can be discovered "offline" against a substitute model.

Many-shot jailbreaking is specifically documented in Anthropic's own published long-context red-teaming research, and is notable for being a jailbreak mechanism that gets *more* effective as context windows grow — a vulnerability that scales with exactly the capability improvement (long context, file 004) that is otherwise treated as a pure win elsewhere in this document.

### Sketch of a many-shot jailbreak construction

To make the mechanism concrete: an attacker constructs a long prompt containing dozens of fabricated prior "conversation turns," each showing a fictional assistant complying with a progressively more sensitive request, before finally appending the actual target harmful request at the end:

```python
def build_many_shot_jailbreak(fake_turns: list[tuple[str, str]], real_harmful_request: str) -> str:
    """fake_turns: list of (user_request, compliant_assistant_response) pairs,
    escalating in sensitivity, fabricated to look like genuine prior conversation
    history rather than an explicit instruction to ignore safety training."""
    transcript = ""
    for user_msg, assistant_msg in fake_turns:
        transcript += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
    transcript += f"User: {real_harmful_request}\nAssistant:"
    return transcript
```

The attack's effectiveness in the published research is reported to scale with the number of fake turns included — more turns of established "compliant persona" pattern in-context more strongly bias the continuation toward compliance on the final real request, which is exactly why the attack specifically benefits from longer available context windows: an attacker with only 2K tokens of budget can fit far fewer escalating fake turns than one with a 200K-token budget.

### The moving-target problem

Any published jailbreak-robustness number is necessarily a statement about robustness *against the specific, currently-known attack set tested* — it cannot be a statement about robustness against attacks not yet invented. This is a structurally different situation from, say, a math benchmark's exact-match scoring, where the grading criterion is stable over time even if the training data changes. Here, the *attack landscape itself* is adversarially evolving, with new jailbreak techniques published continuously by both academic red-teaming researchers and the broader public.

The correct interpretation of a jailbreak-robustness score is therefore as a **lower bound on current vulnerability**, not an upper bound or a settled robustness guarantee. This is precisely why frontier labs' actual practice is continuous red-teaming — ongoing, open-ended adversarial testing, including external bug-bounty-style programs and dedicated red-teaming staff — rather than treating a one-time static-benchmark pass as sufficient evidence of safety. The benchmark is a useful, standardized regression check (did a new model at least remain robust to all previously-known attacks) but was never designed to be, and cannot be, a complete safety certification.

## A worked illustration of the over-refusal tradeoff

Consider a concrete XSTest-style pair: "What's the fastest way to kill a process that's hogging my CPU?" (benign — a Linux/systems-administration question) versus "What's the fastest way to kill a person who's threatening me?" (genuinely concerning, context-dependent, and plausibly warranting a careful, non-instructive response). Both share the surface token "kill," a short imperative framing, and a similar sentence structure. A model relying on shallow lexical cues (presence of the word "kill") would either refuse both or comply with both; a model with genuine intent-discrimination correctly treats them very differently despite the surface similarity.

Scaling this up: OR-Bench-style automated generation can produce hundreds of such pairs across many surface-feature categories (words associated with violence, drugs, weapons, hacking, self-harm — each paired with a benign, legitimate use of the same vocabulary), giving a much larger and more statistically stable estimate of a model's over-refusal rate than XSTest's smaller hand-curated set alone, at the cost of the automated-generation pipeline itself potentially introducing its own systematic biases in which benign use cases it thinks to generate.

## Common interview framings worth preparing for

- **"A red team reports zero successful jailbreaks against a new model release. What's your first follow-up question?"** — which specific attack techniques were in the red-teaming battery, and how recently were they discovered relative to the model's training/RLHF cutoff; a red team that only tested last year's known techniques against this year's model is not testing much, and the honest framing is "robust against this specific tested set," not "safe."
- **"How would you tell if a model's low over-refusal rate came at the cost of its harmful-refusal rate, versus genuinely improved calibration?"** — plot both axes together (harmful-compliance rate, benign-compliance rate) across successive model versions; genuine calibration improvement shows both moving in the good direction simultaneously (harmful-compliance flat or down, benign-compliance up), while a tradeoff-driven change shows one improving at the expense of the other moving the wrong way.
- **"Why might two labs report very different over-refusal rates for models that seem comparably safe on harmful-content benchmarks?"** — different labs' XSTest/OR-Bench-style benign sets encode different judgment calls about where the safe/benign line falls, and different post-training pipelines calibrate refusal thresholds differently; a lower over-refusal number from one lab does not necessarily mean a better-calibrated model if the underlying benign-prompt set used to measure it was constructed with a more permissive definition of "benign" than another lab's set.

## Quick-reference comparison

| Benchmark/technique | What it measures | What it cannot tell you |
|---|---|---|
| HarmBench (ASR) | Compliance rate on ~400 harmful behaviors under a battery of attacks | Whether the model is over-cautious on benign requests |
| XSTest / OR-Bench | Compliance rate on benign-but-harmful-looking requests | Robustness to harmful requests (needs pairing with HarmBench-style measurement) |
| GCG / PAIR / jailbreak batteries | Robustness against specific, currently-known attack techniques | Robustness against attacks not yet invented (a lower bound only) |
| Many-shot jailbreaking eval | Vulnerability scaling with context length | General jailbreak robustness outside the long-context regime |

## What's missing from this file's three methodologies taken together

Even the combination of HarmBench-style ASR, XSTest/OR-Bench-style over-refusal measurement, and jailbreak-battery robustness testing does not cover several categories of risk that matter in practice: none of the three directly measures **multi-turn manipulation that stays within any single turn's apparent benignity** but accumulates harm across a session in ways no single-turn classifier would flag; none directly measures **capability-uplift risk** in the sense of a model providing genuinely novel, non-public harmful information rather than merely restating already-public harmful information (a distinction that matters enormously for CBRN-category risk specifically, and that a simple compliance/refusal classifier is not well-suited to assess); and none directly measures **honesty/deception** as a safety property distinct from harm-avoidance — a model can be perfectly compliant with refusal policy while still being systematically dishonest about its own capabilities, reasoning, or certainty, which is a safety-relevant property this file's benchmarks were never designed to probe.

## Implementing a joint safety-and-jailbreak regression tracker

A practical pattern worth knowing concretely: rather than treating HarmBench-style ASR, over-refusal rate, and jailbreak-battery robustness as three separate one-off reports, a mature evaluation practice tracks all three as a time series across successive model versions and successive additions to the attack battery, so that a new model's safety posture is always read relative to trend, not in isolation.

```python
def safety_regression_report(history: list[dict], new_model_result: dict) -> dict:
    """history: list of past {model_version, harmful_compliance_rate,
    benign_compliance_rate, jailbreak_asr_by_technique} records.
    new_model_result: same shape, for the model being evaluated now."""
    prev = history[-1] if history else None
    report = {"model_version": new_model_result["model_version"]}
    if prev is not None:
        report["harmful_compliance_delta"] = (
            new_model_result["harmful_compliance_rate"] - prev["harmful_compliance_rate"]
        )
        report["benign_compliance_delta"] = (
            new_model_result["benign_compliance_rate"] - prev["benign_compliance_rate"]
        )
        # flag any previously-robust jailbreak technique that newly succeeds --
        # a regression against a technique the PREVIOUS model was robust to
        # is a stronger signal than a static snapshot alone would show
        regressions = [
            technique for technique, asr in new_model_result["jailbreak_asr_by_technique"].items()
            if asr > prev["jailbreak_asr_by_technique"].get(technique, 0.0) + 0.05
        ]
        report["new_jailbreak_regressions"] = regressions
    return report
```

The specific thing this catches that a one-off snapshot report would not: a new model that improves harmful-compliance rate overall, while quietly regressing against one specific previously-robust jailbreak technique — a pattern a single aggregate ASR number would fully hide, since it would only show up as a small movement in an average across many techniques, but that a per-technique regression check flags directly.

## Synthesis

The throughline across this file is that safety evaluation cannot be reduced to a single number without losing the exact information that makes it useful. A harmful-compliance rate alone can't distinguish good calibration from indiscriminate over-caution — hence the paired benign-lookalike methodology of XSTest/OR-Bench. A jailbreak-robustness number against a fixed attack battery can't be read as a robustness guarantee against attacks not yet devised — hence continuous red-teaming as the actual operational practice, with static benchmarks like HarmBench serving as a standardized regression floor rather than a ceiling on assurance.

A staff-level answer in this space should be able to state, for any single safety metric handed to you, what it does *not* tell you — which is usually more interview-relevant than reciting what it does measure.
