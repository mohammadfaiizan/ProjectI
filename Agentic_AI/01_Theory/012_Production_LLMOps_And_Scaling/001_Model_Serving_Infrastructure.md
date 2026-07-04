# Model Serving Infrastructure

## The Question Behind the Question

Every team that puts an LLM into production eventually has to answer a deceptively simple question: should we call someone else's model over an API, or should we run the model ourselves? This sounds like an infrastructure decision, but it's really an economics and control decision wearing an infrastructure costume. Calling a managed API means renting inference by the token from a provider who has already solved batching, GPU scheduling, quantization, and failover at a scale you will probably never reach on your own. Self-hosting means you own the GPUs, the serving stack, the scaling policy, and every 2 a.m. page when a node falls over — in exchange for control over latency, data residency, customization, and, at sufficient volume, unit economics that no API markup can match.

Neither answer is "more advanced" than the other, and a senior engineer's job is to know which one applies to a given product at a given stage, not to have a permanent allegiance to either. But to make that call intelligently, you first need to understand what a serving framework like vLLM, TGI, or TensorRT-LLM is actually doing under the hood, because those internals are precisely what determine whether self-hosting will be cheaper or more painful than just paying the API markup.

## Why Naive Serving Is Wasteful

If you took a Hugging Face `model.generate()` call and wrapped it in a web server, you would have a working but catastrophically inefficient LLM service. The reason is that autoregressive generation is memory-bandwidth-bound, not compute-bound, per request: producing each new token requires reading the entire model's weights (and the growing KV cache) from GPU memory, but the actual matrix multiplication for a single sequence barely uses the GPU's compute throughput. A modern GPU can do orders of magnitude more FLOPs than a single generation stream will ever ask of it — the bottleneck is moving data, not crunching it.

The fix, as with most underutilized-hardware problems, is batching: process many requests' tokens together so that the one expensive weight-read is amortized across many sequences instead of one. But LLM batching has a wrinkle that batching a normal ML model doesn't: requests don't arrive at the same time, and they don't finish at the same time. One user asks a one-sentence question, another pastes in a 4,000-token document, and a third's request is mid-generation when a fourth walks in. A naive implementation either batches too rigidly (wait for a fixed batch window, waste latency for everyone) or not at all (throughput crawls). Everything interesting in modern LLM serving is downstream of solving this scheduling and memory problem well, and that's exactly what continuous batching and PagedAttention were built to do.

## Continuous Batching

Traditional dynamic batching in classic ML serving works by collecting a batch of fixed-shape inputs, running one forward pass, and returning all outputs together — every request in the batch starts and ends at the same time. Applied naively to LLM generation, this is disastrous: if one sequence in the batch needs 500 tokens and another needs 20, the whole batch is stuck running until the longest sequence finishes, and the GPU slots for the already-finished short sequence sit idle the entire time. This is sometimes called static batching, and it can waste 50%+ of GPU capacity on any workload with varied response lengths, which is essentially every real chat workload.

Continuous batching (also called in-flight batching, the term NVIDIA uses in TensorRT-LLM) fixes this by scheduling at the level of individual decoding steps rather than entire requests. After every single token is generated for every sequence currently in the batch, the scheduler checks: has any sequence finished (hit EOS or its max length)? If so, evict it immediately and admit a new request from the waiting queue into that now-free slot — all before the next token-generation step begins. The batch composition can change every single step. This means a GPU is essentially never idle waiting for a straggler; short requests exit as soon as they're done and their capacity is immediately reused by whatever's next in line, rather than being held hostage by the longest request in an arbitrarily-drawn batch.

```python
class ContinuousBatchScheduler:
    """A simplified mental model of what vLLM/TGI do at every decode step.
    Real implementations also handle prefill/decode phase separation,
    priority, and preemption -- this captures the core idea only."""

    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.active = {}     # request_id -> generation state
        self.waiting = []    # queued requests not yet admitted

    def step(self):
        # 1. Admit new requests into any free batch slots
        while len(self.active) < self.max_batch_size and self.waiting:
            req = self.waiting.pop(0)
            self.active[req.id] = req

        # 2. Run one decode step for every active sequence, batched together
        token_outputs = self.run_one_decode_step(list(self.active.values()))

        # 3. Evict any sequence that just finished, freeing its slot immediately
        for req_id, token in token_outputs.items():
            self.active[req_id].append_token(token)
            if self.active[req_id].is_finished():
                self.emit_response(self.active.pop(req_id))

    def run_one_decode_step(self, batch):
        # In reality: one fused forward pass across all sequences' next-token
        # positions, using the KV cache each sequence has already accumulated.
        ...
```

The throughput gain from continuous batching alone, independent of any memory optimization, is typically cited at 2-4x over naive static batching on chat-style workloads with variable response lengths — this was the central result of the Orca paper that introduced the technique, and it's why every serious serving framework built after 2022 implements some version of it.

## PagedAttention and KV Cache Memory Management

Continuous batching solves the scheduling problem, but it exposes a second problem: memory. Every sequence in flight needs its own KV cache — the stored keys and values for every token generated so far, which every subsequent token's attention computation needs to read. The KV cache for a single sequence grows linearly with sequence length, and its size is unknown in advance, because you don't know how long the model will decide to keep generating.

Before PagedAttention, serving frameworks handled this the way you'd handle any array whose final size is unknown: pre-allocate a contiguous block of GPU memory sized for the worst case (the maximum sequence length the server supports), for every sequence in the batch. This is enormously wasteful. If your max sequence length is 4,096 tokens but the average conversation only produces 200 tokens of actual KV cache, you're reserving 20x more memory per sequence than you need, and that reserved-but-unused memory can't be given to any other request. Worse, this fragmentation compounds: as sequences of different lengths are allocated and freed, you get holes in GPU memory that are too small to fit a new sequence's worst-case allocation even though the total free memory would be plenty if it were contiguous. vLLM's own benchmarks found that naive KV cache management wasted 60-80% of allocated memory to fragmentation and over-reservation, and since KV cache memory is what limits how many sequences you can batch together, wasted memory directly translates to wasted throughput.

PagedAttention, introduced by the vLLM project (UC Berkeley), borrows the solution operating systems settled on decades ago for the analogous problem of managing process memory: virtual memory paging. Instead of one contiguous block per sequence, the KV cache is chopped into fixed-size blocks (say, 16 tokens per block), and a sequence's logical KV cache is a list of pointers to physical blocks that don't need to be contiguous in GPU memory at all. Blocks are allocated on demand, one at a time, as a sequence actually generates more tokens — never speculatively for a worst-case length. When a sequence finishes, its blocks are returned to a free pool immediately, and because any block can be assigned to any sequence, there's no fragmentation problem: free memory is always usable regardless of which sequences it was previously allocated to.

```python
class PagedKVCache:
    """Conceptual sketch of block-based KV cache allocation, as in vLLM."""

    def __init__(self, num_blocks, block_size=16):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))   # pool of physical block IDs
        self.block_tables = {}                        # seq_id -> [physical_block_ids]

    def append_token(self, seq_id, current_length):
        needs_new_block = current_length % self.block_size == 0
        if needs_new_block:
            if not self.free_blocks:
                raise MemoryError("KV cache pool exhausted, must preempt or reject")
            block = self.free_blocks.pop()
            self.block_tables.setdefault(seq_id, []).append(block)
        # Attention kernel gathers K/V from the physical blocks in block_tables[seq_id],
        # which may be scattered across GPU memory -- this indirection is the whole trick.

    def free_sequence(self, seq_id):
        self.free_blocks.extend(self.block_tables.pop(seq_id, []))
```

The other capability this unlocks, which turns out to matter enormously in agentic and RAG workloads, is cheap memory sharing. If two requests share a common prefix — the same system prompt, the same few-shot examples, or (in beam search / parallel sampling) the same partial generation — their block tables can simply point to the same physical blocks for the shared portion, using copy-on-write only when a sequence diverges and needs to write new tokens. This is the same mechanism operating systems use for `fork()` and shared libraries, and it's directly why vLLM can implement highly efficient prefix caching: a shared system prompt across thousands of requests occupies exactly one copy of KV cache blocks in GPU memory, not one copy per request.

## The Major Serving Frameworks

### vLLM

vLLM is the framework most associated with PagedAttention, since its authors invented the technique, and it has become something close to the default choice for self-hosting open-weight models (Llama, Mistral, Qwen, DeepSeek, and most other Hugging Face-format checkpoints). Its core value proposition is throughput per GPU-dollar: continuous batching plus PagedAttention plus a highly optimized CUDA attention kernel routinely gets cited at 2-24x higher throughput than a naive Hugging Face Transformers serving loop, with the wide range depending on workload shape (the more variable your sequence lengths and the more concurrent requests, the bigger the win). It exposes an OpenAI-compatible HTTP API out of the box, which matters practically because it means application code written against the OpenAI SDK can point at a self-hosted vLLM server with essentially no changes. It also supports tensor parallelism (splitting a model's weights across multiple GPUs when it doesn't fit on one), quantized weight formats (AWQ, GPTQ, FP8), speculative decoding, and multi-LoRA serving (hosting many fine-tuned adapters on top of one base model and swapping between them per-request without reloading full weights).

### Text Generation Inference (TGI)

TGI is Hugging Face's own production serving framework, and conceptually it converged on many of the same ideas — continuous batching, an efficient attention implementation via FlashAttention/FlashAttention-2, tensor parallelism, and quantization support (bitsandbytes, GPTQ, AWQ). Where it tends to differentiate is tighter native integration with the Hugging Face ecosystem (loading straight from the Hub, first-class support for the model architectures Hugging Face ships quickly), and it's a common default if your organization is already standardized on Hugging Face tooling for training and fine-tuning, since the same checkpoint format flows straight into serving without conversion steps. Historically TGI's license terms were a consideration for commercial use (it briefly moved to a more restrictive license before reverting to Apache 2.0), which is a reminder that license terms for serving frameworks are worth checking explicitly rather than assumed, since they can change between versions.

### TensorRT-LLM and Triton Inference Server

TensorRT-LLM is NVIDIA's answer, and it takes a different approach to getting speed: rather than being a Python-first serving loop with optimized kernels bolted on, it compiles the model into an optimized inference engine ahead of time, fusing operations, choosing optimal CUDA kernels for the specific GPU architecture you're deploying to, and applying aggressive quantization (down to FP8 or even INT4 with acceptable quality on many models). This ahead-of-time compilation step is the source of both its strength and its friction: compiled engines are typically the fastest option on NVIDIA hardware specifically, often meaningfully faster than vLLM on raw tokens/sec for the same model and GPU, but building an engine is slower and less flexible than just pointing a Python server at a checkpoint — you compile per model, per GPU architecture, per precision, and per batch-size configuration, and changing any of those requires rebuilding. TensorRT-LLM is usually paired with Triton Inference Server, NVIDIA's general-purpose model server, which adds the surrounding production concerns: request queuing, dynamic batching orchestration, multi-model hosting on shared GPUs, and its own in-flight batching support for LLM workloads specifically.

The practical framing: TensorRT-LLM is the right choice when you have a small number of stable models running at very high volume on NVIDIA hardware and the engineering effort of a compile step is worth it for the last 20-30% of throughput. vLLM (or TGI) is the right choice when you want to iterate on models quickly, run a heterogeneous fleet of checkpoints, or don't have a dedicated MLOps team to own a compilation pipeline — which describes the large majority of teams self-hosting LLMs today.

## Quantization as a Serving Lever

All three frameworks treat quantization as a first-class serving concern, not just a training-time trick, because it directly attacks the memory-bandwidth bottleneck described earlier: a 4-bit quantized weight moves 4x less data per token than an FP16 weight for the same parameter count, which is a nearly linear win against a bandwidth-bound workload. AWQ and GPTQ are the two dominant post-training quantization schemes for weights-only quantization (keeping activations in higher precision), and FP8 (supported natively on Hopper-generation and newer NVIDIA GPUs) is increasingly used for both weights and activations with minimal quality loss because the hardware has native FP8 tensor cores rather than needing to simulate low precision in software. The practical rule of thumb worth knowing for an interview: quantization mainly buys you the ability to fit a bigger model (or more concurrent requests' KV cache) into the same GPU memory, and it usually costs a small, benchmarkable amount of output quality — the right amount of quantization is a decision you validate against your own eval set, not a number you copy from a blog post, since quality degradation is model- and task-dependent.

## Structured Output as a Serving-Layer Concern

One more capability worth knowing lives inside these serving frameworks, because it's easy to assume structured output (forcing a model to emit valid JSON matching a schema, or output matching a formal grammar) is purely a prompting concern when it's actually, in the best implementations, a serving-layer one. Grammar-constrained decoding works by intersecting the model's token probability distribution at each step with the set of tokens that would keep the output a valid continuation of a target grammar (a JSON schema compiled to a finite-state machine, for instance), masking out any token that would produce invalid output before sampling, rather than sampling freely and hoping the model happens to produce valid JSON. vLLM, TGI, and TensorRT-LLM all support this via integrations with grammar libraries (Outlines, XGrammar, and similar), and because the constraint is applied during decoding rather than as a post-hoc validation-and-retry step, it guarantees syntactic validity while adding only a small amount of decoding overhead — a materially better approach than the older pattern of prompting for JSON and retrying on parse failure, which wastes a full generation on every failure and still offers no hard guarantee.

```python
# Conceptual sketch of what the serving framework does at each decode step
# when a JSON schema constraint is active.
def constrained_sample(logits, grammar_state, tokenizer):
    valid_token_ids = grammar_state.get_valid_next_tokens(tokenizer)
    mask = full_negative_infinity_vector(len(logits))
    mask[valid_token_ids] = 0
    constrained_logits = logits + mask     # invalid tokens can never be sampled
    next_token = sample(constrained_logits)
    grammar_state = grammar_state.advance(next_token)
    return next_token, grammar_state
```

This matters for the self-host-vs-API decision too: managed APIs from major providers now offer this as a first-class "structured output" or "JSON mode" feature with the same guarantee, so the presence of grammar-constrained decoding is not by itself a reason to self-host — but if you need constrained decoding against a custom grammar the managed providers don't support (a domain-specific DSL, a legacy output format), that capability gap is one more concrete item to weigh alongside the cost and data-residency factors below.

## Self-Hosting vs. Managed APIs

With the mechanics out of the way, the actual decision comes down to five questions, and a senior engineer should be able to walk through all five for any given product rather than defaulting to whichever option is more familiar.

**Volume and unit economics.** Managed APIs charge a per-token price that already bakes in the provider's margin on top of their own (very well-optimized) infrastructure cost. Self-hosting has a large fixed cost (GPUs, whether owned or rented by the hour) and a much lower marginal cost per token once utilization is high. This means self-hosting only wins economically past a volume threshold where the fixed GPU cost, amortized over your request volume, undercuts the API's per-token price — and that threshold is highly sensitive to how well you can keep GPU utilization high, which is exactly what continuous batching and PagedAttention are for. A team running a low-traffic internal tool will almost always lose money self-hosting; a team running a high-volume consumer product with steady load may save 50-80% by self-hosting an open-weight model of comparable capability.

```python
def breakeven_requests_per_month(
    gpu_hourly_cost, hours_per_month,
    api_cost_per_request, self_hosted_requests_per_gpu_hour,
):
    """Rough breakeven volume above which self-hosting undercuts an API,
    ignoring engineering/ops cost -- always add that back in qualitatively."""
    monthly_gpu_cost = gpu_hourly_cost * hours_per_month
    monthly_capacity = self_hosted_requests_per_gpu_hour * hours_per_month
    self_hosted_cost_per_request = monthly_gpu_cost / monthly_capacity
    if self_hosted_cost_per_request >= api_cost_per_request:
        return None  # self-hosting never wins at this utilization/config
    # Requests needed for self-hosted fixed cost to be less than equivalent API spend
    return monthly_gpu_cost / api_cost_per_request

print(breakeven_requests_per_month(
    gpu_hourly_cost=3.5, hours_per_month=730,
    api_cost_per_request=0.01, self_hosted_requests_per_gpu_hour=1200,
))
```

**Latency control and predictability.** Managed APIs are subject to shared-tenant load: your P99 latency can spike because of other customers' traffic, not yours, and you have essentially no visibility into why. Self-hosting gives you a dedicated, capacity-planned latency profile that you control entirely — critical for products with hard latency SLAs (real-time voice agents, for instance) where an unpredictable third-party tail latency is unacceptable regardless of average-case performance.

**Data residency, privacy, and compliance.** Some domains (healthcare, finance, government, anything under strict data-residency law) cannot send data to a third-party API at all, full stop, regardless of the provider's compliance certifications. Self-hosting inside your own VPC or on-prem is sometimes not a cost optimization but a hard legal requirement, which makes this the one factor that can override an unfavorable cost calculation entirely.

**Model customization.** If your product depends on a fine-tuned model, a custom architecture, or a model not offered by any managed provider, self-hosting is the only option — there's no decision to make. Conversely, if frontier general capability matters more than customization (complex reasoning, broad world knowledge, cutting-edge multimodal understanding), managed APIs from labs at the capability frontier will usually outperform anything you could self-host and fine-tune yourself, at least for the foreseeable future.

**Operational capability and opportunity cost.** Self-hosting is a genuine ongoing engineering commitment: capacity planning, GPU procurement or cloud reservation strategy, on-call for a serving stack, upgrade cycles as new model versions and serving-framework versions ship. A five-person startup pre-product-market-fit almost always should default to a managed API, because the engineering time spent running GPU infrastructure is engineering time not spent finding product-market fit, and that opportunity cost dwarfs any per-token savings at low volume.

## A Practical Decision Framework

Put together, the sane default posture is: start on managed APIs, because they let you validate product value with near-zero infrastructure investment and the best available model quality. Move to self-hosting only when one of the hard constraints (data residency, custom fine-tune, latency SLA incompatible with shared-tenant infra) forces it, or when volume has grown enough that the cost math clearly favors it *and* you have the operational maturity to run GPU infrastructure reliably — which in practice usually means you already have an SRE/platform function, not just application engineers. Many mature production systems end up in a hybrid: a self-hosted fleet (vLLM, typically) handling the bulk of predictable, high-volume traffic at low marginal cost, with a managed API as both an overflow valve for traffic spikes and a fallback path if the self-hosted fleet degrades — getting the cost benefits of self-hosting without giving up the reliability of a well-run managed provider as a safety net.

```python
def choose_serving_strategy(monthly_volume, has_custom_finetune,
                             data_residency_required, has_platform_team,
                             latency_sla_ms):
    if data_residency_required or has_custom_finetune:
        return "self_host"  # hard constraint, cost math is secondary
    if not has_platform_team:
        return "managed_api"  # opportunity cost of ops outweighs savings
    if monthly_volume < 2_000_000 and latency_sla_ms > 2000:
        return "managed_api"  # not enough volume to justify fixed GPU cost
    return "hybrid_self_host_primary_api_fallback"
```

## Operational Nuances Once You Self-Host

A few production realities show up only once you actually run a serving stack. Cold start is a real cost: loading a 70B-parameter model's weights onto GPUs from disk/network storage can take minutes, which matters for autoscaling — you cannot spin up a new replica in response to a traffic spike the way you can scale a stateless web server, so self-hosted LLM fleets typically over-provision headroom rather than relying on reactive autoscaling. Multi-LoRA serving (vLLM and TGI both support this) lets you host dozens of fine-tuned adapters on one base model's GPU footprint, which is the difference between needing one GPU fleet per customer-specific fine-tune versus one shared fleet serving all of them — a major cost lever for any product offering per-customer customization. Finally, spot/preemptible GPU instances can cut costs 60-70% versus on-demand, but only for workloads that tolerate mid-request interruption gracefully (requiring request-level retry logic and, ideally, request queuing in front of the serving layer so a preempted node's in-flight requests aren't simply dropped).

## Speculative Decoding

Everything above attacks throughput by improving how many sequences share a GPU's memory bandwidth at once. Speculative decoding attacks the other axis: reducing the number of sequential, full-model forward passes needed to produce a given amount of output. The idea is to use a small, fast "draft" model to guess several tokens ahead, then verify all of those guesses in a single forward pass of the large target model — since verifying a token (a single forward pass to check whether the large model would have assigned it reasonably high probability) is cheap relative to generating one autoregressively, and a single large-model pass can verify several draft tokens at once because verification, unlike generation, can be parallelized across positions.

```python
def speculative_decode_step(draft_model, target_model, prefix_tokens, k=4):
    """Simplified single-step sketch of speculative decoding.
    Real implementations (as in vLLM and TensorRT-LLM) handle rejection
    sampling to guarantee the output distribution matches the target model
    exactly, not just approximately."""
    draft_tokens = draft_model.generate(prefix_tokens, num_tokens=k)  # fast, sequential, but cheap model
    # One target-model forward pass scores all k draft positions plus one bonus position
    target_logprobs = target_model.score_positions(prefix_tokens, draft_tokens)

    accepted = []
    for i, token in enumerate(draft_tokens):
        if target_model.accepts(token, target_logprobs[i]):   # rejection-sampling test
            accepted.append(token)
        else:
            # Reject and resample this position from the target's own distribution;
            # everything after this point in the draft is discarded.
            resampled = target_model.resample(target_logprobs[i])
            accepted.append(resampled)
            break
    return accepted
```

The net effect, when the draft model's guesses are frequently correct (which is common for predictable continuations — closing brackets, common phrases, repeated boilerplate), is that you get multiple tokens of real output per expensive target-model forward pass instead of one, cutting latency for a fixed amount of output without changing the output distribution at all (correctly implemented speculative decoding is a lossless technique — a subtlety worth stating explicitly, since it sounds like an approximation but is mathematically exact when the rejection-sampling step is implemented correctly). The catch is that it helps latency and can help throughput at low concurrency, but its benefit shrinks as batch size grows, because the whole point was using otherwise-idle compute headroom for verification, and a fully-batched, already compute-saturated server has no idle headroom left to spend on it — which is why speculative decoding is typically most valuable for latency-sensitive, lower-concurrency deployments rather than the highest-throughput batch scenarios continuous batching targets.

## Disaggregating Prefill and Decode

A more recent architectural refinement, appearing in vLLM's and other frameworks' newer scheduling modes, is splitting the two phases of generation — the compute-heavy prefill pass over the input prompt, and the memory-bandwidth-heavy decode loop that produces output tokens one at a time — onto separate GPU pools entirely, rather than interleaving them on the same GPUs as continuous batching does by default. The motivation is that prefill and decode have almost opposite hardware profiles: prefill wants to saturate compute (it processes many tokens in parallel in one pass) while decode wants to saturate memory bandwidth (it repeatedly reads the whole model and a growing KV cache to produce one token at a time). Running both workloads on the same GPU pool means each interferes with the other's ideal utilization pattern — a long prefill request can stall decode steps for sequences already in flight, directly hurting TPOT (time-per-output-token) for everyone else in the batch.

Disaggregated serving routes incoming requests' prefill computation to a pool of GPUs tuned and scheduled for compute-bound work, then transfers the resulting KV cache over a fast interconnect to a separate pool of GPUs handling the memory-bandwidth-bound decode loop. This adds real complexity (you now need fast KV-cache transfer between machines, and two pools to capacity-plan instead of one) and is generally only worth adopting at a scale where prefill/decode interference is a measurable, material fraction of latency — but it's increasingly the direction high-scale, latency-sensitive frontier-model serving is heading, and it's worth knowing about even if a given team's scale doesn't yet justify the complexity.

## Benchmarking Serving Frameworks Correctly

A common mistake when evaluating vLLM vs. TGI vs. TensorRT-LLM (or any serving configuration change) is to benchmark a single request in isolation and compare latencies — which tells you almost nothing about how the system behaves under the concurrent load a real product actually generates, since the entire value proposition of continuous batching and PagedAttention is about behavior *under concurrency*, not single-request speed. A meaningful benchmark sweeps concurrency (1, 8, 32, 128 simultaneous requests, or whatever range spans your expected production load) and reports both throughput (tokens/sec aggregate) and per-request latency (P50/P95 TTFT and TPOT) at each concurrency level, because the two move in opposite directions as concurrency rises — throughput climbs as the GPU is kept busier, while individual-request latency degrades as more requests compete for the same compute and memory bandwidth. The right operating point is wherever your product's actual latency SLA is still met, read off that curve, not the point of maximum raw throughput.

```python
def summarize_benchmark_sweep(results_by_concurrency):
    """results_by_concurrency: {concurrency: {"throughput_tok_s": x, "ttft_p95_ms": y, "tpot_p95_ms": z}}"""
    for concurrency, metrics in sorted(results_by_concurrency.items()):
        print(f"concurrency={concurrency:4d}  "
              f"throughput={metrics['throughput_tok_s']:8.1f} tok/s  "
              f"TTFT p95={metrics['ttft_p95_ms']:6.0f}ms  "
              f"TPOT p95={metrics['tpot_p95_ms']:5.1f}ms")

def find_max_concurrency_within_sla(results_by_concurrency, ttft_sla_ms):
    ok = [c for c, m in results_by_concurrency.items() if m["ttft_p95_ms"] <= ttft_sla_ms]
    return max(ok) if ok else None
```

## Autoscaling a Self-Hosted GPU Fleet

The cold-start problem mentioned above (a large model's weights taking minutes to load) means a self-hosted LLM fleet cannot autoscale the way a stateless web tier does — you can't wait for a traffic spike to happen and then spin up a fresh replica, because by the time the new replica has finished loading weights, the spike is often long over and users have already had a bad experience. Two patterns address this. The first is keeping a pool of warm (weights-loaded, idle) replicas as headroom above expected peak, scaling that headroom based on leading indicators (queue depth and request-arrival rate trending up) rather than lagging indicators (current CPU/GPU utilization, which only rises after the queue has already backed up). The second, used by teams operating at large scale, is predictive scaling from historical traffic patterns (time-of-day, day-of-week seasonality is often strong and predictable for consumer-facing products), pre-warming replicas ahead of an expected peak rather than reacting to it. Both patterns trade some amount of idle GPU cost for latency reliability, and the right balance is a direct function of how expensive a slow or dropped request is to your product relative to the cost of idle GPU-hours — an equation worth working out explicitly rather than guessing at a headroom percentage.
