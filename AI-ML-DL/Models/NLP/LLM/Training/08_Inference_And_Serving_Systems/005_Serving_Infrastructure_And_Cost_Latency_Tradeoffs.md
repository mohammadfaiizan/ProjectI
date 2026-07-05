## Serving Infrastructure and Cost/Latency Trade-offs

### 1. Prefill and decode are two different workloads wearing one trench coat

Every LLM request passes through two phases with fundamentally different computational character, and conflating them is the single most common source of confused reasoning about serving performance.

**Prefill.** Given a prompt of `n` tokens, the model must compute every layer's activations for all `n` positions before the first output token can be sampled. Because none of these `n` positions' Q/K/V computations depend on each other's *outputs* within the same forward pass (causal masking only restricts what each position can *attend to*, not the order in which the underlying matmuls can be computed — with the full prompt already known, all `n` positions' projections can be computed as one large batched matmul), this phase is embarrassingly parallel across the sequence dimension. A prefill over `n` tokens is, computationally, `n` positions' worth of matmul work executed essentially simultaneously — large matrix-multiply operations with high arithmetic intensity (many FLOPs performed per byte of weight loaded from HBM). This makes prefill **compute-bound** (limited by the GPU's raw FLOP/s throughput) rather than memory-bandwidth-bound, and it is exactly the regime GPUs are best at: big, dense, parallel matmuls.

**Decode.** After prefill produces the first output token, every subsequent token must be generated one at a time: token `t+1`'s computation depends on token `t` having already been sampled (it's part of the input to the next step). Each decode step performs a matmul with a batch dimension but a sequence dimension of exactly 1 new position — tiny arithmetic intensity relative to the amount of data (the full weight set, plus the growing KV cache) that must be read from HBM to perform it. This makes decode **memory-bandwidth-bound**, as established in files 001 and 003: the GPU's compute units are mostly idle, waiting on data movement, and the only lever available to raise utilization is batching many sequences' single-token steps together (file 003) so the fixed cost of reading weights from HBM is amortized across more useful work per read.

**Why this asymmetry drives fundamentally different optimal configurations.** Because prefill is compute-bound, its throughput scales with how much you can keep the GPU's matmul units fed — favoring hardware and kernel configurations tuned for large dense compute, and it benefits comparatively little from bigger batches once a single prompt is already large enough to saturate the GPU (a single long prompt's prefill can already be compute-bound on its own). Because decode is memory-bandwidth-bound, its throughput scales with how many sequences you can batch together per token step, up to the point where the batch itself becomes compute-bound (a large enough decode batch starts behaving like a prefill-shaped matmul) — and it is comparatively insensitive to per-sequence prompt length once that prompt's KV cache is already resident. These are close to opposite optimization targets: prefill wants "as much parallel work as possible, briefly"; decode wants "as much concurrent low-intensity work as possible, sustained over a long period."

### 1b. The roofline model, as the formal version of "compute-bound vs. memory-bound"

The intuition in Section 1 has a standard, formal name in computer architecture: the **roofline model**, which plots achievable performance (FLOP/s) against **arithmetic intensity** — the ratio of FLOPs performed to bytes moved from memory for a given operation. Every piece of hardware has two ceilings: a peak compute throughput (its maximum FLOP/s) and a peak memory bandwidth (its maximum bytes/s). For any operation, its *achievable* performance is capped by whichever ceiling binds first: if the operation's arithmetic intensity is low, it hits the memory-bandwidth ceiling before it can approach peak FLOP/s (memory-bound); if arithmetic intensity is high, it can approach the compute ceiling instead (compute-bound). The crossover point — the arithmetic intensity at which a piece of hardware transitions from memory-bound to compute-bound — is itself a hardware property, roughly `peak_FLOPs / peak_memory_bandwidth`, and it differs across GPU generations (newer accelerators have generally grown compute throughput faster than memory bandwidth across recent generations, pushing the crossover point higher and making memory-bandwidth-bound workloads like decode relatively *more* disadvantaged on newer hardware unless batching or other amortization techniques compensate).

```python
def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    return flops / bytes_moved

def roofline_bound_flops_per_sec(arithmetic_intensity_val: float,
                                   peak_flops_per_sec: float,
                                   peak_bandwidth_bytes_per_sec: float) -> float:
    memory_bound_ceiling = arithmetic_intensity_val * peak_bandwidth_bytes_per_sec
    return min(peak_flops_per_sec, memory_bound_ceiling)

# Illustrative single-token decode step: read the full weight set once, do a small
# matmul against it. Arithmetic intensity here is low -- a handful of FLOPs per weight
# byte read -- so the memory-bound ceiling binds well below peak FLOP/s.
decode_ai = arithmetic_intensity(flops=2 * 70e9, bytes_moved=70e9 * 2)  # ~2 FLOPs/param
print(decode_ai)  # ~2 -- far below a typical GPU's compute/bandwidth crossover point
```

A single decode step's arithmetic intensity (roughly 2 FLOPs per weight byte touched, for a simple matmul-dominated accounting) is far below the crossover point of any modern GPU, confirming quantitatively what Section 1 argued qualitatively: an individual decode step cannot come close to a GPU's peak FLOP/s no matter how fast the arithmetic units are, because the bottleneck is moving the weights, not computing with them. Batching raises the *effective* arithmetic intensity of a decode iteration (more FLOPs performed per byte of weight read, since the same weights now serve many sequences' matmuls in one pass) — which is the formal, roofline-model statement of exactly why batching is the correct lever for decode throughput, and prefill's already-high intrinsic arithmetic intensity is why it doesn't need the same lever nearly as badly.

### 2. Why disaggregating prefill and decode onto separate hardware pools makes sense

A serving system that runs prefill and decode on the *same* GPUs, interleaved (as continuous batching naturally does when it mixes prefill chunks into an ongoing decode batch, file 003 Section 3), creates **interference**: a prefill chunk injected into an iteration that would otherwise be a fast, small decode step forces that iteration's latency up to whatever the prefill chunk costs, delaying every decode step riding along in that same batch. Every currently-streaming user's inter-token latency (Section 3) takes a hit whenever a new, possibly large prompt needs its prefill done, purely because prefill and decode are forced to share the same GPU's iteration schedule.

Research systems (e.g. Splitwise, DistServe, and related work — the specific systems are a fast-moving research area and any one paper's exact reported numbers should be treated as that paper's own result rather than a fixed industry constant) propose **disaggregating** prefill and decode onto physically separate pools of GPUs, each provisioned and autoscaled independently for its own workload's character:

- A **prefill pool**, sized for compute throughput, that receives a new request, runs its prefill, and produces (a) the first output token and (b) the resulting KV cache.
- The completed KV cache is then **transferred** (over fast interconnect — NVLink within a node, or a fast network fabric across nodes) to a **decode pool**, sized for aggregate memory bandwidth and KV-cache capacity, which takes over token-by-token generation for the rest of the request's lifetime.

This buys several things simultaneously: prefill workload no longer disrupts decode iteration latency (they're on different hardware entirely, so there's no shared-batch interference); each pool can be scaled independently to match the actual traffic mix (a burst of new conversations needs more prefill capacity; a large number of long, slow-streaming sessions needs more decode capacity, and these two demands don't have to move in lockstep); and each pool's hardware and kernel configuration can be tuned for its own bottleneck (e.g., a prefill pool might be configured for larger batch-of-prompts throughput and less concerned with KV-cache HBM headroom since requests don't linger there, while a decode pool is configured to maximize concurrently resident sequences per GPU). The cost is the added complexity and latency of the KV-cache transfer step itself (a real engineering cost — that transfer has to be fast enough not to meaningfully add to time-to-first-token) and running two differently-shaped fleets instead of one homogeneous one (added operational complexity, capacity-planning complexity, and the need for a control-plane component that hands a request off between pools). Whether this trade-off is worth it depends on scale: at small scale, the operational overhead of two pools plus a handoff protocol can outweigh the interference-avoidance benefit; at the scale of a major LLM product serving a large, heterogeneous mix of short-chat and long-context/agentic traffic, the interference cost of colocated prefill/decode becomes large enough that disaggregation is an increasingly common design choice among serving teams operating at that scale. Treat the specific decision of whether any given production system uses disaggregation as an engineering judgment call informed by traffic mix and scale, not a universally correct default.

### 3. Latency metrics that matter, and why they're optimized somewhat independently

Two numbers dominate production LLM latency reporting, and they correspond directly to the prefill/decode split above:

- **Time-to-first-token (TTFT)** — the delay between a request arriving and the first output token being returned to the client. TTFT is dominated by (a) any queueing delay before the request is admitted into a running batch, and (b) the prefill pass itself, whose cost scales with prompt length (a long prompt's prefill genuinely takes longer, being a bigger compute-bound matmul). TTFT is what a user perceives as "how long before anything happens" — critical for interactive, conversational use cases where a visible delay before the response even starts feels broken, even if the eventual full response arrives reasonably fast.
- **Time-per-output-token (TPOT)**, also called **inter-token latency (ITL)** — the steady-state delay between successive output tokens once generation is underway. TPOT is governed by decode-phase cost: how memory-bandwidth-bound the current batch's iteration is, which depends on batch size, KV-cache read volume (growing with context length, file 001), and whatever else is competing for the same GPU's iteration schedule (e.g., a colocated prefill chunk, Section 2). TPOT is what determines the user-perceived *streaming speed* of the response — whether tokens appear at a comfortable reading pace or noticeably lag.

**Why these are optimized somewhat independently, and can trade off against each other.** Batching more aggressively is close to a pure win for *throughput* (more tokens generated per GPU-second, hence lower cost per token) but directly *hurts* TPOT: a larger decode batch means each iteration's matmul is bigger, so each individual token step takes longer wall-clock time even though more tokens are produced per step in aggregate — an individual streaming user experiences slightly slower per-token pacing as the batch around them grows, even though the server's aggregate efficiency is improving. Similarly, prioritizing low TTFT (e.g., preempting or delaying decode iterations to rush a new arrival's prefill through immediately) directly competes with TPOT for already-streaming users (Section 2's interference argument). A serving system's scheduling policy is, at bottom, choosing a point on a TTFT/TPOT/throughput trade-off surface, and different products reasonably choose different points: a live conversational assistant cares enormously about both low TTFT (feels responsive) and low TPOT (feels fluent while streaming), while a bulk-offline-summarization job might not care about either individually and only cares about aggregate throughput/cost, tolerating both higher TTFT (batched, queued processing) and higher TPOT (large batches) in exchange for a much lower cost per token. This is a large part of *why* production systems expose distinct "priority" or "batch" API tiers with different pricing — they are literally selling different points on this trade-off surface.

**p50 vs p99, and why tail latency is a different problem from median latency.** Median TTFT/TPOT reflects the common case; p99 reflects what happens under queueing pressure, memory contention (a request landing right when the KV-cache pool is nearly exhausted and preemption/eviction logic kicks in, file 003 Section 7), or an unlucky collision with a very large concurrent prefill. A staff-level diagnosis of a latency regression has to be able to distinguish "the whole distribution shifted" (a systemic capacity or model-change issue) from "the tail got fatter while the median held" (a scheduling, admission-control, or resource-contention issue that only bites under specific unlucky conditions) — these point to different root causes and different fixes, and conflating them is a common shallow-analysis mistake.

### 4. The cost-per-token math

A serving deployment's economics reduce to a small number of quantities a staff engineer should be able to combine on demand:

```
cost_per_token = (dollars_per_gpu_hour * num_gpus) / (tokens_per_second * 3600)
```

Worked example: suppose an 8xH100 node costs roughly $25-30/hour on a typical cloud on-demand rate (a real, if provider- and moment-dependent, figure — treat any specific dollar figure as illustrative, since cloud GPU pricing shifts and varies by provider, commitment level, and region), and that node, running a quantized 70B-class model with continuous batching and a well-tuned batch size, sustains an aggregate decode throughput on the order of a few thousand output tokens/second across all concurrently-batched sequences (again, illustrative — the real number is a direct function of the batch size the KV-cache budget allows, per file 001's math, and the model's own per-token compute cost). At, say, $28/hour and 3,000 tokens/second sustained:

```python
dollars_per_hour = 28
tokens_per_second = 3000
cost_per_million_tokens = dollars_per_hour / (tokens_per_second * 3600) * 1_000_000
print(f"${cost_per_million_tokens:.3f} per million output tokens")
# 28 / (3000*3600) * 1e6 = 28 / 10,800,000 * 1e6 ≈ $2.59 per million tokens
```

The point of running this arithmetic is not to memorize a number (real deployment costs vary by an order of magnitude depending on model size, quantization, hardware generation, and achieved batch size) but to internalize the **structure**: cost per token is *hardware cost rate divided by achieved throughput*, and every technique in this module is a lever on the denominator. Quantization (file 002) raises achievable batch size (more HBM free for KV cache) and lowers per-step compute/bandwidth cost, both directly raising `tokens_per_second`. Continuous batching and PagedAttention (file 003) raise achieved batch size toward the theoretical maximum by eliminating scheduling and memory waste. Speculative decoding (file 004) raises effective tokens-per-target-model-call. Every one of these is, in the final accounting, a way of pushing `cost_per_token` down for the *same* underlying hardware rate — which is why a staff engineer's mental model of "which technique matters most" should always be traceable back to this one ratio.

**A crucial asymmetry: prefill tokens and decode tokens do not cost the same amount to produce**, and production API pricing increasingly reflects this directly (many commercial APIs price input/prompt tokens and output/generated tokens differently, often with input tokens priced lower). This falls directly out of Section 1: prefill tokens are processed in one large, efficient, compute-bound parallel pass — cheap per token, especially with prefix caching (file 004 Section 6) eliminating redundant recomputation entirely for shared content. Decode tokens are each individually expensive relative to their prefill counterparts because each one, by itself, triggers a full memory-bandwidth-bound pass through the entire model's weights and KV cache, only partially amortized by whatever batch size is achievable at that moment. A cost model that treats "a token" as a single undifferentiated unit, rather than separately accounting for prefill-token cost and decode-token cost, will misestimate the economics of any workload whose input/output token ratio differs from whatever ratio the estimator implicitly assumed — e.g., a long-document-summarization workload (huge prefill, short decode) has a very different cost profile per total token than a long-creative-writing workload (short prefill, huge decode), even at an identical total token count.

### 5. Provisioning and capacity planning, briefly

Putting Sections 1-4 together into an operational picture: a staff engineer provisioning a serving deployment is really solving a joint allocation problem across (a) how many GPUs, of what generation/memory size; (b) how they're partitioned between prefill-oriented and decode-oriented pools (Section 2), if disaggregating; (c) what quantization scheme (file 002) is applied, trading a small, validated quality cost for materially more HBM headroom; (d) what batching and scheduling policy (file 003) is used, and where on the TTFT/TPOT/throughput trade-off surface (Section 3) the product's actual latency requirements sit; and (e) whether speculative decoding or prefix caching (file 004) are applicable to the workload's actual traffic pattern (structured/predictable generation and/or heavily shared prompts benefit; highly open-ended, low-prefix-overlap traffic benefits less). None of these decisions is made in isolation — a change to the quantization scheme changes the achievable batch size, which changes the TTFT/TPOT trade-off surface, which changes what scheduling policy is optimal, which changes what the resulting cost-per-token actually is. Treating this as one coupled system, rather than optimizing each lever independently against a fixed assumption about the others, is exactly the kind of systems thinking a staff-level serving discussion is expected to demonstrate.

### 6. Little's Law and sizing a replica pool from a traffic forecast

A concrete, checkable tool for the provisioning problem in Section 5: **Little's Law**, a basic queueing-theory identity stating that, in steady state, `L = lambda * W`, where `L` is the average number of requests in a system (in flight, i.e. concurrently occupying serving capacity), `lambda` is the average arrival rate (requests/second), and `W` is the average time a request spends in the system (queueing delay plus service time — for an LLM request, roughly TTFT plus total generation time). This gives a direct way to translate a traffic forecast into a required concurrency budget, which in turn (via file 001's KV-cache math) tells you how many GPUs you need.

```python
def required_concurrency(arrival_rate_per_sec: float, avg_time_in_system_sec: float) -> float:
    """Little's Law: average number of requests the server must be able to hold
    concurrently, in steady state, to keep up with the given arrival rate."""
    return arrival_rate_per_sec * avg_time_in_system_sec

def replicas_needed(required_concurrency_val: float, max_concurrent_per_replica: int,
                     headroom_factor: float = 1.3) -> int:
    """headroom_factor > 1 reserves margin against burstiness -- traffic arrival is
    rarely perfectly steady, and Little's Law describes steady-state averages, not
    worst-case bursts, so provisioning to the exact average invites p99 queueing
    regressions the moment real traffic deviates from its average (Section 3's p99
    discussion)."""
    import math
    return math.ceil(required_concurrency_val * headroom_factor / max_concurrent_per_replica)

if __name__ == "__main__":
    lam = 50          # 50 requests/second average arrival rate
    avg_ttft = 0.3     # seconds
    avg_tokens_out = 400
    avg_tpot = 0.02    # seconds/token
    avg_time_in_system = avg_ttft + avg_tokens_out * avg_tpot   # ~8.3 s end-to-end

    L = required_concurrency(lam, avg_time_in_system)
    n_replicas = replicas_needed(L, max_concurrent_per_replica=48)  # file 001's worked example
    print(f"steady-state concurrency needed: {L:.1f} requests")
    print(f"replicas needed (with headroom): {n_replicas}")
```

This is the connective tissue between a product's traffic forecast (arrivals/second, expected output length) and the GPU fleet size a staff engineer actually has to request — and it makes explicit why average-case provisioning is dangerous: Little's Law describes the steady-state average, but real traffic arrives in bursts, and provisioning to the bare average concurrency with no headroom guarantees that any burst above average immediately produces a growing queue and the p99-latency regression pattern described in Section 3 and diagnosed in file 007/008's TTFT-regression scenario questions.

### 7. Hardware selection considerations

Section 1b's roofline framing gives the right lens for comparing accelerator choices, but it's worth naming the concrete axes a staff engineer actually compares when picking hardware for a given workload, since "just use the newest GPU" is not always the right answer:

- **HBM capacity per device** directly sets the file-001 crossover point (how much KV cache fits before needing to shard across devices) and the largest model that fits without sharding at all — a workload dominated by very long contexts cares about this axis more than raw compute.
- **Memory bandwidth** directly sets decode throughput per file 001b's roofline argument — for a decode-heavy, latency-sensitive workload, bandwidth-per-dollar is frequently a more relevant comparison metric than FLOPs-per-dollar.
- **Peak compute (FLOP/s), especially at lower precision (fp8/int8) if the hardware has native low-precision tensor cores** — matters most for prefill-heavy workloads and for compute-bound large-batch decode, and is the axis quantization (file 002) and disaggregated prefill pools (Section 2) are specifically trying to exploit well.
- **Interconnect bandwidth (NVLink/NVSwitch within a node, network fabric across nodes)** — matters for tensor-parallel sharding (file 001 Section 12) and for the KV-cache transfer step in disaggregated serving (Section 2); a hardware choice with excellent per-GPU specs but weak interconnect can bottleneck exactly the multi-GPU configurations a large model requires.
- **Availability and cost structure** (on-demand vs. reserved/committed pricing, spot/preemptible capacity) — a purely technical hardware comparison that ignores actual achievable cost-per-hour at the commitment level a business is willing to make is incomplete; the cost-per-token math in Section 4 is only as accurate as the `dollars_per_gpu_hour` figure fed into it, and that figure varies enormously by commitment structure, not just by GPU generation.

None of these axes dominates unconditionally — a long-context-heavy, latency-sensitive chat product should weight HBM capacity and bandwidth most heavily; a bulk-offline-summarization pipeline dominated by large-batch prefill should weight raw low-precision compute throughput and cost-per-hour most heavily — which is exactly why "what hardware should we use" is a workload-conditional question, not a single universal ranking of accelerators.

### 8. Designing latency SLAs around the metrics in Section 3

A product team defining a latency SLA (service-level agreement, or an internal SLO/service-level-objective if not externally committed) for an LLM feature has to decide, explicitly, what to commit to and at what percentile — a decision that should be derived from the actual user experience the feature needs to deliver, not picked arbitrarily.

- **Commit to TTFT at a stated percentile** (commonly p95 or p99, since median alone hides exactly the tail-latency risk Section 3 describes) appropriate to how "instant" the surface needs to feel — an autocomplete-style surface might commit to a p99 TTFT in the tens of milliseconds; a conversational assistant might commit to a p99 in the low hundreds of milliseconds; a bulk batch API might have no TTFT commitment at all.
- **Commit to TPOT/ITL separately**, since (Section 3) it is optimized somewhat independently and a product can fail its users on TPOT even while meeting a TTFT commitment perfectly (fast to start, uncomfortably slow to keep streaming).
- **Explicitly decide what happens on breach** — a pure monitoring/alerting response (page an on-call engineer), an automatic mitigation (shed load, escalate priority, or fail over to a different pool/provider), or, for an externally-facing SLA, a contractual remedy — and wire the decision back into the admission-control and autoscaling logic (file 003 Section 7-8) rather than treating the SLA purely as a reporting artifact disconnected from the systems that actually determine whether it's met.
- **Revisit the SLA whenever the underlying serving configuration changes materially** (a new quantization scheme, a batch-size policy change, a hardware migration, file 002 Section 10's evaluation discipline applied to latency rather than quality) — an SLA calibrated against one serving configuration silently becomes wrong the moment that configuration changes underneath it, exactly the same staleness risk file 006 Section 2 describes for a router calibrated against a since-changed model pairing.

### 9. Utilization is never 100%, and the cost model needs to say so

Section 4's cost-per-token formula implicitly assumed a GPU sustains its measured `tokens_per_second` continuously — but real fleets are never at peak achievable throughput around the clock, for reasons worth separating explicitly rather than folding into one vague "overhead" fudge factor:

- **Traffic is not uniform over time.** Almost every real product has a diurnal (and often weekly) traffic pattern — peak hours see far more concurrent load than off-peak hours — and a fleet sized for peak load necessarily runs under-utilized during off-peak hours unless autoscaling can shrink the fleet fast enough to track demand (and, per file 006 Section 3, LLM replica autoscaling reacts more slowly than typical stateless-service autoscaling because loading model weights into a new replica's HBM takes real time). A fleet that cannot scale down at all (e.g., reserved/committed capacity with no elastic component) pays the peak-provisioned cost rate around the clock regardless of actual load — the *effective* cost-per-token for the business is therefore higher than the peak-throughput cost-per-token computed in Section 4 by roughly the inverse of the fleet's average utilization rate.
- **Headroom reserved for burst absorption and preemption avoidance** (Section 6's headroom factor, file 003 Section 7-8's admission-control conservatism) is, by design, capacity that sits partially idle in the common case specifically so it's available in the uncommon case — a deliberate trade of average-case efficiency for tail-latency protection, not a wasted inefficiency to be optimized away.
- **Canary and rollout capacity** (file 006 Section 3) — a fleet running a canary split, or maintaining a rollback-ready previous version alongside a new one during a staged rollout, is running some capacity that isn't purely serving peak-efficiency production traffic at all times.

```python
def effective_cost_per_million_tokens(peak_cost_per_million: float,
                                       average_utilization_fraction: float) -> float:
    """average_utilization_fraction: fleet's average achieved throughput as a fraction
    of its peak achievable throughput, averaged over a representative time window
    (e.g. a full week, to capture diurnal patterns)."""
    return peak_cost_per_million / average_utilization_fraction

if __name__ == "__main__":
    peak_cost = 2.59   # from Section 4's worked example
    for util in (0.9, 0.7, 0.5, 0.3):
        eff = effective_cost_per_million_tokens(peak_cost, util)
        print(f"avg utilization={util:.0%} -> effective cost: ${eff:.2f} / 1M tokens")
```

At 90% average utilization, the effective cost is close to the peak-throughput figure; at 30% average utilization (a fleet heavily over-provisioned for a rarely-hit peak, with poor autoscaling responsiveness), the effective cost more than triples relative to the peak-throughput number — meaning the *actual* cost lever a staff engineer should often be optimizing hardest is not squeezing out the last few percent of peak throughput, but improving average utilization through better autoscaling responsiveness, better traffic shaping (e.g., steering flexible/batch-tolerant traffic into off-peak windows), and right-sizing the headroom-versus-efficiency trade-off in Section 6.

### 10. A fully worked, end-to-end capacity and cost exercise

Chaining every tool this file has introduced into one connected worked example, exactly the form a staff interview whiteboard exercise is likely to take:

```python
GiB = 1024 ** 3

# --- Step 1: traffic forecast -> required concurrency (Section 6) ---
lam = 80                       # requests/second, forecasted peak arrival rate
avg_prompt_tokens = 2_000
avg_output_tokens = 500
avg_tpot_sec = 0.02
avg_ttft_sec = 0.25
avg_time_in_system = avg_ttft_sec + avg_output_tokens * avg_tpot_sec
L = lam * avg_time_in_system    # Little's Law

# --- Step 2: per-sequence KV-cache cost -> max concurrency per GPU node (file 001) ---
def kv_bytes_per_seq(n_layers, n_kv_heads, head_dim, seq_len, bytes_per_value=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * bytes_per_value

avg_ctx = avg_prompt_tokens + avg_output_tokens
per_seq = kv_bytes_per_seq(n_layers=80, n_kv_heads=8, head_dim=128, seq_len=avg_ctx)
node_hbm, weight_bytes, activation_reserve = 8 * 80 * GiB, 35 * GiB, 20 * GiB  # int4 weights
max_per_node = int((node_hbm - weight_bytes - activation_reserve) // per_seq)

# --- Step 3: replicas needed, with burst headroom (Section 6) ---
import math
headroom = 1.3
n_nodes = math.ceil(L * headroom / max_per_node)

# --- Step 4: cost, at peak and at realistic average utilization (Section 9) ---
dollars_per_node_hour = 28
tokens_per_sec_per_node = max_per_node / avg_tpot_sec   # rough decode-bound estimate
peak_cost_per_million = (dollars_per_node_hour / (tokens_per_sec_per_node * 3600)) * 1_000_000
avg_utilization = 0.55
effective_cost_per_million = peak_cost_per_million / avg_utilization

print(f"required concurrency (Little's Law): {L:.1f}")
print(f"max concurrent sequences per node:   {max_per_node}")
print(f"nodes needed (with headroom):        {n_nodes}")
print(f"peak cost per 1M tokens:              ${peak_cost_per_million:.3f}")
print(f"effective cost per 1M tokens:         ${effective_cost_per_million:.3f}")
```

Every number in this script traces back to a section of this file or file 001: the traffic forecast and Little's Law (Section 6), the KV-cache-driven per-node concurrency ceiling (file 001, restated in Step 2), the burst headroom (Section 6), and the utilization-adjusted cost (Section 9). This is deliberately the shape a real capacity-planning exercise takes — a chain of individually simple calculations, each depending on the output of the previous one, rather than one opaque formula — and being able to walk an interviewer through this chain live, adjusting any one input and re-deriving the downstream consequences, is a stronger demonstration of staff-level fluency than quoting any single memorized number.

### 11. What changes if you re-run this exercise with a different lever pulled

The real value of having the Section 10 chain assembled explicitly is being able to re-run it mentally (or literally) with one input changed, and reason about which downstream numbers move and why — exactly the kind of follow-up an interviewer is likely to press on after an initial worked answer.

- **Doubling `avg_output_tokens`** (a product change that produces longer responses on average) increases both `avg_time_in_system` (more decode steps per request) and `per_seq` (more KV-cache tokens per request, file 001), compounding: required concurrency `L` rises from the longer time-in-system, *and* the per-node capacity `max_per_node` falls from the larger per-sequence cache cost — both effects push toward needing more nodes, and naively estimating only one of the two effects understates the true node count needed.
- **Switching from int4 back to fp16 weights** frees nothing but instead consumes more of `node_hbm` for `weight_bytes`, directly shrinking `max_per_node` (file 002 Section 8's "memory freed for batching" argument, run in reverse) — a visible, quantifiable illustration of why quantization's batching benefit is not a rounding-error-sized effect.
- **Improving average utilization from 55% to 85%** (better autoscaling, better traffic shaping, Section 9) leaves the *node count* required for peak traffic completely unchanged (peak capacity still has to exist) but substantially lowers the *effective* cost-per-token, because it changes how much of that peak-provisioned capacity sits idle on average rather than changing how much capacity is provisioned at all — a reminder that "cost" and "capacity" are related but distinct optimization targets, and a fix to one does not automatically fix the other.
- **Adding disaggregation** (Section 2) doesn't change the Step-1/Step-2 arithmetic directly (the aggregate concurrency and cache-capacity requirements are the same either way) but changes *how* that capacity is partitioned across two differently-shaped pools, and — if it successfully removes prefill/decode interference — can shift the achievable `avg_tpot_sec` and `avg_ttft_sec` inputs to Step 1 favorably, which then feeds back into a lower required `L`. This is the clearest illustration in the whole exercise of Section 5's closing point: every lever in this module is coupled to every other, and a change justified purely on latency grounds (disaggregation, motivated by Section 2's interference argument) can still show up as a capacity and cost change once you trace it all the way through this chain.

### 12. Geographic distribution: a brief note on a further axis this file has otherwise held fixed

Everything above implicitly assumed a single regional deployment; a global product generally serves from multiple geographic regions, adding a further axis to the trade-off surface that's worth flagging even briefly. Placing capacity closer to users reduces network round-trip time, which is a real, additive contribution to TTFT that is entirely separate from anything in Sections 1-3 (a perfectly-optimized serving stack still pays the speed-of-light-bound network latency between a user and the nearest available serving region) — for a global, latency-sensitive product, multi-region deployment is often a larger lever on *observed* TTFT for geographically distant users than any of the model-serving optimizations in this module. The cost of that benefit is fragmenting the capacity-planning problem in Section 10 into multiple, independently-sized regional pools, each needing its own headroom margin, and a harder version of the Section 9 utilization problem (traffic isn't just non-uniform over time, it's non-uniform across regions too, and a global follow-the-sun traffic pattern means peak load shifts geographically over a 24-hour cycle) — a genuinely additional layer of the same joint-optimization problem Section 5 describes, not a separate problem requiring different tools.

### 13. Summary checklist for a staff-level serving-economics discussion

- State the prefill/decode distinction precisely (compute-bound vs. memory-bandwidth-bound, Section 1) before reaching for any downstream conclusion — nearly every other claim in this file is downstream of getting this distinction right.
- Be able to justify, not just assert, when disaggregation is and isn't worth its added complexity (Section 2, file 007/008's scenario questions on this).
- Keep TTFT and TPOT as separate, independently-tracked metrics, at both median and tail percentiles (Section 3), and never report or reason about a single blended "latency" number.
- Be able to write down the cost-per-token formula from memory and use it live (Section 4), including the prefill/decode cost asymmetry.
- Be able to connect a traffic forecast to a required GPU fleet size via Little's Law (Section 6), and to a cost figure that accounts for realistic average utilization, not just peak throughput (Section 9).
- Treat every lever in this module — quantization, batching, disaggregation, speculative decoding, hardware choice — as coupled to every other lever through the shared capacity-and-cost accounting worked out in Section 10, not as independently optimizable knobs.
- Remember that average utilization (Section 9), not just peak throughput, determines the cost figure a business actually pays, and that improving utilization and improving peak throughput are related but distinct levers with different fixes.
- Hold the geographic dimension (Section 12) in reserve for any question about a genuinely global product — it is easy to omit entirely if the discussion has stayed implicitly single-region throughout, and naming it explicitly signals a more complete picture of the problem.

A quick self-test worth being able to pass without notes: (1) explain why prefill and decode sit at opposite ends of the roofline model's arithmetic-intensity axis, using the specific term "arithmetic intensity"; (2) write the cost-per-token formula from memory and compute a rough number given a hypothetical GPU price and throughput; (3) state Little's Law and use it to convert a given arrival rate and average time-in-system into a required concurrency figure; (4) explain why a fleet's effective cost-per-token is higher than its peak-throughput cost-per-token, and name the two largest contributors to that gap; and (5) give one concrete example of a product decision that should prioritize TTFT over TPOT, and one that should prioritize the reverse, with a one-sentence justification for each.
