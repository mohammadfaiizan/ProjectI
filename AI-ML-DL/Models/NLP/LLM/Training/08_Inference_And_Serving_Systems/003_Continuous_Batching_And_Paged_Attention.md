## Continuous Batching and PagedAttention

### 1. Why batching matters for decode throughput

File 001 established that decode is memory-bandwidth-bound: generating one token requires loading the entire model's weights (and reading the KV cache) from HBM into the GPU's compute units, doing a comparatively tiny amount of arithmetic (a `[batch, 1, d_model]`-shaped set of matmuls) relative to how much data just moved, and writing the result back. The GPU's arithmetic units sit mostly idle waiting for data; the bottleneck is HBM-to-SRAM bandwidth, not FLOPs.

The fix, wherever a bottleneck is "moving X once and doing too little work with it," is to do more work per move: instead of decoding one request's next token per forward pass, decode `batch_size` requests' next tokens in the *same* forward pass. The weights are loaded from HBM exactly once per layer regardless of batch size (up to the point where the batch itself becomes large enough to be compute-bound rather than memory-bound), so batching amortizes the fixed weight-loading cost across many requests' worth of useful work, converting an inherently memory-bound single-request operation into something closer to compute-bound, high-utilization work as batch size grows. This is *why* batch size is the single most important lever for decode throughput and cost-per-token — and it is also exactly why KV cache size (file 001), which caps how many requests you can hold concurrently, is the binding constraint on how much of this benefit you can actually realize.

### 2. Static batching and why it wastes capacity

The naive way to batch autoregressive generation: collect a fixed group of `N` requests, pad every prompt to the same length, run prefill on the padded batch, then run decode step by step — at every step, every one of the `N` sequences advances by exactly one token, in lockstep, because the physical batch is one tensor and every row of a tensor is processed together — until the *entire batch* satisfies its stopping condition (hits an EOS token or its max-length limit). Only once every single sequence in the batch has finished can the server retire the batch and admit a new group of `N` requests.

This wastes GPU capacity in two independent ways:

- **Tail idling.** Different requests need very different numbers of output tokens — one user asks a yes/no question (5 tokens), another asks for a long essay (2,000 tokens). In a static batch, the short request finishes at step 5 but its slot cannot be reused; the framework must keep "generating" for it (typically by masking it out or generating and discarding tokens) purely to keep the batch tensor's shape consistent, until the longest-running request in the batch finally finishes at step 2,000. For a large fraction of that time, a meaningful chunk of the batch's slots are doing no useful work at all, while still consuming the full compute/memory cost of a batch that size.
- **No admission until the whole batch drains.** Even if you're willing to tolerate the tail-idling waste, new requests that arrive while a batch is running have to *queue* — they cannot be inserted into the currently-running batch (its shape and its KV cache allocations are fixed for the batch's lifetime in a static-batching implementation), so they wait until the entire batch retires, which could be however long the single longest request in that batch takes. This directly inflates the queueing component of time-to-first-token (file 005) for unlucky arrivals, and it means average GPU utilization across a full serving window is much lower than the utilization you'd measure by looking only at a mid-batch snapshot.

Both problems have the same root cause: **the unit of scheduling is the whole batch**, not the individual request or the individual token-generation step.

### 3. Continuous batching: scheduling at iteration granularity

Continuous batching (also called in-flight batching, or iteration-level scheduling — the term used in the Orca paper that introduced the technique, and the mechanism vLLM and essentially every modern serving engine implements) changes the unit of scheduling from "the whole batch, for its entire lifetime" to **"one forward pass, i.e. one decode step, for whichever requests are currently active."**

The scheduler's loop looks like this, at a conceptual level:

```
active_requests = []          # currently in the running batch
waiting_queue = []            # arrived but not yet admitted

loop forever:
    # 1. Retire anything that finished last step
    for req in active_requests:
        if req.is_finished():           # hit EOS or max_tokens
            release_kv_cache(req)
            active_requests.remove(req)
            return_result_to_client(req)

    # 2. Admit new requests into now-free capacity
    while has_free_capacity() and waiting_queue:
        req = waiting_queue.pop(0)
        allocate_kv_cache(req)
        active_requests.append(req)     # will get its prefill done this step

    # 3. Run exactly one forward pass over the current active set
    #    (a mix of prefill for newly admitted requests and decode
    #    for already-running requests -- see "chunked prefill" below)
    run_one_iteration(active_requests)

    # 4. Append each request's newly generated token to its own sequence
    for req in active_requests:
        req.append_token(sampled_token[req])
```

Crucially, this loop's granularity is a *single forward pass over the model*, not "run until this group finishes." A request that finishes at step 5 is evicted from `active_requests` immediately after step 5, freeing its KV-cache slot and its place in the batch that very iteration; a new request from the queue can be admitted into that freed slot on the very next iteration, without waiting for anyone else in the batch to finish. This eliminates both static-batching failure modes at once: there is no tail-idling (a finished request's slot is reused immediately, not held open), and there is no batch-lifetime admission barrier (new requests join at iteration granularity, bounded by how long a *single* forward pass takes — milliseconds — rather than by how long the slowest request in an entire cohort takes).

**Prefill and decode coexisting in one batch.** A subtlety continuous batching has to handle: prefill (processing a new request's entire prompt, which needs a full parallel forward pass over potentially thousands of tokens) and decode (advancing many already-running requests by exactly one token each) are computationally very different operations, and naively mixing a large prefill into the same iteration as many decode steps can spike that iteration's latency badly, delaying every decode step riding along with it (this directly hurts inter-token latency for already-in-flight requests, file 005). **Chunked prefill** (splitting a long prompt's prefill into several smaller chunks, each scheduled alongside ongoing decode iterations rather than as one giant blocking iteration) is the standard mitigation, letting the scheduler smooth a large prefill's cost across several iterations instead of injecting one large latency spike.

### 4. PagedAttention: the memory-management problem continuous batching creates

Continuous batching solves the *scheduling* problem, but it creates a new *memory-allocation* problem. In a naive implementation, when a request is admitted, the server must decide how much contiguous KV-cache memory to reserve for it — and since you don't know in advance how many tokens it will ultimately generate, the naive answer is "reserve a contiguous buffer sized for the maximum possible sequence length" (the model's context window). This is exactly analogous to a program that doesn't know how much memory it'll need and simply mallocs the theoretical maximum up front.

This wastes memory in the same way over-provisioned static memory allocation always does: most requests never reach the maximum context length, so most of each request's pre-reserved contiguous buffer sits empty for the request's entire lifetime — **internal fragmentation**, in operating-systems terms, and at the scale of a 100K+-token context window reserved per request, the waste is enormous (recall from file 001 that a single 128K-token slot can be hundreds of GiB; reserving that for every admitted request regardless of how long it actually turns out to run is untenable at any real concurrency). There's a second fragmentation mode too: even ignoring per-request over-reservation, as requests of varying lengths are admitted and retired over time, contiguous free space in HBM becomes fragmented into gaps too small to fit a new request's reservation even when the *total* free memory would be more than sufficient — **external fragmentation**.

**PagedAttention** (introduced by the vLLM project) is a direct application of the OS virtual-memory paging solution to exactly this problem.

- The KV cache for a sequence is not one contiguous buffer. It is divided into fixed-size **blocks** (analogous to OS memory pages — commonly holding, say, 16 tokens' worth of K/V per block), physically scattered anywhere in HBM.
- Each sequence has a **block table** (analogous to a page table) — a small, per-sequence list mapping *logical* token positions (0, 1, 2, ... in generation order) to the *physical* block index actually holding that range of tokens' K/V data. Attention computation at each decode step walks the block table to gather the (non-contiguous) physical blocks that make up the sequence's full KV history, computes attention against them, and appends the new token's K/V into whichever block currently has a free slot (allocating a fresh block from a global free-block pool when the current block fills up).
- Because blocks are allocated **on demand, one at a time, as a sequence actually grows**, a request never reserves more memory than it has actually used *so far* — there is no more up-front worst-case reservation, and both internal fragmentation (over-reserving for a request that turns out short) and external fragmentation (leftover gaps too small to use) are eliminated in the same way paged virtual memory eliminates them for general-purpose programs: any free block, anywhere in the block pool, can satisfy any sequence's next allocation, because sequences never need contiguity.

```python
BLOCK_SIZE = 16  # tokens per physical block, a serving-engine configuration constant

class BlockTable:
    """Maps a sequence's logical token positions to physical KV-cache block indices."""
    def __init__(self):
        self.blocks: list[int] = []   # blocks[i] = physical block id holding tokens
                                       # [i*BLOCK_SIZE, (i+1)*BLOCK_SIZE)

    def physical_block_for(self, logical_token_pos: int) -> tuple[int, int]:
        block_idx = logical_token_pos // BLOCK_SIZE
        offset_in_block = logical_token_pos % BLOCK_SIZE
        return self.blocks[block_idx], offset_in_block

class BlockAllocator:
    def __init__(self, num_physical_blocks: int):
        self.free_blocks = list(range(num_physical_blocks))
        self.ref_counts = [0] * num_physical_blocks   # for copy-on-write, Section 5

    def allocate(self) -> int:
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id: int):
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)

def append_token_kv(seq_block_table: BlockTable, allocator: BlockAllocator,
                     token_pos: int, k_vec, v_vec, kv_store: dict[int, list]):
    needs_new_block = (token_pos % BLOCK_SIZE == 0)
    if needs_new_block:
        new_block = allocator.allocate()
        seq_block_table.blocks.append(new_block)
    block_id, offset = seq_block_table.physical_block_for(token_pos)
    kv_store[block_id][offset] = (k_vec, v_vec)   # write into the (possibly fresh) block
```

The practical effect vLLM reported (numbers as originally published in the vLLM paper; treat these as the paper's own reported figures, subject to the usual caveat about self-reported benchmarks against specific baselines rather than universal constants) was that eliminating this fragmentation allowed dramatically higher achievable batch sizes for the same HBM budget — the paper reports roughly 2-4x higher throughput than then-contemporary systems (FasterTransformer, Orca-style implementations without paged memory) at equivalent latency, purely from memory-management efficiency, stacked on top of continuous batching's scheduling efficiency. The exact multiplier is workload- and hardware-dependent and has shifted as the whole field's baselines improved since that paper; the mechanism and its qualitative effect (near-zero fragmentation → higher safe batch size → higher throughput) is the durable, transferable fact.

### 5. Copy-on-write: sharing a common prefix across requests

Paging KV cache into blocks unlocks a second, independently valuable capability: **sharing physical blocks between different sequences** whenever those sequences happen to have identical token history over some prefix — most commonly, a shared system prompt, a shared few-shot template, or (in a "sample N completions from the same prompt" workload, e.g. beam search or best-of-N sampling) a shared user prompt with multiple candidate continuations branching off it.

Because a sequence's identity is just its block table (a list of physical block indices), two sequences that share an identical token prefix can simply have their block tables point at the **same physical blocks** for that shared portion — no data is duplicated, and no extra memory is consumed for the shared region beyond the single copy already computed. This is exactly the *fork* pattern from OS process memory management (`fork()` sharing pages between parent and child until one writes), and PagedAttention uses the identical mechanism: **copy-on-write (CoW)**.

- Each physical block carries a reference count (`ref_counts` in the sketch above). A block shared by `k` sequences has `ref_count = k`.
- As long as all `k` sequences are only *reading* that block (i.e., attending over it — every decode step reads the shared prefix's K/V but does not need to modify it), no copying is needed; all `k` block tables simply point at the same physical memory.
- The moment one sequence's generation **diverges** — it needs to *write* new tokens whose position would fall into what was, until now, a shared block (this happens once a sequence's own newly generated tokens start filling a block that had been shared from the common prefix) — that one sequence gets a **private copy** of the block, decrements the shared block's reference count, and continues writing into its own copy from that point forward. The other sequences sharing the original block are entirely unaffected.

```python
def fork_sequence(parent_table: BlockTable, allocator: BlockAllocator) -> BlockTable:
    """Create a new sequence sharing all of parent's current blocks (no copy yet)."""
    child_table = BlockTable()
    child_table.blocks = list(parent_table.blocks)   # share physical block ids
    for block_id in child_table.blocks:
        allocator.ref_counts[block_id] += 1
    return child_table

def write_with_cow(seq_table: BlockTable, allocator: BlockAllocator,
                    logical_block_idx: int, kv_store: dict[int, list]):
    block_id = seq_table.blocks[logical_block_idx]
    if allocator.ref_counts[block_id] > 1:
        # Shared block about to be mutated -> copy it privately first.
        new_block_id = allocator.allocate()
        kv_store[new_block_id] = list(kv_store[block_id])   # duplicate contents
        allocator.free(block_id)                             # drop this sequence's share
        seq_table.blocks[logical_block_idx] = new_block_id
        block_id = new_block_id
    return block_id   # caller writes new K/V into kv_store[block_id]
```

The practical value of this is large in exactly the workloads that dominate real production traffic: a long, shared system prompt (a few thousand tokens of instructions, tool schemas, and few-shot examples, common to *every* request against a given deployment) can have its KV cache computed once and its physical blocks shared by every concurrent request using that system prompt, rather than every request separately paying the prefill compute *and* the cache memory for identical content. This is the mechanical foundation underneath what's marketed as "prompt caching" or "prefix caching" in commercial LLM APIs (file 004 discusses prefix caching as a technique in its own right, and this is precisely the underlying implementation vLLM-style systems use to realize it): the saving is not merely "skip recomputing the shared prefix's forward pass," it is also "do not even allocate separate memory for it" — both the compute and the memory cost of the shared portion are paid exactly once, no matter how many concurrent requests share it, until and unless a given request's generation actually diverges.

### 6. Composing the two techniques

Continuous batching and PagedAttention solve genuinely different problems and compose cleanly: continuous batching answers "at this instant, which requests should be in the running batch" (a scheduling question, decided every iteration); PagedAttention answers "where does each of those requests' KV cache actually live in HBM, and how efficiently is that memory being used" (an allocation question, decided per token/block). A scheduler that does continuous batching over a naively-allocated contiguous-buffer-per-request memory layout still leaves most of the fragmentation waste on the table; a paged-memory allocator sitting under a static, whole-batch scheduler still leaves the tail-idling and admission-barrier waste on the table. Realizing the full throughput gain requires both simultaneously — which is exactly the design vLLM (and, by now, essentially every serious LLM serving engine: TensorRT-LLM, TGI, SGLang, and others, each with its own implementation details) ships as its baseline architecture. Everything else this module covers — quantization (file 002, frees HBM for more paged blocks), speculative decoding and prefix caching (file 004, reduces work per decode iteration or reuses cached blocks), and prefill/decode disaggregation (file 005, changes which hardware pool runs which kind of iteration) — sits on top of this scheduling-plus-memory-management foundation rather than replacing it.

### 7. Scheduling policy: fairness and priority beyond FCFS

Section 3's scheduler sketch used the simplest possible admission rule — pop from the front of a waiting queue whenever a slot frees up (first-come-first-served, FCFS). Real production schedulers generally need something richer, for reasons that show up the moment traffic becomes heterogeneous.

Pure FCFS treats every request identically regardless of size or importance, which creates two concrete problems. **Head-of-line blocking at admission**: if the request at the front of the queue needs a very large KV-cache reservation (a long prompt) and the server is close to its cache budget, FCFS will make every request behind it wait for that one large reservation to become admittable, even if several smaller requests behind it in the queue could be admitted immediately into currently-available headroom — a scheduler that instead scans the queue for *any* admittable request (not just the front one) can keep the server busier, at the cost of a more complex fairness story (a request can now be "skipped over" indefinitely if it's unlucky enough to always be the largest one waiting). **No product-tier differentiation**: a single FCFS queue gives no way to express that some traffic (an interactive chat request) should preferentially skip ahead of other traffic (a bulk batch job) — real systems generally implement multiple priority classes with weighted admission (e.g., reserve some fraction of slots for high-priority traffic, or use a priority-weighted queue rather than strict FCFS), directly implementing the different points on the cost/latency trade-off surface different product tiers are willing to pay for (file 005 Section 3's TTFT/TPOT/throughput discussion, and file 006 Section 1's cost/latency tiering across whole models — the same tiering idea recurring one level down, inside a single model's own scheduler).

### 8. Preemption and swapping under memory pressure

Section 4's admission-control discussion (and file 008 Part 2 Q6's admission controller) assumed a request is either admitted or queued — but what happens to an *already-running* request if the pool's aggregate KV-cache demand grows past capacity mid-flight, e.g. because several already-admitted requests turned out to need more tokens than initially expected, or a burst of new long-context requests was admitted optimistically? Two standard responses:

- **Preemption with recomputation.** Evict a running request's KV-cache blocks entirely, return it to the waiting queue, and when it's eventually re-admitted, recompute its prefill from scratch (its own context, including everything generated so far, is treated as a fresh prompt). This is simple to implement and doesn't need any additional storage tier, at the cost of wasted compute (redoing prefill work that had already been done once) — acceptable if preemption is rare and prompts aren't too long, expensive if it becomes frequent under sustained memory pressure.
- **Swapping to CPU memory.** Instead of discarding a preempted request's KV-cache blocks, copy them out to host (CPU) RAM, freeing the GPU HBM blocks for other use, and copy them back in (swap in) when the request is re-admitted — avoiding the recomputation cost at the price of a data-transfer cost (over PCIe or similar) each way, and requiring the serving stack to manage a second tier of storage for cache blocks, analogous to OS-level swap space backing virtual memory. This is a direct, deliberate extension of the paging analogy from Section 4: PagedAttention already treats KV-cache blocks like OS memory pages; swapping extends that analogy one step further by adding a slower, larger backing store beneath HBM, exactly as OS virtual memory backs RAM with disk.

Which is preferable depends on the relative cost of recomputation versus transfer for the specific workload (long prompts make recomputation expensive and swapping relatively more attractive; short prompts make recomputation cheap enough that the added engineering complexity of a swap tier may not be worth it) — another instance of a serving decision that has no universally correct answer, only a workload-conditional one.

### 9. Chunked prefill, made concrete

Section 3 described chunked prefill qualitatively; it's worth seeing the scheduling logic explicitly, since the interaction between prefill chunks and ongoing decode steps is exactly the kind of mechanism a staff interview may ask you to sketch at a whiteboard.

```python
from dataclasses import dataclass, field

@dataclass
class PrefillJob:
    request_id: int
    total_prompt_tokens: int
    tokens_processed: int = 0

    def remaining(self) -> int:
        return self.total_prompt_tokens - self.tokens_processed

    def is_done(self) -> bool:
        return self.remaining() == 0


class ChunkedPrefillScheduler:
    def __init__(self, max_prefill_tokens_per_iteration: int):
        """Caps how many NEW prompt tokens may be processed in a single iteration,
        regardless of how many decode steps for other requests run alongside it --
        this is the knob that bounds the worst-case latency spike a prefill can
        inject into a shared iteration (Section 3)."""
        self.chunk_budget = max_prefill_tokens_per_iteration
        self.pending_prefills: list[PrefillJob] = []
        self.decode_requests: set[int] = set()

    def submit_prefill(self, request_id: int, prompt_tokens: int):
        self.pending_prefills.append(PrefillJob(request_id, prompt_tokens))

    def run_iteration(self):
        budget = self.chunk_budget
        finished_this_iteration = []

        # Spend the prefill budget on pending jobs, in order, until it's exhausted.
        for job in self.pending_prefills:
            if budget <= 0:
                break
            take = min(budget, job.remaining())
            job.tokens_processed += take
            budget -= take
            if job.is_done():
                finished_this_iteration.append(job.request_id)
                self.decode_requests.add(job.request_id)

        self.pending_prefills = [j for j in self.pending_prefills if not j.is_done()]

        # Every already-decoding request advances by exactly one token this iteration,
        # regardless of how much of the prefill budget was just spent -- decode
        # iterations are not blocked waiting for a large prefill to finish in one shot.
        for rid in self.decode_requests:
            pass  # advance_decode_step(rid) in a real implementation

        return finished_this_iteration


if __name__ == "__main__":
    sched = ChunkedPrefillScheduler(max_prefill_tokens_per_iteration=512)
    sched.submit_prefill(request_id=1, prompt_tokens=2_000)   # a long prompt
    sched.submit_prefill(request_id=2, prompt_tokens=100)      # a short prompt

    for i in range(6):
        done = sched.run_iteration()
        print(f"iteration {i}: finished prefill for {done}, "
              f"decode set so far: {sched.decode_requests}")
```

Request 2's short prompt finishes prefill on the very first iteration (well within the 512-token chunk budget), joining the decode set immediately, while request 1's 2,000-token prompt is spread across four iterations (2000/512 rounded up) — and critically, every iteration in between is still available to run one decode step for request 2 (and for any other already-decoding request), rather than being entirely consumed by request 1's prefill. This is the mechanical realization of the TTFT-for-the-new-request-versus-TPOT-for-everyone-else trade-off named qualitatively in Section 3 and file 005 Section 2.

### 10. Reasoning about the reported throughput gains without overclaiming

Section 4 cited vLLM's own reported 2-4x throughput improvement over pre-PagedAttention baselines, with an explicit caveat that the exact multiplier is workload- and hardware-dependent. It's worth being able to explain *why* the gain is bounded the way it is, not just cite the number: the theoretical ceiling on the improvement from eliminating fragmentation alone is bounded by how much fragmentation the *baseline* system actually had — a baseline that already pre-allocates cache reasonably tightly (e.g., using a max-length setting close to the actual traffic's typical length, rather than the full context window) leaves less fragmentation on the table for PagedAttention to recover, so the realized gain in any specific comparison is a function of how wasteful the specific baseline being compared against was, not a fixed property of PagedAttention itself. This is a generally useful skeptical habit for any reported "our system is Nx faster than baseline Y" claim in the serving literature: the honest question is always "faster than what baseline, configured how, on what traffic," not just "is the number big."

### 11. What a staff-level understanding of this needs to include

Beyond the mechanism, be ready to reason about the failure modes and tuning knobs a real deployment has to manage:

- **Block size trade-off.** Smaller blocks reduce internal fragmentation (less wasted space in a partially-filled final block per sequence) but increase block-table bookkeeping overhead and the number of non-contiguous memory accesses the attention kernel must gather from per step (more, smaller reads instead of fewer, larger ones) — a real speed/memory trade-off, not a free lunch; typical implementations settle on a block size (commonly a small power of two, e.g. 16) empirically balancing the two.
- **Scheduling policy inside continuous batching still matters.** "Admit whenever there's free capacity" is the simplest policy; real schedulers also have to decide *fairness* (should a long-waiting request be prioritized over a request that just arrived, to bound worst-case queueing latency — first-come-first-served vs. more sophisticated policies) and *how aggressively to run chunked prefill* (a scheduler that always prioritizes draining the decode batch smoothly versus one that opportunistically slots in prefill chunks trades throughput against tail latency for newly arriving requests, file 005's TTFT-vs-TPOT tension).
- **Preemption.** If the active batch's aggregate KV-cache demand exceeds available blocks (e.g. a burst of long-context requests all arrive together), the scheduler must decide which already-running request(s) to evict back to the waiting queue (or discard/recompute, depending on the implementation) — this is a genuine admission-control and QoS decision, not just an implementation detail, and directly determines the tail-latency behavior of the system under load.
