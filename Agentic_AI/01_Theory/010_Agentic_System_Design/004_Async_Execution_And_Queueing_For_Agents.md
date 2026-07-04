# Async Execution and Queueing for Agents

## Why Synchronous Request/Response Breaks Down

The default mental model most engineers bring to API design is synchronous: a client sends a
request, the server does some work, and a response comes back on the same connection a short time
later. This model is so ingrained that it's the natural first instinct even for agentic systems —
call an endpoint, wait, get the agent's answer. It works fine for a single LLM completion that takes
a couple of seconds. It falls apart for agentic workloads, and it's worth being precise about
exactly why, because the reasons generalize into design requirements.

The first reason is simply duration. An agent that has to retrieve context, plan, call several
tools, run tests, and iterate can easily run for minutes; an agent doing genuinely open-ended
research, a multi-file refactor, or a long document-generation task can run for tens of minutes or
hours. Nearly every piece of standard web infrastructure has a timeout well short of that: browsers
time out idle connections, load balancers (e.g., many managed ALB/API Gateway configurations)
enforce hard request timeouts around 30-60 seconds, and even a raw TCP connection held open across a
flaky mobile network for twenty minutes is asking for trouble. You cannot reliably hold a
synchronous HTTP connection open for the duration of a long agent run; something in the stack will
kill it before the agent finishes.

The second reason is resource efficiency. A synchronous model ties up a request-handling thread or
process for the entire duration of the work, which means your concurrency ceiling is bounded by how
many long-lived connections your web tier can hold open simultaneously — a deeply wasteful way to
spend server resources on work that's mostly "waiting for an LLM API or a tool call to respond," not
actually consuming CPU.

The third reason is resilience. If the process handling a synchronous request crashes, restarts, or
is redeployed while a 10-minute agent task is halfway through, that work is simply gone, with no
record of how far it got and no way to resume — completely unacceptable for anything that took real
money and time to produce partway through.

The fix for all three problems is the same architectural shift used throughout distributed systems
for exactly this class of problem: decouple *accepting* the work from *doing* the work, using a
durable queue and a separate pool of workers, and give the client a way to track progress and
retrieve results asynchronously rather than blocking on a single request.

## The Core Architecture: Queue and Worker Pool

```
 Client
   |
   |  1. POST /tasks  {task description}
   v
+-----------------+       2. enqueue job         +------------------+
|  API Gateway /   | ---------------------------> |   Job Queue      |
|  Task Service    |                               |  (durable,      |
|  (returns task_id|                               |   persisted)    |
|   immediately)   |                               +--------+--------+
+-----------------+                                         |
   ^                                                          | 3. dequeue
   | 4. poll / subscribe for status                           v
   |                                                 +------------------+
   |                                                 |   Worker Pool    |
   +------------------ status/result updates ------- |  (runs the agent |
                                                       |   loop)         |
                                                       +--------+--------+
                                                                |
                                                    5. checkpoint progress,
                                                       write partial/final
                                                       results
                                                                v
                                                       +------------------+
                                                       |   Result /       |
                                                       |   State Store    |
                                                       +------------------+
```

The client calls an endpoint that does the minimum necessary work synchronously — validate the
request, write a job record, enqueue it — and returns immediately with a task ID. A separate,
horizontally scalable pool of worker processes pulls jobs off the queue and actually runs the agent
loop, writing progress and results to a durable store as it goes, independent of whether the
original client connection is even still open. The client then either polls a status endpoint,
subscribes to a push channel, or is notified via webhook/callback when the task completes.

This separation is what gives you the properties a synchronous model can't: the worker pool can run
for as long as the task genuinely needs, independent of any HTTP timeout; you scale by adding
workers, not by holding more connections open; and because the job and its progress live in durable
storage rather than in a single process's memory, a worker crash means "pick this job back up," not
"the work is lost."

## Choosing a Queue Technology

The right queueing technology depends mostly on two things: how strict your durability and ordering
requirements are, and how large your team's existing operational footprint already is (adopting a
new piece of infrastructure has a real cost, so "what do we already run well" is a legitimate
factor, not a cop-out).

**Managed cloud queues** (SQS, Cloud Tasks, Azure Queue Storage) are the default choice for most
teams: they're durable, handle at-least-once delivery, support visibility timeouts (a message
becomes invisible to other consumers while being processed, and reappears if not acknowledged in
time — which is exactly the mechanism you want for "if the worker crashes mid-task, someone else
should retry it"), and require essentially no operational overhead. The trade-off is fewer
guarantees around ordering and no native support for complex routing.

**Redis-based queues** (via libraries like Celery, RQ, or BullMQ) are a common choice when the team
already runs Redis for caching or session state, and they're fast and simple, but Redis's durability
guarantees are weaker than a purpose-built queue unless configured carefully (AOF persistence,
replication) — worth an explicit gut-check for tasks where losing a queued job silently would be a
real problem.

**Kafka or a similar log-based system** is the right choice when you need strict ordering
guarantees, want to replay history (useful for debugging "what sequence of events led to this
agent's decision"), or need multiple independent consumers to process the same stream of events
differently — heavier to operate, and usually overkill unless you already have it for other reasons.

For most agentic task platforms, a managed cloud queue plus a durable state store for
progress/results is the pragmatic default, reserving Kafka for cases where the audit/replay
properties are a genuine requirement rather than a nice-to-have.

## The Worker Loop

A worker is a long-running process that pulls a job, executes the agent loop against it, and reports
results — and the loop itself needs to be resilient to the worker process itself dying mid-task,
which is the scenario synchronous architectures can't handle at all and queue-based ones are
specifically designed for.

```python
def worker_main_loop(queue, agent_runner, state_store):
    while True:
        message = queue.receive(visibility_timeout_s=60)  # short timeout; extend while working
        if message is None:
            continue

        job = deserialize(message.body)
        state_store.mark_status(job.id, "in_progress")

        try:
            for checkpoint in agent_runner.run_streaming(job):
                # extend the queue visibility timeout so another worker
                # doesn't pick this job up while we're still actively on it
                queue.extend_visibility(message, additional_s=60)
                state_store.save_checkpoint(job.id, checkpoint)

            state_store.mark_status(job.id, "completed")
            queue.delete(message)  # acknowledge -- only after fully done

        except RetryableError as e:
            state_store.record_attempt_failure(job.id, str(e))
            # do NOT delete the message -- let visibility timeout expire
            # so it becomes available for another attempt/worker
        except FatalError as e:
            state_store.mark_status(job.id, "failed", reason=str(e))
            queue.delete(message)  # don't retry something that can't succeed
```

Two details in this loop matter more than they look. First, **acknowledging the message only after
real completion** (not after dequeuing it) is what gives you at-least-once processing — if the
worker dies between dequeue and completion, the message becomes visible again and another worker
retries it. Second, **extending the visibility timeout while actively working** prevents the
opposite failure: a long-running task getting silently duplicated because its original visibility
timeout expired while it was still legitimately in progress, causing a second worker to pick up "the
same" job and run it concurrently. Getting this wrong is one of the most common bugs in queue-based
agent systems, and it connects directly to the idempotency discussion in the next chapter — because
at-least-once delivery means your job processing logic has to tolerate occasionally running the same
job twice, by design, not by exception.

## Streaming Partial Results Back to the Client

Decoupling the request from the response doesn't mean the user has to stare at a spinner until the
whole task finishes. Most production agent platforms give users incremental visibility into
progress, and there are three common mechanisms, each suited to different situations.

**Polling** is the simplest: the client periodically calls `GET /tasks/{id}` and receives the
current status and any partial output accumulated so far. It's easy to implement and works
everywhere, including behind restrictive corporate proxies, at the cost of some latency in noticing
state changes (bounded by your poll interval) and some wasted load from polling tasks that haven't
changed.

**Server-sent events (SSE) or WebSockets** let the server push updates to the client as they happen
— token-by-token generation, tool-call progress, intermediate reasoning steps — giving a much more
responsive feel, at the cost of needing to manage persistent connections (which reintroduces some of
the connection-lifetime concerns from earlier, though at a much smaller and more manageable scale
than holding open a connection for the *entire* task, since the connection here is just for
status/streaming, and the actual work continues independently in the worker even if the connection
drops and the client has to reconnect).

**Webhooks/callbacks** are the right mechanism for tasks with no live user watching — a CI-triggered
coding agent, a nightly report-generation task — where the calling system provides a callback URL
and gets a single POST when the task completes (or at defined milestones), rather than needing to
maintain any connection at all.

```python
async def stream_task_status(task_id: str, websocket):
    last_seen_seq = 0
    async for update in state_store.subscribe(task_id):
        if update.sequence <= last_seen_seq:
            continue  # already sent; connection resumed after a drop
        await websocket.send_json({
            "sequence": update.sequence,
            "status": update.status,
            "partial_output": update.partial_output,
        })
        last_seen_seq = update.sequence
        if update.status in ("completed", "failed"):
            break
```

A detail worth designing in deliberately rather than discovering in production: the streaming
channel should be resumable. If a client's WebSocket drops (mobile network, laptop sleep) and
reconnects, it should be able to say "give me everything since sequence number 40" rather than
either missing updates or receiving a confusing full replay — this is why the example above tracks a
monotonic sequence number in the state store rather than treating the live connection as the only
record of progress.

## Handling Tasks That Run for Minutes or Hours

For agent tasks at the long end of the duration spectrum, a few additional design elements become
necessary rather than optional.

**Checkpointing** means periodically persisting enough state to resume from a specific point rather
than from scratch — for an agent loop, this is typically the conversation/reasoning history plus a
record of which steps have already produced durable side effects (files written, API calls made)
versus which are still pending. Without checkpointing, a worker crash at minute 40 of a 45-minute
task means redoing all 40 minutes; with it, a resumed worker picks up from the last saved point.

**Heartbeats** are a signal, separate from checkpoints, that a worker is still alive and making
progress even if it hasn't hit a natural checkpoint boundary recently — a monitoring system watching
for "no heartbeat in N minutes" is how you detect a hung worker (as opposed to a
slow-but-progressing one) without waiting for a hard timeout to fire on the task itself.

**Cost and runaway-loop caps scale in importance with task duration.** A task that's allowed to run
for hours needs a hard ceiling on cost, iteration count, and possibly wall-clock time independent of
whether it's "making progress," because a subtly looping agent that keeps calling tools without
converging can otherwise run — and rack up cost — for far longer than anyone intended before a human
notices. Building this cap into the worker loop itself (not just as an external monitoring alert)
means the system self-terminates rather than relying on someone to catch the alert in time.

**Cancellation needs a real mechanism**, not just "the client stops polling." Because the worker is
decoupled from the original request, the user needs an explicit way to say "stop this task," which
means a cancellation flag in the state store that the worker checks between steps (and ideally
mid-step for anything that itself takes a long time, like a long-running tool call), plus cleanup
logic for whatever partial side effects exist at the point of cancellation — an agent that's told to
stop mid-task should leave the system in a clean, understandable state, not an ambiguous
half-finished one.

```python
def should_continue(job_id: str, state_store) -> bool:
    status = state_store.get_status(job_id)
    if status.cancellation_requested:
        return False
    if status.elapsed_seconds > status.max_runtime_seconds:
        return False
    if status.total_cost_usd > status.max_cost_usd:
        return False
    return True
```

## Trade-off: Latency of the Async Model Itself

It's worth acknowledging directly that decoupling request from response adds its own latency and
complexity floor: enqueueing, dequeuing, and the poll/push round trip all add some delay compared to
a hypothetical synchronous call that could somehow return instantly, and building a robust
status/streaming layer is real engineering effort that a synchronous endpoint doesn't need. This is
why short, bounded tasks (a single tool-augmented Q&A turn that reliably finishes in a few seconds)
are usually still served synchronously — async infrastructure is a tool for tasks whose duration or
reliability profile genuinely requires it, not a default to reach for on every endpoint. The
practical rule of thumb: if a task's p99 duration comfortably fits within your infrastructure's
connection timeout with margin to spare, keep it synchronous for the simplicity; once p99 duration
approaches or exceeds that ceiling, or once task duration becomes unpredictable enough that a small
fraction of requests would blow through any reasonable timeout, that's the signal to move it to the
queue-and-worker model described in this chapter.

