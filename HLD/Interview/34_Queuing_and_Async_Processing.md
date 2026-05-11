# 34 — Queuing and Async Processing

---

## Easy (Q1–Q7)

---

### Q1. Why does async processing improve resilience and throughput?

**Answer:**

Asynchronous processing decouples the rate at which work arrives from the rate at which it is processed, providing three core benefits: resilience, throughput, and scalability.

**Resilience improvement:**
```
Synchronous (no queue):
  API Server ─── direct call ──► Email Service (down)
  API Server gets error → user sees "Internal Server Error"
  Downstream failure = upstream failure

Asynchronous (with queue):
  API Server ─── enqueue ──► Queue ──► Email Worker (restarts, retries)
  API Server gets "accepted" → user sees success
  Email Worker crashes → message stays in queue → worker restarts → reprocesses
  Downstream failure = delayed processing (not lost)
```

**Throughput improvement:**
```
Without queue (burst traffic problem):
  Black Friday: 10,000 orders/sec arrive simultaneously
  Payment processor: handles 1,000/sec max
  Result: 9,000 requests fail with timeout/overload errors

With queue (traffic smoothing):
  10,000 orders/sec → enqueue → Queue buffers the burst
  Payment workers: drain queue at 1,000/sec
  Result: All 10,000 orders processed over 10 seconds, none lost
  
Queue acts as a shock absorber
```

**Scalability:**
```
Without queue: To double throughput, must double API capacity AND worker capacity simultaneously
With queue:    Scale API and workers independently
               API scales for write bursts
               Workers scale for processing capacity
               They never need to be the same size
```

**Concrete latency improvement:**
```
Synchronous checkout (user waits for all steps):
  Validate order:   50ms
  Reserve inventory: 100ms
  Charge payment:   800ms
  Send email:       500ms
  Update analytics: 200ms
  Total user wait:  1,650ms

Async checkout (only critical path is synchronous):
  Validate order:   50ms  ← sync (user waits)
  Charge payment:   800ms ← sync (user waits)
  Everything else → enqueued, processed after response
  Total user wait:  850ms (48% faster)
```

---

### Q2. What is the difference between a job queue, message queue, and event stream?

**Answer:**

These three terms describe different models for async communication, often confused because the same tool (Kafka, RabbitMQ) can implement all three.

**Job Queue:**
```
Purpose: Execute a unit of work once, exactly by one worker
Model: Producer adds job; ONE consumer claims and executes it
Job is "consumed" (removed) after completion

Use cases: Send email, resize image, generate PDF, run report

Example (Celery/Sidekiq):
  Producer: email_queue.enqueue(send_welcome_email, user_id=42)
  Worker 1: picks up job → sends email → job deleted from queue
  Worker 2: (job already taken, doesn't process it)

Key properties:
  - Competing consumers (load balancing)
  - At-least-once or exactly-once processing
  - Dead letter queue for failed jobs
```

**Message Queue:**
```
Purpose: Decouple sender from receiver; sender doesn't know who processes
Model: Producer sends message; consumer(s) receive and process

Use cases: Order processing, command dispatch, work distribution

Example (RabbitMQ/SQS):
  Producer: queue.send({ "type": "order_placed", "order_id": 42 })
  Consumer: reads message, processes order, ACKs message
  
Key difference from job queue:
  - Message may be routed to different consumers by type
  - Consumer explicitly acknowledges before deletion
  - Visibility timeout: message hidden during processing
```

**Event Stream:**
```
Purpose: Publish facts (things that happened); multiple consumers read independently
Model: Producer appends event; each consumer group reads at its own offset
Events are NOT deleted after consumption — retained for replay

Use cases: Audit log, event sourcing, real-time analytics, multiple downstream systems

Example (Kafka):
  Producer: topic.publish("order.placed", { order_id: 42, ... })
  
  Payment Consumer Group:  reads at offset 42 → charges payment
  Analytics Consumer Group: reads at offset 42 → updates dashboard
  Notification Group:      reads at offset 42 → sends email
  
  Each group independently tracks its position (offset)
  Event stays in Kafka for retention period (7 days by default)

Key difference: Multiple independent consumers, retention, replay capability
```

**Summary:**

| Property | Job Queue | Message Queue | Event Stream |
|----------|-----------|---------------|--------------|
| Consumers | One (competing) | One (exclusive) or fan-out | Many (independent) |
| After read | Deleted | Deleted after ACK | Retained (offset-based) |
| Replay | No | No | Yes |
| Ordering | Optional | FIFO optional | Partition-ordered |
| Examples | Celery, Sidekiq | RabbitMQ, SQS | Kafka, Kinesis |
| Use for | Task execution | Command routing | Event-driven systems |

---

### Q3. What is a worker pool pattern and how do you size the worker pool?

**Answer:**

A worker pool maintains a fixed number of worker goroutines/threads/processes that pull jobs from a queue and execute them, preventing resource exhaustion from unbounded concurrent work.

**Without worker pool (unbounded):**
```
1000 jobs arrive → 1000 goroutines spawn → 1000 concurrent DB connections
→ DB connection pool exhausted → cascading failure
```

**Worker pool:**
```
Queue: [job1, job2, job3, ..., job1000]
         │
         ▼
┌──────────────────────────────────────────┐
│  Worker Pool (10 workers)                │
│  Worker 1: processing job 1              │
│  Worker 2: processing job 2              │
│  ...                                     │
│  Worker 10: processing job 10            │
│  (remaining 990 jobs wait in queue)      │
└──────────────────────────────────────────┘
```

**Implementation (Go):**
```go
func NewWorkerPool(queueURL string, workerCount int) *WorkerPool {
    pool := &WorkerPool{
        jobs:    make(chan Job, workerCount*2),
        done:    make(chan struct{}),
    }
    
    // Start N workers
    for i := 0; i < workerCount; i++ {
        go pool.worker(i)
    }
    
    return pool
}

func (p *WorkerPool) worker(id int) {
    for job := range p.jobs {
        start := time.Now()
        err := processJob(job)
        duration := time.Since(start)
        
        metrics.RecordJobDuration(duration)
        if err != nil {
            metrics.IncrementJobFailures()
            requeueWithBackoff(job)
        } else {
            metrics.IncrementJobSuccess()
        }
    }
}
```

**Sizing the worker pool:**

The optimal worker count depends on whether the workload is CPU-bound or I/O-bound:

```
CPU-bound jobs (image processing, encryption):
  Workers = CPU cores (or CPU cores - 1 for headroom)
  Adding more workers adds context-switching overhead, not throughput
  
  Rule: workers = os.cpu_count()

I/O-bound jobs (HTTP calls, DB queries, file I/O):
  Workers = much higher (waiting on I/O, not using CPU)
  
  Formula (Little's Law):
    Optimal workers = throughput × average_latency
    
  Example:
    Target: 100 jobs/sec
    Average job latency: 500ms (DB query + HTTP call)
    Optimal workers = 100 × 0.5 = 50 workers
    
  Add 20-30% buffer: 60-65 workers
```

**Dynamic worker pool (autoscaling):**
```python
class DynamicWorkerPool:
    def __init__(self, min_workers=5, max_workers=100):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
    
    async def autoscale(self):
        while True:
            queue_depth = await queue.depth()
            processing_rate = metrics.get_processing_rate()
            
            if queue_depth > self.current_workers * 10:
                # Scale up: queue backing up
                new_count = min(self.current_workers * 2, self.max_workers)
                await self.set_worker_count(new_count)
            elif queue_depth < self.current_workers * 2:
                # Scale down: workers idle
                new_count = max(self.current_workers // 2, self.min_workers)
                await self.set_worker_count(new_count)
            
            await asyncio.sleep(30)
```

---

### Q4. What is at-least-once delivery and how do you make job handlers idempotent?

**Answer:**

**At-least-once delivery** guarantees that a message will be delivered to a consumer at least once — but possibly more than once in failure scenarios (worker crash after processing but before acknowledging, network partition during ACK).

**Why at-least-once (not exactly-once):**
```
Exactly-once requires distributed coordination (expensive, complex)
Most systems offer at-least-once because:
  - Network can fail between "process" and "ack"
  - Worker can crash after processing but before ack
  - Broker redelivers unacknowledged messages
  
Risk: Job executed twice → charging user twice, sending duplicate email
Solution: Make handlers idempotent (safe to run twice)
```

**Idempotency patterns:**

**Pattern 1: Database deduplication key**
```python
async def handle_send_email(job: Job):
    idempotency_key = f"email:{job.data['email_type']}:{job.data['user_id']}"
    
    # Try to claim this job (atomic INSERT with unique constraint)
    try:
        await db.execute(
            """
            INSERT INTO processed_jobs (idempotency_key, processed_at)
            VALUES ($1, NOW())
            """,
            idempotency_key
        )
    except UniqueViolationError:
        logger.info(f"Duplicate job detected: {idempotency_key}, skipping")
        return  # Already processed, safe to ignore
    
    # Safe to process now
    await email_service.send(
        to=job.data['user_email'],
        template=job.data['email_type']
    )
```

**Pattern 2: Idempotency key on external API**
```python
async def handle_charge_payment(job: Job):
    # Use order_id as idempotency key for payment API
    # Payment provider will return same result if called twice with same key
    response = await stripe.charge.create(
        amount=job.data['amount'],
        currency="usd",
        customer=job.data['stripe_customer_id'],
        idempotency_key=f"order:{job.data['order_id']}"  # Stripe deduplicates
    )
    
    await db.update_order_payment(job.data['order_id'], response.id)
```

**Pattern 3: Check-then-act with optimistic locking**
```python
async def handle_fulfill_order(job: Job):
    order = await db.get_order(job.data['order_id'])
    
    if order.status != 'pending':
        logger.info(f"Order {order.id} already processed: {order.status}")
        return  # Idempotent: don't re-fulfill
    
    # CAS update: only update if status is still 'pending'
    updated = await db.execute(
        """
        UPDATE orders SET status = 'fulfilled'
        WHERE id = $1 AND status = 'pending'
        """,
        order.id
    )
    
    if updated == 0:
        return  # Another worker already updated
    
    await inventory_service.ship(order)
```

**SQS visibility timeout (prevents duplicate processing):**
```python
# SQS: message becomes "invisible" during processing
# If not deleted within visibility_timeout, it reappears for another worker
# Set visibility_timeout = expected_processing_time × 3 (safety margin)

await sqs.change_message_visibility(
    receipt_handle=message.receipt_handle,
    visibility_timeout=300  # 5 minutes for long job
)
```

---

### Q5. What is a dead letter queue (DLQ) and when does a job go there?

**Answer:**

A dead letter queue (DLQ) is a separate queue where messages are sent after they have failed processing a maximum number of times. It captures "poisoned" messages without blocking the main queue.

**Why DLQ is essential:**
```
Without DLQ:
  Malformed message → worker fails → message returns to queue
  → worker picks it up → fails → returns → loops forever
  → main queue clogged with poison messages
  → all workers stuck retrying bad messages
  → real work not processing

With DLQ:
  Message fails 3 times → moved to DLQ automatically
  → Main queue continues flowing
  → DLQ allows manual inspection and replay
```

**SQS DLQ configuration:**
```json
{
  "QueueName": "order-processing",
  "Attributes": {
    "RedrivePolicy": "{\"deadLetterTargetArn\": \"arn:aws:sqs:us-east-1:123:order-processing-dlq\",
                      \"maxReceiveCount\": \"3\"}"
  }
}
```

**Causes for a message landing in DLQ:**
```
1. Processing exceptions (bug in handler code)
2. Deserialization failure (malformed JSON/Protobuf)
3. Dependency unavailable after all retries (DB permanently down)
4. Business logic rejection (insufficient funds, invalid state)
5. Timeout exceeded (job takes longer than visibility_timeout)
6. Schema validation failure (consumer can't parse new event format)
```

**DLQ monitoring + alerting:**
```python
# Alert: DLQ depth > 0 (any failure is significant)
class DLQMonitor:
    async def check(self):
        depth = await sqs.get_queue_attributes(
            QueueUrl=DLQ_URL,
            AttributeNames=['ApproximateNumberOfMessages']
        )
        count = int(depth['Attributes']['ApproximateNumberOfMessages'])
        
        if count > 0:
            await alert_slack(
                f"DLQ has {count} failed messages — investigation needed\n"
                f"Queue: order-processing-dlq"
            )
        
        metrics.gauge("dlq.depth", count, tags={"queue": "order-processing"})
```

**DLQ replay after fixing the bug:**
```python
async def replay_dlq():
    """Move all DLQ messages back to main queue after fixing the bug."""
    while True:
        messages = await sqs.receive_message(
            QueueUrl=DLQ_URL,
            MaxNumberOfMessages=10
        )
        
        if not messages.get('Messages'):
            break
        
        for msg in messages['Messages']:
            # Re-send to main queue
            await sqs.send_message(
                QueueUrl=MAIN_QUEUE_URL,
                MessageBody=msg['Body']
            )
            # Delete from DLQ
            await sqs.delete_message(
                QueueUrl=DLQ_URL,
                ReceiptHandle=msg['ReceiptHandle']
            )
        
        logger.info(f"Replayed {len(messages['Messages'])} messages from DLQ")
```

---

### Q6. Explain retry with exponential backoff and jitter. Why is jitter important?

**Answer:**

Exponential backoff increases the wait time between retries exponentially to reduce load on a struggling service. Jitter adds randomness to prevent synchronized retry storms.

**Without jitter (thundering herd problem):**
```
1000 workers all fail at t=0:
  Retry 1: all retry at t=2s   → 1000 simultaneous requests → service overloads again
  Retry 2: all retry at t=4s   → 1000 simultaneous requests → fails again
  Retry 3: all retry at t=8s   → 1000 simultaneous requests → fails again
  
Workers are perfectly synchronized → they all hammer the service at the same time
```

**With jitter (spread the load):**
```
1000 workers fail at t=0:
  Retry 1: each waits 2s ± random(0, 2s) → spread over 4-second window
           → ~250 requests/sec instead of 1000/sec all at once
  Service recovers; retries succeed
```

**Implementation:**
```python
import random
import asyncio

async def retry_with_backoff(
    func,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True
):
    """
    Retry with exponential backoff + full jitter.
    Delay formula: random(0, min(cap, base * 2^attempt))
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except RetryableError as e:
            last_exception = e
            
            if attempt == max_retries:
                raise MaxRetriesExceeded(f"Failed after {max_retries} retries") from e
            
            # Exponential backoff
            delay = min(max_delay, base_delay * (backoff_multiplier ** attempt))
            
            if jitter:
                # Full jitter: random in [0, delay]
                delay = random.uniform(0, delay)
                # Alternative: decorrelated jitter (often better)
                # delay = random.uniform(base_delay, prev_delay * 3)
            
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay:.2f}s"
            )
            await asyncio.sleep(delay)
    
    raise last_exception

# Usage:
await retry_with_backoff(
    lambda: payment_service.charge(order_id=42),
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0
)
```

**Retry delay schedule (base=1s, multiplier=2, with jitter):**

| Attempt | Without Jitter | With Full Jitter |
|---------|---------------|-----------------|
| 1 | 1s | 0-1s |
| 2 | 2s | 0-2s |
| 3 | 4s | 0-4s |
| 4 | 8s | 0-8s |
| 5 | 16s | 0-16s |

**Which errors to retry:**
```python
RETRYABLE_ERRORS = {
    429,  # Too Many Requests
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

NON_RETRYABLE_ERRORS = {
    400,  # Bad Request (bug in caller, retrying won't help)
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not Found
    422,  # Validation Error
}
```

---

### Q7. What is a priority queue for async jobs? How do you implement it with multiple queues?

**Answer:**

A priority queue ensures high-priority jobs are processed before low-priority ones, even if low-priority jobs arrived first.

**Use cases:**
- P1: Password reset emails (time-sensitive, user is waiting)
- P2: Order confirmation emails
- P3: Marketing newsletters
- P4: Batch analytics exports

**Multi-queue approach (simplest and most common):**
```
High priority queue:   [P1 jobs]  ←── Worker checks first
Medium priority queue: [P2 jobs]  ←── Worker checks second
Low priority queue:    [P3 jobs]  ←── Worker checks last
Background queue:      [P4 jobs]  ←── Separate workers, can be starved

Worker algorithm:
  while running:
    job = high_queue.pop() or medium_queue.pop() or low_queue.pop()
    process(job)
```

**Implementation:**
```python
from redis import Redis

class PriorityJobQueue:
    QUEUES = {
        'critical': 'jobs:p1',
        'high':     'jobs:p2',
        'normal':   'jobs:p3',
        'low':      'jobs:p4',
    }
    
    def __init__(self):
        self.redis = Redis()
    
    def enqueue(self, job: dict, priority: str = 'normal'):
        queue_key = self.QUEUES[priority]
        self.redis.rpush(queue_key, json.dumps(job))
    
    def dequeue(self) -> dict | None:
        """Poll queues in priority order (strict priority)."""
        for queue_key in self.QUEUES.values():
            job_json = self.redis.lpop(queue_key)
            if job_json:
                return json.loads(job_json)
        return None  # All queues empty

# Workers with weighted allocation (prevent starvation):
class WeightedWorkerPool:
    """
    Allocate workers by priority:
    P1: 40% of workers always dedicated to critical
    P2: 30% of workers
    P3: 20% of workers
    P4: 10% of workers (background tasks)
    """
    WORKER_WEIGHTS = {'p1': 4, 'p2': 3, 'p3': 2, 'p4': 1}
    
    def start_workers(self, total_workers: int = 20):
        total = sum(self.WORKER_WEIGHTS.values())  # 10
        for priority, weight in self.WORKER_WEIGHTS.items():
            count = int(total_workers * weight / total)
            for _ in range(count):
                threading.Thread(
                    target=self.worker_loop,
                    args=(priority,),
                    daemon=True
                ).start()
```

**Redis sorted set for single-queue priority:**
```python
# Use Redis ZADD with priority score
# Lower score = higher priority (ZPOPMIN gets lowest score = highest priority)
redis.zadd('jobs', {job_json: priority_score})

# Worker:
job = redis.zpopmin('jobs', count=1)  # Gets highest priority job
```

---

## Medium (Q8–Q15)

---

### Q8. How do you schedule future and delayed jobs?

**Answer:**

Delayed jobs need to be queued but not processed until a future time (send email in 30 minutes, retry in 5 minutes, run report at 2am).

**Approach 1: Priority queue with future timestamp (Redis)**
```python
import time
import redis

class DelayedJobQueue:
    def __init__(self):
        self.redis = Redis()
        self.QUEUE_KEY = 'delayed_jobs'
        self.PROCESSING_KEY = 'processing_jobs'
    
    def schedule(self, job: dict, run_at: float):
        """Schedule job to run at unix timestamp run_at."""
        self.redis.zadd(self.QUEUE_KEY, {json.dumps(job): run_at})
    
    def schedule_in(self, job: dict, delay_seconds: float):
        """Schedule job to run after delay_seconds from now."""
        self.schedule(job, time.time() + delay_seconds)
    
    def poll_ready_jobs(self) -> list[dict]:
        """Atomically get all jobs whose run_at <= now."""
        now = time.time()
        
        # Lua script: atomic get + remove for jobs ready to run
        lua_script = """
        local jobs = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
        if #jobs > 0 then
            redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
        end
        return jobs
        """
        ready = self.redis.eval(lua_script, 1, self.QUEUE_KEY, now)
        return [json.loads(j) for j in ready]

# Scheduler loop (runs every second):
async def scheduler_loop(queue: DelayedJobQueue, work_queue: WorkQueue):
    while True:
        ready_jobs = queue.poll_ready_jobs()
        for job in ready_jobs:
            await work_queue.enqueue(job)  # Move to immediate processing queue
        await asyncio.sleep(1)
```

**Approach 2: SQS delay (built-in, up to 15 minutes)**
```python
await sqs.send_message(
    QueueUrl=QUEUE_URL,
    MessageBody=json.dumps(job),
    DelaySeconds=900  # Max: 900 seconds = 15 minutes
)
# SQS hides the message for 900 seconds before making it available
```

**Approach 3: PostgreSQL-based scheduled jobs (SKIP LOCKED)**
```sql
CREATE TABLE scheduled_jobs (
    id          BIGSERIAL PRIMARY KEY,
    payload     JSONB NOT NULL,
    run_at      TIMESTAMPTZ NOT NULL,
    status      VARCHAR(20) DEFAULT 'scheduled',
    attempts    INT DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scheduled_jobs_run_at ON scheduled_jobs(run_at)
WHERE status = 'scheduled';

-- Worker picks up ready jobs (SKIP LOCKED prevents double-processing)
BEGIN;
UPDATE scheduled_jobs
SET status = 'processing'
WHERE id IN (
    SELECT id FROM scheduled_jobs
    WHERE run_at <= NOW()
      AND status = 'scheduled'
    ORDER BY run_at
    LIMIT 10
    FOR UPDATE SKIP LOCKED
)
RETURNING *;
-- Process the returned jobs
COMMIT;
```

**Cron-based scheduling (APScheduler/Celery Beat):**
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

# Fixed interval
scheduler.add_job(run_daily_report, 'interval', hours=24)

# Cron expression (9am every weekday)
scheduler.add_job(send_weekly_digest, 'cron', 
                  day_of_week='mon-fri', hour=9, minute=0)

# One-time future job
scheduler.add_job(send_reminder, 'date',
                  run_date=datetime(2025, 6, 1, 9, 0))

scheduler.start()
```

---

### Q9. How do you implement job progress reporting? Compare polling vs WebSocket push.

**Answer:**

Long-running jobs (video encoding, data export) need a mechanism to report progress to waiting clients without blocking the HTTP connection.

**Pattern 1: Async job + polling endpoint**
```python
# Submit job: returns immediately with job ID
@app.post("/exports")
async def create_export(request: ExportRequest) -> dict:
    job_id = str(uuid.uuid4())
    
    await redis.hset(f"job:{job_id}", mapping={
        "status": "queued",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat()
    })
    
    # Enqueue background work
    await queue.enqueue({"type": "export", "job_id": job_id, **request.dict()})
    
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}

# Poll endpoint: client checks every few seconds
@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    data = await redis.hgetall(f"job:{job_id}")
    if not data:
        raise HTTPException(404, "Job not found")
    
    return {
        "job_id": job_id,
        "status": data["status"],    # queued|processing|completed|failed
        "progress": int(data["progress"]),  # 0-100
        "result_url": data.get("result_url"),
        "error": data.get("error")
    }

# Worker updates progress
async def process_export_job(job: dict):
    job_id = job["job_id"]
    total_rows = await count_rows(job["filters"])
    
    processed = 0
    async for batch in fetch_batches(job["filters"]):
        await write_to_csv(batch)
        processed += len(batch)
        
        # Update progress
        progress = int(processed / total_rows * 100)
        await redis.hset(f"job:{job_id}", mapping={
            "status": "processing",
            "progress": progress
        })
    
    result_url = await upload_to_s3(job_id)
    await redis.hset(f"job:{job_id}", mapping={
        "status": "completed",
        "progress": 100,
        "result_url": result_url
    })
```

**Polling client (JavaScript):**
```javascript
async function pollJobStatus(jobId) {
    const interval = setInterval(async () => {
        const response = await fetch(`/jobs/${jobId}`);
        const job = await response.json();
        
        updateProgressBar(job.progress);
        
        if (job.status === 'completed') {
            clearInterval(interval);
            window.location.href = job.result_url;  // Download
        } else if (job.status === 'failed') {
            clearInterval(interval);
            showError(job.error);
        }
    }, 2000);  // Poll every 2 seconds
}
```

**Pattern 2: Server-Sent Events (SSE) — one-way push from server**
```python
@app.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    async def event_generator():
        while True:
            data = await redis.hgetall(f"job:{job_id}")
            
            yield f"data: {json.dumps(data)}\n\n"
            
            if data["status"] in ("completed", "failed"):
                break
            
            await asyncio.sleep(1)
    
    return EventSourceResponse(event_generator())
```

**Pattern 3: WebSocket (bidirectional, low latency)**
```python
@app.websocket("/jobs/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    
    # Subscribe to Redis pub/sub for this job
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"job:{job_id}:progress")
    
    async for message in pubsub.listen():
        if message["type"] == "message":
            await websocket.send_json(json.loads(message["data"]))
    
    await websocket.close()

# Worker publishes progress updates
await redis.publish(f"job:{job_id}:progress",
                    json.dumps({"progress": 45, "status": "processing"}))
```

**Comparison:**

| Method | Use When | Complexity | Connection Overhead |
|--------|---------|------------|---------------------|
| Polling | Simple jobs, infrequent updates | Low | Per-poll request |
| SSE | Frequent updates, one-way | Low | One persistent connection |
| WebSocket | Real-time, bidirectional | Medium | One persistent connection |

---

### Q10. Why is exactly-once job execution hard? How do you approximate it?

**Answer:**

True exactly-once execution requires a distributed transaction across the message broker and the application side-effects — impossible without heavy coordination.

**The fundamental problem:**
```
Worker receives message → processes job → must:
  A: Delete message from queue (broker operation)
  B: Commit result to database (app operation)

Between A and B, anything can fail:
  - Worker crashes after DB commit but before queue delete
    → Message redelivered → job runs twice
  - Worker crashes after queue delete but before DB commit
    → Message gone → job never ran (at-most-once)

You cannot atomically commit to TWO different systems.
```

**Why exactly-once is "impossible" in the general case:**
```
CAP theorem implication:
  Distributed systems with network partitions cannot have:
    - Consistency (exactly-once) AND
    - Availability (always process)
    - Partition tolerance (network can fail)
    
  Most queuing systems choose availability + partition tolerance
  → At-least-once with idempotency is the practical solution
```

**Approximating exactly-once:**

**Approach 1: Transactional outbox + at-least-once + idempotency**
```python
# "Exactly once" = at-least-once + idempotent processing
async def process_order_payment(job: Job):
    # Guard: atomic claim prevents double-processing
    async with db.transaction():
        # Check if already processed (idempotency check inside transaction)
        already_done = await db.fetchval(
            "SELECT EXISTS(SELECT 1 FROM processed_jobs WHERE job_id = $1)",
            job.id
        )
        if already_done:
            return  # Idempotent skip
        
        # Process business logic
        payment = await charge_payment(job.data)
        
        # Record completion IN THE SAME TRANSACTION
        await db.execute(
            "INSERT INTO processed_jobs (job_id, result) VALUES ($1, $2)",
            job.id, payment.id
        )
    
    # Now safe to ACK the message
    await queue.ack(job.receipt_handle)
```

**Approach 2: Kafka exactly-once (Kafka Streams / transactions)**
```python
# Kafka supports exactly-once with transactions (read-process-write)
producer = KafkaProducer(
    enable_idempotence=True,           # Dedup at producer level
    transactional_id="payment-processor-1"
)

producer.init_transactions()

consumer = KafkaConsumer(
    isolation_level="read_committed"  # Only read committed messages
)

for message in consumer:
    producer.begin_transaction()
    try:
        result = process(message)
        
        # Write result and commit offset atomically
        producer.send("output-topic", result)
        producer.send_offsets_to_transaction(
            {TopicPartition("input", 0): OffsetAndMetadata(message.offset + 1, "")},
            consumer.config["group_id"]
        )
        producer.commit_transaction()
        
    except Exception:
        producer.abort_transaction()
```

**Approach 3: Outbox pattern (database + queue atomically)**
```sql
-- Write to DB and "outbox" table in one transaction
BEGIN;
UPDATE orders SET status = 'fulfilled' WHERE id = 42;
INSERT INTO outbox (topic, key, payload) 
VALUES ('order.fulfilled', '42', '{"order_id": 42, "status": "fulfilled"}');
COMMIT;

-- Outbox worker publishes to Kafka and deletes from outbox
-- If publish fails: retry from outbox (DB record persists)
-- If ACK fails after publish: DB record already there, dedup on consumer side
```

---

### Q11. How do you handle long-running jobs with heartbeats and idempotent resume?

**Answer:**

Jobs that take minutes or hours need: a mechanism to signal they're still alive (heartbeat), automatic timeout if they stall, and the ability to resume from where they left off after a crash.

**The problem:**
```
Job: "Export 10 million rows to CSV" (estimated: 15 minutes)
Worker crashes at minute 7 (processed 4.7M rows)
Queue's visibility_timeout = 10 minutes
→ Message becomes visible → another worker picks it up
→ Starts from beginning → wastes 7 minutes → may crash again
```

**Heartbeat implementation (SQS):**
```python
import asyncio

class LongRunningJobHandler:
    HEARTBEAT_INTERVAL = 30  # seconds
    VISIBILITY_TIMEOUT = 60  # Must be > HEARTBEAT_INTERVAL
    
    async def process(self, message: SQSMessage):
        # Start heartbeat task in background
        heartbeat_task = asyncio.create_task(
            self._heartbeat(message.receipt_handle)
        )
        
        try:
            await self._do_work(message.body)
            await sqs.delete_message(message.receipt_handle)
        except Exception as e:
            logger.error(f"Job failed: {e}")
            raise
        finally:
            heartbeat_task.cancel()
    
    async def _heartbeat(self, receipt_handle: str):
        """Periodically extend message visibility to prevent timeout."""
        while True:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            try:
                await sqs.change_message_visibility(
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=self.VISIBILITY_TIMEOUT
                )
                logger.debug("Heartbeat sent — job still running")
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                # Job may be re-queued — prepare for graceful shutdown
```

**Idempotent resume with checkpoint:**
```python
async def export_large_dataset(job: dict):
    export_id = job['export_id']
    
    # Load checkpoint (where did we leave off?)
    checkpoint = await redis.hget(f"export:{export_id}:checkpoint", "last_offset")
    start_offset = int(checkpoint) if checkpoint else 0
    
    total = await db.fetchval("SELECT COUNT(*) FROM records WHERE filter=$1", job['filter'])
    
    with open(f"/tmp/export-{export_id}.csv", 'a' if start_offset > 0 else 'w') as f:
        writer = csv.writer(f)
        
        offset = start_offset
        batch_size = 1000
        
        while offset < total:
            # Fetch batch
            batch = await db.fetch(
                "SELECT * FROM records WHERE filter=$1 LIMIT $2 OFFSET $3",
                job['filter'], batch_size, offset
            )
            
            for row in batch:
                writer.writerow(row)
            
            offset += len(batch)
            
            # Save checkpoint every batch
            await redis.hset(f"export:{export_id}:checkpoint", "last_offset", offset)
        
    # Upload completed file
    result_url = await s3_upload(f"/tmp/export-{export_id}.csv")
    await redis.delete(f"export:{export_id}:checkpoint")  # Clean up
    return result_url

# If worker crashes at offset 4700:
# New worker picks up job → loads checkpoint (4700) → resumes from row 4700
# Rows 0-4699 not re-processed
```

---

### Q12. Explain the fan-out job pattern (map-reduce model for async work).

**Answer:**

Fan-out jobs split one large unit of work into many parallel sub-jobs, then aggregate the results — the async equivalent of MapReduce.

**Use case:** Generate a personalized weekly digest email for 1 million users.

```
Without fan-out:
  One job processes all 1M users sequentially
  Time: 1M × 50ms = ~14 hours

With fan-out:
  Parent job: splits into 1000 child jobs (1000 users each)
  1000 workers process in parallel
  Time: 1000 × 50ms + coordination ≈ 2 minutes
  Speedup: 420x
```

**Fan-out pattern:**
```python
class WeeklyDigestFanout:
    
    async def create_parent_job(self) -> str:
        job_id = str(uuid.uuid4())
        
        # Count total users
        total_users = await db.fetchval("SELECT COUNT(*) FROM users WHERE digest_enabled=true")
        batch_size = 1000
        total_batches = math.ceil(total_users / batch_size)
        
        # Track completion
        await redis.hset(f"fanout:{job_id}", mapping={
            "total": total_batches,
            "completed": 0,
            "failed": 0,
            "status": "running"
        })
        
        # Enqueue child jobs
        for batch_num in range(total_batches):
            await queue.enqueue({
                "type": "digest_batch",
                "parent_job_id": job_id,
                "batch_num": batch_num,
                "offset": batch_num * batch_size,
                "limit": batch_size
            })
        
        return job_id
    
    async def process_child_job(self, job: dict):
        """Process one batch of users."""
        parent_id = job["parent_job_id"]
        
        # Fetch user batch
        users = await db.fetch(
            "SELECT * FROM users WHERE digest_enabled=true LIMIT $1 OFFSET $2",
            job["limit"], job["offset"]
        )
        
        # Process each user
        for user in users:
            digest = await generate_digest(user)
            await email_queue.enqueue({"to": user.email, "body": digest})
        
        # Atomically increment completed count
        completed = await redis.hincrby(f"fanout:{parent_id}", "completed", 1)
        total = int(await redis.hget(f"fanout:{parent_id}", "total"))
        
        # Check if all children done
        if completed == total:
            await redis.hset(f"fanout:{parent_id}", "status", "completed")
            await notify_parent_job_complete(parent_id)
    
    async def process_child_failure(self, job: dict, error: Exception):
        """Handle child job failure."""
        parent_id = job["parent_job_id"]
        
        failed = await redis.hincrby(f"fanout:{parent_id}", "failed", 1)
        completed = await redis.hincrby(f"fanout:{parent_id}", "completed", 1)
        total = int(await redis.hget(f"fanout:{parent_id}", "total"))
        
        logger.error(f"Child job {job['batch_num']} failed: {error}")
        
        if completed == total:
            status = "completed_with_failures" if failed > 0 else "completed"
            await redis.hset(f"fanout:{parent_id}", "status", status)
```

**Airflow DAG equivalent:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

with DAG("weekly_digest", schedule_interval="0 9 * * MON") as dag:
    
    with TaskGroup("generate_digests") as generate:
        for batch_num in range(100):  # 100 batches of 10,000 users
            PythonOperator(
                task_id=f"batch_{batch_num}",
                python_callable=process_user_batch,
                op_kwargs={"batch": batch_num}
            )
    
    send_emails = PythonOperator(
        task_id="send_emails",
        python_callable=flush_email_queue
    )
    
    generate >> send_emails
```

---

### Q13. How do you manage job queue backlog when queue depth grows unbounded?

**Answer:**

An unbounded queue backlog means producers are adding jobs faster than consumers process them. If left unchecked, the queue grows until it exhausts memory, disk, or processing lag exceeds business SLAs.

**Detecting a backlog:**
```
Queue depth metrics:
  Normal:   queue_depth < 1000, lag < 30s
  Warning:  queue_depth > 10,000, lag > 5 minutes
  Critical: queue_depth > 100,000, lag > 1 hour (SLA violated)
```

**Strategies for managing backlog:**

**Strategy 1: Scale consumers (fastest response)**
```python
class AutoscalingWorkerPool:
    async def scale_on_backlog(self):
        depth = await queue.depth()
        current_workers = self.worker_count
        
        if depth > 10_000 and current_workers < 100:
            # Add workers proportionally
            new_workers = min(100, current_workers * 2)
            await self.set_worker_count(new_workers)
            logger.info(f"Scaled workers {current_workers} → {new_workers}")
```

**Strategy 2: Load shedding (drop low-priority jobs)**
```python
async def enqueue_with_backpressure(job: Job):
    depth = await queue.depth()
    
    # Drop low-priority jobs when queue is deep
    if depth > 100_000 and job.priority >= 3:
        metrics.increment("jobs.dropped", tags={"priority": job.priority})
        logger.warning(f"Dropping P{job.priority} job due to backlog")
        return  # Graceful drop
    
    if depth > 500_000:
        raise QueueFullError("Queue critically overloaded — all new jobs rejected")
    
    await queue.enqueue(job)
```

**Strategy 3: Job expiry (TTL-based)**
```python
async def enqueue_with_ttl(job: Job, ttl_seconds: int = 3600):
    job.expires_at = time.time() + ttl_seconds
    await queue.enqueue(job)

async def process_job_with_expiry(job: Job):
    # Check if job is still relevant
    if job.expires_at and time.time() > job.expires_at:
        metrics.increment("jobs.expired")
        logger.info(f"Job {job.id} expired — skipping")
        return  # Skip stale job
    
    await process(job)
```

**Strategy 4: Batch processing (increase throughput per worker)**
```python
async def batch_worker(queue, batch_size: int = 100):
    """Process multiple jobs in one operation."""
    while True:
        # Dequeue up to 100 jobs at once
        jobs = await queue.dequeue_batch(max_count=batch_size)
        
        if not jobs:
            await asyncio.sleep(1)
            continue
        
        # Process as batch (e.g., bulk DB insert vs 100 individual inserts)
        await process_batch(jobs)
```

**Strategy 5: Queue compression (deduplicate)**
```python
# For idempotent jobs: if same job already queued, don't add duplicate
async def enqueue_unique(job: Job):
    key = f"job_dedup:{job.type}:{job.entity_id}"
    
    # Only enqueue if not already queued
    was_set = await redis.set(key, 1, nx=True, ex=300)  # 5-minute dedup window
    
    if was_set:
        await queue.enqueue(job)
    else:
        metrics.increment("jobs.deduplicated")
```

**Backlog monitoring dashboard:**
```
Grafana panels:
  - Queue depth (rate + current) by queue name
  - Worker count vs. queue depth (correlation)
  - Processing rate (jobs/sec)
  - Consumer lag (oldest unprocessed message age)
  - DLQ depth
  - Jobs dropped/expired rate

Alert: consumer lag > 30 minutes → page oncall
Alert: DLQ depth > 100 → investigate
```

---

### Q14. How do you design a distributed cron scheduler with at-most-once execution?

**Answer:**

A distributed cron scheduler must execute jobs on schedule across multiple nodes, ensuring that even though many nodes run the scheduler code, each job fires exactly once (at-most-once: better to skip than to double-fire for idempotent jobs).

**Key challenge:**
```
3 scheduler nodes all running the same cron configuration
09:00:00 → all 3 nodes see "run daily_report" at the same time
→ Without coordination: 3 executions of daily_report
```

**Approach 1: Leader election (Redis-based)**
```python
import redis
import time
from datetime import datetime

LOCK_TTL = 5  # seconds — leader must renew every 5s

class DistributedCronScheduler:
    def __init__(self, node_id: str):
        self.redis = Redis()
        self.node_id = node_id
        self.is_leader = False
    
    async def run_leader_election(self):
        """Continuously compete for leadership."""
        while True:
            # Try to acquire leader lock (SET NX PX)
            acquired = self.redis.set(
                "cron:leader",
                self.node_id,
                nx=True,       # Only set if not exists
                px=5000        # 5 second TTL (auto-expires if node crashes)
            )
            
            if acquired:
                self.is_leader = True
                # Renew lock every 2 seconds
                asyncio.create_task(self._renew_lock())
            else:
                self.is_leader = False
            
            await asyncio.sleep(3)
    
    async def _renew_lock(self):
        """Leader renews its lock periodically."""
        while self.is_leader:
            current_leader = self.redis.get("cron:leader")
            if current_leader == self.node_id.encode():
                self.redis.expire("cron:leader", 5)  # Extend TTL
            else:
                self.is_leader = False  # Lost leadership
                break
            await asyncio.sleep(2)
    
    async def run_scheduler(self):
        """Only leader runs scheduled jobs."""
        await self.run_leader_election()  # Background task
        
        while True:
            if self.is_leader:
                await self.check_and_fire_due_jobs()
            await asyncio.sleep(1)  # Check every second
    
    async def check_and_fire_due_jobs(self):
        now = datetime.utcnow()
        for job in self.cron_jobs:
            if job.is_due(now):
                await self._fire_job_once(job, now)
    
    async def _fire_job_once(self, job, scheduled_time: datetime):
        """Prevent double-firing with Redis lock."""
        execution_key = f"cron:executed:{job.name}:{scheduled_time.strftime('%Y%m%d%H%M')}"
        
        # Atomic claim: only one node fires this minute's job
        claimed = self.redis.set(execution_key, self.node_id, nx=True, ex=120)
        
        if claimed:
            await self.work_queue.enqueue({
                "type": job.name,
                "scheduled_at": scheduled_time.isoformat()
            })
            logger.info(f"Cron fired: {job.name} at {scheduled_time}")
```

**Approach 2: Database-locked cron entries**
```sql
CREATE TABLE cron_jobs (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) UNIQUE,
    schedule    VARCHAR(100),    -- cron expression: '0 9 * * 1'
    last_run_at TIMESTAMPTZ,
    locked_by   VARCHAR(100),    -- node that's currently executing
    locked_at   TIMESTAMPTZ
);

-- Atomic claim: only one node updates successfully
UPDATE cron_jobs
SET locked_by = $1, locked_at = NOW()
WHERE name = 'daily_report'
  AND (locked_at IS NULL OR locked_at < NOW() - INTERVAL '10 minutes')
  AND (last_run_at IS NULL OR last_run_at < NOW() - INTERVAL '23 hours')
  -- Only if no recent run AND lock expired
```

**Missed job handling (clock drift, node down):**
```python
async def check_missed_jobs(self):
    """Run on startup: catch up any jobs that were missed while scheduler was down."""
    for job in self.cron_jobs:
        missed_executions = job.get_missed_executions(since=last_shutdown_time)
        for execution_time in missed_executions:
            if job.should_backfill:
                await self._fire_job_once(job, execution_time)
            else:
                logger.warning(f"Skipped missed execution: {job.name} at {execution_time}")
```

---

### Q15. Explain the database-as-a-queue anti-pattern. When is it acceptable?

**Answer:**

The database-as-a-queue anti-pattern uses a relational database table as a job queue, with workers polling for pending rows. This pattern has serious performance issues at scale but is acceptable in specific circumstances.

**The implementation:**
```sql
CREATE TABLE jobs (
    id          BIGSERIAL PRIMARY KEY,
    payload     JSONB,
    status      VARCHAR(20) DEFAULT 'pending',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    worker_id   VARCHAR(100)
);

-- Worker polling loop (anti-pattern at scale):
LOOP:
  UPDATE jobs
  SET status = 'processing', worker_id = $worker_id
  WHERE id = (
    SELECT id FROM jobs WHERE status = 'pending'
    ORDER BY created_at LIMIT 1
    FOR UPDATE SKIP LOCKED
  )
  RETURNING *;
```

**Why it's an anti-pattern:**

| Problem | Description |
|---------|-------------|
| Polling overhead | Workers constantly query DB even when queue is empty ("thundering herd" on DB) |
| Table bloat | Old completed/failed rows accumulate; VACUUM pressure |
| Index degradation | Partial index on status degrades as rows accumulate |
| Row locking | High write + read on same table causes lock contention |
| Throughput limit | DB typically handles 10k-50k jobs/sec max (vs Kafka: millions/sec) |
| No fan-out | One consumer gets each message; no built-in pub/sub |

**When it IS acceptable:**

```
Acceptable scenarios:
  1. Low volume: < 1,000 jobs/day (no performance concern)
  2. Transactional consistency required:
     "Job creation MUST be in same transaction as business data"
     Example: Create order → atomically enqueue order_placed job
     → If order creation fails, job is also rolled back (no orphaned jobs)
  
  3. Already using PostgreSQL, no budget for Redis/Kafka
  4. Simple retry/DLQ requirements
  5. Audit trail important (all job history in DB, easily queryable)
```

**When to migrate to a real queue:**
```
Migrate when:
  - Workers spend > 20% of time idle-polling
  - Job throughput > 1,000/minute
  - Need pub/sub (multiple consumers same message)
  - Need delayed jobs > 15 seconds
  - DB CPU > 60% and jobs are a significant contributor
```

**Acceptable version (with SKIP LOCKED — PostgreSQL's proper queue primitive):**
```sql
-- SKIP LOCKED prevents workers from blocking each other
BEGIN;
SELECT * FROM jobs
WHERE status = 'pending'
ORDER BY priority DESC, created_at ASC
LIMIT 10
FOR UPDATE SKIP LOCKED;

UPDATE jobs SET status = 'processing', worker_id = 'worker-1'
WHERE id IN (selected_ids);
COMMIT;

-- After processing:
UPDATE jobs SET status = 'completed' WHERE id = $1;

-- Cleanup: archive or delete completed jobs regularly
DELETE FROM jobs WHERE status = 'completed' AND processed_at < NOW() - INTERVAL '7 days';
```

SKIP LOCKED was specifically added to PostgreSQL 9.5 to enable the queue pattern efficiently. Without it, the pattern is especially harmful.

---

## Hard (Q16–Q20)

---

### Q16. Design a distributed job queue system handling 100,000 jobs per second with priority, idempotency, and observability.

**Answer:**

At 100k jobs/sec, a single queue node is a bottleneck. The architecture requires partitioning, distributed workers, and careful observability.

**Architecture:**
```
Producers (API servers)
    │
    ▼
Kafka (10 partitions × 10k jobs/sec/partition = 100k total)
    ├── p0:  user_id % 10 == 0
    ├── p1:  user_id % 10 == 1
    │...
    └── p9:  user_id % 10 == 9

Consumer Groups (auto-scaled workers)
    ├── Critical Worker Group (P1 jobs): 20 pods
    ├── High Worker Group (P2 jobs):     30 pods
    ├── Normal Worker Group (P3 jobs):   40 pods
    └── Low Worker Group (P4 jobs):      10 pods

Results:
    Redis (job status, idempotency keys)
    PostgreSQL (completed job audit trail)
    
Monitoring:
    Prometheus → Grafana (queue depth, processing rate, error rate)
    Kafka consumer lag → PagerDuty alert if lag > 10,000
```

**Priority routing with Kafka:**
```python
# Priority encoded in Kafka topic name
PRIORITY_TOPICS = {
    1: "jobs.critical",   # Always processed first
    2: "jobs.high",
    3: "jobs.normal",
    4: "jobs.low"
}

class JobProducer:
    def __init__(self):
        self.kafka = KafkaProducer(
            bootstrap_servers="kafka:9092",
            value_serializer=lambda v: json.dumps(v).encode()
        )
    
    async def submit(self, job_type: str, payload: dict, priority: int = 3) -> str:
        job_id = ulid.new().str  # ULID: sortable + unique
        
        message = {
            "job_id": job_id,
            "type": job_type,
            "payload": payload,
            "priority": priority,
            "submitted_at": datetime.utcnow().isoformat(),
            "idempotency_key": payload.get("idempotency_key", job_id)
        }
        
        topic = PRIORITY_TOPICS[priority]
        
        # Use entity_id as partition key for ordering
        partition_key = str(payload.get("user_id", job_id))
        
        await self.kafka.send(topic, key=partition_key.encode(), value=message)
        
        # Track job status in Redis
        await redis.setex(
            f"job:{job_id}:status",
            86400,  # 24-hour TTL
            json.dumps({"status": "queued", "priority": priority})
        )
        
        return job_id
```

**Idempotency with Redis:**
```python
class IdempotentJobConsumer:
    IDEMPOTENCY_TTL = 86400  # 24 hours
    
    async def handle(self, message: KafkaMessage):
        job = json.loads(message.value)
        idem_key = f"idem:{job['idempotency_key']}"
        
        # Atomic SET NX: only one worker processes per idempotency key
        claimed = await redis.set(idem_key, job['job_id'],
                                   nx=True, ex=self.IDEMPOTENCY_TTL)
        
        if not claimed:
            existing = await redis.get(idem_key)
            logger.info(f"Duplicate job detected: {job['job_id']} → idempotency key already processed by {existing}")
            return  # Skip duplicate
        
        try:
            await self._process(job)
            await self._mark_complete(job['job_id'])
        except Exception as e:
            await redis.delete(idem_key)  # Release idempotency claim on failure
            raise
```

**Observability instrumentation:**
```python
from prometheus_client import Counter, Histogram, Gauge

jobs_processed = Counter("jobs_processed_total",
                          "Total jobs processed",
                          labelnames=["job_type", "priority", "status"])
job_duration = Histogram("job_processing_seconds",
                          "Job processing time",
                          labelnames=["job_type", "priority"],
                          buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 30])
queue_depth = Gauge("queue_depth",
                     "Current queue depth",
                     labelnames=["topic", "priority"])

async def process_with_metrics(job: dict):
    with job_duration.labels(job['type'], job['priority']).time():
        try:
            await process_job(job)
            jobs_processed.labels(job['type'], job['priority'], 'success').inc()
        except Exception:
            jobs_processed.labels(job['type'], job['priority'], 'failure').inc()
            raise
```

---

### Q17. Design a job pipeline for processing 1 billion video transcoding jobs with dependencies, retries, and cost optimization.

**Answer:**

Video transcoding at scale requires: dependency management, cost-efficient worker types, intelligent retries, and progress reporting.

**Job DAG (Directed Acyclic Graph):**
```
upload_received
      │
      ▼
validate_format ──── (fails) ──► notify_upload_error
      │
      ▼
extract_metadata
      │
      ├──► generate_thumbnail (no dependencies)
      │
      ├──► transcode_360p  ─┐
      ├──► transcode_720p  ─┼──► package_hls ──► update_cdn
      ├──► transcode_1080p ─┘
      │
      └──► generate_subtitles (AI, slow)
                │
                ▼
           update_video_record (wait for all above)
```

**Dependency tracking with Redis:**
```python
class JobDAGManager:
    async def submit_dag(self, video_id: str, dag: DAG) -> str:
        dag_id = str(uuid.uuid4())
        
        # Store all jobs with their dependency counts
        for job in dag.jobs:
            deps = len(dag.get_dependencies(job.id))
            await redis.hset(f"dag:{dag_id}:job:{job.id}", mapping={
                "status": "waiting" if deps > 0 else "ready",
                "pending_deps": deps,
                "payload": json.dumps(job.payload)
            })
        
        # Submit all "ready" jobs (no dependencies) immediately
        for job in dag.get_root_jobs():
            await job_queue.enqueue(job, dag_id=dag_id)
        
        return dag_id
    
    async def on_job_completed(self, dag_id: str, job_id: str):
        """When a job completes, unblock its dependents."""
        dag = await self.load_dag(dag_id)
        
        for dependent_id in dag.get_dependents(job_id):
            # Atomic decrement: when pending_deps reaches 0, job is ready
            remaining = await redis.hincrby(
                f"dag:{dag_id}:job:{dependent_id}",
                "pending_deps", -1
            )
            
            if remaining == 0:
                job_data = await redis.hgetall(f"dag:{dag_id}:job:{dependent_id}")
                await job_queue.enqueue(json.loads(job_data["payload"]))
                await redis.hset(f"dag:{dag_id}:job:{dependent_id}", "status", "queued")
```

**Cost-optimized worker selection:**
```python
JOB_WORKER_TYPES = {
    "validate_format":    "cpu-small",      # Fast, cheap EC2
    "transcode_360p":     "cpu-spot",       # Can use spot instances
    "transcode_720p":     "cpu-spot",       # Spot: 70% cheaper
    "transcode_1080p":    "cpu-4xlarge",    # Needs more CPU, use spot
    "generate_thumbnail": "cpu-small",
    "generate_subtitles": "gpu-spot",       # GPU for AI transcription
    "package_hls":        "cpu-small",
    "update_cdn":         "cpu-small"
}

class CostAwareJobRouter:
    def route(self, job: Job) -> str:
        worker_type = JOB_WORKER_TYPES.get(job.type, "cpu-small")
        
        # Check spot availability
        if "spot" in worker_type and not self.spot_available(worker_type):
            # Fallback to on-demand if spot unavailable
            worker_type = worker_type.replace("spot", "ondemand")
        
        return QUEUE_NAMES[worker_type]
```

**Retry strategy with escalating worker size:**
```python
RETRY_STRATEGY = {
    "transcode_1080p": [
        {"max_attempts": 3, "worker": "cpu-4xlarge-spot",  "backoff": "exponential"},
        {"max_attempts": 2, "worker": "cpu-8xlarge-ondemand", "backoff": "fixed:60"},
        {"max_attempts": 1, "worker": "cpu-16xlarge-ondemand", "backoff": "none"},
    ]
}

async def handle_job_failure(job: Job, error: Exception, attempt: int):
    strategy = RETRY_STRATEGY.get(job.type, [DEFAULT_RETRY])
    
    tier = min(attempt, len(strategy) - 1)
    retry_config = strategy[tier]
    
    if attempt >= sum(t["max_attempts"] for t in strategy):
        await move_to_dlq(job, error)
        return
    
    await job_queue.enqueue(
        job,
        worker_type=retry_config["worker"],
        delay=calculate_backoff(retry_config["backoff"], attempt)
    )
```

---

### Q18. How do you implement job cancellation that propagates to in-progress workers?

**Answer:**

Job cancellation must handle three states: queued (easy — just delete), in-progress (hard — must signal worker), and completed (no-op).

**Three-phase cancellation:**

**Phase 1: Mark as cancelled (before worker picks it up)**
```python
async def cancel_job(job_id: str) -> CancellationResult:
    async with db.transaction():
        job = await db.fetchrow(
            "SELECT * FROM jobs WHERE id = $1 FOR UPDATE",
            job_id
        )
        
        if job['status'] == 'pending':
            # Easy: delete from queue before worker sees it
            await db.execute("UPDATE jobs SET status='cancelled' WHERE id=$1", job_id)
            await queue.remove(job_id)  # Remove from Redis/SQS
            return CancellationResult.CANCELLED_BEFORE_START
        
        elif job['status'] == 'processing':
            # Hard: signal running worker
            await db.execute(
                "UPDATE jobs SET status='cancellation_requested' WHERE id=$1",
                job_id
            )
            # Publish cancellation signal
            await redis.publish(f"job:{job_id}:cancel", "cancel")
            return CancellationResult.CANCELLATION_REQUESTED
        
        elif job['status'] in ('completed', 'failed', 'cancelled'):
            return CancellationResult.ALREADY_TERMINAL
```

**Phase 2: Worker polls for cancellation during processing**
```python
class CancellableJobWorker:
    def __init__(self):
        self.cancel_subscriptions: dict[str, asyncio.Event] = {}
    
    async def process_job(self, job: Job):
        job_id = job.id
        
        # Subscribe to cancellation channel
        cancel_event = asyncio.Event()
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"job:{job_id}:cancel")
        
        async def listen_for_cancel():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    cancel_event.set()
        
        cancel_task = asyncio.create_task(listen_for_cancel())
        
        try:
            await self._process_with_cancellation(job, cancel_event)
        finally:
            cancel_task.cancel()
            await pubsub.unsubscribe(f"job:{job_id}:cancel")
    
    async def _process_with_cancellation(self, job: Job, cancel: asyncio.Event):
        """Long-running job with periodic cancellation check."""
        
        for batch in await get_batches(job.data):
            # Check for cancellation between batches
            if cancel.is_set():
                logger.info(f"Job {job.id} cancelled during processing")
                await self._cleanup(job)
                await db.execute(
                    "UPDATE jobs SET status='cancelled' WHERE id=$1",
                    job.id
                )
                return
            
            await process_batch(batch)
            await checkpoint(job.id, batch.offset)
```

**Phase 3: Cascading cancellation (cancel parent → cancel all children)**
```python
async def cancel_dag(dag_id: str):
    """Cancel all jobs in a DAG."""
    jobs = await db.fetch(
        "SELECT id, status FROM jobs WHERE dag_id = $1",
        dag_id
    )
    
    cancellation_results = await asyncio.gather(*[
        cancel_job(job['id'])
        for job in jobs
        if job['status'] not in ('completed', 'failed')
    ])
    
    cancelled_count = sum(1 for r in cancellation_results
                          if r != CancellationResult.ALREADY_TERMINAL)
    
    await db.execute(
        "UPDATE dags SET status='cancelled' WHERE id=$1",
        dag_id
    )
    
    return {"cancelled": cancelled_count, "total": len(jobs)}
```

---

### Q19. Design a real-time notification system using async processing that guarantees delivery across 5 channels (email, SMS, push, in-app, Slack).

**Answer:**

**Architecture:**
```
Event Sources → Event Bus → Notification Fanout → Channel Workers
                            (Kafka)
  Order Service → order.shipped ─────────────────► Email Worker
  Auth Service  → password.reset ─────────────────► SMS Worker
  Any Service   → any.event ───────────────────────► Push Worker
                                                  ──► In-App Worker
                                                  ──► Slack Worker
```

**Event to notification mapping:**
```python
NOTIFICATION_RULES = {
    "order.shipped": [
        {"channel": "email",  "template": "order_shipped_email",  "priority": 2},
        {"channel": "push",   "template": "order_shipped_push",   "priority": 2},
        {"channel": "in_app", "template": "order_shipped_inapp",  "priority": 2},
    ],
    "password.reset": [
        {"channel": "email",  "template": "password_reset_email", "priority": 1},
        # SMS only if email fails
    ],
    "payment.failed": [
        {"channel": "email",  "template": "payment_failed_email", "priority": 1},
        {"channel": "sms",    "template": "payment_failed_sms",   "priority": 1},
        {"channel": "push",   "template": "payment_failed_push",  "priority": 1},
    ]
}
```

**Fanout consumer (Kafka → per-channel queues):**
```python
class NotificationFanout:
    async def handle_event(self, event: dict):
        user = await user_service.get(event['user_id'])
        prefs = await get_user_notification_preferences(user.id, event['type'])
        
        rules = NOTIFICATION_RULES.get(event['type'], [])
        
        tasks = []
        for rule in rules:
            channel = rule['channel']
            
            # Check user preferences
            if not prefs.get(channel, True):
                continue  # User disabled this channel
            
            # Check quiet hours
            if is_quiet_hours(user.timezone, user.quiet_hours) and rule['priority'] > 1:
                # Schedule for end of quiet hours
                send_at = next_quiet_hours_end(user.timezone, user.quiet_hours)
            else:
                send_at = None  # Send now
            
            notification = await create_notification(
                user_id=user.id,
                channel=channel,
                template=rule['template'],
                variables=event,
                priority=rule['priority'],
                send_at=send_at
            )
            
            tasks.append(
                channel_queue[channel].enqueue(
                    notification.id,
                    priority=rule['priority'],
                    delay=calculate_delay(send_at)
                )
            )
        
        await asyncio.gather(*tasks)
```

**Per-channel worker with retry + fallback:**
```python
class EmailWorker:
    PROVIDERS = ["sendgrid", "mailgun", "ses"]  # Fallback chain
    
    async def process(self, notification_id: str):
        notif = await db.get_notification(notification_id)
        
        for provider_name in self.PROVIDERS:
            try:
                provider = get_provider(provider_name)
                result = await asyncio.wait_for(
                    provider.send(
                        to=notif.user_email,
                        subject=notif.subject,
                        html=notif.body
                    ),
                    timeout=10.0
                )
                
                await db.log_delivery(notification_id, 'sent',
                                       provider=provider_name,
                                       provider_msg_id=result.message_id)
                return  # Success
                
            except (ProviderError, asyncio.TimeoutError) as e:
                await db.log_delivery(notification_id, 'provider_failed',
                                       provider=provider_name, error=str(e))
                logger.warning(f"Provider {provider_name} failed, trying next")
                continue
        
        # All providers failed
        await db.log_delivery(notification_id, 'failed',
                               error="All providers exhausted")
        raise AllProvidersFailedError()
```

**Delivery guarantee tracking:**
```sql
-- Notification with all channel delivery statuses
SELECT
    n.id,
    n.type,
    ndl.channel,
    ndl.status,
    ndl.provider,
    ndl.timestamp
FROM notifications n
JOIN notification_delivery_log ndl ON n.id = ndl.notification_id
WHERE n.user_id = 42
ORDER BY n.created_at DESC, ndl.timestamp ASC;
```

---

### Q20. Design a job system for a financial institution that requires exactly-once execution, audit trail, and regulatory compliance.

**Answer:**

Financial job systems must be: exactly-once (no duplicate payments), fully auditable (every state change recorded), regulatorily compliant (data residency, retention), and recoverable (no data loss).

**Core architecture:**
```
API Server
    ↓ (transactional outbox)
PostgreSQL ────── Job table + Outbox table ──────────────────► Kafka
                  (same transaction)                               │
                                                                   ▼
                                                           Financial Job Workers
                                                                   │
                                                          ┌────────┴────────────┐
                                                          ▼                     ▼
                                                   Audit Log DB          Payment Processor
                                                   (append-only)          (idempotent API)
```

**Transactional outbox (true exactly-once at submission):**
```python
async def submit_payment_job(payment: Payment) -> str:
    job_id = snowflake_id()  # Sortable unique ID
    
    async with db.transaction():
        # Insert payment record
        await db.execute(
            "INSERT INTO payments (id, amount, from_account, to_account, status) "
            "VALUES ($1, $2, $3, $4, 'pending')",
            payment.id, payment.amount, payment.from_acct, payment.to_acct
        )
        
        # Insert job in SAME transaction
        await db.execute(
            "INSERT INTO payment_jobs (id, payment_id, payload, status, created_at) "
            "VALUES ($1, $2, $3, 'queued', NOW())",
            job_id, payment.id, json.dumps(payment.dict())
        )
        
        # Outbox entry (same transaction!)
        await db.execute(
            "INSERT INTO outbox (topic, key, payload) "
            "VALUES ('payment.jobs', $1, $2)",
            str(payment.id), json.dumps({"job_id": job_id, "payload": payment.dict()})
        )
    # If transaction commits: all three rows exist, consistent
    # If transaction fails: none exist, no orphaned job
    
    return job_id
```

**Outbox relay (CDC to Kafka):**
```python
# Debezium or custom relay: poll outbox and publish to Kafka
async def outbox_relay():
    while True:
        async with db.transaction():
            rows = await db.fetch(
                "SELECT * FROM outbox WHERE published = FALSE "
                "ORDER BY id LIMIT 100 FOR UPDATE SKIP LOCKED"
            )
            
            for row in rows:
                await kafka.produce(
                    topic=row['topic'],
                    key=row['key'],
                    value=row['payload']
                )
                await db.execute(
                    "UPDATE outbox SET published = TRUE WHERE id = $1",
                    row['id']
                )
        
        await asyncio.sleep(0.1)
```

**Exactly-once payment execution:**
```python
class PaymentJobWorker:
    async def process(self, job: dict):
        payment_id = job['payment_id']
        job_id = job['job_id']
        
        async with db.transaction():
            # Lock payment row and check state
            payment = await db.fetchrow(
                "SELECT * FROM payments WHERE id = $1 FOR UPDATE",
                payment_id
            )
            
            if payment['status'] != 'pending':
                logger.info(f"Payment {payment_id} already processed: {payment['status']}")
                return  # Idempotent: skip duplicate
            
            # Execute with payment processor idempotency key
            try:
                result = await payment_processor.execute(
                    amount=payment['amount'],
                    from_account=payment['from_account'],
                    to_account=payment['to_account'],
                    idempotency_key=f"payment:{payment_id}"  # Provider deduplicates
                )
                
                # Update state IN SAME TRANSACTION
                await db.execute(
                    "UPDATE payments SET status='completed', processor_ref=$1 WHERE id=$2",
                    result.reference, payment_id
                )
                await db.execute(
                    "UPDATE payment_jobs SET status='completed' WHERE id=$1",
                    job_id
                )
                
                # Immutable audit entry
                await db.execute(
                    "INSERT INTO payment_audit_log "
                    "(payment_id, event, actor, details, recorded_at) "
                    "VALUES ($1, 'payment_completed', 'system', $2, NOW())",
                    payment_id, json.dumps(result.dict())
                )
                
            except PaymentProcessorError as e:
                await db.execute(
                    "UPDATE payments SET status='failed', failure_reason=$1 WHERE id=$2",
                    str(e), payment_id
                )
                await db.execute(
                    "INSERT INTO payment_audit_log "
                    "(payment_id, event, actor, details, recorded_at) "
                    "VALUES ($1, 'payment_failed', 'system', $2, NOW())",
                    payment_id, json.dumps({"error": str(e)})
                )
                raise
```

**Regulatory compliance features:**
```python
# Data residency: jobs routed to workers in correct geographic region
class RegionAwareJobRouter:
    def route(self, job: PaymentJob) -> str:
        region = get_data_residency_region(job.from_account)
        return REGIONAL_QUEUES[region]

# Retention: audit logs retained for 7 years (SOX/PCI)
CREATE TABLE payment_audit_log (
    -- ... columns ...
    retention_until DATE GENERATED ALWAYS AS (recorded_at::DATE + INTERVAL '7 years') STORED
);

# GDPR: personal data in separate table, deletable
# Audit log references account_id (not name/email)
# PII vault handles deletion independently

# Reconciliation job (daily): verify all payments match processor records
@scheduler.cron("0 2 * * *")  # 2am daily
async def reconcile_payments():
    processor_records = await payment_processor.get_settlements(date=yesterday())
    our_records = await db.fetch(
        "SELECT id, amount FROM payments WHERE DATE(completed_at) = $1",
        yesterday()
    )
    
    discrepancies = reconcile(our_records, processor_records)
    if discrepancies:
        await alert_compliance_team(discrepancies)
```

---

## Quick Reference

### Queue Type Selection

| Scenario | Tool | Why |
|----------|------|-----|
| Simple background jobs | Celery + Redis | Easy setup, mature |
| High-throughput event streaming | Kafka | Retention, replay, fan-out |
| AWS-native queuing | SQS + Lambda | Managed, serverless |
| Exactly-once financial | PostgreSQL outbox + Kafka | ACID + streaming |
| Priority jobs | Redis sorted sets | Fast, flexible priority |
| Scheduled/delayed | Redis ZADD + poller | Simple, no extra infra |

### Retry Decision Tree

```
Job failed?
  Is it retryable? (500, timeout, network)
    YES → Apply exponential backoff with jitter
    NO  → Move to DLQ (400, validation error, business rejection)
  
  Max retries exceeded?
    YES → DLQ + alert
    NO  → Re-enqueue with backoff
```

### Worker Pool Sizing Formula

```
CPU-bound:  workers = CPU cores
I/O-bound:  workers = target_throughput_per_sec × avg_latency_sec × 1.2 (buffer)

Example: 200 jobs/sec, 300ms avg latency
  workers = 200 × 0.3 × 1.2 = 72 workers
```

### Job Lifecycle States

```
submitted → queued → processing → completed
                  ↘            ↘ failed → retry → processing...
                   ↘                             ↘ DLQ (max retries)
                    cancelled (any state except completed)
```

### Idempotency Patterns

| Pattern | Best For | Storage |
|---------|---------|---------|
| Unique constraint (DB) | DB operations | DB row |
| Redis SET NX | Any operation | Redis key (TTL) |
| External API idempotency key | Payment APIs, emails | Provider-side |
| Check-then-act (CAS) | State machine transitions | DB row status |
| Outbox pattern | DB write + queue publish | DB outbox table |

### Queue Health Metrics

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Queue depth | > 10k | > 100k | Scale workers |
| Consumer lag | > 5 min | > 30 min | Investigate or scale |
| DLQ depth | > 0 | > 100 | Fix bug + replay |
| Processing rate | Drops 50% | Drops 80% | Check worker health |
| Job error rate | > 1% | > 5% | Investigate failures |
