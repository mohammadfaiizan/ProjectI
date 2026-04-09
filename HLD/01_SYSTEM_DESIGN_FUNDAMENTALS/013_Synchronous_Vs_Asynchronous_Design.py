"""
SYNCHRONOUS VS ASYNCHRONOUS DESIGN
=====================================

Problem Statement:
Services can communicate synchronously (caller blocks and waits) or
asynchronously (caller submits work and continues). The choice has major
implications for latency, throughput, coupling, and fault tolerance.

Communication Patterns:
  SYNC:  Client → calls Service → waits → gets response
  ASYNC: Client → puts message in Queue → returns immediately
                  Worker reads Queue → processes → stores result

Trade-offs:
  Synchronous : Simple, immediate feedback, tight coupling, back-pressure problem
  Asynchronous: Decoupled, resilient, higher throughput, eventual consistency

When to Use:
  SYNC  → User-facing reads, payment confirmation, auth, short operations
  ASYNC → Email sending, image resizing, video transcoding, batch processing
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import time
import uuid
import threading
import queue
import random


class ProcessingMode(Enum):
    SYNC                = "synchronous"
    ASYNC_FIRE_FORGET   = "async_fire_forget"
    ASYNC_WITH_CALLBACK = "async_with_callback"
    ASYNC_WITH_FUTURE   = "async_with_future"
    MESSAGE_QUEUE       = "message_queue"


@dataclass
class Job:
    job_id    : str
    payload   : str
    submitted_at: float = field(default_factory=time.time)
    started_at  : Optional[float] = None
    completed_at: Optional[float] = None
    result      : Optional[str]  = None
    status      : str = "pending"

    @property
    def wait_time_ms(self) -> float:
        if self.started_at:
            return (self.started_at - self.submitted_at) * 1000
        return 0.0

    @property
    def processing_time_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    @property
    def total_time_ms(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.submitted_at) * 1000
        return 0.0


# ─────────────────────────────────────────────
# SYNCHRONOUS PROCESSING
# ─────────────────────────────────────────────

class SyncImageUploadService:
    """
    Processes image uploads synchronously.
    Caller blocks until resize + store is complete.
    Upload → resize → store → return URL  (all in one request)
    """

    def __init__(self, resize_latency_ms: float = 500):
        self.resize_latency_ms = resize_latency_ms
        self.jobs: List[Job] = []

    def upload_and_resize(self, image_name: str) -> Dict:
        job = Job(job_id=str(uuid.uuid4())[:8], payload=image_name)
        job.started_at = time.time()
        # Simulate resize (blocking)
        time.sleep(self.resize_latency_ms / 1000.0)
        job.completed_at = time.time()
        job.result       = f"https://cdn.example.com/{image_name}_thumb.jpg"
        job.status       = "completed"
        self.jobs.append(job)
        return {"url": job.result, "latency_ms": job.total_time_ms}


# ─────────────────────────────────────────────
# ASYNC PROCESSING WITH QUEUE
# ─────────────────────────────────────────────

class MessageQueueAsync:
    """
    Decoupled async processing via message queue.
    Client submits job and gets job_id immediately.
    Background workers process in parallel.
    """

    def __init__(self, worker_count: int = 3, process_latency_ms: float = 500):
        self._queue : queue.Queue = queue.Queue()
        self._jobs  : Dict[str, Job] = {}
        self._workers: List[threading.Thread] = []
        self.process_latency_ms = process_latency_ms

        for _ in range(worker_count):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self):
        while True:
            try:
                job_id = self._queue.get(timeout=2.0)
                job    = self._jobs[job_id]
                job.started_at = time.time()
                job.status     = "processing"
                time.sleep(self.process_latency_ms / 1000.0)
                job.completed_at = time.time()
                job.result       = f"https://cdn.example.com/{job.payload}_thumb.jpg"
                job.status       = "completed"
                self._queue.task_done()
            except queue.Empty:
                break

    def submit(self, image_name: str) -> str:
        job    = Job(job_id=str(uuid.uuid4())[:8], payload=image_name)
        self._jobs[job.job_id] = job
        self._queue.put(job.job_id)
        return job.job_id   # return immediately!

    def get_status(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def wait_all(self):
        self._queue.join()


# ─────────────────────────────────────────────
# FUTURE / PROMISE PATTERN
# ─────────────────────────────────────────────

class FutureResult:
    """
    A promise-like object: caller gets a future immediately,
    polls or awaits for result later.
    """

    def __init__(self, job_id: str):
        self.job_id    = job_id
        self._result   : Optional[str] = None
        self._done     = threading.Event()

    def set_result(self, result: str):
        self._result = result
        self._done.set()

    def get(self, timeout_s: float = 5.0) -> Optional[str]:
        if self._done.wait(timeout=timeout_s):
            return self._result
        return None   # timeout

    @property
    def is_done(self) -> bool:
        return self._done.is_set()


class AsyncWithFutureService:
    """Submit and get a Future back immediately."""

    def __init__(self, workers: int = 2, latency_ms: float = 300):
        self._futures : Dict[str, FutureResult] = {}
        self._queue   : queue.Queue = queue.Queue()
        self.latency_ms = latency_ms
        for _ in range(workers):
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            try:
                job_id, payload = self._queue.get(timeout=3.0)
                time.sleep(self.latency_ms / 1000.0)
                result = f"https://cdn.example.com/{payload}_processed.jpg"
                self._futures[job_id].set_result(result)
                self._queue.task_done()
            except queue.Empty:
                break

    def process(self, payload: str) -> FutureResult:
        job_id = str(uuid.uuid4())[:8]
        future = FutureResult(job_id)
        self._futures[job_id] = future
        self._queue.put((job_id, payload))
        return future   # caller gets this immediately


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_synchronous_vs_asynchronous():
    print("=" * 65)
    print("SYNCHRONOUS VS ASYNCHRONOUS DESIGN")
    print("Scenario: 5 image uploads each taking 300ms to resize")
    print("=" * 65)

    images = [f"photo_{i}.jpg" for i in range(1, 6)]

    # ── Synchronous ───────────────────────────
    print("\n[1] SYNCHRONOUS UPLOAD (blocks on each)")
    print("─" * 50)
    sync_svc = SyncImageUploadService(resize_latency_ms=300)
    t_start  = time.time()
    for img in images:
        result = sync_svc.upload_and_resize(img)
        print(f"  ✅ {img} → {result['url'][:50]}  ({result['latency_ms']:.0f}ms)")
    sync_total = (time.time() - t_start) * 1000
    print(f"\n  Total wall-clock time: {sync_total:.0f}ms  ← 5 × 300ms serial")

    # ── Async Queue ───────────────────────────
    print("\n\n[2] ASYNC QUEUE (fire-and-forget)")
    print("─" * 50)
    async_svc = MessageQueueAsync(worker_count=3, process_latency_ms=300)
    t_start   = time.time()
    job_ids   = []
    for img in images:
        jid = async_svc.submit(img)
        job_ids.append((img, jid))
        submit_ms = (time.time() - t_start) * 1000
        print(f"  📬 {img} submitted → job_id={jid}  ({submit_ms:.1f}ms)")

    submit_total = (time.time() - t_start) * 1000
    print(f"\n  All 5 jobs submitted in {submit_total:.1f}ms  ← returns immediately!")
    print("  Processing happens in background…")

    # Wait for completion
    async_svc.wait_all()
    async_total = (time.time() - t_start) * 1000

    for img, jid in job_ids:
        job = async_svc.get_status(jid)
        print(f"  ✅ {img} done: wait={job.wait_time_ms:.0f}ms  process={job.processing_time_ms:.0f}ms  total={job.total_time_ms:.0f}ms")
    print(f"\n  Total wall-clock time: {async_total:.0f}ms  ← ~300ms (parallel workers)")

    # ── Future pattern ────────────────────────
    print("\n\n[3] ASYNC WITH FUTURE PATTERN")
    print("─" * 50)
    future_svc = AsyncWithFutureService(workers=3, latency_ms=200)
    futures    = []
    for img in images:
        f = future_svc.process(img)
        futures.append((img, f))
        print(f"  📬 {img} → future received (not done yet: {not f.is_done})")

    print("\n  Doing other work while images process…")
    time.sleep(0.05)

    print("\n  Collecting results from futures:")
    for img, f in futures:
        result = f.get(timeout_s=3.0)
        print(f"  ✅ {img}: {result}")

    # ── Comparison ────────────────────────────
    print("\n\n[4] WHEN TO USE EACH PATTERN")
    print("─" * 50)
    rows = [
        ("Sync",              "User login/auth",             "Result needed immediately"),
        ("Sync",              "Payment processing",          "Must confirm success or failure"),
        ("Async (queue)",     "Send welcome email",          "User doesn't wait for email"),
        ("Async (queue)",     "Resize uploaded image",       "Background job, user gets URL later"),
        ("Async (queue)",     "Video transcoding",           "Might take minutes — never sync"),
        ("Async (future)",    "Parallel DB lookups",         "Fan-out then join results"),
        ("Async (fire-forget)","Analytics event logging",    "Low priority, can drop under load"),
    ]
    print(f"  {'Pattern':<20} {'Example':<30} {'Reason'}")
    print(f"  {'─'*80}")
    for pattern, example, reason in rows:
        print(f"  {pattern:<20} {example:<30} {reason}")


if __name__ == "__main__":
    demonstrate_synchronous_vs_asynchronous()
