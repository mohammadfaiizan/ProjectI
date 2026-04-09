"""
STREAM PROCESSING DESIGN
==========================

Problem Statement:
Batch processing reads a finite dataset, processes it, produces output.
Stream processing operates on an unbounded, continuous sequence of events.
Design challenge: produce low-latency, accurate results over infinite data.

Core Concepts:
  Event Time vs Processing Time:
    Event time:      When the event actually occurred (in the source system).
    Processing time: When the event is processed by the stream processor.
    Difference (skew/lag): network delay, consumer backlog, late arrivals.

  Windows:
    Tumbling window:  Fixed non-overlapping intervals. [0-60s], [60-120s], ...
                      Simple, exactly one window per event.
    Sliding window:   Fixed size, advances by slide interval. [0-60s], [30-90s], ...
                      Events appear in multiple windows. Overlap = window/slide windows per event.
    Session window:   Variable size, ends after inactivity gap.
                      Groups user activity bursts. Complex but natural for user sessions.
    Hopping window:   Alias for sliding window in some frameworks.

  Watermarks:
    Mechanism to track event-time progress in the presence of out-of-order data.
    Watermark W(t): "I believe no events with event_time < t will arrive."
    Windows are closed and emitted once the watermark passes the window end.
    Aggressive watermark (low lag allowance) → faster results, more late data dropped.
    Conservative watermark → slower results, fewer late events missed.

  Late Data Handling:
    Options: drop, re-emit window update, accumulate in side output.

  State Management:
    Stateless operators: filter, map (no memory of past events).
    Stateful operators:  aggregations, joins, deduplication (maintain state per key).
    State backends:      in-memory (fast, not fault-tolerant), RocksDB (persistent).

  Fault Tolerance:
    Checkpointing: periodically snapshot operator state → replay from last checkpoint on failure.
    Exactly-once: requires both checkpointing + idempotent sinks.

Frameworks: Apache Flink, Apache Spark Structured Streaming, Kafka Streams, AWS Kinesis Data Analytics.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple
from collections import defaultdict
import time
import uuid
import threading
import random
import math


# ─────────────────────────────────────────────
# STREAM EVENT
# ─────────────────────────────────────────────

@dataclass
class StreamEvent:
    event_id  : str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    key       : str   = ""
    value     : Any   = None
    event_time: float = field(default_factory=time.time)   # when it happened
    proc_time : float = field(default_factory=time.time)   # when we received it


# ─────────────────────────────────────────────
# TUMBLING WINDOW AGGREGATOR
# ─────────────────────────────────────────────

class TumblingWindowAggregator:
    """
    Aggregates events into fixed non-overlapping time windows.
    Uses event time. Emits window result once watermark passes window end.
    """

    def __init__(self, window_size_s: float, watermark_delay_s: float = 0.0):
        self.window_size_s    = window_size_s
        self.watermark_delay_s = watermark_delay_s
        self._buckets         : Dict[float, Dict[str, Any]] = defaultdict(lambda: defaultdict(float))
        self._watermark       : float = 0.0
        self._emitted         : List[Dict] = []

    def _window_start(self, event_time: float) -> float:
        return math.floor(event_time / self.window_size_s) * self.window_size_s

    def process(self, event: StreamEvent):
        win_start = self._window_start(event.event_time)
        self._buckets[win_start][event.key] += event.value

        # Advance watermark
        new_watermark = event.proc_time - self.watermark_delay_s
        if new_watermark > self._watermark:
            self._watermark = new_watermark
            self._emit_ready_windows()

    def _emit_ready_windows(self):
        ready = [ws for ws in self._buckets
                 if ws + self.window_size_s <= self._watermark]
        for ws in sorted(ready):
            self._emitted.append({
                "window_start": ws,
                "window_end"  : ws + self.window_size_s,
                "counts"      : dict(self._buckets.pop(ws)),
            })

    def flush(self):
        """Force-emit all remaining windows (end of stream)."""
        for ws in sorted(self._buckets):
            self._emitted.append({
                "window_start": ws,
                "window_end"  : ws + self.window_size_s,
                "counts"      : dict(self._buckets.pop(ws)),
            })
        self._buckets.clear()

    @property
    def emitted_windows(self) -> List[Dict]:
        return self._emitted


# ─────────────────────────────────────────────
# SLIDING WINDOW AGGREGATOR
# ─────────────────────────────────────────────

class SlidingWindowAggregator:
    """
    Sliding window: each event appears in (window_size / slide_size) windows.
    Useful for moving averages, rolling counts.
    """

    def __init__(self, window_size_s: float, slide_s: float):
        self.window_size_s = window_size_s
        self.slide_s       = slide_s
        self._events       : List[StreamEvent] = []

    def add(self, event: StreamEvent):
        self._events.append(event)

    def query(self, key: str, at_time: float) -> float:
        """Sum of key's values in the window ending at at_time."""
        window_start = at_time - self.window_size_s
        total = 0.0
        for e in self._events:
            if e.key == key and window_start <= e.event_time <= at_time:
                total += e.value
        return total

    def windows(self, key: str, start: float, end: float) -> List[Tuple[float, float, float]]:
        """Generate all slide-aligned windows in [start, end] with sum for key."""
        results = []
        t = start
        while t + self.window_size_s <= end + self.slide_s:
            win_end = t + self.window_size_s
            total = self.query(key, win_end)
            results.append((t, win_end, total))
            t += self.slide_s
        return results


# ─────────────────────────────────────────────
# SESSION WINDOW
# ─────────────────────────────────────────────

@dataclass
class Session:
    session_id  : str
    key         : str
    start_time  : float
    end_time    : float
    events      : List[StreamEvent] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def event_count(self) -> int:
        return len(self.events)


class SessionWindowAggregator:
    """
    Groups events by key into sessions separated by an inactivity gap.
    A new event extends the session; gap > timeout closes the session.
    """

    def __init__(self, gap_timeout_s: float):
        self.gap_timeout_s = gap_timeout_s
        self._open_sessions: Dict[str, Session] = {}
        self._closed        : List[Session]      = []

    def process(self, event: StreamEvent) -> Optional[Session]:
        key     = event.key
        closed  = None

        if key in self._open_sessions:
            session = self._open_sessions[key]
            # Check if gap exceeded
            if event.event_time - session.end_time > self.gap_timeout_s:
                closed = session
                del self._open_sessions[key]
                self._closed.append(closed)
                # Start new session
                self._open_sessions[key] = Session(
                    session_id = str(uuid.uuid4())[:8],
                    key        = key,
                    start_time = event.event_time,
                    end_time   = event.event_time,
                    events     = [event],
                )
            else:
                session.end_time = event.event_time
                session.events.append(event)
        else:
            self._open_sessions[key] = Session(
                session_id = str(uuid.uuid4())[:8],
                key        = key,
                start_time = event.event_time,
                end_time   = event.event_time,
                events     = [event],
            )
        return closed

    def close_all(self) -> List[Session]:
        sessions = list(self._open_sessions.values())
        self._closed.extend(sessions)
        self._open_sessions.clear()
        return sessions

    @property
    def closed_sessions(self) -> List[Session]:
        return self._closed


# ─────────────────────────────────────────────
# STATEFUL OPERATOR: RUNNING COUNT (per-key)
# ─────────────────────────────────────────────

class StatefulCountOperator:
    """Per-key running count. Represents a stateful stream operator."""

    def __init__(self):
        self._counts: Dict[str, int] = defaultdict(int)
        self._checkpoints: List[Dict[str, int]] = []

    def process(self, event: StreamEvent) -> int:
        self._counts[event.key] += 1
        return self._counts[event.key]

    def checkpoint(self):
        """Snapshot state for fault tolerance."""
        self._checkpoints.append(dict(self._counts))

    def restore(self, checkpoint_index: int = -1):
        """Restore from checkpoint (simulates recovery after failure)."""
        if self._checkpoints:
            self._counts = defaultdict(int, self._checkpoints[checkpoint_index])

    def state_size(self) -> int:
        return len(self._counts)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_stream_processing():
    print("=" * 65)
    print("STREAM PROCESSING DESIGN")
    print("=" * 65)

    random.seed(99)
    base_time = 1000.0   # use fixed base time for reproducibility

    # ── Tumbling Windows ─────────────────────────
    print("\n[1] TUMBLING WINDOW — 10-SECOND WINDOWS")
    print("─" * 55)

    tumbling = TumblingWindowAggregator(window_size_s=10, watermark_delay_s=0)
    # Simulate 30 events across 3 windows [1000-1010, 1010-1020, 1020-1030]
    pages = ["home", "product", "checkout"]
    for i in range(30):
        t   = base_time + i   # one event per second
        key = random.choice(pages)
        evt = StreamEvent(key=key, value=1, event_time=t, proc_time=t + 0.1)
        tumbling.process(evt)

    tumbling.flush()
    print(f"  Emitted {len(tumbling.emitted_windows)} windows:")
    for w in tumbling.emitted_windows:
        ws = w['window_start'] - base_time
        we = w['window_end']   - base_time
        print(f"    [{ws:.0f}s-{we:.0f}s]: {w['counts']}")

    # ── Late Events + Watermark ───────────────────
    print("\n\n[2] WATERMARK — HANDLING LATE DATA")
    print("─" * 55)

    watermarked = TumblingWindowAggregator(window_size_s=10, watermark_delay_s=3)
    # Publish events with late arrivals
    events = [
        (base_time + 2,  "A", 1, base_time + 2.1),   # on-time
        (base_time + 8,  "B", 1, base_time + 8.1),   # on-time
        (base_time + 12, "A", 1, base_time + 12.1),  # advances watermark
        (base_time + 3,  "C", 1, base_time + 15.0),  # LATE — event_time=3 but arrives at 15
        (base_time + 22, "B", 1, base_time + 22.1),  # well ahead
    ]
    for evt_t, key, val, proc_t in events:
        e = StreamEvent(key=key, value=val, event_time=evt_t, proc_time=proc_t)
        watermarked.process(e)
        print(f"    event_time=+{evt_t-base_time:.0f}s proc_time=+{proc_t-base_time:.0f}s "
              f"key={key}  watermark=+{watermarked._watermark-base_time:.1f}s  "
              f"windows_emitted={len(watermarked.emitted_windows)}")

    watermarked.flush()
    print(f"\n  Emitted {len(watermarked.emitted_windows)} windows:")
    for w in watermarked.emitted_windows:
        ws = w['window_start'] - base_time
        we = w['window_end']   - base_time
        print(f"    [{ws:.0f}s-{we:.0f}s]: {w['counts']}")

    # ── Sliding Windows ───────────────────────────
    print("\n\n[3] SLIDING WINDOW — 10s WINDOW, 5s SLIDE")
    print("─" * 55)

    sliding = SlidingWindowAggregator(window_size_s=10, slide_s=5)
    # Revenue events for user "alice"
    revenue_events = [(base_time + t, random.uniform(10, 100)) for t in range(0, 25, 3)]
    for t, val in revenue_events:
        sliding.add(StreamEvent(key="alice", value=val, event_time=t, proc_time=t))

    windows = sliding.windows("alice", start=base_time, end=base_time + 25)
    print(f"  Revenue windows for 'alice' ({len(windows)} windows):")
    for ws, we, total in windows:
        print(f"    [{ws-base_time:.0f}s-{we-base_time:.0f}s]: ${total:.2f}")

    # ── Session Windows ───────────────────────────
    print("\n\n[4] SESSION WINDOW — 5s INACTIVITY GAP")
    print("─" * 55)

    session_agg = SessionWindowAggregator(gap_timeout_s=5)
    # User "bob" has two browsing bursts with a 10s gap between them
    bob_events = [0, 1, 2, 3, 15, 16, 18, 19, 20]   # seconds relative to base
    for t in bob_events:
        evt = StreamEvent(key="bob", value=1, event_time=base_time + t,
                          proc_time=base_time + t)
        closed = session_agg.process(evt)
        if closed:
            print(f"    Session closed: {closed.event_count} events  "
                  f"duration={closed.duration_s:.1f}s")

    # Close remaining open sessions
    remaining = session_agg.close_all()
    for s in remaining:
        print(f"    Session closed (flush): {s.event_count} events  "
              f"duration={s.duration_s:.1f}s")

    total_sessions = len(session_agg.closed_sessions)
    print(f"  Total sessions for 'bob': {total_sessions} "
          f"(expected 2 — one per burst)")

    # ── Stateful Operator + Checkpoint ───────────
    print("\n\n[5] STATEFUL OPERATOR — CHECKPOINT & RESTORE")
    print("─" * 55)

    counter = StatefulCountOperator()
    pages2  = ["home"] * 5 + ["checkout"] * 3 + ["product"] * 4
    random.shuffle(pages2)

    for i, page in enumerate(pages2[:8]):
        count = counter.process(StreamEvent(key=page, value=1,
                                             event_time=base_time + i))
        if i == 7:
            counter.checkpoint()
            print(f"  Checkpoint taken at event {i+1}. State: {dict(counter._counts)}")

    # Process more events
    for i, page in enumerate(pages2[8:]):
        counter.process(StreamEvent(key=page, value=1,
                                     event_time=base_time + 8 + i))

    print(f"  After 12 total events: {dict(counter._counts)}")
    counter.restore()
    print(f"  After restore to checkpoint: {dict(counter._counts)}")

    # ── Pattern Summary ───────────────────────────
    print("\n\n[6] WINDOW TYPE SELECTION GUIDE")
    print("─" * 55)
    guide = [
        ("Tumbling",  "Non-overlapping fixed",   "Hourly report, per-minute billing"),
        ("Sliding",   "Overlapping, step < size", "Moving average, rolling sum"),
        ("Session",   "Activity-gap based",       "User session length, click funnels"),
    ]
    print(f"  {'Type':<10} {'Nature':<28} {'Use Case'}")
    print(f"  {'─'*65}")
    for wtype, nature, use in guide:
        print(f"  {wtype:<10} {nature:<28} {use}")

    print()
    concepts = [
        ("Event time",     "Use for correctness — tied to reality"),
        ("Proc time",       "Use for simplicity when ordering isn't critical"),
        ("Watermark",       "Controls latency vs completeness tradeoff"),
        ("Stateful op",     "Count, sum, join — needs checkpointing for FT"),
        ("Checkpoint",      "Snapshot state → replay from here on failure"),
        ("Late data",       "Drop, side-output, or recompute window"),
    ]
    print(f"\n  {'Concept':<20} {'Guidance'}")
    print(f"  {'─'*55}")
    for concept, guidance in concepts:
        print(f"  {concept:<20} {guidance}")


if __name__ == "__main__":
    demonstrate_stream_processing()
