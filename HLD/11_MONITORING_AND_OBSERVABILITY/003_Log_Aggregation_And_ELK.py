"""
LOG AGGREGATION AND ELK STACK
==============================

Problem Statement:
Microservices each write logs locally. Without aggregation, debugging requires
SSHing to each node. Log aggregation centralizes logs for searching, alerting,
and analysis.

ELK / Elastic Stack:
  E = Elasticsearch   Full-text search + analytics engine (Lucene-based).
                      Documents stored in indices; sharded across nodes.
  L = Logstash        Log pipeline: input → filter (parse/enrich) → output.
                      High CPU/RAM; often replaced by Filebeat/Fluentd.
  K = Kibana          Visualization: dashboards, Discover (ad-hoc search),
                      Lens, Maps, Alerting.

Modern Stack Variants:
  EFK:  Elasticsearch + Fluentd  + Kibana  (K8s standard)
  PLG:  Promtail     + Loki      + Grafana (lightweight, label-based, no FTS)
  OTel: OpenTelemetry Collector can receive/export logs to any backend.

Elasticsearch Concepts:
  Index:      Collection of documents (like a DB table).
  Shard:      Horizontal partition of an index (default 1 primary + 1 replica).
  Mapping:    Schema for fields: keyword vs text, date, long, nested.
  Analyzer:   Tokenizer + filters applied at index/query time.
  ILM:        Index Lifecycle Management — hot → warm → cold → delete phases.
              hot-7d: 1 primary shard, SSD; warm: read-only, merge; delete: 30d.
  Query DSL:  JSON-based query language.
              match, term, range, bool (must/should/must_not/filter).

Log Processing Pipeline:
  App → stdout/file → Filebeat/Fluentd → Kafka (buffer) → Logstash → ES → Kibana
                                                        ↓ (direct for low-vol)
                                              Elasticsearch

Structured Logging Best Practices:
  - JSON format always. Human-readable text wastes parse CPU.
  - Include: timestamp, level, service, trace_id, span_id, user_id (hashed),
             message, error.type, error.stack, duration_ms.
  - Use correlation IDs to link logs across services.
  - Do NOT log PII (email, SSN, card numbers). Use tokenization.
  - Log levels: ERROR > WARN > INFO > DEBUG > TRACE.
    Production: INFO minimum; DEBUG only in dev.

Loki vs Elasticsearch:
  Loki: index only labels (like Prometheus). Log lines stored compressed.
        Cheap storage; fast for label queries; no full-text index.
        Good for "show logs for pod=X, env=prod, last 15m".
  ES:   Full-text search; field extraction; aggregations.
        Good for "find all requests where user_agent contains 'bot'".
"""

from __future__ import annotations

import json
import re
import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator
from enum import Enum
from collections import defaultdict


# ─────────────────────────────────────────────
# LOG LEVEL
# ─────────────────────────────────────────────

class LogLevel(Enum):
    TRACE = 0
    DEBUG = 1
    INFO  = 2
    WARN  = 3
    ERROR = 4
    FATAL = 5

    def __ge__(self, other): return self.value >= other.value
    def __gt__(self, other): return self.value >  other.value


# ─────────────────────────────────────────────
# STRUCTURED LOG ENTRY
# ─────────────────────────────────────────────

@dataclass
class LogEntry:
    timestamp:  float
    level:      LogLevel
    service:    str
    message:    str
    fields:     Dict[str, Any] = field(default_factory=dict)
    trace_id:   Optional[str]  = None
    span_id:    Optional[str]  = None

    def to_json(self) -> str:
        doc = {
            "@timestamp": self.timestamp,
            "level":      self.level.name,
            "service":    self.service,
            "message":    self.message,
            **self.fields,
        }
        if self.trace_id: doc["trace_id"] = self.trace_id
        if self.span_id:  doc["span_id"]  = self.span_id
        return json.dumps(doc)

    @classmethod
    def from_json(cls, s: str) -> "LogEntry":
        d = json.loads(s)
        level = LogLevel[d.pop("level", "INFO")]
        ts    = d.pop("@timestamp", time.time())
        svc   = d.pop("service", "unknown")
        msg   = d.pop("message", "")
        tid   = d.pop("trace_id", None)
        sid   = d.pop("span_id",  None)
        return cls(ts, level, svc, msg, d, tid, sid)


# ─────────────────────────────────────────────
# LOG PIPELINE STAGE
# ─────────────────────────────────────────────

class PipelineStage:
    """Base class for Logstash-like filter stages."""
    def process(self, entry: LogEntry) -> Optional[LogEntry]:
        raise NotImplementedError


class GrokParser(PipelineStage):
    """
    Parse unstructured log lines into structured fields.
    Simplified: uses named regex groups.
    """

    COMMON_PATTERNS = {
        "IP":       r"\d{1,3}(?:\.\d{1,3}){3}",
        "TIMESTAMP":r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        "WORD":     r"\S+",
        "NUMBER":   r"\d+(?:\.\d+)?",
        "STATUS":   r"\d{3}",
    }

    def __init__(self, pattern: str, source_field: str = "message"):
        self._pattern      = pattern
        self._source_field = source_field
        # Replace %{NAME:field} → named groups
        regex = re.sub(
            r"%\{(\w+):(\w+)\}",
            lambda m: f"(?P<{m.group(2)}>{self.COMMON_PATTERNS.get(m.group(1), r'\\S+')})",
            pattern,
        )
        self._re = re.compile(regex)

    def process(self, entry: LogEntry) -> Optional[LogEntry]:
        src = entry.fields.get(self._source_field, entry.message)
        m   = self._re.search(src)
        if m:
            entry.fields.update(m.groupdict())
        return entry


class FieldEnricher(PipelineStage):
    """Add computed/static fields to every log entry."""

    def __init__(self, static_fields: Dict[str, Any]):
        self._fields = static_fields

    def process(self, entry: LogEntry) -> Optional[LogEntry]:
        for k, v in self._fields.items():
            if k not in entry.fields:
                entry.fields[k] = v
        return entry


class PIIScrubber(PipelineStage):
    """Remove or mask PII before indexing."""

    PATTERNS = [
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
        (re.compile(r"\b\d{16}\b"),         "[CARD]"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    ]

    def process(self, entry: LogEntry) -> Optional[LogEntry]:
        for pat, replacement in self.PATTERNS:
            entry.message = pat.sub(replacement, entry.message)
            for k, v in entry.fields.items():
                if isinstance(v, str):
                    entry.fields[k] = pat.sub(replacement, v)
        return entry


class LevelFilter(PipelineStage):
    """Drop entries below minimum level."""

    def __init__(self, min_level: LogLevel):
        self._min = min_level

    def process(self, entry: LogEntry) -> Optional[LogEntry]:
        return entry if entry.level >= self._min else None


class LogPipeline:
    """Logstash-like processing pipeline."""

    def __init__(self):
        self._stages: List[PipelineStage] = []

    def add_stage(self, stage: PipelineStage) -> "LogPipeline":
        self._stages.append(stage)
        return self

    def run(self, entry: LogEntry) -> Optional[LogEntry]:
        for stage in self._stages:
            if entry is None:
                return None
            entry = stage.process(entry)
        return entry


# ─────────────────────────────────────────────
# ELASTICSEARCH-LIKE INDEX
# ─────────────────────────────────────────────

@dataclass
class ESMapping:
    """Field mappings for an index."""
    keyword_fields: List[str] = field(default_factory=list)  # exact match
    text_fields:    List[str] = field(default_factory=list)   # full-text
    numeric_fields: List[str] = field(default_factory=list)


class ElasticsearchIndex:
    """
    Simplified in-memory Elasticsearch index.
    Supports:
    - Document indexing
    - term query (keyword exact match)
    - match query (text contains, case-insensitive)
    - range query on numeric fields
    - bool query (must/should/must_not)
    - aggregations (terms, stats)
    """

    def __init__(self, name: str, mapping: ESMapping):
        self.name     = name
        self.mapping  = mapping
        self._docs:   List[Dict]              = []
        self._keyword_index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self._text_index:    Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self._lock    = threading.Lock()
        self._doc_count = 0

    def index(self, doc: Dict) -> str:
        """Add document; returns doc ID."""
        with self._lock:
            doc_id = str(self._doc_count)
            self._doc_count += 1
            self._docs.append({"_id": doc_id, **doc})

            # Build keyword inverted index
            for field_name in self.mapping.keyword_fields:
                val = doc.get(field_name)
                if val is not None:
                    self._keyword_index[field_name][str(val)].append(len(self._docs) - 1)

            # Build text inverted index (tokenize)
            for field_name in self.mapping.text_fields:
                val = doc.get(field_name)
                if val and isinstance(val, str):
                    tokens = re.findall(r"\w+", val.lower())
                    for tok in set(tokens):
                        self._text_index[field_name][tok].append(len(self._docs) - 1)

            return doc_id

    def _resolve_indices(self, indices: List[int]) -> List[Dict]:
        return [self._docs[i] for i in indices if i < len(self._docs)]

    def query(self, q: Dict) -> List[Dict]:
        """Execute a simplified query DSL."""
        if "term" in q:
            field_name, value = next(iter(q["term"].items()))
            indices = self._keyword_index[field_name].get(str(value), [])
            return self._resolve_indices(indices)

        if "match" in q:
            field_name, value = next(iter(q["match"].items()))
            tokens = re.findall(r"\w+", str(value).lower())
            result_sets = [set(self._text_index[field_name].get(t, []))
                           for t in tokens]
            if not result_sets:
                return []
            indices = list(result_sets[0].intersection(*result_sets[1:]) if len(result_sets) > 1 else result_sets[0])
            return self._resolve_indices(sorted(indices))

        if "range" in q:
            field_name, bounds = next(iter(q["range"].items()))
            gte = bounds.get("gte")
            lte = bounds.get("lte")
            results = []
            for doc in self._docs:
                v = doc.get(field_name)
                if v is None:
                    continue
                if gte is not None and v < gte:
                    continue
                if lte is not None and v > lte:
                    continue
                results.append(doc)
            return results

        if "bool" in q:
            bool_q = q["bool"]
            must     = bool_q.get("must",     [])
            should   = bool_q.get("should",   [])
            must_not = bool_q.get("must_not", [])

            # must: intersection
            must_sets = []
            for sub in must:
                docs = self.query(sub)
                must_sets.append({d["_id"] for d in docs})

            must_ids = must_sets[0] if must_sets else {d["_id"] for d in self._docs}
            for s in must_sets[1:]:
                must_ids &= s

            # must_not: subtract
            for sub in must_not:
                docs = self.query(sub)
                must_ids -= {d["_id"] for d in docs}

            # should: union (at least one if must is empty)
            if should and not must:
                should_ids: set = set()
                for sub in should:
                    docs = self.query(sub)
                    should_ids |= {d["_id"] for d in docs}
                must_ids &= should_ids

            return [d for d in self._docs if d["_id"] in must_ids]

        return list(self._docs)

    def aggregate_terms(self, field_name: str, size: int = 10) -> Dict[str, int]:
        """Count unique values for a field (like ES terms aggregation)."""
        counts: Dict[str, int] = defaultdict(int)
        for doc in self._docs:
            val = doc.get(field_name)
            if val is not None:
                counts[str(val)] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:size])

    def aggregate_stats(self, field_name: str) -> Dict:
        """Stats aggregation: min/max/avg/sum/count."""
        values = [doc[field_name] for doc in self._docs
                  if isinstance(doc.get(field_name), (int, float))]
        if not values:
            return {}
        return {
            "min":   min(values),
            "max":   max(values),
            "avg":   sum(values) / len(values),
            "sum":   sum(values),
            "count": len(values),
        }

    def count(self) -> int:
        return len(self._docs)


# ─────────────────────────────────────────────
# ILM (Index Lifecycle Management) Simulator
# ─────────────────────────────────────────────

class ILMPhase(Enum):
    HOT   = "HOT"     # active writes, SSD
    WARM  = "WARM"    # read-only, merged shards, HDD
    COLD  = "COLD"    # infrequent access, object storage
    DELETE= "DELETE"  # gone

@dataclass
class ILMPolicy:
    hot_days:    int = 7
    warm_days:   int = 30
    cold_days:   int = 90
    delete_days: int = 180


class ILMManager:
    def __init__(self, policy: ILMPolicy):
        self._policy = policy

    def current_phase(self, index_age_days: float) -> ILMPhase:
        p = self._policy
        if index_age_days < p.hot_days:     return ILMPhase.HOT
        if index_age_days < p.warm_days:    return ILMPhase.WARM
        if index_age_days < p.cold_days:    return ILMPhase.COLD
        if index_age_days < p.delete_days:  return ILMPhase.COLD
        return ILMPhase.DELETE

    def storage_cost_usd_gb_month(self, phase: ILMPhase) -> float:
        return {
            ILMPhase.HOT:    0.20,   # SSD
            ILMPhase.WARM:   0.05,   # HDD
            ILMPhase.COLD:   0.004,  # S3
            ILMPhase.DELETE: 0.0,
        }[phase]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_elk():
    print("=" * 65)
    print("LOG AGGREGATION AND ELK STACK")
    print("=" * 65)

    # ── Log Pipeline ──────────────────────────
    print("\n[1] LOG PIPELINE (Logstash-like)")
    print("─" * 55)

    pipeline = LogPipeline()
    pipeline.add_stage(GrokParser(
        r"%{IP:client_ip} \"%{WORD:http_method} %{WORD:path}\" %{STATUS:http_status} %{NUMBER:response_time_ms}",
        source_field="message"
    ))
    pipeline.add_stage(FieldEnricher({"env": "production", "cluster": "us-east-1"}))
    pipeline.add_stage(PIIScrubber())
    pipeline.add_stage(LevelFilter(LogLevel.INFO))

    raw_logs = [
        LogEntry(time.time(), LogLevel.INFO,  "nginx",
                 '10.0.0.1 "GET /api/users" 200 45.2'),
        LogEntry(time.time(), LogLevel.ERROR, "nginx",
                 '10.0.0.2 "POST /api/login" 500 120.5',
                 {"user_email": "user@example.com"}),
        LogEntry(time.time(), LogLevel.DEBUG, "nginx",
                 '10.0.0.3 "GET /health" 200 2.1'),   # filtered out
    ]

    processed = []
    for entry in raw_logs:
        result = pipeline.run(entry)
        if result:
            processed.append(result)
            print(f"  [{result.level.name}] {result.service}: {result.message[:50]}")
            extras = {k: v for k, v in result.fields.items()
                      if k in ("client_ip", "http_method", "http_status", "user_email", "env")}
            print(f"    fields: {extras}")

    print(f"  {len(raw_logs)} input → {len(processed)} output (1 DEBUG filtered)")

    # ── Elasticsearch Indexing ─────────────────
    print("\n[2] ELASTICSEARCH INDEXING AND QUERY")
    print("─" * 55)

    mapping = ESMapping(
        keyword_fields=["level", "service", "http_status", "env"],
        text_fields=["message"],
        numeric_fields=["response_time_ms"],
    )
    idx = ElasticsearchIndex("logs-2024.01.15", mapping)

    import random
    random.seed(42)

    services  = ["api", "auth", "payment", "search"]
    levels    = ["INFO", "WARN", "ERROR"]
    messages  = [
        "User login successful",
        "Payment processed",
        "Database connection failed",
        "Cache miss for key",
        "Request timeout exceeded",
        "Authentication failed invalid token",
    ]

    for _ in range(100):
        svc   = random.choice(services)
        level = random.choices(levels, weights=[80, 15, 5])[0]
        msg   = random.choice(messages)
        rt    = random.uniform(1, 500)
        idx.index({
            "level":           level,
            "service":         svc,
            "message":         msg,
            "response_time_ms": round(rt, 1),
            "http_status":     "500" if level == "ERROR" else "200",
            "env":             "production",
        })

    print(f"  Indexed {idx.count()} documents")

    # term query
    errors = idx.query({"term": {"level": "ERROR"}})
    print(f"\n  term(level=ERROR):       {len(errors)} results")

    # match query
    auth_failures = idx.query({"match": {"message": "authentication failed"}})
    print(f"  match(message=auth):     {len(auth_failures)} results")

    # bool query
    error_payments = idx.query({
        "bool": {
            "must": [
                {"term": {"level": "ERROR"}},
                {"term": {"service": "payment"}},
            ]
        }
    })
    print(f"  bool(ERROR + payment):   {len(error_payments)} results")

    # range query
    slow = idx.query({"range": {"response_time_ms": {"gte": 400}}})
    print(f"  range(rt>=400ms):        {len(slow)} results")

    # ── Aggregations ──────────────────────────
    print("\n[3] AGGREGATIONS")
    print("─" * 55)

    print("  terms(level):")
    for val, cnt in idx.aggregate_terms("level").items():
        print(f"    {val:<8} {cnt}")

    print("  terms(service):")
    for val, cnt in idx.aggregate_terms("service").items():
        print(f"    {val:<10} {cnt}")

    stats = idx.aggregate_stats("response_time_ms")
    print(f"  stats(response_time_ms): "
          f"min={stats['min']:.1f}ms  avg={stats['avg']:.1f}ms  max={stats['max']:.1f}ms")

    # ── ILM ───────────────────────────────────
    print("\n[4] INDEX LIFECYCLE MANAGEMENT (ILM)")
    print("─" * 55)

    ilm = ILMManager(ILMPolicy())
    ages = [1, 5, 15, 60, 120, 200]
    print(f"  {'Age (days)':<14} {'Phase':<10} {'Cost ($/GB/mo)'}")
    print(f"  {'─'*14} {'─'*10} {'─'*14}")
    for age in ages:
        phase = ilm.current_phase(age)
        cost  = ilm.storage_cost_usd_gb_month(phase)
        print(f"  {age:<14} {phase.value:<10} ${cost:.3f}")

    # ── Loki vs Elasticsearch ─────────────────
    print("\n[5] LOKI vs ELASTICSEARCH")
    print("─" * 55)

    comparison = [
        ("Index method",    "Labels only (like Prometheus)",    "Inverted index on all fields"),
        ("Storage cost",    "Very low (compressed chunks)",     "High (index overhead ~30%)"),
        ("Query speed",     "Fast by label; slow full-text",    "Fast full-text search"),
        ("Schema",          "Schema-free; only labels indexed", "Explicit mappings required"),
        ("Use case",        "K8s/container log tail",           "Security/compliance/analytics"),
        ("Cardinality",     "Low-cardinality labels only",      "Handles high cardinality"),
        ("Retention cost",  "~$0.004/GB (S3)",                  "~$0.05-0.20/GB (SSD/HDD)"),
        ("Setup complexity","Low (Promtail agent + Loki)",       "Medium-High (beats/logstash)"),
    ]
    print(f"  {'Aspect':<22} {'Loki':<35} {'Elasticsearch'}")
    print("  " + "─" * 80)
    for aspect, loki, es in comparison:
        print(f"  {aspect:<22} {loki:<35} {es}")


if __name__ == "__main__":
    demonstrate_elk()
