# 36 — Feature Flags and Progressive Delivery

## Easy (Q1–Q7)

---

### Q1. What is a feature flag (feature toggle) and why do engineering teams use them?

A **feature flag** (also called a feature toggle, feature switch, or feature gate) is a conditional code block that allows a feature to be enabled or disabled at runtime without deploying new code. The feature's behavior is controlled by an external configuration value rather than a hard-coded `if`.

```python
# Without feature flag — every user sees new checkout
def checkout(cart):
    return new_checkout_flow(cart)

# With feature flag — controlled rollout
def checkout(cart, user):
    if feature_flags.is_enabled("new_checkout_flow", user):
        return new_checkout_flow(cart)
    return old_checkout_flow(cart)
```

**Why teams use them:**

| Reason | Explanation |
|---|---|
| Decouple deploy from release | Code ships to production dark; feature turns on separately |
| Progressive rollout | Enable for 1% → 10% → 100% of users |
| Instant rollback | Flip the flag; no re-deploy needed |
| Kill switch | Disable a breaking feature in seconds |
| A/B testing | Compare two variants with production traffic |
| Permission flags | Beta users, internal staff, paid tier only |
| Dark launching | Run new code path, discard output — test in prod silently |

**Core value proposition:** Feature flags separate the **deployment** event (shipping code) from the **release** event (exposing the feature to users). This lets teams practice continuous deployment while still controlling when users see changes, dramatically reducing the risk of each deployment.

---

### Q2. What are the four main types of feature flags and when is each used?

Feature flags are not one-size-fits-all. Pete Hodgson's taxonomy (from Martin Fowler's blog) identifies four primary types based on their **lifetime** and **dynamism**.

```
FLAG TYPES
┌──────────────────────────────────────────────────────────────┐
│  Type           │ Lifetime │ Who Changes It │ Example        │
├──────────────────────────────────────────────────────────────┤
│  Release flag   │ Days–wks │ Engineers      │ new_checkout   │
│  Experiment flag│ Days–wks │ Data/Product   │ ab_test_cta    │
│  Ops flag       │ Long     │ Ops/SRE        │ maintenance_mode│
│  Permission flag│ Permanent│ Business rules │ premium_feature │
└──────────────────────────────────────────────────────────────┘
```

**1. Release Flags (Short-lived)**
Used during trunk-based development to hide incomplete work. The flag should be deleted once the rollout is complete. Keeping them around creates flag debt.

**2. Experiment Flags (A/B Test flags)**
Control which variant a user sees in an experiment. They need consistent assignment (same user → same variant) and statistical significance tracking. Lifetime: 1–4 weeks.

**3. Ops Flags (Operational flags)**
Long-lived flags for operational control. Examples: disable email notifications during an outage, enable maintenance mode, throttle a downstream API call. Changed by SREs not developers.

**4. Permission Flags (Entitlement flags)**
Control access to paid features, beta programs, or internal tools. These are permanent by design — the "flag" is really user entitlement. Example: `pro_analytics_dashboard` is on for Pro subscribers, off for free users.

Understanding the type helps you decide: Who can change it? Where is it stored (code, DB, feature flag service)? When should it be deleted?

---

### Q3. How do you implement a simple feature flag service — what are the core components?

A feature flag service has three layers: **storage**, **evaluation engine**, and **SDK/client**.

```
┌──────────────────────────────────────────────────────┐
│                FEATURE FLAG SERVICE                  │
│                                                      │
│  Admin UI / API  ──►  Flag Config Store              │
│                        (PostgreSQL + Redis Cache)    │
│                              │                       │
│                         SDK Polling                  │
│                              │                       │
│              Application Process Memory              │
│              (local flag config cache)               │
│                              │                       │
│            Evaluation Engine (rules engine)          │
│                              │                       │
│                    Boolean / String Result           │
└──────────────────────────────────────────────────────┘
```

**Flag Config in Database:**
```sql
CREATE TABLE feature_flags (
    flag_key        VARCHAR(100) PRIMARY KEY,
    enabled         BOOLEAN DEFAULT FALSE,
    rollout_percent INT DEFAULT 0,           -- 0-100
    targeting_rules JSONB,                   -- user segments
    default_variant VARCHAR(50),
    created_at      TIMESTAMP,
    updated_at      TIMESTAMP
);
```

**Redis Cache Layer:**
```python
def get_flag(flag_key: str) -> dict:
    cached = redis.get(f"flag:{flag_key}")
    if cached:
        return json.loads(cached)
    flag = db.query("SELECT * FROM feature_flags WHERE flag_key = %s", flag_key)
    redis.setex(f"flag:{flag_key}", 30, json.dumps(flag))  # 30s TTL
    return flag
```

**SDK Evaluation (in application):**
```python
class FeatureFlagClient:
    def __init__(self):
        self.flags = {}
        self._start_polling()   # background thread refreshes every 30s

    def is_enabled(self, flag_key: str, user_context: dict) -> bool:
        flag = self.flags.get(flag_key)
        if not flag or not flag["enabled"]:
            return False
        # Check targeting rules first
        if self._matches_targeting(flag, user_context):
            return True
        # Then percentage rollout
        return self._in_rollout(flag_key, user_context["user_id"],
                                flag["rollout_percent"])
```

Key design decisions: local caching in the SDK (no network call on each request), polling vs streaming (SSE/WebSocket) for flag updates, and graceful degradation when the flag service is down (use last known values).

---

### Q4. How does a gradual (percentage-based) rollout work — 5% → 25% → 50% → 100%?

A gradual rollout incrementally exposes a new feature to larger percentages of users, monitoring for errors or performance regressions at each stage before expanding.

**Consistent User Bucketing:**
The key requirement is that a user always sees the same experience across requests. This is achieved by hashing `(flag_key + user_id)` and mapping the hash into a 0–100 bucket:

```python
import hashlib

def get_user_bucket(flag_key: str, user_id: str) -> int:
    """Returns a stable bucket 0-99 for a given user and flag."""
    hash_input = f"{flag_key}:{user_id}".encode()
    hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
    return hash_value % 100

def is_in_rollout(flag_key: str, user_id: str, rollout_percent: int) -> bool:
    bucket = get_user_bucket(flag_key, user_id)
    return bucket < rollout_percent
```

**Rollout Schedule:**
```
Day 1:  rollout_percent = 5   (internal team + 5% of users)
        → Monitor: error rate, p99 latency, conversion
Day 2:  rollout_percent = 25  (if metrics are healthy)
Day 4:  rollout_percent = 50
Day 7:  rollout_percent = 100 (full release)
        → Delete flag (flag debt prevention)
```

**Rollout with Staged Gate:**
```python
ROLLOUT_STAGES = [5, 25, 50, 100]

def advance_rollout(flag_key: str, current_percent: int) -> int:
    """Returns next rollout stage if metrics are healthy."""
    if not metrics_healthy(flag_key):
        alert_on_call(f"Rollout paused for {flag_key}")
        return current_percent  # hold current stage
    try:
        next_stage = next(p for p in ROLLOUT_STAGES if p > current_percent)
        return next_stage
    except StopIteration:
        return 100
```

The benefit of hashing over random assignment is **stickiness** — user 42 is always in or always out regardless of when they make their request, avoiding jarring UI switches mid-session.

---

### Q5. What is the difference between a canary deployment and a feature flag — when do you use each?

Both canary deployments and feature flags are risk-reduction techniques, but they operate at different layers of the stack.

```
CANARY DEPLOYMENT                    FEATURE FLAG
─────────────────                    ────────────
Infra / deployment layer             Code / application layer
New binary runs on some servers      New code path runs on all servers
Traffic split by load balancer       Traffic split by flag evaluation
Affects ALL features in new build    Affects ONE specific feature
Rollback = shift traffic back        Rollback = flip flag (seconds)
Good for: platform upgrades,         Good for: product features,
dependency bumps, infrastructure     A/B tests, operational controls
changes                              user-specific targeting
```

**Canary Deployment — How it works:**
```
                    ┌──────────┐
Users ──► LB ──────►│ v1.2 (90%)│ (stable)
                    └──────────┘
              └────►│ v1.3 (10%)│ (canary — new binary)
                    └──────────┘
```
10% of requests hit the new version. If error rates spike on v1.3, the load balancer shifts all traffic back to v1.2.

**Feature Flag — How it works:**
```
All servers run v1.3
┌──────────────────────────────────┐
│  if flag("new_algo", user):      │
│      return new_ranking()   ←──  │── 10% of users
│  return old_ranking()       ←──  │── 90% of users
└──────────────────────────────────┘
```

**Decision Guide:**
- Upgrading a framework dependency → **canary deployment**
- Rolling out a new checkout flow → **feature flag**
- Changing JVM version → **canary deployment**
- A/B testing a UI change → **feature flag**
- Both together: deploy new binary as canary, AND gate new feature behind a flag for maximum safety

---

### Q6. What is dark launching, and how does it help teams test new code paths in production?

**Dark launching** means running a new code path in production with real traffic, but discarding or ignoring the output — users never see the result. The goal is to test the new path under real load without exposing it to users.

```python
def get_product_recommendations(user_id: str) -> List[Product]:
    # Always return old results to the user
    old_results = old_recommendation_engine(user_id)

    # Dark launch: run new engine, discard results, but measure it
    if feature_flags.is_enabled("dark_launch_new_rec_engine", user_id):
        try:
            start = time.time()
            new_results = new_recommendation_engine(user_id)
            latency = time.time() - start

            # Shadow compare — log differences for analysis
            metrics.record("new_rec_engine.latency", latency)
            metrics.record("new_rec_engine.result_overlap",
                           overlap_score(old_results, new_results))
        except Exception as e:
            metrics.increment("new_rec_engine.errors")
            logger.error(f"Dark launch error: {e}")

    return old_results   # User always sees old results
```

**Benefits:**
| Benefit | Description |
|---|---|
| Real load testing | Synthetic load tests miss real data edge cases |
| Latency profiling | Measure actual p99 with production data distribution |
| Correctness checking | Compare outputs of old vs new algorithm |
| No user impact | Bugs in new path don't affect anyone |
| Cache warming | Pre-populate caches before actual launch |

**Common use cases:** New database query replacing a slow one (compare results + latency), new ML model (compare recommendations against production model), new payment processor (run both, compare responses, use old for actual charge).

---

### Q7. What is a kill switch and how do you use feature flags for operational control?

A **kill switch** is a feature flag configured to be **on by default** and used to **disable** a feature during an incident. Unlike a release flag (off → on), a kill switch starts enabled and is flipped off in an emergency.

```python
# Kill switch pattern: flag name conveys "circuit breaker" intent
def process_payment(payment: Payment) -> Result:
    # Kill switch: ops team can disable in seconds if payments are failing
    if not feature_flags.is_enabled("payment_processing_enabled"):
        return Result.SERVICE_UNAVAILABLE("Payment processing temporarily disabled")

    return payment_gateway.charge(payment)
```

**Operational Flag Examples:**
```python
# Disable expensive background job during peak traffic
if feature_flags.is_enabled("run_analytics_aggregation"):
    analytics.run_daily_aggregation()

# Degrade gracefully: show cached results instead of live search
if feature_flags.is_enabled("live_search"):
    return search_engine.query(term)
return cached_search_results(term)

# Throttle downstream API calls
rate = feature_flags.get_int("downstream_api_rate_limit", default=1000)
rate_limiter.set_limit("downstream_api", rate)
```

**Kill Switch Hierarchy:**
```
Global kill switch    → disables entire service for everyone
Service kill switch   → disables one feature for everyone
Segment kill switch   → disables for one region/datacenter
User kill switch      → disables for specific problematic account
```

**Key operational practice:** Ops flags should be documented in a runbook. Every flag should have: what it does when flipped, who can flip it, and what the expected impact is. SREs should be able to flip an ops flag in a dashboard without code access.

---

## Medium (Q8–Q15)

---

### Q8. How does the feature flag SDK evaluation order work — from targeting rules through rollout percentage to default?

The evaluation order in a feature flag SDK follows a strict priority chain. Understanding this chain is critical for debugging unexpected flag behavior.

```
SDK EVALUATION CHAIN
─────────────────────────────────────────────
Input: flag_key + user_context

Step 1: Is flag enabled at all?
        if flag.enabled == False → return DEFAULT
        ↓
Step 2: Does user match individual targeting rules?
        (specific user IDs, email list, beta users)
        if match → return TARGETED VARIANT (bypass rollout %)
        ↓
Step 3: Does user match segment targeting?
        (org tier = "enterprise", country = "US", plan = "pro")
        if match → return SEGMENT VARIANT
        ↓
Step 4: Does user fall within rollout percentage?
        bucket = hash(flag_key + user_id) % 100
        if bucket < rollout_percent → return ON variant
        ↓
Step 5: Return DEFAULT variant (flag is "off" for this user)
─────────────────────────────────────────────
```

```python
class FlagEvaluator:
    def evaluate(self, flag: Flag, user: UserContext) -> Variant:
        # Step 1: Global kill switch
        if not flag.enabled:
            return flag.default_variant

        # Step 2: Individual user targeting (highest priority)
        for rule in flag.individual_rules:
            if user.user_id in rule.user_ids:
                return rule.variant

        # Step 3: Segment targeting
        for rule in flag.segment_rules:
            if self._user_matches_segment(user, rule.segment):
                return rule.variant

        # Step 4: Percentage rollout
        bucket = self._get_bucket(flag.key, user.user_id)
        if bucket < flag.rollout_percent:
            return flag.on_variant

        # Step 5: Default (flag off for this user)
        return flag.default_variant

    def _get_bucket(self, flag_key: str, user_id: str) -> int:
        seed = f"{flag_key}:{user_id}"
        return int(hashlib.sha256(seed.encode()).hexdigest(), 16) % 100
```

**Why this order matters:**
- Individual rules allow "always enable for internal users" regardless of rollout %
- Segment rules allow "all enterprise customers get early access"
- Rollout % controls the gradual release
- Default is the safe fallback

---

### Q9. How do you implement A/B testing with feature flags — how is consistent user splitting achieved?

A/B testing with feature flags requires that users are consistently assigned to the same variant across all their sessions and requests (sticky assignment), the split is statistically random (no systematic bias), and the assignment is server-side (can't be gamed by client).

**Consistent Hashing for A/B Split:**
```python
def assign_ab_variant(experiment_key: str, user_id: str,
                       variants: list[dict]) -> str:
    """
    variants = [{"name": "control", "weight": 50},
                {"name": "treatment", "weight": 50}]
    """
    # Stable hash: same user always gets same bucket
    seed = f"{experiment_key}:{user_id}"
    bucket = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % 100

    # Map bucket to variant by cumulative weight
    cumulative = 0
    for variant in variants:
        cumulative += variant["weight"]
        if bucket < cumulative:
            return variant["name"]
    return variants[-1]["name"]

# Usage
variant = assign_ab_variant(
    "homepage_cta_test",
    user_id="user_123",
    variants=[
        {"name": "control",   "weight": 50},
        {"name": "treatment", "weight": 50}
    ]
)
# user_123 always gets "treatment" for this experiment
```

**Multi-Variant (A/B/C/n) Split:**
```
Bucket 0-33  → variant A
Bucket 34-66 → variant B
Bucket 67-99 → variant C
```

**Tracking Assignment for Analysis:**
```python
# Log every assignment for downstream statistical analysis
analytics.track("experiment_assignment", {
    "experiment_key": experiment_key,
    "user_id": user_id,
    "variant": variant,
    "timestamp": datetime.utcnow().isoformat()
})
```

**Statistical Requirements:**
- Minimum sample size per variant before declaring significance
- Use Chi-squared or t-test for conversion metrics
- Watch for novelty effect (users interact with "new" things more initially)
- Run for at least 2 full business cycles (week Mon–Sun pattern)
- Avoid running too many experiments on overlapping user populations

---

### Q10. How do you handle feature flag debt — cleaning up stale flags?

**Feature flag debt** occurs when flags are not removed after the rollout completes. Over time, accumulated stale flags make the codebase hard to read, test, and reason about. A flag left for 6 months may have outlived both its variants and the engineers who created it.

**Detection Strategy:**
```sql
-- Find flags not updated in 30+ days that are at 100% rollout
SELECT flag_key, rollout_percent, updated_at,
       EXTRACT(DAY FROM NOW() - updated_at) AS days_stale
FROM feature_flags
WHERE rollout_percent = 100
  AND updated_at < NOW() - INTERVAL '30 days'
ORDER BY days_stale DESC;
```

**Flag Lifecycle States:**
```
CREATED → ROLLING_OUT → FULLY_ROLLED_OUT → SCHEDULED_FOR_REMOVAL → DELETED
                                                    ↑
                                    Set this state + create cleanup ticket
```

**Code Cleanup Process:**
```python
# Before cleanup: flag everywhere
def render_new_ui(user):
    if flags.is_enabled("new_dashboard_v2", user):
        return new_dashboard()
    return old_dashboard()

# After cleanup: remove the flag and the dead branch
def render_new_ui(user):
    return new_dashboard()   # old_dashboard() deleted
```

**Organizational Practices:**
| Practice | Description |
|---|---|
| Expiry date on creation | Every flag gets a `expires_at` date |
| Automated PR creation | Bot creates PR to remove expired flags |
| Ownership field | Every flag has an owner team |
| Dashboard of stale flags | Visible in engineering metrics |
| Sprint cleanup | Team reviews stale flags each sprint |

**Cost of not cleaning up:**
- Each stale flag is a conditional branch that must be tested in both states
- N flags = 2^N possible combinations to test (combinatorial explosion)
- Dead code accumulates, making refactoring harder

---

### Q11. How do feature flags work across microservices — how do you coordinate flag state?

In a microservices architecture, a user request may touch 5–10 services. Each service may independently evaluate a feature flag. Without coordination, **service A** may think the flag is on for a user while **service B** thinks it is off, causing split-brain behavior.

```
USER REQUEST
     │
     ▼
 API Gateway ──► User Service ──► Order Service ──► Inventory Service
     │                │                │                    │
  evaluates        evaluates        evaluates            evaluates
  flag=ON          flag=ON          flag=OFF             flag=ON
                                       ↑
                              Cache miss, got stale value!
```

**Solutions:**

**Option 1: Centralized Flag Evaluation (API Gateway evaluates once)**
```python
# API Gateway evaluates all flags for the user once
# and passes the resolved flag values in a request header
def middleware(request):
    user = get_user(request)
    flag_context = flag_service.evaluate_all(user)
    # Pass as header to downstream services
    request.headers["X-Feature-Flags"] = json.dumps(flag_context)
    return forward(request)

# Downstream service reads pre-evaluated flag from header
def order_service_handler(request):
    flags = json.loads(request.headers.get("X-Feature-Flags", "{}"))
    if flags.get("new_checkout_flow"):
        return new_checkout()
    return old_checkout()
```

**Option 2: SDK with Shared Cache (all services poll same Redis)**
```
All services → flag SDK → Redis (shared) → PostgreSQL
                              ↑
                     30-second TTL, all instances see same values
```

**Option 3: Flag Context Propagation via OpenTelemetry Baggage**
```python
# Propagate flag state in distributed trace baggage
span.set_baggage("feature.new_checkout", "true")
# Downstream services read from trace context
```

**Best Practice:** Evaluate flags as close to the user boundary as possible (API gateway or BFF layer), then propagate the resolved values. This guarantees consistency within a single request's lifecycle.

---

### Q12. How do you implement feature flags in database migrations using the expand-contract pattern?

Database migrations are the riskiest part of any deployment because they can't be rolled back without data loss. The **expand-contract** (or parallel-change) pattern combined with feature flags allows zero-downtime schema migrations.

**The Three Phases:**

```
PHASE 1: EXPAND (backward-compatible schema change)
──────────────────────────────────────────────────
ALTER TABLE users ADD COLUMN phone_v2 VARCHAR(20);  -- new column nullable
-- Both old column (phone) and new column (phone_v2) exist
-- Feature flag "use_phone_v2" = OFF
-- Writes go to BOTH columns; reads from old column

PHASE 2: MIGRATE (backfill + dual-write)
──────────────────────────────────────────────────
-- Backfill existing data
UPDATE users SET phone_v2 = normalize_phone(phone) WHERE phone_v2 IS NULL;
-- Feature flag "use_phone_v2" still = OFF
-- Monitor: confirm phone_v2 has same coverage as phone

PHASE 3: CONTRACT (remove old column)
──────────────────────────────────────────────────
-- Feature flag "use_phone_v2" = ON (new column is authoritative)
-- Remove dual-write to old column
-- Schedule: ALTER TABLE users DROP COLUMN phone;
```

**Code with Feature Flag Guard:**
```python
def save_user_phone(user_id: str, phone: str):
    normalized = normalize_phone(phone)

    if flags.is_enabled("use_phone_v2"):
        # Write only to new column
        db.execute("UPDATE users SET phone_v2 = %s WHERE id = %s",
                   normalized, user_id)
    else:
        # Dual-write: write to both columns during migration
        db.execute("""
            UPDATE users SET phone = %s, phone_v2 = %s WHERE id = %s
        """, phone, normalized, user_id)

def get_user_phone(user_id: str) -> str:
    if flags.is_enabled("use_phone_v2"):
        return db.scalar("SELECT phone_v2 FROM users WHERE id = %s", user_id)
    return db.scalar("SELECT phone FROM users WHERE id = %s", user_id)
```

**Rollback Safety:** If Phase 3 reveals a bug, flip the flag back to OFF — the old column is still intact. This is impossible if you drop the column immediately.

---

### Q13. How does LaunchDarkly / Unleash architecture work internally?

Understanding how managed feature flag services are built helps you design your own or evaluate third-party options intelligently.

```
LAUNCHDARKLY ARCHITECTURE
─────────────────────────────────────────────────────────────
  Dashboard / API  ──► Config DB (flag rules) ──► Event Bus
                                                     │
                                              Streaming Service
                                           (Server-Sent Events)
                                                     │
                                    ┌────────────────┴────────┐
                                    │                         │
                               SDK (Go)                  SDK (Python)
                            local flag store          local flag store
                                    │                         │
                             Application A            Application B
                            (evaluates locally)      (evaluates locally)
                                    │
                              Event Sink (analytics events)
```

**Key Architecture Decisions:**

**1. Streaming Updates (not polling):**
LaunchDarkly uses Server-Sent Events (SSE) to push flag updates to all SDK instances within ~200ms of a change. This is critical for kill switches where 30-second polling is too slow.

```
Flag changes in DB
      │
      ▼
Streaming server pushes SSE event to all connected SDKs
      │
      ▼
SDK updates local in-memory flag store
      │
      ▼
Next flag evaluation uses new value immediately
```

**2. Local Evaluation (not RPC per request):**
The SDK downloads ALL flag rules to local memory. Flag evaluation is a local function call, not a network request. This means:
- Zero added latency to flag evaluation
- Works even if LaunchDarkly servers are unreachable (last known config)

**3. Analytics Event Batching:**
Every flag evaluation generates an analytics event (for experiment tracking). These are batched and sent asynchronously to avoid impacting request latency.

**4. Unleash (Open-source alternative):**
```
Unleash Server (Node.js) ← Admin UI
        │
  PostgreSQL (flag config)
        │
  Unleash SDK (polls every 15s or uses SSE)
        │
  Application (local evaluation)
```

Unleash supports the same pattern but is self-hosted, making it suitable for air-gapped environments.

---

### Q14. What metrics should you track during a feature flag rollout to detect regressions early?

A gradual rollout is only as good as your ability to detect problems. The metrics framework should compare the flag=ON cohort vs flag=OFF cohort in real time.

**Metric Framework (RADE):**

```
R — Reliability  (error rate, 5xx rate, exception rate)
A — Application  (business metrics: conversion, revenue, completion)
D — Duration     (latency: p50, p95, p99)
E — Engagement   (user actions: clicks, sessions, page views)
```

**Per-Cohort Metric Comparison:**
```python
# Tag all metrics with flag variant
metrics.increment("checkout.attempts",
                  tags={"flag_new_checkout": str(flag_value)})
metrics.histogram("checkout.latency_ms", latency,
                  tags={"flag_new_checkout": str(flag_value)})
metrics.increment("checkout.errors",
                  tags={"flag_new_checkout": str(flag_value)})

# Dashboards compare ON vs OFF cohorts
```

**Automated Rollout Gate:**
```python
def should_advance_rollout(flag_key: str, current_percent: int) -> bool:
    on_metrics  = metrics.get_cohort(flag_key, variant="on",  window="1h")
    off_metrics = metrics.get_cohort(flag_key, variant="off", window="1h")

    # Regression criteria
    if on_metrics.error_rate > off_metrics.error_rate * 1.1:  # 10% worse
        alert(f"Error rate regression in {flag_key}")
        return False
    if on_metrics.p99_latency > off_metrics.p99_latency * 1.2:  # 20% slower
        alert(f"Latency regression in {flag_key}")
        return False
    if on_metrics.conversion < off_metrics.conversion * 0.95:  # 5% drop
        alert(f"Conversion regression in {flag_key}")
        return False
    return True
```

**Minimum rollout observation time:** At least 30 minutes at each stage (to account for traffic patterns) before auto-advancing. Some teams require 24 hours at 5% before moving to 25%.

---

### Q15. How do you ensure feature flag consistency — the same user always sees the same variant?

**Consistency** means that a user who sees variant A on their first request continues to see variant A on every subsequent request, even across different servers, regions, and sessions. Inconsistency creates jarring UX — for example, a checkout button that appears and disappears.

**Mechanisms for Sticky Assignment:**

**1. Deterministic Hashing (stateless, most common):**
```python
# No database needed — hash always produces the same bucket
def get_bucket(flag_key: str, user_id: str) -> int:
    return int(hashlib.sha256(f"{flag_key}:{user_id}".encode()).hexdigest(), 16) % 100
```
**Pro:** Stateless, consistent across all servers without coordination.
**Con:** Cannot manually move a user between variants.

**2. Assignment Table (stateful):**
```sql
CREATE TABLE flag_assignments (
    flag_key   VARCHAR(100),
    user_id    VARCHAR(100),
    variant    VARCHAR(50),
    assigned_at TIMESTAMP,
    PRIMARY KEY (flag_key, user_id)
);
-- Lookup before evaluation; write on first assignment
```
**Pro:** Can re-assign users, supports holdout groups.
**Con:** Database lookup per flag evaluation; needs caching.

**3. Session Stickiness (edge case):**
For anonymous users without a stable `user_id`, use a cookie-based device ID:
```python
def get_or_create_device_id(request) -> str:
    device_id = request.cookies.get("__device_id")
    if not device_id:
        device_id = str(uuid.uuid4())
        set_cookie(response, "__device_id", device_id, max_age=365*24*3600)
    return device_id
```

**Consistency Requirements Checklist:**
- Same user on mobile app and web → same variant (requires `user_id`, not `session_id`)
- User logs out, logs back in → same variant (server-side assignment, not client-side)
- Server A and Server B → same variant (hash function, not random)
- Before and after flag config update → same variant (maintain assignments in DB, or use stable hash)

---

## Hard (Q16–Q20)

---

### Q16. How do you design a feature flag system that handles 100,000 flag evaluations per second with sub-millisecond latency?

At 100K evaluations/second, you cannot make a network call per evaluation. The architecture must push all flag data to the application's local memory and evaluate entirely in-process.

```
HIGH-PERFORMANCE FLAG ARCHITECTURE
────────────────────────────────────────────────────────────────
                    ┌─────────────────┐
Admin Dashboard ──► │   Flag Config   │
                    │   Service (Go)  │
                    │  + PostgreSQL   │
                    └────────┬────────┘
                             │  SSE stream (flag diffs)
                    ┌────────▼────────┐
                    │  Streaming Edge │  (Pub/Sub gateway)
                    │  Layer          │
                    └────────┬────────┘
                        SSE push on change (~200ms propagation)
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   App Server 1      App Server 2       App Server N
   in-memory map     in-memory map     in-memory map
   (all flag rules)  (all flag rules)  (all flag rules)
          │
   Local evaluation: O(1) hash lookup + rule matching
   Zero network calls. Sub-microsecond evaluation.
```

**In-Memory Flag Store:**
```go
// Go implementation — evaluation is in-process
type FlagStore struct {
    mu    sync.RWMutex
    flags map[string]*Flag   // keyed by flag_key
}

func (s *FlagStore) Evaluate(flagKey string, user User) Variant {
    s.mu.RLock()
    flag := s.flags[flagKey]
    s.mu.RUnlock()

    if flag == nil || !flag.Enabled {
        return DefaultVariant
    }
    // All evaluation logic: O(rules) per flag, typically < 20 rules
    return flag.Evaluate(user)
}
```

**Streaming Update Protocol:**
```json
// SSE event: only send deltas, not full flag list
{
  "type": "flag_updated",
  "flag_key": "new_checkout_flow",
  "version": 1047,
  "payload": { "enabled": true, "rollout_percent": 25 }
}
```

**Capacity math:**
- 1M flags × 64 bytes avg config = 64MB per server (acceptable in-memory footprint for even large flag sets)
- Evaluation: hash lookup (O(1)) + rule matching (O(k) where k = number of rules, typically < 10)
- 100K evaluations/sec = 10μs per evaluation budget — local evaluation easily fits in < 1μs

**Failure Mode:** If the Flag Config Service is unreachable, the SDK continues serving the last known flag values from its in-memory store. This is better than failing open or failing closed with a network error.

---

### Q17. How do you design a blue-green / canary / rolling / feature flag hybrid deployment strategy for maximum safety?

Modern deployment pipelines use all four techniques in combination, not as alternatives. Understanding what each layer protects against allows you to compose them correctly.

```
DEPLOYMENT SAFETY LAYERS
──────────────────────────────────────────────────────────────
Layer 1: Feature Flag (code layer)
  → Protects: new product logic visible to users
  → Rollback time: < 1 second (flip flag)
  → Granularity: per-user, per-segment

Layer 2: Canary Deployment (infra layer)
  → Protects: new binary / framework / dependency regressions
  → Rollback time: 1–5 minutes (shift LB weight)
  → Granularity: per-request (% of traffic to new binary)

Layer 3: Rolling Deployment (fleet layer)
  → Protects: gradual binary rollout without downtime
  → Rollback time: 10–30 minutes (re-deploy old version)
  → Granularity: per-server (replace N servers at a time)

Layer 4: Blue-Green (environment layer)
  → Protects: infrastructure/DB schema changes
  → Rollback time: < 1 minute (switch DNS/LB to green)
  → Granularity: entire environment
```

**Combined Strategy for a Major Release:**

```
Step 1: Deploy new code to GREEN environment (all flags OFF)
        GREEN is not receiving production traffic
        ↓
Step 2: Run smoke tests, integration tests on GREEN
        ↓
Step 3: Canary: shift 5% of traffic to GREEN
        Monitor error rate and latency vs BLUE
        ↓
Step 4: Advance canary: 5% → 25% → 50% → 100% (shift LB)
        At 100%, GREEN is now ACTIVE, BLUE is standby
        ↓
Step 5: Feature flag rollout: turn on new feature for 5% of users
        Monitor business metrics
        ↓
Step 6: Advance flag rollout: 5% → 25% → 50% → 100%
        ↓
Step 7: Decommission flag. BLUE becomes next deployment's GREEN.
```

**Rollback Decision Tree:**
```
Error spike detected
      │
      ├─ Is it a business logic bug?
      │    └─ Flip feature flag OFF → instant, zero infra change
      │
      ├─ Is it in the new binary (affects all features)?
      │    └─ Shift canary weight back to BLUE → 1-5 min
      │
      └─ Is it a DB schema / infra issue?
           └─ Switch DNS back to BLUE (blue-green) → < 1 min
```

---

### Q18. How do you implement a self-hosted feature flag system using PostgreSQL and Redis without a third-party service?

A production-grade self-hosted feature flag system requires: a flag config store, a fast evaluation cache, SDK client libraries, an admin API, and an audit log.

**Database Schema:**
```sql
-- Flag definitions
CREATE TABLE feature_flags (
    flag_key         VARCHAR(100) PRIMARY KEY,
    description      TEXT,
    flag_type        VARCHAR(20) NOT NULL,  -- 'release', 'experiment', 'ops', 'permission'
    enabled          BOOLEAN DEFAULT FALSE,
    rollout_percent  INT DEFAULT 0 CHECK (rollout_percent BETWEEN 0 AND 100),
    targeting_rules  JSONB DEFAULT '[]',
    variants         JSONB DEFAULT '{"on": true, "off": false}',
    owner_team       VARCHAR(100),
    expires_at       TIMESTAMP,
    created_at       TIMESTAMP DEFAULT NOW(),
    updated_at       TIMESTAMP DEFAULT NOW()
);

-- Audit log
CREATE TABLE flag_audit_log (
    id            BIGSERIAL PRIMARY KEY,
    flag_key      VARCHAR(100) NOT NULL,
    changed_by    VARCHAR(100) NOT NULL,
    change_type   VARCHAR(20) NOT NULL,  -- 'created', 'updated', 'deleted'
    old_value     JSONB,
    new_value     JSONB,
    changed_at    TIMESTAMP DEFAULT NOW()
);

-- User assignments (for experiments)
CREATE TABLE flag_user_assignments (
    flag_key    VARCHAR(100) NOT NULL,
    user_id     VARCHAR(100) NOT NULL,
    variant     VARCHAR(50)  NOT NULL,
    assigned_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (flag_key, user_id)
);
```

**Redis Caching Layer:**
```python
class FlagCache:
    FLAG_PREFIX = "ff:"
    ALL_FLAGS_KEY = "ff:all"
    DEFAULT_TTL = 30  # seconds

    def get_flag(self, flag_key: str) -> dict | None:
        data = self.redis.get(f"{self.FLAG_PREFIX}{flag_key}")
        return json.loads(data) if data else None

    def set_flag(self, flag_key: str, flag_data: dict):
        self.redis.setex(f"{self.FLAG_PREFIX}{flag_key}",
                         self.DEFAULT_TTL, json.dumps(flag_data))

    def invalidate_flag(self, flag_key: str):
        self.redis.delete(f"{self.FLAG_PREFIX}{flag_key}")
        # Also publish change event for SDK streaming
        self.redis.publish("flag_changes", json.dumps({"flag_key": flag_key}))
```

**Admin API:**
```python
@app.patch("/flags/{flag_key}/rollout")
def update_rollout(flag_key: str, percent: int, user=Depends(require_auth)):
    old_flag = db.get_flag(flag_key)
    db.update_flag(flag_key, rollout_percent=percent)
    cache.invalidate_flag(flag_key)
    # Write audit log
    db.insert_audit(flag_key, user.email, "updated",
                    old_value=old_flag, new_value={"rollout_percent": percent})
    return {"status": "updated"}
```

**SDK Polling with Redis Pub/Sub:**
```python
class FlagSDK:
    def __init__(self):
        self._local_cache = {}
        self._subscribe_to_changes()  # Redis pub/sub

    def _subscribe_to_changes(self):
        pubsub = self.redis.pubsub()
        pubsub.subscribe("flag_changes")
        threading.Thread(target=self._listen, args=(pubsub,), daemon=True).start()

    def _listen(self, pubsub):
        for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                self._local_cache.pop(data["flag_key"], None)  # evict stale entry
```

This gives you real-time flag propagation (< 100ms) without requiring polling.

---

### Q19. How do feature flags interact with testing strategies — how do you ensure both branches of a flag are tested?

Feature flags introduce **dead code risk**: if you only test the "on" state, the "off" branch accumulates bugs that surface during rollback. If you only test the "off" state, the "on" branch has never been tested. Both branches must be exercised.

**Unit Testing — Mock the Flag:**
```python
# Parameterized test: run the same test for both flag states
@pytest.mark.parametrize("flag_enabled", [True, False])
def test_checkout_flow(flag_enabled, mocker):
    mocker.patch("flags.is_enabled",
                 side_effect=lambda key, user: flag_enabled if key == "new_checkout" else False)

    cart = create_test_cart()
    result = checkout(cart, user=test_user())

    if flag_enabled:
        assert result.flow_version == "v2"
    else:
        assert result.flow_version == "v1"
    # Both flows complete without error
    assert result.status == "success"
```

**Integration Testing — Test Matrix:**
```
Test Suite Matrix for "new_checkout_flow" flag:
┌──────────────────────────────────────────────────────────┐
│ Test Case          │ Flag State │ Expected Behavior        │
├──────────────────────────────────────────────────────────┤
│ happy_path_v1      │ OFF        │ old checkout completes   │
│ happy_path_v2      │ ON         │ new checkout completes   │
│ payment_failure_v1 │ OFF        │ error handled in old UI  │
│ payment_failure_v2 │ ON         │ error handled in new UI  │
│ edge_case_empty_v1 │ OFF        │ empty cart handled       │
│ edge_case_empty_v2 │ ON         │ empty cart handled       │
└──────────────────────────────────────────────────────────┘
```

**CI/CD Pipeline Integration:**
```yaml
# Run tests in both flag states in CI pipeline
test_with_flag_on:
  env:
    FEATURE_FLAGS_OVERRIDE: '{"new_checkout_flow": true}'
  run: pytest tests/checkout/

test_with_flag_off:
  env:
    FEATURE_FLAGS_OVERRIDE: '{"new_checkout_flow": false}'
  run: pytest tests/checkout/
```

**Preventing Untested Flag Branches — Static Analysis:**
```python
# Custom lint rule: flag keys must have corresponding test coverage
# Scan for is_enabled("flag_key") → require test with both True/False mock
def check_flag_coverage(source_file, test_file):
    flags_used = extract_flag_calls(source_file)
    flags_tested = extract_flag_mocks(test_file)
    untested = flags_used - flags_tested
    if untested:
        raise LintError(f"Flags without test coverage: {untested}")
```

**E2E Testing with Overrides:**
Allow QA to set flag overrides via a special header (only in non-production):
```
X-Feature-Flag-Override: new_checkout_flow=true
```
This lets QA test the new checkout flow on staging even before it's rolled out.

---

### Q20. How do you design an instant rollback strategy — flag rollback vs code rollback — and which is faster?

When an incident occurs, every second counts. Understanding the rollback options and their time-to-recovery (TTR) is essential for designing a safe deployment strategy.

**Rollback Options Comparison:**
```
ROLLBACK STRATEGY COMPARISON
────────────────────────────────────────────────────────────────────
Method            │ TTR         │ Who can do it │ Risk             │
──────────────────────────────────────────────────────────────────
Feature flag flip │ < 5 seconds │ SRE, PM, Eng  │ Very low         │
Canary revert     │ 1–5 min     │ SRE            │ Low              │
Blue-green switch │ 1–2 min     │ SRE            │ Low              │
Git revert + CI   │ 15–45 min   │ Engineer       │ Medium (new code)│
Hotfix deploy     │ 30–120 min  │ Engineer       │ Medium–High      │
DB rollback       │ 30–480 min  │ DBA            │ Very High (data) │
────────────────────────────────────────────────────────────────────
```

**Feature Flag Rollback Architecture:**
```
Incident detected (error spike on new_checkout_flow=ON cohort)
                │
                ▼ (SRE opens dashboard, takes 5 seconds)
         Set rollout_percent = 0 for "new_checkout_flow"
                │
                ▼ Cache invalidation propagates via Redis pub/sub
                │ (< 100ms to all SDK instances)
                ▼
         All users now see old checkout flow
                │
                ▼ Error spike resolves within 1–2 minutes
                  (in-flight requests complete on old path)
```

**Automated Rollback:**
```python
class RolloutWatchdog:
    def __init__(self, flag_key: str, threshold: float):
        self.flag_key = flag_key
        self.error_threshold = threshold  # e.g., 0.05 = 5% error rate

    def check_and_rollback(self):
        on_cohort  = metrics.error_rate(self.flag_key, variant="on",  window="5m")
        off_cohort = metrics.error_rate(self.flag_key, variant="off", window="5m")

        if on_cohort > off_cohort + self.error_threshold:
            logger.critical(f"Auto-rollback: {self.flag_key} error rate "
                            f"{on_cohort:.2%} vs baseline {off_cohort:.2%}")
            flag_service.set_rollout(self.flag_key, 0)
            pagerduty.alert(f"Auto-rollback executed for {self.flag_key}")
```

**Code Rollback Timing (when flag rollback is not enough):**
If the bug exists in shared code (not behind a flag), a code rollback is required. This takes 15–45 minutes minimum through CI/CD. Best practice: always wrap risky changes in a flag so rollback is always < 30 seconds.

**Rollback Decision Tree:**
```
Error spike detected
      │
      ├─► Bug is in code behind a feature flag?
      │         YES → Flip flag OFF (< 5 seconds)
      │
      ├─► Bug is in canary binary affecting all features?
      │         YES → Shift canary traffic back (< 5 min)
      │
      └─► Bug is in infrastructure / shared code?
                YES → Blue-green switch or git revert + deploy
```

**Key Principle:** Design every risky change to be behind a flag. "Every commit is instantly rollbackable" is the goal.

---

## Quick Reference

```
FEATURE FLAG TYPES
──────────────────────────────────────────────────────
Release flag     → short-lived, hide incomplete work
Experiment flag  → A/B test, analyze with stats
Ops flag         → kill switch, circuit breaker, rate control
Permission flag  → entitlement, beta access, tier gating

EVALUATION ORDER
──────────────────────────────────────────────────────
1. Flag globally disabled?  → return default
2. Individual targeting?    → return targeted variant
3. Segment targeting?       → return segment variant
4. In rollout %?            → return ON variant
5. Default                  → return OFF variant

ROLLOUT STAGES
──────────────────────────────────────────────────────
Internal (0% public) → 5% → 25% → 50% → 100% → delete flag

BUCKET FORMULA
──────────────────────────────────────────────────────
bucket = SHA256(flag_key + ":" + user_id) % 100
user sees ON if bucket < rollout_percent

ROLLBACK SPEED
──────────────────────────────────────────────────────
Feature flag flip  →  < 5 seconds
Canary revert      →  1–5 minutes
Blue-green switch  →  1–2 minutes
Git revert + CI    →  15–45 minutes

DARK LAUNCH PATTERN
──────────────────────────────────────────────────────
Run new code path → measure → discard output → return old result
Goal: test performance/correctness with real prod traffic safely

FLAG DEBT PREVENTION
──────────────────────────────────────────────────────
Set expires_at on creation
Review stale flags (rollout_percent=100 AND updated_at > 30 days)
Auto-create cleanup PRs
Delete flag + dead branch as a unit

TESTING RULE
──────────────────────────────────────────────────────
For every flag, test BOTH branches in CI
Parameterize tests with flag=True and flag=False

SELF-HOSTED STACK
──────────────────────────────────────────────────────
PostgreSQL (config + audit) → Redis (30s TTL cache + pub/sub)
→ SDK (local in-memory, SSE/pub/sub refresh)
→ Evaluate locally (zero network calls per evaluation)

KEY TOOLS
──────────────────────────────────────────────────────
LaunchDarkly    → managed, streaming, enterprise
Split           → managed, experiment-focused
Unleash         → open-source, self-hosted
Flagsmith       → open-source, self-hosted
Flipt           → open-source, gRPC API
GrowthBook      → open-source, A/B testing focus
```
