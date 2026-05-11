# 31 — Reliability Engineering and SRE

---

## Easy (Q1–Q7)

---

### Q1. What is Site Reliability Engineering (SRE) and how does it differ from traditional DevOps?

**Answer:**

Site Reliability Engineering (SRE) is a discipline pioneered by Google that applies software engineering principles to infrastructure and operations problems. The goal is to create scalable, reliable software systems by treating reliability as a feature — not an afterthought.

**SRE vs DevOps:**

| Dimension         | SRE                                       | DevOps                                      |
|-------------------|-------------------------------------------|---------------------------------------------|
| Origin            | Google (2003, Ben Treynor Sloss)          | Community movement (2009)                   |
| Prescriptiveness  | Prescriptive — specific practices (error budgets, SLOs) | Philosophical — principles and culture     |
| Role              | Dedicated SRE role with coding mandate    | Often cross-functional responsibility       |
| Toil limit        | Enforced 50% toil cap                     | Not formally defined                        |
| Reliability metric| Error budgets, SLIs, SLOs                 | Deployment frequency, MTTR                  |
| Focus             | Production reliability + capacity planning| CI/CD pipeline, collaboration, culture      |

**Core SRE tenets:**
1. SREs write code — they are software engineers who happen to work on operations
2. Error budgets govern release velocity
3. Postmortems are blameless
4. Toil is bounded to 50% of work
5. Reliability is defined explicitly via SLOs

```
SRE Philosophy:
  "If a human operator needs to touch your system during normal operations,
   you have a bug. The definition of normal changes as your systems grow."
                                                      — Ben Treynor Sloss
```

DevOps is broader and cultural; SRE is a concrete implementation of DevOps principles with specific tools and metrics.

---

### Q2. What is an error budget and how do teams use it to balance features vs reliability?

**Answer:**

An error budget is the maximum allowable unreliability for a service over a rolling window (typically 30 days). It is derived directly from the SLO:

```
Error Budget = 1 - SLO target
Example: SLO = 99.9% availability
Error Budget = 0.1% of time = 43.8 minutes/month of downtime allowed
```

**How it works in practice:**

```
Month starts: Full error budget available (43.8 minutes)
  |
  |--- Incident 1: 10-min outage  --> 33.8 min remaining
  |--- Deploy bug: 5-min degradation --> 28.8 min remaining
  |--- Planned maintenance: 15 min --> 13.8 min remaining
  |
Budget almost exhausted --> Engineering freezes feature work
                        --> Focus shifts to reliability improvements
```

**Decision framework:**

| Budget Status    | Action                                              |
|------------------|-----------------------------------------------------|
| Budget healthy   | Teams can take risks, ship features aggressively    |
| Budget 50% gone  | Review upcoming deploys, increase testing           |
| Budget exhausted | Feature freeze; SRE and dev focus on reliability    |
| Consistently full| SLO may be too loose; tighten it                    |

**Benefits of error budgets:**
- Removes the adversarial relationship between Dev (ship fast) and Ops (keep stable)
- Makes reliability a shared responsibility — both teams burn the same budget
- Creates a neutral, data-driven conversation: "We have X minutes left, do we want to risk this deploy?"
- Incentivizes devs to write reliable code since they share budget ownership

Error budgets are the single most powerful concept in SRE because they quantify exactly how much risk a team is allowed to take.

---

### Q3. Explain the SLI/SLO/SLA hierarchy with concrete examples.

**Answer:**

These three terms form a layered reliability contract:

```
  ┌─────────────────────────────────────────┐
  │  SLA (Service Level Agreement)          │  ← Customer-facing legal contract
  │    "99.5% availability or we refund"    │
  └────────────────┬────────────────────────┘
                   │ must be weaker than
  ┌────────────────▼────────────────────────┐
  │  SLO (Service Level Objective)          │  ← Internal engineering target
  │    "99.9% availability"                 │
  └────────────────┬────────────────────────┘
                   │ measured by
  ┌────────────────▼────────────────────────┐
  │  SLI (Service Level Indicator)          │  ← Raw metric / measurement
  │    "% of requests < 200ms & 2xx"        │
  └─────────────────────────────────────────┘
```

**Concrete example — Payment API:**

| Layer | Definition | Example Value |
|-------|-----------|---------------|
| SLI   | Ratio of successful requests (2xx, < 500ms) to total requests | Measured continuously |
| SLO   | 99.9% of requests are successful over 30 days | 99.9% |
| SLA   | 99.5% availability guaranteed to customers with credits if missed | 99.5% |

**Why SLO > SLA:**
The SLO (99.9%) is tighter than the SLA (99.5%) to provide a safety buffer. If the team hits 99.7%, they've breached the SLO internally but not the SLA — alerting them to fix issues before customers are harmed.

**Latency SLI example:**
```
SLI = P99 latency of /checkout endpoint
SLO = P99 latency < 300ms for 99.5% of requests in any 28-day window
```

Good SLIs are: meaningful to users, unambiguous, measurable with existing infrastructure.

---

### Q4. What is toil in SRE and why do teams limit it to 50% of work?

**Answer:**

**Toil** is manual, repetitive, automatable operational work that scales linearly with service growth and provides no lasting value.

**Characteristics of toil:**
- **Manual** — a human must do it
- **Repetitive** — done over and over
- **Automatable** — a machine could do it
- **Tactical** — reactive, not strategic
- **No enduring value** — service is not improved by the work
- **Scales with service** — more traffic = more toil

**Examples:**
```
Toil:                          Not Toil:
- Manually restarting pods     - Writing the auto-restart script
- Resizing disk every month    - Building auto-scaling storage
- Responding to false alerts   - Fixing the alert threshold
- Copy-pasting deploy scripts  - Building a CD pipeline
- Handling password resets     - Building self-service password portal
```

**Why the 50% cap:**
```
Time Allocation Target:
  [████████████████████░░░░░░░░░░░░░░░░░░░░]
  50% Toil (max)              50% Engineering work (min)

If toil > 50%:
  - SRE team becomes a traditional ops team
  - No time to automate → toil grows → more toil (vicious cycle)
  - Team morale degrades
  - Reliability doesn't improve
```

**Google's rule:** If toil exceeds 50%, SRE management escalates to product/dev teams to fix the underlying causes. This creates strong incentive for dev teams to write reliable, self-healing services.

Tracking toil: SREs log all toil work with estimated hours in a spreadsheet or JIRA project monthly.

---

### Q5. What are the four golden signals for monitoring a service?

**Answer:**

The four golden signals (from Google's SRE book) are the minimum set of metrics needed to understand a service's health:

```
┌─────────────┬──────────────────────────────────────────────────────────┐
│  Signal     │  Definition + Example                                    │
├─────────────┼──────────────────────────────────────────────────────────┤
│  Latency    │  Time to handle a request                                │
│             │  Track successful AND failed latency separately          │
│             │  Example: P50=50ms, P99=300ms, P999=2s                   │
├─────────────┼──────────────────────────────────────────────────────────┤
│  Traffic    │  Volume of demand on the system                          │
│             │  HTTP: requests/sec; Streaming: bytes/sec; DB: queries/s │
│             │  Example: 12,000 RPS peak                                │
├─────────────┼──────────────────────────────────────────────────────────┤
│  Errors     │  Rate of failed requests                                 │
│             │  Explicit (5xx) + implicit (200 with wrong data)         │
│             │  Example: 0.01% error rate                               │
├─────────────┼──────────────────────────────────────────────────────────┤
│  Saturation │  How full/loaded the service is                          │
│             │  CPU %, memory %, queue depth, disk I/O                  │
│             │  Example: 78% CPU — approaching limit                    │
└─────────────┴──────────────────────────────────────────────────────────┘
```

**Comparison with other frameworks:**

| Framework | Signals | Best For |
|-----------|---------|----------|
| Four Golden Signals | Latency, Traffic, Errors, Saturation | User-facing services |
| RED | Rate, Errors, Duration | Microservices |
| USE | Utilization, Saturation, Errors | Infrastructure/hardware |

**Dashboard setup:**
```
Recommended: One dashboard per service with:
  Row 1: Traffic (RPS graph)
  Row 2: Error rate (% graph, red line at SLO threshold)
  Row 3: Latency (P50/P95/P99 time series)
  Row 4: Saturation (CPU/memory heatmap)
```

If you can only monitor 4 things per service, monitor these four.

---

### Q6. Define MTTD, MTTR, and MTBF. Why do teams track each?

**Answer:**

These three metrics describe different phases of a system failure lifecycle:

```
Timeline of an incident:
─────────────────────────────────────────────────────────────────────►
│                  │                  │              │
Failure starts     Alert fires        Fix deployed   System stable
│                  │                  │              │
│←── MTTD ────────►│                  │              │
│←────────────── MTTR ───────────────►│              │
│                                                    │
│◄────────────────────── MTBF (from prev fix) ───────│
```

**Definitions:**

| Metric | Full Name | Formula | Goal |
|--------|-----------|---------|------|
| MTTD | Mean Time To Detect | Avg time from failure to alert/detection | Minimize |
| MTTR | Mean Time To Recover | Avg time from detection to full recovery | Minimize |
| MTBF | Mean Time Between Failures | Avg time between two incidents | Maximize |

**Typical values by tier:**

| Tier | MTTD | MTTR | MTBF |
|------|------|------|------|
| Tier 0 (critical) | < 2 min | < 15 min | > 30 days |
| Tier 1 | < 5 min | < 1 hour | > 7 days |
| Tier 2 | < 30 min | < 4 hours | > 1 day |

**Why each matters:**
- **MTTD**: Long detection = users suffering before anyone knows. Improve with synthetic monitors, anomaly detection, error budget burn alerts
- **MTTR**: Time from "we know" to "we're fixed." Improve with runbooks, rollback automation, oncall training
- **MTBF**: How often failures occur. Improve with chaos testing, code quality, redundancy

Teams report these monthly in SRE reviews to trend over time.

---

### Q7. What is change failure rate and how does deployment frequency relate to reliability?

**Answer:**

**Change Failure Rate (CFR)** is the percentage of deployments that cause a production incident requiring a rollback, hotfix, or service degradation.

```
CFR = (# of failed deployments) / (total deployments) × 100%
```

**DORA metrics and CFR:**

| DORA Metric | Elite | High | Medium | Low |
|-------------|-------|------|--------|-----|
| Deployment Frequency | Multiple/day | Daily | Weekly | Monthly |
| Change Failure Rate | 0–5% | 5–10% | 10–15% | > 15% |
| MTTR | < 1 hour | < 1 day | < 1 week | > 6 months |

**Deployment frequency and blast radius:**

```
Large infrequent deployments:
  [1000 changes] ──deploy──► incident (which of 1000 changes caused it?)
                              MTTR: hours/days (hard to isolate)

Small frequent deployments:
  [5 changes] ──deploy──► incident (much easier to find cause)
                           MTTR: minutes (roll back one small change)
```

**Why more frequent deployments improve reliability:**
1. Smaller blast radius — fewer changes per deploy
2. Faster rollback — only one or two commits to revert
3. Earlier feedback — bugs found sooner in the release cycle
4. Lower cognitive load — engineers remember what changed

**Tracking CFR in practice:**
```python
# Tag every deployment in your CD system
deployment_events = get_deployments(last_30_days)
failed = [d for d in deployment_events if d.caused_incident]
cfr = len(failed) / len(deployment_events) * 100
```

Target: < 5% CFR with daily deploys is an elite engineering organization.

---

## Medium (Q8–Q15)

---

### Q8. Explain burn rate alerts — 1-hour window (14.4x) vs 6-hour window (6x) — and how to set up multi-window alerting.

**Answer:**

Burn rate measures how fast your error budget is being consumed relative to the "normal" rate.

**Burn rate = 1.0** means you're consuming the error budget at exactly the rate that will exhaust it at the end of the 30-day window.

```
Error Budget = 0.1% (for 99.9% SLO)
30-day window = 43,200 minutes

Burn rate 1x:   consuming 0.1%  errors over 30 days → budget exhausted in 30 days
Burn rate 14.4x: consuming 1.44% errors per hour    → budget exhausted in 2.08 days
Burn rate 6x:   consuming 0.6%  errors per hour     → budget exhausted in 5 days
```

**Multi-window alerting (Google's recommended approach):**

The trick is using TWO windows per alert to avoid alert storms from brief spikes:

```
Alert 1 (Critical — Page oncall immediately):
  condition: burn_rate_1h > 14.4 AND burn_rate_5min > 14.4
  interpretation: "We're burning through the budget 14x fast — page NOW"
  budget consumed if no action: ~2% in 1 hour = significant

Alert 2 (Warning — Ticket, no page):
  condition: burn_rate_6h > 6 AND burn_rate_30min > 6
  interpretation: "Sustained elevated burn — fix within the day"
  budget consumed if no action: ~5% in 6 hours

Alert 3 (Info — Weekly review):
  condition: burn_rate_3d > 1
  interpretation: "Slow burn — investigate before week ends"
```

**Prometheus/Alertmanager example:**
```yaml
# Alert: high burn rate (critical)
- alert: HighErrorBudgetBurnRate
  expr: |
    (
      sum(rate(http_requests_total{status=~"5.."}[1h]))
      /
      sum(rate(http_requests_total[1h]))
    ) > (14.4 * 0.001)
    and
    (
      sum(rate(http_requests_total{status=~"5.."}[5m]))
      /
      sum(rate(http_requests_total[5m]))
    ) > (14.4 * 0.001)
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Error budget burning at 14.4x — page oncall"
```

**Why two windows?**
- Short window (1h/5m): detects fast burn
- Long window (6h/30m): confirms it's sustained (not a blip)
- Using both prevents noisy pages from transient spikes

---

### Q9. What is a blameless postmortem? Describe its structure and why blame is counterproductive.

**Answer:**

A blameless postmortem is a structured analysis of an incident focused on understanding **what happened and why**, not **who made a mistake**. The core insight is that humans make mistakes in systems designed to allow those mistakes — the system is at fault, not the individual.

**Why blameless culture matters:**
```
Blame culture:                     Blameless culture:
  Engineers hide mistakes            Engineers report issues openly
  Root causes stay unfixed           Root causes are addressed
  Fear of oncall                     Psychological safety
  Individuals punished               Systems improved
  Next incident guaranteed           Incident less likely to recur
```

**Blameless postmortem structure:**

```
1. INCIDENT SUMMARY (2-3 sentences)
   - Service affected, duration, customer impact, severity

2. TIMELINE (chronological, factual)
   - 14:02 UTC - Alert fired: P99 latency > 2s
   - 14:07 UTC - On-call engineer paged
   - 14:23 UTC - Root cause identified: config change at 13:45
   - 14:35 UTC - Rollback deployed
   - 14:42 UTC - Service fully recovered
   [Use "we" not "Alice" or "Bob"]

3. ROOT CAUSE ANALYSIS
   - Use 5 Whys or fishbone diagram
   - Why did latency spike? → DB connection pool exhausted
   - Why was pool exhausted? → Query introduced N+1 problem
   - Why wasn't it caught? → No query performance test in CI
   - Why no test? → No standard for query testing
   - Root cause: Missing engineering standard

4. IMPACT
   - X% of users affected for Y minutes
   - Z requests failed
   - Error budget consumed: N minutes

5. ACTION ITEMS (SMART goals)
   - Add query performance test to CI pipeline [Owner: Alice, Due: 2025-06-01]
   - Set DB connection pool alarm [Owner: Bob, Due: 2025-05-20]
   - Document connection pool sizing runbook [Owner: Team, Due: 2025-05-25]

6. LESSONS LEARNED
   - What went well? (monitoring caught it; rollback was fast)
   - What went poorly? (took 21 min to page; no runbook existed)
   - Where did we get lucky?
```

Postmortems should be written within 48-72 hours of incident resolution while memory is fresh. They should be shared broadly — they are learning artifacts, not blame documents.

---

### Q10. How do you reduce MTTR? Describe at least four concrete strategies.

**Answer:**

MTTR (Mean Time to Recover) is reduced by making the **detect → diagnose → fix → verify** cycle faster.

**Strategy 1: Observability (Reduce diagnosis time)**
```
Three pillars:
  Metrics  → dashboards, anomaly detection
  Logs     → structured logs with trace IDs, ERROR level sampling
  Traces   → distributed tracing (Jaeger/Zipkin) to find slow spans

Before: Engineer reads 10 log files across 5 services manually
After:  Engineer searches trace ID in Jaeger, sees slow DB query in 2 min
```

**Strategy 2: Runbooks (Reduce cognitive load during incidents)**
```
Runbook for "DB Connection Pool Exhausted":
  1. Run: kubectl describe pod <db-pod>
  2. Check: SELECT count(*) FROM pg_stat_activity;
  3. Kill long-running queries: SELECT pg_terminate_backend(pid) WHERE duration > interval '5 min'
  4. If not resolved: scale horizontally → kubectl scale deployment db --replicas=3
  5. Escalate: page DBA team
```
Runbooks convert expert knowledge into step-by-step scripts. MTTR drops from 1 hour to 15 minutes.

**Strategy 3: Automated rollback**
```
CD pipeline with automatic rollback:
  Deploy v2.1 → Monitor error rate for 5 min
    If error_rate > SLO threshold:
      Auto-rollback to v2.0
      Page oncall with: "Rollback triggered at 14:32 — v2.1 caused 2% error rate"
```

**Strategy 4: Chaos engineering + game days**
Practice incident response before real incidents. Run a game day monthly: inject a failure, have oncall follow runbooks, measure MTTR. Engineers who've practiced respond faster under pressure.

**Strategy 5: Feature flags for instant kill switches**
```python
if feature_flag_enabled("new_payment_flow"):
    run_new_flow()
else:
    run_old_flow()
# Flip flag = instant rollback without deployment
```

**Typical MTTR improvements:**

| Action | MTTR Before | MTTR After |
|--------|-------------|------------|
| Add runbooks | 60 min | 20 min |
| Add distributed tracing | 30 min | 10 min |
| Auto-rollback pipeline | 45 min | 8 min |
| Game days (training) | 90 min | 30 min |

---

### Q11. What is chaos engineering? Describe its principles, game days, and blast radius containment.

**Answer:**

Chaos engineering is the practice of deliberately injecting failures into production (or staging) systems to discover weaknesses before they become incidents.

**Netflix Chaos Monkey** — the original: randomly terminated EC2 instances in production to force engineers to build resilient services.

**Core principles (from Principles of Chaos Engineering):**
```
1. Define steady state (what "normal" looks like — error rate, latency)
2. Hypothesize steady state continues under fault
3. Introduce variables that reflect real-world failures
4. Try to disprove the hypothesis
```

**Types of chaos experiments:**

| Experiment | What it tests |
|------------|--------------|
| Kill random pod | Pod restart resilience, K8s self-healing |
| Introduce 500ms network latency | Timeout handling, retry logic |
| Fill disk to 95% | Disk-full error handling |
| Drop 10% of packets | Packet loss tolerance |
| Kill one DB replica | Failover behavior |
| Max out CPU on one node | Autoscaler response |

**Game day structure:**
```
Before:
  - Define hypothesis ("service will remain below 0.1% error rate during single AZ failure")
  - Notify all stakeholders
  - Have rollback plan ready
  - Start recording

During:
  - Inject failure
  - Observe metrics in real-time
  - Allow oncall to respond (don't intervene unless safety trigger hit)

After:
  - Write postmortem-style findings
  - Create action items for each discovered weakness
```

**Blast radius containment:**
```
Start small:
  1. Single unit test (mock failure)
  2. Staging environment
  3. 1% of production traffic (canary)
  4. Single availability zone
  5. All production

Never run chaos experiments without:
  - A way to stop immediately (kill switch)
  - Monitoring active
  - Business hours (not Friday afternoon)
  - Prior notification to support/sales
```

Tools: Chaos Monkey, Gremlin, AWS Fault Injection Simulator, LitmusChaos (Kubernetes).

---

### Q12. Explain progressive delivery: canary releases, feature flags, and dark launches.

**Answer:**

Progressive delivery is the practice of rolling out changes to a small subset of users before full rollout, allowing real-world validation with limited blast radius.

**Canary Release:**
```
Traffic routing:
  All users → v1.0 (stable)
       │
       └── 5% → v2.0 (canary)
       
Monitor for 30 min:
  If error_rate(v2.0) < threshold:
    Increment to 10% → 25% → 50% → 100%
  Else:
    Route 100% back to v1.0, investigate
```

Kubernetes canary with Argo Rollouts:
```yaml
strategy:
  canary:
    steps:
    - setWeight: 5
    - pause: { duration: 10m }
    - setWeight: 20
    - pause: { duration: 10m }
    - setWeight: 50
    - pause: { duration: 10m }
    analysis:
      templates:
      - templateName: success-rate
      args:
      - name: service-name
        value: payment-service
```

**Feature Flags:**
```python
# LaunchDarkly / Unleash / custom implementation
user_in_experiment = feature_flag.is_enabled(
    flag="new_checkout_flow",
    user_id=user.id,
    rollout_percentage=10  # 10% of users
)

if user_in_experiment:
    response = new_checkout_flow(cart)
else:
    response = old_checkout_flow(cart)
```

Benefits:
- Separate deploy from release (code ships dark, flag enables it)
- Instant rollback (flip flag, no redeploy)
- A/B testing built-in

**Dark Launches:**
```
Production traffic → Real handler (returns result to user)
                  ↘ Shadow handler (reads same request, result discarded)
                   
Use case: Testing new ML model or DB migration under real load
          without affecting users. Compare outputs side by side.
```

**Comparison:**

| Technique | Risk | Complexity | User Impact |
|-----------|------|------------|-------------|
| Canary release | Low | Medium | None (if caught early) |
| Feature flag | Very low | Low | None |
| Dark launch | None | Medium | Zero |

---

### Q13. What is load shedding and how do you gracefully drop low-priority traffic under overload?

**Answer:**

Load shedding is the practice of intentionally rejecting or dropping requests when a system is overloaded to protect core functionality and prevent complete failure.

**The alternative — no load shedding:**
```
Without load shedding:
  System overloaded → All requests queued → All requests slow
  → Cascading failure → Complete outage affecting ALL users

With load shedding:
  System overloaded → Low-priority requests rejected (503)
  → High-priority requests still served with normal latency
  → Graceful degradation instead of total failure
```

**Priority classification:**
```
Priority 1 (NEVER shed): Payment processing, core auth
Priority 2 (Shed last): User reads, dashboard loads  
Priority 3 (Shed first): Recommendation engine, analytics
Priority 4 (Always shed when overloaded): Batch jobs, reporting
```

**Implementation approaches:**

```python
# Token bucket for admission control
class AdmissionController:
    def __init__(self, capacity, refill_rate):
        self.tokens = capacity
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens/second
    
    def allow_request(self, priority: int) -> bool:
        cost = {1: 1, 2: 2, 3: 3, 4: 5}[priority]
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        if priority == 1:  # Never reject P1
            return True
        return False  # Shed this request
```

**HTTP-level shedding:**
```nginx
# Nginx: limit request rate
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;
limit_req zone=api burst=20 nodelay;
# Returns 503 when exceeded
```

**CPU-based shedding:**
```python
import psutil

def should_shed(priority: int) -> bool:
    cpu_usage = psutil.cpu_percent(interval=0.1)
    if cpu_usage > 90 and priority >= 3:
        return True  # Shed P3/P4 when CPU > 90%
    if cpu_usage > 95 and priority >= 2:
        return True  # Shed P2/P3/P4 when CPU > 95%
    return False
```

**Graceful 503 response:**
```json
{
  "error": "service_overloaded",
  "message": "System is temporarily at capacity. Retry in 5 seconds.",
  "retry_after": 5
}
```

Include `Retry-After` header so clients implement proper backoff.

---

### Q14. Explain request hedging and how it reduces tail latency. How does Google use it?

**Answer:**

Request hedging (also called "backup requests") is the technique of sending a duplicate request to a second server after a short delay, then using whichever response arrives first and canceling the other.

**The tail latency problem:**
```
P50 latency: 10ms  ← most requests
P95 latency: 50ms
P99 latency: 500ms  ← slow server, GC pause, hot shard
P999 latency: 2000ms

In a microservice making 100 parallel fan-out calls:
P(at least one > P99) = 1 - (0.99)^100 = 63%
→ 63% of top-level requests hit P99 latency even if P99 = 1%!
```

**How hedging works:**
```
t=0ms:   Send request to Server A
         ↓
t=5ms:   (5ms hedge delay — p95 threshold)
         No response yet from A
         → Also send same request to Server B
         ↓
t=8ms:   Server B responds ← Use this response
t=12ms:  Server A responds ← Cancel this (or just ignore)

Result: Effective latency = 8ms instead of potentially 500ms
```

**Implementation:**
```python
import asyncio

async def hedged_request(url, hedge_delay_ms=5):
    """Send hedged requests, return first response."""
    
    async def single_request(delay=0):
        await asyncio.sleep(delay / 1000)
        return await http_client.get(url)
    
    tasks = [
        asyncio.create_task(single_request(0)),         # Primary
        asyncio.create_task(single_request(hedge_delay_ms))  # Hedge
    ]
    
    # Return first completed, cancel others
    done, pending = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    
    return done.pop().result()
```

**Google's use:**
Google's paper "The Tail at Scale" (Jeff Dean, 2013) describes using hedging in Bigtable and GFS. They send duplicate requests after the 95th-percentile expected latency passes. This adds only ~5% extra load but eliminates most tail latency.

**Cost-benefit:**
```
Without hedging: 1% P99 = 500ms tail latency, catastrophic for fan-out
With hedging:    ~5% extra requests, 90% reduction in P99 tail latency
```

Use hedge delay = P95 latency. Higher percentile = fewer extra requests but less benefit.

---

### Q15. How do you set up fair on-call rotations? Explain follow-the-sun and rotation schedules.

**Answer:**

On-call design must balance: complete coverage, fairness, engineer wellbeing, and effective incident response.

**Rotation types:**

```
1. Simple rotation (weekly):
   Week 1: Alice
   Week 2: Bob
   Week 3: Carol
   → Simple but week-long stretches are exhausting

2. Split rotation (weekday/weekend separate):
   Weekday primary: Alice → Bob → Carol (weekly cycle)
   Weekend primary: Dave → Eve (bi-weekly)
   → Reduces burden; different people do weekends

3. Follow-the-sun (global teams):
   08:00–16:00 UTC: APAC team (Singapore/Tokyo)
   16:00–00:00 UTC: EMEA team (London/Amsterdam)
   00:00–08:00 UTC: US team (San Francisco/New York)
   → Each region oncall only during business hours
   → Requires handoff process at each boundary
```

**Follow-the-sun handoff protocol:**
```
14:55 UTC (5 min before handoff):
  Outgoing team (APAC) posts in #incidents:
    "Handoff to EMEA: Active incident P2 on payment-service
     Current status: DB replica lag at 45s, investigating
     Runbook: https://wiki/payment-db-lag
     Next step: Check if lag improves after 15:15 UTC deploy"
     
15:00 UTC:
  EMEA team acknowledges in thread
  APAC team available for 30-min overlap questions
```

**PagerDuty rotation configuration:**
```yaml
schedule:
  name: "Payment Service On-Call"
  time_zone: "UTC"
  layers:
    - name: "Primary"
      rotation_virtual_start: "2025-01-01T00:00:00Z"
      rotation_turn_length_seconds: 604800  # 1 week
      users: [alice, bob, carol, dave]
    - name: "Secondary (backup)"
      rotation_turn_length_seconds: 604800
      users: [eve, frank, grace, henry]
      
escalation_policy:
  - escalate_after: 5_minutes
    target: primary_oncall
  - escalate_after: 15_minutes
    target: secondary_oncall
  - escalate_after: 30_minutes
    target: engineering_manager
```

**Fairness practices:**
- Count interrupts per person per quarter, rebalance if uneven
- Give "oncall recovery day" — day off after a heavy oncall week
- Track overnight pages separately from daytime; reduce overnight load
- New engineers shadow before going primary
- "Oncall debt" — people who joined late get lighter first rotations

---

## Hard (Q16–Q20)

---

### Q16. Design a full SRE observability stack for a microservices platform with burn rate alerting, distributed tracing, and SLO dashboards.

**Answer:**

**Architecture overview:**
```
Services (instrumented)
    │
    ├── Metrics → Prometheus (scrape) → Thanos (long-term) → Grafana
    ├── Logs    → Fluentd/Vector → Elasticsearch/Loki → Kibana/Grafana
    └── Traces  → OpenTelemetry SDK → Jaeger/Tempo → Grafana
    
Alerts: Prometheus Alertmanager → PagerDuty → Slack
SLO tracking: Pyrra / Sloth (SLO-as-code)
```

**Step 1: Instrument services (OpenTelemetry)**
```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
import time

# Metrics
meter = metrics.get_meter("payment-service")
request_counter = meter.create_counter("http_requests_total")
request_duration = meter.create_histogram(
    "http_request_duration_seconds",
    boundaries=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
)

# Middleware
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    labels = {
        "method": request.method,
        "path": request.url.path,
        "status": str(response.status_code)
    }
    request_counter.add(1, labels)
    request_duration.record(duration, labels)
    return response
```

**Step 2: SLO definition with Sloth (Prometheus SLO-as-code)**
```yaml
# payment-service.slo.yaml
version: "prometheus/v1"
service: "payment-service"
labels:
  team: payments
  tier: "1"
slos:
  - name: requests-availability
    objective: 99.9
    description: "99.9% of payment requests should succeed"
    sli:
      events:
        error_query: |
          sum(rate(http_requests_total{service="payment",status=~"5.."}[{{.window}}]))
        total_query: |
          sum(rate(http_requests_total{service="payment"}[{{.window}}]))
    alerting:
      name: PaymentServiceHighErrorRate
      page_alert:
        labels:
          severity: critical
      ticket_alert:
        labels:
          severity: warning
```

**Step 3: Grafana SLO dashboard**
```
Panel 1: SLO compliance (gauge)
  - Current 30d availability
  - Green if > SLO, red if breached

Panel 2: Error budget remaining (bar gauge)
  - Minutes remaining this month
  - Color: green > 50%, yellow 10-50%, red < 10%

Panel 3: Burn rate (time series)
  - 1h burn rate (red line at 14.4x)
  - 6h burn rate (orange line at 6x)

Panel 4: Latency SLO (heatmap)
  - P99 latency vs SLO threshold

Panel 5: Recent incidents (table)
  - Incident, duration, budget consumed, status
```

**Step 4: Multi-window burn rate alerts**
```yaml
groups:
  - name: slo_alerts
    rules:
      - alert: PaymentHighBurnRate_Critical
        expr: |
          (
            sum(rate(http_requests_total{service="payment",status=~"5.."}[1h]))
            / sum(rate(http_requests_total{service="payment"}[1h]))
          ) > 0.0144
          and
          (
            sum(rate(http_requests_total{service="payment",status=~"5.."}[5m]))
            / sum(rate(http_requests_total{service="payment"}[5m]))
          ) > 0.0144
        for: 2m
        annotations:
          summary: "Payment service burning error budget 14.4x — page now"
          runbook: "https://runbooks.internal/payment-high-error-rate"
```

This stack covers all aspects: real-time detection, distributed tracing for diagnosis, and SLO dashboards for trend analysis.

---

### Q17. Design a capacity planning system that forecasts and provisions infrastructure ahead of demand.

**Answer:**

Capacity planning prevents two failure modes: over-provisioning (wasted cost) and under-provisioning (incidents during traffic spikes).

**Four-step framework:**

```
1. DEMAND FORECASTING
   ↓
2. CAPACITY MODELING (demand → resources needed)
   ↓
3. LEAD TIME AWARENESS (how far ahead to provision)
   ↓
4. AUTOMATED PROVISIONING + ALERTING
```

**Step 1: Demand forecasting**
```python
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

def forecast_traffic(historical_rps: pd.DataFrame, days_ahead: int):
    """
    Forecast future RPS using Facebook Prophet (handles seasonality).
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(historical_rps.rename(columns={"timestamp": "ds", "rps": "y"}))
    
    future = model.make_future_dataframe(periods=days_ahead * 24, freq="H")
    forecast = model.predict(future)
    
    # Add safety margin
    forecast["rps_with_buffer"] = forecast["yhat_upper"] * 1.2  # 20% headroom
    return forecast
```

**Step 2: Resource modeling**
```
Capacity Model for API Server:
  - Each server: 2 vCPU, 4GB RAM
  - Max RPS per server at 70% CPU: 500 RPS
  - Target CPU utilization: 60% (30% headroom)
  - Max safe RPS per server: 300 RPS

Given forecasted peak: 12,000 RPS
  Servers needed = ceil(12,000 / 300) = 40 servers
  With N+2 redundancy: 42 servers
  With cross-region: 42 × 2 = 84 servers total
```

**Step 3: Lead time awareness by resource type**
```
Resource               Lead Time    Action
──────────────────────────────────────────────────────
EC2 Spot instance      2-5 min      Autoscaling covers
Reserved instance      1 day        Pre-purchase 30 days out
Physical servers       8-12 weeks   Forecast 3-6 months out
Cross-region capacity  2-4 weeks    Forecast quarterly
Database replicas      30-60 min    Autoscaling covers
CDN edge capacity      Minutes      CDN handles automatically
```

**Step 4: Automated provisioning with Terraform + alerts**
```python
# Capacity alert: trigger when headroom < 30%
class CapacityMonitor:
    def __init__(self, threshold_pct=70):
        self.threshold = threshold_pct
    
    def check_and_provision(self):
        current_rps = get_current_rps()
        current_capacity = get_current_max_rps()
        utilization_pct = (current_rps / current_capacity) * 100
        
        if utilization_pct > self.threshold:
            # Scale out
            new_count = math.ceil(current_rps / (SERVERS_RPS_LIMIT * 0.6))
            trigger_terraform_apply(desired_count=new_count + 2)
            alert_capacity_team(
                f"Scaling from {current_count} to {new_count+2} servers"
            )
```

**Capacity review cadence:**

| Horizon | Cadence | Accuracy | Action |
|---------|---------|----------|--------|
| 1 week | Weekly | ±10% | Autoscaling adjustments |
| 1 month | Monthly | ±20% | Reserved instance purchases |
| 1 quarter | Quarterly | ±35% | Data center capacity planning |
| 1 year | Annually | ±50% | Budget + architecture decisions |

**Special events (traffic spikes):**
```
Event-driven capacity:
  1. Marketing sends "Black Friday" traffic forecast 2 weeks ahead
  2. SRE pre-warms: launch extra instances, warm CDN caches
  3. Enable pre-scaling (don't wait for autoscaling trigger)
  4. Arrange vendor support SLA during event
  5. Freeze deploys during event window
```

---

### Q18. Explain the trade-offs between reliability and performance when adding caches and read replicas.

**Answer:**

Adding infrastructure for performance can simultaneously improve reliability — but introduces new failure modes. Understanding the full trade-off space is critical.

**Cache: Impact on reliability and performance**

```
Without cache:
  Read traffic: [All reads] → Primary DB → Disk I/O
  
With Redis cache:
  Read traffic: [Cache hit ~80%] → Redis → Sub-millisecond
               [Cache miss ~20%] → Primary DB → Disk I/O

Performance gain: P99 latency drops from 100ms to 5ms for cache hits
Reliability gain: DB load reduced by 80% → DB less likely to overload
```

**New failure modes introduced by cache:**

| Failure Mode | Description | Mitigation |
|---|---|---|
| Cache stampede | Cache expires → N requests hit DB simultaneously | Probabilistic early expiry, locking |
| Cache poisoning | Stale/incorrect data served | Short TTLs, cache invalidation on write |
| Redis OOM | Cache runs out of memory | Set maxmemory-policy allkeys-lru |
| Cache unavailable | Redis down → all traffic hits DB | Circuit breaker with cache-aside fallback |
| Thundering herd | Cache restart → cold cache | Gradual warm-up, cache-aside pattern |

**Cache-aside with circuit breaker:**
```python
class CacheLayer:
    def __init__(self):
        self.cache = Redis()
        self.cache_circuit_breaker = CircuitBreaker(threshold=5, timeout=30)
    
    def get(self, key: str, loader_fn):
        # Try cache first
        if self.cache_circuit_breaker.is_closed():
            try:
                value = self.cache.get(key)
                if value:
                    return value
            except RedisException:
                self.cache_circuit_breaker.record_failure()
        
        # Cache miss or circuit open: load from DB
        value = loader_fn()
        
        # Write to cache if circuit closed
        if self.cache_circuit_breaker.is_closed():
            self.cache.setex(key, ttl=300, value=value)
        
        return value
```

**Read replicas: reliability and performance trade-offs**
```
Primary DB: handles all writes + some reads
Read Replica 1: handles read traffic for analytics
Read Replica 2: handles read traffic for user-facing reads
Read Replica 3: hot standby (promoted if primary fails)

Performance gain: Read traffic distributed → primary not overloaded
Reliability gain: Failover to replica in ~30s if primary dies

Replication lag problem:
  Write to primary: user updates profile
  Immediately read from replica: old profile returned (lag = 50ms-2s)
  
Solution: Read-your-writes routing
  - Route writes and immediate reads to primary
  - Route all other reads to replicas
  - OR use synchronous replication (higher latency but consistent)
```

**When adding infrastructure hurts reliability:**
```
Anti-pattern: Adding a distributed cache makes system MORE complex
  Consistency failure: Cache and DB diverge during network partition
  Operational burden: More services = more things that can fail
  Debugging: Cache miss bugs are hard to reproduce

Rule: Only add a caching layer if:
  1. You have proven DB is the bottleneck (measure first)
  2. You can tolerate some staleness
  3. You have cache invalidation logic
  4. You have monitoring + fallback
```

---

### Q19. Design a runbook automation system that converts manual SRE steps into automated remediation.

**Answer:**

Runbook automation converts human-executed troubleshooting steps into code that runs automatically when an alert fires, reducing MTTR from minutes to seconds.

**Architecture:**
```
Alert fires (PagerDuty/Alertmanager)
    │
    ▼
Runbook Automation Engine
    ├── Step 1: Diagnose (query metrics/logs)
    ├── Step 2: Remediate (restart pod/scale/rollback)
    ├── Step 3: Verify (check metrics improved)
    └── Step 4: Notify (Slack + ticket with full audit trail)
    
If automation fails:
    └── Escalate to human oncall with full diagnosis report
```

**Runbook definition format:**
```yaml
# runbooks/high-memory-usage.yaml
name: HighMemoryUsage
trigger:
  alert: PodMemoryUsageHigh
  threshold: "memory_usage > 90%"

steps:
  - id: diagnose
    type: kubectl
    command: "kubectl top pods -n {{ .namespace }} --sort-by=memory"
    timeout: 30s
    
  - id: check_oom_killer
    type: kubectl
    command: "kubectl describe pod {{ .pod_name }} -n {{ .namespace }}"
    parse:
      look_for: "OOMKilled"
      
  - id: remediate_restart
    type: kubectl
    command: "kubectl delete pod {{ .pod_name }} -n {{ .namespace }}"
    condition: "{{ eq .check_oom_killer.found true }}"
    rollback: false
    
  - id: check_memory_leak
    type: prometheus_query
    query: |
      increase(container_memory_usage_bytes{pod="{{ .pod_name }}"}[1h])
    condition: "{{ gt .result 500000000 }}"  # 500MB increase in 1h
    
  - id: scale_out
    type: kubectl
    command: "kubectl scale deployment {{ .deployment }} --replicas={{ add .current_replicas 2 }}"
    condition: "{{ .check_memory_leak.triggered }}"
    max_replicas: 20
    
  - id: verify
    type: prometheus_query
    query: "avg(container_memory_working_set_bytes{deployment='{{ .deployment }}'})"
    wait_for: 120s
    success_threshold: "{{ lt .result 0.80 }}"
    
  - id: notify
    type: slack
    channel: "#incidents"
    message: |
      Runbook `HighMemoryUsage` executed for {{ .pod_name }}
      Steps taken: {{ .executed_steps }}
      Memory now: {{ .verify.result }}%
      Status: {{ if .verify.success }}RESOLVED{{ else }}ESCALATING TO HUMAN{{ end }}
      
on_failure:
  escalate_to: pagerduty
  include_diagnosis: true
```

**Execution engine (Python pseudocode):**
```python
class RunbookEngine:
    async def execute(self, runbook: Runbook, context: dict):
        audit_log = []
        
        for step in runbook.steps:
            # Check condition
            if step.condition and not eval_condition(step.condition, context):
                audit_log.append(f"SKIP {step.id}: condition not met")
                continue
            
            try:
                result = await self.execute_step(step, context)
                context[step.id] = result
                audit_log.append(f"OK   {step.id}: {result.summary}")
                
                # Safety gate: stop if service degraded further
                if await self.check_service_degraded(context):
                    raise SafetyGateTriggered(f"Service worsened at step {step.id}")
                    
            except Exception as e:
                audit_log.append(f"FAIL {step.id}: {e}")
                await self.escalate_to_human(runbook, context, audit_log, e)
                return
        
        # Store complete audit trail for postmortem
        await self.save_execution_record(runbook.name, context, audit_log)
```

**Guardrails for automation safety:**
```
1. Dry-run mode: Log what would happen, don't execute (verify first)
2. Human approval gate: Some steps require Slack approval before executing
   - "Will restart 5 pods — approve? [yes/no]"
3. Blast radius limit: Never auto-scale > 2x current size
4. Time window: Only run automated remediation during business hours
   - Overnight: diagnose only, wake oncall for execution
5. Rate limit: Max 3 automated remediations per service per hour
```

---

### Q20. Explain the full reliability model for a global service: how do you achieve 99.99% availability when each component has 99.9% availability?

**Answer:**

Achieving 99.99% ("four nines" = 52 min/year downtime) from components that are individually 99.9% requires careful redundancy architecture.

**The math of serial vs parallel reliability:**
```
Serial (all components must work):
  System availability = A1 × A2 × A3 × ... × An
  
  Example: 3 components each at 99.9%
  System = 0.999 × 0.999 × 0.999 = 0.997 = 99.7%
  → WORSE than any individual component!

Parallel (any component can serve):
  System availability = 1 - (1-A)^n
  
  Example: 2 parallel components each at 99.9%
  System = 1 - (1 - 0.999)^2 = 1 - 0.000001 = 99.9999%
  → MUCH BETTER than individual!
```

**Achieving 99.99% for a global API:**

```
Global Architecture:
                    ┌─────────────────────────────────────┐
                    │  Anycast DNS / Global Load Balancer  │
                    │    (Route53 health checks)           │
                    └────────┬───────────────┬────────────┘
                             │               │
              ┌──────────────▼──┐     ┌─────▼──────────────┐
              │   Region: US-E  │     │   Region: EU-W      │
              │                 │     │                      │
              │  AZ-1   AZ-2   │     │  AZ-1   AZ-2        │
              │  [API] [API]   │     │  [API] [API]         │
              │  [DB]  [DB]    │     │  [DB]  [DB]          │
              └─────────────────┘     └──────────────────────┘
              
Each AZ: 99.9% availability
Two AZs in parallel: 99.9999%
Two regions: Even higher, latency routing handles failover
```

**Dependency risk analysis:**

```
For each external dependency, calculate impact:
  
Dependency: Single-region PostgreSQL (99.9% = 8.7h/year downtime)
Impact: Complete service outage if DB is down
Risk: Unacceptable for 99.99% target

Mitigation:
  - Multi-AZ RDS: automatic failover in 60-90s
  - Read replicas in multiple AZs
  - Global Aurora: cross-region replication with < 1s lag
  
Dependency: Redis cache (99.9%)
Impact: 80% of requests degrade to DB (3x latency)
Mitigation:
  - Redis Cluster (3 primaries + 3 replicas)
  - Application-level fallback (cache miss = DB hit, not failure)
```

**Error budget allocation for 99.99%:**
```
99.99% availability = 52.6 minutes downtime per year = 4.4 min/month

Budget allocation:
  Planned maintenance: 1 min/month (use blue-green, aim for zero)
  Unplanned incidents: 2 min/month
  Deployment risk: 1 min/month (canary releases)
  Third-party failures: 0.4 min/month
  
Total: 4.4 min/month = 99.99% target
```

**The five reliability pillars for 99.99%:**

| Pillar | Mechanism | Contribution |
|--------|-----------|--------------|
| Redundancy | Multi-AZ, multi-region | Eliminates single points of failure |
| Observability | MTTD < 2 min | Detects failures before users notice |
| Automation | Auto-rollback, auto-scaling | MTTR < 5 min without human |
| Testing | Chaos engineering, load tests | Discovers failures pre-production |
| Progressive delivery | Canary + feature flags | Blast radius < 1% for any deploy |

**Reliability budget as a system design constraint:**
```python
def check_architecture_availability(components: list[float]) -> float:
    """
    Given list of serial components, each with parallel redundancy,
    calculate system availability.
    """
    availability = 1.0
    for single_component_avail, parallel_count in components:
        # Parallel availability
        component_system_avail = 1 - (1 - single_component_avail) ** parallel_count
        # Multiply through for serial chain
        availability *= component_system_avail
    return availability

# Example:
components = [
    (0.999, 3),   # Load balancer: 3 instances
    (0.999, 2),   # App servers: 2 AZs
    (0.999, 2),   # Cache: 2 replicas
    (0.999, 3),   # DB: Multi-AZ with replica
]
# System availability: check_architecture_availability(components)
# Result: ~99.9999% — well above 99.99% target
```

The key insight: design with failure as a default assumption, not an exception. Every component will fail; the system must not.

---

## Quick Reference

### SRE Core Metrics

| Metric | Formula | Target (Tier 1) |
|--------|---------|-----------------|
| Availability SLO | (good_requests / total) × 100 | 99.9% |
| Error Budget | (1 - SLO) × window | > 0 remaining |
| Change Failure Rate | failed_deploys / total_deploys | < 5% |
| MTTD | avg(detect_time - failure_time) | < 2 min |
| MTTR | avg(recover_time - detect_time) | < 15 min |
| Toil percentage | toil_hours / total_hours | < 50% |

### Burn Rate Thresholds

| Alert Level | Window | Burn Rate | Budget Consumed | Action |
|-------------|--------|-----------|-----------------|--------|
| Critical | 1h + 5m | 14.4x | 2% in 1h | Page immediately |
| Warning | 6h + 30m | 6x | 5% in 6h | Create ticket |
| Info | 3d + 6h | 1x | 10% in 3d | Weekly review |

### Downtime Budget Table

| SLO | Annual Downtime | Monthly Downtime |
|-----|----------------|------------------|
| 99% | 87.6 hours | 7.3 hours |
| 99.9% | 8.76 hours | 43.8 minutes |
| 99.95% | 4.38 hours | 21.9 minutes |
| 99.99% | 52.6 minutes | 4.4 minutes |
| 99.999% | 5.26 minutes | 26 seconds |

### Four Golden Signals vs RED vs USE

| Signal | Four Golden Signals | RED | USE |
|--------|--------------------|----|-----|
| Request rate | Traffic | Rate | — |
| Errors | Errors | Errors | Errors |
| Latency | Latency | Duration | — |
| Resource saturation | Saturation | — | Utilization + Saturation |

### Progressive Delivery Comparison

| Method | Blast Radius | Rollback Speed | Complexity |
|--------|-------------|----------------|------------|
| Full deploy | 100% | Redeploy (min) | Low |
| Blue-green | 100% → 0% | DNS switch (sec) | Medium |
| Canary | 1-10% | Weight to 0% (sec) | Medium |
| Feature flag | 1-100% configurable | Flag flip (ms) | Low |
| Dark launch | 0% (shadow) | Just stop shadowing | High |
