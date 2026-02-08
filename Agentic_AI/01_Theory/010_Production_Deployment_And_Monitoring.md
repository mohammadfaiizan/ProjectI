# Production Deployment and Monitoring

## Table of Contents

1. From Prototype to Production
2. Architecture Patterns
3. Containerization and Packaging
4. Scaling Strategies
5. Cost Optimization
6. Latency Management
7. Observability and Logging
8. Tracing Agent Execution
9. Monitoring and Alerting
10. CI/CD for Agents
11. LLMOps
12. Disaster Recovery

---

## 1. From Prototype to Production

### The Production Gap

| Aspect | Prototype | Production |
|--------|-----------|------------|
| Reliability | "Works on my machine" | 99.9% uptime SLA |
| Scale | 1-10 users | 1,000-100,000+ users |
| Cost | Unlimited budget | Cost per request matters |
| Latency | Seconds acceptable | P95 < 5 seconds |
| Security | Basic/none | Defense in depth |
| Monitoring | Print statements | Full observability stack |
| Error handling | Crash and restart | Graceful degradation |
| Testing | Manual | Automated CI/CD pipeline |
| Data | Test data | Real user data, PII |
| Compliance | Not considered | GDPR, SOC2, HIPAA |

### Production Readiness Checklist

```
[ ] Error handling for all failure modes
[ ] Rate limiting on API calls
[ ] Retry logic with exponential backoff
[ ] Input validation and sanitization
[ ] Output validation and content filtering
[ ] PII detection and redaction
[ ] Cost tracking and budget alerts
[ ] Logging and tracing infrastructure
[ ] Health check endpoints
[ ] Kill switch mechanism
[ ] Backup model/provider fallback
[ ] Load testing completed
[ ] Security review completed
[ ] Compliance review completed
[ ] Runbook documentation
[ ] Incident response plan
```

---

## 2. Architecture Patterns

### Single-Agent Service

```
+--------+     +---------+     +----------+     +---------+
| Client | --> | API     | --> | Agent    | --> | LLM     |
|        |     | Gateway |     | Service  |     | Provider|
+--------+     +---------+     +----+-----+     +---------+
                                    |
                              +-----+-----+
                              |           |
                         +----v---+  +----v---+
                         | Tool   |  | Memory |
                         | Service|  | Store  |
                         +--------+  +--------+
```

### Multi-Agent Service

```
+--------+     +---------+     +-------------+
| Client | --> | API     | --> | Orchestrator|
|        |     | Gateway |     | Service     |
+--------+     +---------+     +------+------+
                                      |
                         +------------+------------+
                         |            |            |
                   +-----v---+  +----v----+  +----v----+
                   | Agent A |  | Agent B |  | Agent C |
                   | Service |  | Service |  | Service |
                   +---------+  +---------+  +---------+
                         |            |            |
                   +-----v---+  +----v----+  +----v----+
                   | Queue   |  | Cache   |  | DB      |
                   +---------+  +---------+  +---------+
```

### Microservice-Based Agent Architecture

```python
# Agent Service (FastAPI)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid

app = FastAPI()

class Agent_Request(BaseModel):
    user_id: str
    session_id: str = None
    message: str
    context: dict = {}

class Agent_Response(BaseModel):
    session_id: str
    response: str
    tool_calls: list = []
    latency_ms: float
    tokens_used: int
    cost: float

@app.post("/agent/chat", response_model=Agent_Response)
async def Chat(request: Agent_Request, background_tasks: BackgroundTasks):
    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.time()

    try:
        # Load session state
        state = await State_Store.Load(session_id)

        # Run agent
        result = await Agent.Run(
            message=request.message,
            state=state,
            context=request.context,
        )

        # Save updated state
        await State_Store.Save(session_id, result.state)

        latency_ms = (time.time() - start_time) * 1000

        # Log async
        background_tasks.add_task(
            Log_Interaction, request, result, latency_ms
        )

        return Agent_Response(
            session_id=session_id,
            response=result.response,
            tool_calls=result.tool_calls,
            latency_ms=latency_ms,
            tokens_used=result.tokens_used,
            cost=result.cost,
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        background_tasks.add_task(Log_Error, request, str(e), latency_ms)
        raise HTTPException(status_code=500, detail="Agent processing failed")


@app.get("/health")
async def Health_Check():
    checks = {
        "llm_provider": await Check_LLM_Provider(),
        "state_store": await Check_State_Store(),
        "vector_db": await Check_Vector_DB(),
    }
    healthy = all(v["status"] == "ok" for v in checks.values())
    return {"status": "healthy" if healthy else "degraded", "checks": checks}
```

---

## 3. Containerization and Packaging

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose (Development)

```yaml
version: "3.8"

services:
  agent-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/agentdb
    depends_on:
      - redis
      - postgres
      - chromadb

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: agentdb
    volumes:
      - postgres-data:/var/lib/postgresql/data

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma-data:/chroma/chroma

volumes:
  redis-data:
  postgres-data:
  chroma-data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-service
  template:
    metadata:
      labels:
        app: agent-service
    spec:
      containers:
        - name: agent
          image: myregistry/agent-service:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: openai-api-key
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent-service
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 4. Scaling Strategies

### Horizontal Scaling

```python
class Agent_Pool:
    def __init__(self, agent_factory, min_agents=2, max_agents=10):
        self.factory = agent_factory
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.agents = [self.factory.Create() for _ in range(min_agents)]
        self.queue = []

    def Get_Available_Agent(self):
        for agent in self.agents:
            if not agent.is_busy:
                agent.is_busy = True
                return agent

        if len(self.agents) < self.max_agents:
            agent = self.factory.Create()
            agent.is_busy = True
            self.agents.append(agent)
            return agent

        return None  # All busy, need to queue

    def Release_Agent(self, agent):
        agent.is_busy = False
        # Process queued requests
        if self.queue:
            request = self.queue.pop(0)
            self.Process(request)

    def Scale_Down(self):
        idle_agents = [a for a in self.agents if not a.is_busy]
        while len(idle_agents) > self.min_agents and len(self.agents) > self.min_agents:
            agent = idle_agents.pop()
            self.agents.remove(agent)
```

### Request Queuing

```python
import asyncio
from collections import deque

class Request_Queue:
    def __init__(self, max_concurrent=10, max_queue_size=100):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = deque(maxlen=max_queue_size)
        self.processing = 0

    async def Submit(self, task_fn, *args, **kwargs):
        if self.processing >= self.semaphore._value and len(self.queue) >= self.queue.maxlen:
            raise Exception("Queue full, try again later")

        async with self.semaphore:
            self.processing += 1
            try:
                result = await task_fn(*args, **kwargs)
                return result
            finally:
                self.processing -= 1

    def Get_Stats(self):
        return {
            "processing": self.processing,
            "queued": len(self.queue),
            "available_slots": self.semaphore._value - self.processing,
        }
```

### Caching Layer

```python
import hashlib

class Agent_Cache:
    def __init__(self, redis_client, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def Get_Cache_Key(self, message, context=None):
        content = f"{message}:{json.dumps(context or {}, sort_keys=True)}"
        return f"agent:cache:{hashlib.sha256(content.encode()).hexdigest()}"

    async def Get(self, message, context=None):
        key = self.Get_Cache_Key(message, context)
        cached = self.redis.get(key)

        if cached:
            self.hits += 1
            return json.loads(cached)

        self.misses += 1
        return None

    async def Set(self, message, response, context=None):
        key = self.Get_Cache_Key(message, context)
        self.redis.setex(key, self.ttl, json.dumps(response))

    def Get_Hit_Rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
```

---

## 5. Cost Optimization

### Cost Tracking

```python
class Cost_Tracker:
    MODEL_PRICING = {
        "gpt-4": {"input": 30.0 / 1_000_000, "output": 60.0 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.0 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "claude-sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
        "claude-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    }

    def __init__(self, budget_limit=100.0):
        self.total_cost = 0.0
        self.budget_limit = budget_limit
        self.cost_log = []

    def Calculate_Cost(self, model, input_tokens, output_tokens):
        pricing = self.MODEL_PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

        self.total_cost += cost
        self.cost_log.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "cumulative": self.total_cost,
            "timestamp": datetime.now().isoformat(),
        })

        if self.total_cost > self.budget_limit * 0.8:
            self.Alert_Budget_Warning()

        return cost

    def Alert_Budget_Warning(self):
        usage_pct = (self.total_cost / self.budget_limit) * 100
        print(f"[BUDGET WARNING] {usage_pct:.1f}% of budget used (${self.total_cost:.2f}/${self.budget_limit:.2f})")

    def Get_Daily_Report(self):
        today = datetime.now().date().isoformat()
        today_entries = [
            e for e in self.cost_log if e["timestamp"].startswith(today)
        ]
        return {
            "date": today,
            "total_cost": sum(e["cost"] for e in today_entries),
            "total_calls": len(today_entries),
            "by_model": self.Group_By_Model(today_entries),
        }

    def Group_By_Model(self, entries):
        groups = {}
        for e in entries:
            model = e["model"]
            if model not in groups:
                groups[model] = {"calls": 0, "cost": 0, "tokens": 0}
            groups[model]["calls"] += 1
            groups[model]["cost"] += e["cost"]
            groups[model]["tokens"] += e["input_tokens"] + e["output_tokens"]
        return groups
```

### Cost Optimization Strategies

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| Model tiering | 50-90% | Use cheaper models for simple tasks |
| Response caching | 20-40% | Cache frequent queries |
| Prompt optimization | 10-30% | Shorter prompts, fewer examples |
| Batch processing | 15-25% | Group similar requests |
| Early stopping | 10-20% | Stop when quality threshold met |
| Token budgeting | 5-15% | Limit max tokens per request |

### Model Router

```python
class Model_Router:
    def __init__(self, models):
        self.models = models

    def Select_Model(self, task_complexity, budget_remaining):
        if task_complexity == "simple":
            return self.Get_Cheapest_Model()
        elif task_complexity == "moderate":
            return self.Get_Mid_Tier_Model()
        elif task_complexity == "complex":
            if budget_remaining > 1.0:
                return self.Get_Best_Model()
            else:
                return self.Get_Mid_Tier_Model()
        return self.Get_Mid_Tier_Model()

    def Classify_Complexity(self, message, llm):
        response = llm.generate(f"""
        Classify complexity: simple, moderate, or complex.
        Message: {message}
        Return one word only.
        """)
        return response.strip().lower()

    def Get_Cheapest_Model(self):
        return min(self.models, key=lambda m: m["cost_per_1k"])

    def Get_Best_Model(self):
        return max(self.models, key=lambda m: m["quality_score"])

    def Get_Mid_Tier_Model(self):
        sorted_models = sorted(self.models, key=lambda m: m["cost_per_1k"])
        return sorted_models[len(sorted_models) // 2]
```

---

## 6. Latency Management

### Latency Breakdown

```
Typical Agent Request Latency:
+-----------------------------------------------------------+
| Network to API Gateway:          10-50ms                   |
| Input validation/guardrails:     5-20ms                    |
| State loading (Redis):           2-10ms                    |
| RAG retrieval (vector search):   50-200ms                  |
| LLM call #1 (planning):         500-3000ms                |
| Tool execution:                  100-5000ms                |
| LLM call #2 (response):         500-3000ms                |
| Output validation:               5-20ms                    |
| State saving:                    2-10ms                    |
| Network back to client:          10-50ms                   |
+-----------------------------------------------------------+
| TOTAL:                           1,200-11,300ms            |
+-----------------------------------------------------------+
```

### Latency Optimization Techniques

```python
class Latency_Optimizer:
    @staticmethod
    async def Parallel_Retrieval(queries, vector_store):
        """Run multiple retrievals in parallel."""
        tasks = [vector_store.Search_Async(q) for q in queries]
        return await asyncio.gather(*tasks)

    @staticmethod
    def Streaming_Response(llm, messages, stream_callback):
        """Stream response tokens to client as they are generated."""
        stream = llm.create(messages=messages, stream=True)
        full_response = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_response += token
            stream_callback(token)
        return full_response

    @staticmethod
    def Speculative_Execution(agent, message, likely_tools):
        """Pre-fetch tool results for likely tool calls."""
        import concurrent.futures

        # Start tool pre-fetching in background
        with concurrent.futures.ThreadPoolExecutor() as executor:
            prefetch_futures = {}
            for tool, default_params in likely_tools:
                future = executor.submit(tool.Execute, **default_params)
                prefetch_futures[tool.name] = future

            # Run LLM call
            llm_result = agent.llm.generate(message)

            # If LLM chose a pre-fetched tool, use cached result
            if llm_result.tool_call and llm_result.tool_call.name in prefetch_futures:
                return prefetch_futures[llm_result.tool_call.name].result()

        return None
```

### Streaming Implementation

```python
from fastapi.responses import StreamingResponse

@app.post("/agent/stream")
async def Stream_Chat(request: Agent_Request):
    async def Generate():
        async for chunk in Agent.Stream(
            message=request.message,
            session_id=request.session_id,
        ):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        Generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
```

---

## 7. Observability and Logging

### Structured Logging

```python
import logging
import json

class Agent_Logger:
    def __init__(self, service_name="agent-service"):
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    def Log(self, level, event, **kwargs):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            **kwargs,
        }
        getattr(self.logger, level)(json.dumps(entry))

    def Log_Request(self, request_id, user_id, message):
        self.Log("info", "request_received",
                 request_id=request_id, user_id=user_id,
                 message_length=len(message))

    def Log_LLM_Call(self, request_id, model, input_tokens, output_tokens, latency_ms, cost):
        self.Log("info", "llm_call",
                 request_id=request_id, model=model,
                 input_tokens=input_tokens, output_tokens=output_tokens,
                 latency_ms=latency_ms, cost=cost)

    def Log_Tool_Call(self, request_id, tool_name, success, latency_ms):
        self.Log("info", "tool_call",
                 request_id=request_id, tool=tool_name,
                 success=success, latency_ms=latency_ms)

    def Log_Error(self, request_id, error_type, message, stack_trace=None):
        self.Log("error", "agent_error",
                 request_id=request_id, error_type=error_type,
                 message=message, stack_trace=stack_trace)

    def Log_Guardrail(self, request_id, guard_name, triggered, reason=None):
        self.Log("warning" if triggered else "info", "guardrail",
                 request_id=request_id, guard=guard_name,
                 triggered=triggered, reason=reason)
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics definitions
REQUEST_COUNT = Counter(
    "agent_requests_total", "Total agent requests",
    ["status", "model"]
)

REQUEST_LATENCY = Histogram(
    "agent_request_duration_seconds", "Request latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

LLM_CALL_LATENCY = Histogram(
    "agent_llm_call_duration_seconds", "LLM call latency",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
)

TOOL_CALL_COUNT = Counter(
    "agent_tool_calls_total", "Tool call count",
    ["tool_name", "status"]
)

ACTIVE_SESSIONS = Gauge(
    "agent_active_sessions", "Number of active sessions"
)

TOKEN_USAGE = Counter(
    "agent_tokens_total", "Total tokens used",
    ["model", "direction"]  # direction: input/output
)

COST_TOTAL = Counter(
    "agent_cost_dollars_total", "Total cost in dollars",
    ["model"]
)

class Metrics_Collector:
    @staticmethod
    def Record_Request(status, model, duration):
        REQUEST_COUNT.labels(status=status, model=model).inc()
        REQUEST_LATENCY.observe(duration)

    @staticmethod
    def Record_LLM_Call(model, duration, input_tokens, output_tokens, cost):
        LLM_CALL_LATENCY.labels(model=model).observe(duration)
        TOKEN_USAGE.labels(model=model, direction="input").inc(input_tokens)
        TOKEN_USAGE.labels(model=model, direction="output").inc(output_tokens)
        COST_TOTAL.labels(model=model).inc(cost)

    @staticmethod
    def Record_Tool_Call(tool_name, success):
        TOOL_CALL_COUNT.labels(
            tool_name=tool_name,
            status="success" if success else "failure"
        ).inc()
```

---

## 8. Tracing Agent Execution

### Distributed Tracing

```python
import uuid

class Trace_Context:
    def __init__(self, trace_id=None, parent_span_id=None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.spans = []

    def Start_Span(self, name, metadata=None):
        span = {
            "span_id": str(uuid.uuid4()),
            "parent_span_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "name": name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "metadata": metadata or {},
            "events": [],
        }
        self.spans.append(span)
        return span

    def End_Span(self, span, status="ok", error=None):
        span["end_time"] = datetime.now().isoformat()
        span["status"] = status
        if error:
            span["error"] = error

    def Add_Event(self, span, event_name, data=None):
        span["events"].append({
            "name": event_name,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        })

    def Get_Full_Trace(self):
        return {
            "trace_id": self.trace_id,
            "spans": self.spans,
            "total_spans": len(self.spans),
        }


class Traced_Agent:
    def __init__(self, agent, logger):
        self.agent = agent
        self.logger = logger

    async def Run(self, message, session_id):
        trace = Trace_Context()

        # Root span
        root = trace.Start_Span("agent_request", {"message": message[:100]})

        try:
            # Input validation span
            val_span = trace.Start_Span("input_validation")
            validated = self.Validate_Input(message)
            trace.End_Span(val_span)

            # State loading span
            state_span = trace.Start_Span("load_state", {"session_id": session_id})
            state = await self.Load_State(session_id)
            trace.End_Span(state_span)

            # LLM reasoning span
            llm_span = trace.Start_Span("llm_reasoning", {"model": self.agent.model})
            response = await self.agent.Think(message, state)
            trace.Add_Event(llm_span, "tokens_used", {
                "input": response.input_tokens,
                "output": response.output_tokens,
            })
            trace.End_Span(llm_span)

            # Tool execution spans
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_span = trace.Start_Span("tool_execution", {
                        "tool": tool_call.name,
                    })
                    result = await self.Execute_Tool(tool_call)
                    trace.End_Span(tool_span, status="ok" if result.success else "error")

            # Output validation span
            out_span = trace.Start_Span("output_validation")
            final = self.Validate_Output(response.text)
            trace.End_Span(out_span)

            trace.End_Span(root, status="ok")

        except Exception as e:
            trace.End_Span(root, status="error", error=str(e))
            raise

        finally:
            self.logger.Log("info", "trace_complete", trace=trace.Get_Full_Trace())

        return final
```

### Trace Visualization

```
Trace: abc-123-def
|
+-- agent_request (0ms - 3500ms) [OK]
    |
    +-- input_validation (5ms - 15ms) [OK]
    |
    +-- load_state (16ms - 25ms) [OK]
    |
    +-- llm_reasoning (26ms - 2100ms) [OK]
    |   tokens: input=450, output=120
    |
    +-- tool_execution: search_web (2101ms - 2800ms) [OK]
    |
    +-- tool_execution: summarize (2801ms - 3200ms) [OK]
    |
    +-- llm_reasoning (3201ms - 3450ms) [OK]
    |   tokens: input=800, output=200
    |
    +-- output_validation (3451ms - 3460ms) [OK]
```

---

## 9. Monitoring and Alerting

### Dashboard Metrics

| Category | Metric | Normal Range | Alert Threshold |
|----------|--------|-------------|-----------------|
| Availability | Uptime % | > 99.9% | < 99.5% |
| Performance | P50 latency | < 2s | > 5s |
| Performance | P95 latency | < 5s | > 15s |
| Performance | P99 latency | < 10s | > 30s |
| Quality | Task completion rate | > 90% | < 80% |
| Quality | Hallucination rate | < 5% | > 10% |
| Cost | Daily spend | Budget/30 | > 150% daily budget |
| Cost | Cost per request | < $0.10 | > $0.50 |
| Volume | Requests per minute | Varies | > 2x normal |
| Errors | Error rate | < 2% | > 5% |
| Safety | Guardrails triggered | < 5% | > 15% |
| Resources | CPU usage | < 70% | > 90% |
| Resources | Memory usage | < 70% | > 85% |

### Alert Configuration

```python
class Alert_Manager:
    def __init__(self, notification_channels):
        self.channels = notification_channels
        self.rules = []
        self.cooldowns = {}  # Prevent alert spam

    def Add_Rule(self, name, metric, condition, severity, cooldown_minutes=15):
        self.rules.append({
            "name": name,
            "metric": metric,
            "condition": condition,
            "severity": severity,
            "cooldown_minutes": cooldown_minutes,
        })

    def Check_All(self, current_metrics):
        alerts = []
        for rule in self.rules:
            metric_value = current_metrics.get(rule["metric"])
            if metric_value is not None and rule["condition"](metric_value):
                if self.Should_Alert(rule["name"], rule["cooldown_minutes"]):
                    alert = {
                        "rule": rule["name"],
                        "metric": rule["metric"],
                        "value": metric_value,
                        "severity": rule["severity"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    alerts.append(alert)
                    self.Send_Alert(alert)
                    self.cooldowns[rule["name"]] = datetime.now()

        return alerts

    def Should_Alert(self, rule_name, cooldown_minutes):
        if rule_name not in self.cooldowns:
            return True
        elapsed = (datetime.now() - self.cooldowns[rule_name]).total_seconds() / 60
        return elapsed >= cooldown_minutes

    def Send_Alert(self, alert):
        for channel in self.channels:
            channel.Send(alert)

# Setup
alerts = Alert_Manager(notification_channels=[slack_channel, email_channel])

alerts.Add_Rule("high_latency", "p95_latency_ms", lambda v: v > 15000, "warning")
alerts.Add_Rule("error_spike", "error_rate", lambda v: v > 0.05, "critical")
alerts.Add_Rule("cost_overrun", "daily_cost", lambda v: v > 50.0, "warning")
alerts.Add_Rule("low_completion", "completion_rate", lambda v: v < 0.8, "critical")
```

---

## 10. CI/CD for Agents

### Agent CI/CD Pipeline

```
+--------+     +-------+     +--------+     +---------+     +--------+
| Code   | --> | Build | --> | Test   | --> | Staging | --> | Prod   |
| Change |     | & Lint|     | Suite  |     | Deploy  |     | Deploy |
+--------+     +-------+     +--------+     +---------+     +--------+
                                |
                    +-----------+-----------+
                    |           |           |
               +----v---+ +----v---+ +----v----+
               | Unit   | | Eval  | | Safety  |
               | Tests  | | Suite | | Tests   |
               +--------+ +-------+ +---------+
```

### GitHub Actions Pipeline

```yaml
name: Agent CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Lint
        run: ruff check .

      - name: Unit tests
        run: pytest tests/unit/ -v

      - name: Agent evaluation suite
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/evaluation/ -v --tb=short

      - name: Safety tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/safety/ -v

  deploy-staging:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying to staging..."
          # kubectl apply -f k8s/staging/

  deploy-production:
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # kubectl apply -f k8s/production/
```

### Prompt Version Management

```python
class Prompt_Manager:
    def __init__(self, storage):
        self.storage = storage

    def Save_Version(self, prompt_name, content, metadata=None):
        version = self.Get_Latest_Version(prompt_name) + 1
        self.storage.Store({
            "name": prompt_name,
            "version": version,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "active": False,
        })
        return version

    def Activate_Version(self, prompt_name, version):
        # Deactivate current active version
        current = self.Get_Active(prompt_name)
        if current:
            current["active"] = False
            self.storage.Update(current)

        # Activate new version
        target = self.storage.Get(prompt_name, version)
        target["active"] = True
        self.storage.Update(target)

    def Get_Active(self, prompt_name):
        return self.storage.Query({"name": prompt_name, "active": True})

    def Rollback(self, prompt_name):
        current = self.Get_Active(prompt_name)
        if current and current["version"] > 1:
            self.Activate_Version(prompt_name, current["version"] - 1)

    def Get_Latest_Version(self, prompt_name):
        versions = self.storage.Query({"name": prompt_name})
        if not versions:
            return 0
        return max(v["version"] for v in versions)
```

---

## 11. LLMOps

### LLMOps Lifecycle

```
+-------------------------------------------------------------------+
|                       LLMOps LIFECYCLE                            |
|                                                                   |
|  +----------+   +----------+   +---------+   +----------+        |
|  | Develop  |-->| Evaluate |-->| Deploy  |-->| Monitor  |        |
|  | - Prompts|   | - Quality|   | - CI/CD |   | - Metrics|        |
|  | - Tools  |   | - Safety |   | - Blue/ |   | - Costs  |        |
|  | - Memory |   | - Cost   |   |   Green |   | - Quality|        |
|  +----------+   +----------+   +---------+   +-----+----+        |
|       ^                                            |              |
|       |              +----------+                  |              |
|       +--------------| Iterate  |<-----------------+              |
|                      | - Prompt |                                 |
|                      |   tuning |                                 |
|                      | - Model  |                                 |
|                      |   switch |                                 |
|                      +----------+                                 |
+-------------------------------------------------------------------+
```

### A/B Testing Agents

```python
class AB_Test_Manager:
    def __init__(self):
        self.experiments = {}

    def Create_Experiment(self, name, variants, traffic_split):
        self.experiments[name] = {
            "variants": variants,
            "traffic_split": traffic_split,
            "results": {v: {"requests": 0, "successes": 0, "total_latency": 0, "total_cost": 0} for v in variants},
            "active": True,
        }

    def Route_Request(self, experiment_name, request_id):
        import random
        exp = self.experiments[experiment_name]
        rand = random.random()

        cumulative = 0
        for variant, split in zip(exp["variants"], exp["traffic_split"]):
            cumulative += split
            if rand <= cumulative:
                return variant

        return exp["variants"][-1]

    def Record_Result(self, experiment_name, variant, success, latency, cost):
        results = self.experiments[experiment_name]["results"][variant]
        results["requests"] += 1
        if success:
            results["successes"] += 1
        results["total_latency"] += latency
        results["total_cost"] += cost

    def Get_Results(self, experiment_name):
        exp = self.experiments[experiment_name]
        summary = {}

        for variant, results in exp["results"].items():
            n = results["requests"]
            if n == 0:
                continue
            summary[variant] = {
                "requests": n,
                "success_rate": results["successes"] / n,
                "avg_latency": results["total_latency"] / n,
                "avg_cost": results["total_cost"] / n,
            }

        return summary
```

---

## 12. Disaster Recovery

### Failure Modes and Mitigations

| Failure Mode | Impact | Mitigation |
|-------------|--------|------------|
| LLM provider outage | Agent cannot respond | Model fallback chain |
| Database failure | No state/memory access | Read replicas, local cache |
| Vector DB failure | No RAG retrieval | Fallback to direct LLM |
| High latency | Poor user experience | Timeout + cached fallback |
| Cost spike | Budget overrun | Rate limiting, kill switch |
| Data corruption | Bad agent behavior | Checkpoints, rollback |
| Security breach | Data exposure | Kill switch, incident response |

### Model Fallback Chain

```python
class Model_Fallback_Chain:
    def __init__(self, models):
        self.models = models  # Ordered by preference

    async def Call(self, messages, **kwargs):
        last_error = None

        for model in self.models:
            try:
                response = await model.Generate(messages, **kwargs)
                return response
            except Exception as e:
                last_error = e
                print(f"Model {model.name} failed: {e}, trying next...")
                continue

        raise Exception(f"All models failed. Last error: {last_error}")

# Usage
fallback = Model_Fallback_Chain([
    Model("gpt-4o", provider="openai"),
    Model("claude-sonnet", provider="anthropic"),
    Model("gpt-4o-mini", provider="openai"),
    Model("llama-3-70b", provider="together"),
])
```

### Graceful Degradation

```python
class Degradation_Manager:
    def __init__(self):
        self.service_status = {
            "llm": "healthy",
            "vector_db": "healthy",
            "tools": "healthy",
            "memory": "healthy",
        }

    def Update_Status(self, service, status):
        self.service_status[service] = status

    def Get_Available_Features(self):
        features = {
            "full_agent": all(s == "healthy" for s in self.service_status.values()),
            "basic_chat": self.service_status["llm"] == "healthy",
            "rag_enabled": self.service_status["vector_db"] == "healthy",
            "tools_enabled": self.service_status["tools"] == "healthy",
            "memory_enabled": self.service_status["memory"] == "healthy",
        }
        return features

    def Get_Degraded_Response(self, request):
        features = self.Get_Available_Features()

        if features["full_agent"]:
            return None  # No degradation

        if not features["basic_chat"]:
            return "Our AI service is temporarily unavailable. Please try again later."

        if not features["rag_enabled"]:
            return "I can help, but my knowledge base is temporarily unavailable. My answers may be less accurate."

        if not features["tools_enabled"]:
            return "I can discuss topics but cannot perform actions (search, calculations) right now."

        return None
```

---

## Summary

Deploying AI agents to production requires treating them as serious software systems. Key principles:

1. **Architecture first**: Design for scalability, reliability, and maintainability from the start
2. **Containerize everything**: Use Docker and Kubernetes for consistent deployments
3. **Scale horizontally**: Add more agent instances, not bigger instances
4. **Optimize costs relentlessly**: Model tiering, caching, and prompt optimization compound savings
5. **Measure everything**: Request latency, token usage, cost, quality, and errors
6. **Trace every step**: Full distributed tracing through the agent execution pipeline
7. **Alert proactively**: Set thresholds and be notified before users notice problems
8. **Automate deployment**: CI/CD pipeline with automated testing gates
9. **Plan for failure**: Fallback chains, graceful degradation, and kill switches
10. **Iterate continuously**: A/B test prompts, models, and architectures based on production data
