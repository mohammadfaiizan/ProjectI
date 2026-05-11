# Cloud Architecture Patterns — Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. What is the shared responsibility model in cloud computing?

The **shared responsibility model** defines which security and operational responsibilities belong to the cloud provider and which belong to the customer. Misunderstanding this model is one of the most common sources of cloud security incidents.

```
Cloud Provider Responsibilities ("security OF the cloud"):
  ├── Physical datacenter security (locks, guards, cameras)
  ├── Hardware (servers, networking equipment)
  ├── Hypervisor / virtualization layer
  ├── Managed service availability (RDS, S3, Lambda uptime)
  └── Global network infrastructure

Customer Responsibilities ("security IN the cloud"):
  ├── Data (classification, encryption at rest and in transit)
  ├── Identity & Access Management (IAM policies, MFA)
  ├── Application code and configuration
  ├── Operating system (for IaaS: patching, hardening)
  ├── Network configuration (VPCs, security groups, NACLs)
  └── Compliance (GDPR, HIPAA — customer must configure appropriately)
```

**Responsibility shifts by service type:**

| Area                | IaaS (EC2)         | PaaS (Elastic Beanstalk) | SaaS (Salesforce)  |
|---------------------|--------------------|--------------------------|--------------------|
| Physical hardware   | Provider           | Provider                 | Provider           |
| Hypervisor          | Provider           | Provider                 | Provider           |
| OS patches          | **Customer**       | Provider                 | Provider           |
| Runtime/middleware  | **Customer**       | Provider                 | Provider           |
| Application code    | **Customer**       | **Customer**             | Provider           |
| Data & encryption   | **Customer**       | **Customer**             | **Customer**       |
| IAM / access        | **Customer**       | **Customer**             | **Customer**       |

The most common mistake: assuming the cloud provider handles data encryption or access control. These are always the customer's responsibility regardless of service type. An S3 bucket left publicly accessible is a customer misconfiguration, not a provider failure.

---

### Q2. What is the difference between IaaS, PaaS, and SaaS, and when should you use each?

**IaaS (Infrastructure as a Service):** Cloud provides virtualized hardware — VMs, storage, networking. Customer manages everything from the OS upward.

**PaaS (Platform as a Service):** Cloud provides the runtime environment. Customer deploys application code; provider manages OS, runtime, scaling infrastructure.

**SaaS (Software as a Service):** Fully managed software delivered over the internet. Customer uses the software with configuration only.

```
Abstraction Stack:
  
  Application  ← SaaS manages everything here and below
  Data         
  Runtime      ← PaaS manages everything here and below
  Middleware   
  OS           ← IaaS manages everything here and below
  Hypervisor   
  Hardware     
```

**Examples and when to use:**

| Service Type | AWS Example            | Use Case                                       |
|--------------|------------------------|------------------------------------------------|
| IaaS         | EC2 instances          | Custom OS requirements, specific kernel config |
| IaaS         | EBS volumes            | Block storage for VMs                          |
| PaaS         | Elastic Beanstalk      | Web apps without wanting to manage servers     |
| PaaS         | RDS (managed DB)       | Production database without DBA overhead       |
| PaaS         | Lambda                 | Event-driven functions, no server management   |
| SaaS         | Snowflake              | Data warehouse — just use SQL                  |
| SaaS         | GitHub                 | Source control — no infrastructure to manage   |

**Decision guide:**
- Use **IaaS** when you need specific OS control, custom networking, or existing licenses
- Use **PaaS** for standard web/API workloads where managed services reduce operational burden
- Use **SaaS** for business tools (email, CRM, monitoring) — build what differentiates, buy commodities

---

### Q3. What is auto-scaling and what is the difference between horizontal and vertical scaling?

**Auto-scaling** automatically adjusts the number or size of compute resources in response to demand, ensuring performance at low load and cost efficiency at high load.

**Horizontal scaling (scale out/in):** Add or remove instances. Works for stateless services.
```
Normal load:   [App1] [App2]           (2 instances)
High load:     [App1] [App2] [App3] [App4]  (4 instances auto-added)
Low load:      [App1]                  (scale in to 1)

Kubernetes HPA example:
  When CPU > 70%: add pod (up to maxReplicas=20)
  When CPU < 30%: remove pod (down to minReplicas=2)
```

**Vertical scaling (scale up/down):** Resize the instance to a larger or smaller type.
```
Before: db.t3.medium (2 vCPU, 4GB RAM)
After:  db.r6g.large (2 vCPU, 16GB RAM) ← scale up RAM for large dataset
```

**Comparison:**

| Dimension         | Horizontal (Scale Out)           | Vertical (Scale Up)           |
|-------------------|----------------------------------|-------------------------------|
| Limit             | Theoretically unlimited          | Largest instance type         |
| Downtime required | None (add instances behind LB)   | Usually brief restart         |
| Application design| Must be stateless                | Any app (even stateful)       |
| Cost at scale     | Linear with instances            | Exponential (large = costly)  |
| Failure blast radius | One instance = small impact  | One large instance = big      |
| Best for          | App servers, microservices       | Databases, single-threaded    |

**AWS auto-scaling configuration:**
```
Auto Scaling Group:
  min_size = 2        # Always keep 2 for HA
  desired = 4         # Normal traffic
  max_size = 20       # Cap for cost control
  
Scale-out policy: Add 2 instances when CPU > 70% for 3 minutes
Scale-in policy:  Remove 1 instance when CPU < 30% for 15 minutes
  (asymmetric: scale out fast, scale in slowly to avoid flapping)
```

---

### Q4. What is serverless computing and when should you use Lambda or Cloud Functions?

**Serverless** means you deploy code (functions) without managing any server infrastructure. The cloud provider provisions, scales, and manages the execution environment. You pay only for actual execution time (milliseconds × memory × invocations).

```
Traditional server:
  [EC2 instance running 24/7] ← paying even when idle

Serverless:
  Event → [Lambda invoked] → Code runs → [Lambda sleeps]
  Billing: only during execution (ms granularity)
  At 0 requests: pay $0 (vs EC2 which costs ~$30/month idle)
```

**Lambda characteristics:**
- Execution timeout: up to 15 minutes (AWS Lambda)
- Memory: 128MB to 10GB (CPU scales proportionally)
- Stateless: no persistent memory between invocations
- Cold start: 100–3000ms for first invocation after idle

**When to use Lambda:**

| Use Case                        | Good Fit? | Reason                                       |
|---------------------------------|-----------|----------------------------------------------|
| Event processing (S3 → resize)  | Yes       | Bursty, event-driven, short duration         |
| API backend (low traffic)       | Yes       | Pay-per-use saves cost vs always-on server   |
| Cron jobs / scheduled tasks     | Yes       | Replace EC2 instances running cron           |
| WebSocket long connections      | No        | Lambda is stateless, has timeout             |
| ML model training               | No        | Too long, needs GPU, too expensive per-ms    |
| High-throughput APIs (> 10K RPS)| Maybe     | Cold starts and concurrency limits matter    |
| Database connection pooling     | No        | Each Lambda opens new DB connections         |

**Cost comparison (hypothetical):**
```
10 req/day, each 100ms at 256MB:
  Lambda: 10 × 0.1s × $0.0000166/GB-s = $0.000004/day ≈ $0/month
  EC2 t3.micro: $0.0104/hour × 720h = $7.49/month

1M req/day (high traffic), each 10ms at 256MB:
  Lambda: 1M × 0.01s × $0.0000166/GB-s = $0.0415/day = $1.25/month
  EC2 t3.medium: 2 instances × $0.0416/hour × 720h = $59.9/month
  → Lambda wins even at 1M req/day for short executions
```

---

### Q5. What is a VPC and what are its core components?

A **VPC (Virtual Private Cloud)** is an isolated, private network within the cloud provider's infrastructure. It gives full control over IP addressing, routing, and network access.

```
VPC: 10.0.0.0/16 (65,536 IP addresses)
  │
  ├── Public Subnet: 10.0.1.0/24 (internet-accessible)
  │     ├── Internet Gateway (IGW) ← attached to VPC
  │     ├── Load Balancer (public IP)
  │     └── Bastion Host / NAT Gateway
  │
  ├── Private Subnet: 10.0.2.0/24 (no direct internet)
  │     ├── App Servers (no public IP)
  │     └── NAT Gateway used for outbound internet
  │
  └── Private Subnet: 10.0.3.0/24 (database tier)
        └── RDS Databases (isolated, no internet access)
```

**Core components:**

| Component         | Purpose                                                       |
|-------------------|---------------------------------------------------------------|
| Internet Gateway  | Allows public subnet resources to reach/be reached from internet |
| NAT Gateway       | Allows private subnet resources to reach internet (outbound only) |
| Security Groups   | Stateful firewall at instance level (rules: inbound + outbound) |
| NACLs             | Stateless firewall at subnet level (rules evaluated in order) |
| Route Tables      | Direct traffic between subnets and gateways                  |
| VPC Peering       | Connect two VPCs privately (same or different accounts)      |
| VPN / Direct Connect | Connect on-premises to VPC                               |

**Security Group vs NACL:**
```
Security Group (instance-level, stateful):
  Inbound: allow 443 from 0.0.0.0/0
  Return traffic automatically allowed (stateful)
  Default: deny all inbound, allow all outbound

NACL (subnet-level, stateless):
  Rule 100: Allow inbound 443 from 0.0.0.0/0
  Rule 200: Allow inbound ephemeral ports 1024-65535 (for return traffic)
  Must explicitly allow return traffic (stateless)
  Evaluated in rule number order; first match wins
```

---

### Q6. What are the different cloud storage tiers and when do you use each?

Cloud providers offer storage at different price/performance/durability points. AWS S3 is the most common example.

```
S3 Storage Classes (hot → cold):
  
  S3 Standard
  │ Cost: $0.023/GB/month
  │ Retrieval: instant, no fee
  │ Use: active data, frequently accessed
  │
  S3 Intelligent-Tiering
  │ Cost: $0.023/GB + monitoring fee
  │ Retrieval: instant (auto-moves objects between tiers)
  │ Use: unknown or changing access patterns
  │
  S3 Standard-IA (Infrequent Access)
  │ Cost: $0.0125/GB/month + $0.01/GB retrieval
  │ Retrieval: instant but with per-GB fee
  │ Use: backups accessed once a month
  │
  S3 One Zone-IA
  │ Cost: $0.01/GB/month (single AZ only)
  │ Use: non-critical reproductible data
  │
  S3 Glacier Instant Retrieval
  │ Cost: $0.004/GB/month
  │ Retrieval: milliseconds (but expensive)
  │ Use: archive data accessed quarterly
  │
  S3 Glacier Flexible Retrieval
  │ Cost: $0.0036/GB/month
  │ Retrieval: minutes to 12 hours
  │ Use: DR backups, yearly access
  │
  S3 Glacier Deep Archive
    Cost: $0.00099/GB/month (cheapest)
    Retrieval: 12–48 hours
    Use: 7-year compliance archives, rarely accessed
```

**Lifecycle policy to automate tiering:**
```json
{
  "Rules": [{
    "Status": "Enabled",
    "Transitions": [
      {"Days": 30,  "StorageClass": "STANDARD_IA"},
      {"Days": 90,  "StorageClass": "GLACIER"},
      {"Days": 365, "StorageClass": "DEEP_ARCHIVE"}
    ],
    "Expiration": {"Days": 2555}  // Delete after 7 years
  }]
}
```

---

### Q7. What is Infrastructure as Code (IaC) and why does it matter for scale?

**Infrastructure as Code** means describing cloud resources (servers, databases, networks) in code files rather than clicking through a console. The code is version-controlled, reviewable, and repeatable.

**Why it matters:**

1. **Reproducibility:** Spin up identical environments (dev, staging, prod) from the same code
2. **Version control:** Every infrastructure change tracked in Git with author, timestamp, reason
3. **Review process:** Infrastructure changes go through PR review like application code
4. **Disaster recovery:** Rebuild entire infrastructure from scratch in minutes after catastrophic failure
5. **Drift prevention:** Detect when manual changes diverge from declared state

**Terraform example:**
```hcl
# main.tf - declarative infrastructure definition
resource "aws_db_instance" "primary" {
  identifier        = "prod-postgres"
  engine            = "postgres"
  engine_version    = "15.4"
  instance_class    = "db.r6g.large"
  allocated_storage = 100

  multi_az               = true
  deletion_protection    = true
  backup_retention_period = 7
  storage_encrypted      = true

  tags = {
    Environment = "production"
    Team        = "platform"
  }
}

# Outputs the endpoint for use by applications
output "db_endpoint" {
  value = aws_db_instance.primary.endpoint
}
```

**Terraform workflow:**
```
terraform plan   → Show what will change (dry run)
terraform apply  → Make the changes
terraform destroy → Remove all resources (use with care!)

State file: terraform.tfstate tracks real resource IDs
→ Store in S3 with DynamoDB locking for team use
```

**CDK (AWS Cloud Development Kit):**
Uses real programming languages (Python, TypeScript) for infrastructure, enabling loops, conditionals, and reusable constructs.

At scale, IaC is non-negotiable: manually clicking 200 resources into existence cannot be reviewed, audited, or reproduced reliably.

---

## Medium (Q8–Q15)

---

### Q8. What is the serverless cold start problem and how do you mitigate it?

A **cold start** occurs when a Lambda function (or Cloud Function) is invoked after a period of inactivity and the cloud provider must initialize a new execution environment before running the function code. This adds latency that does not exist for "warm" invocations.

**Cold start anatomy:**
```
Cold start (first invocation after ~15min idle):
  1. Download code package from S3 (50-500ms)
  2. Start container/micro-VM (50-200ms)
  3. Initialize runtime (Node.js/Python/JVM) (50ms-2s for JVM)
  4. Run initialization code (global scope, DB connections) (variable)
  5. Execute function handler
  
  Total additional latency: 100ms (simple Node.js) → 3000ms (JVM Spring Boot)

Warm invocation (container already running):
  1. Execute function handler only
  Total: no additional latency
```

**Mitigation strategies:**

**1. Provisioned Concurrency (AWS Lambda):**
```yaml
# Pre-warm N containers, kept permanently warm
aws lambda put-provisioned-concurrency-config \
  --function-name my-api \
  --qualifier production \
  --provisioned-concurrent-executions 10

# Cost: pay for 10 idle containers (~$30/month for 10 × 256MB)
# Benefit: zero cold starts for first 10 concurrent requests
```

**2. Choose a faster runtime:**
```
Cold start times by runtime (typical):
  Python:     100-300ms
  Node.js:    100-300ms
  Go:         50-200ms  (fastest, compiled binary)
  Java (JVM): 1000-3000ms (worst, JVM startup)
  Java (GraalVM native): 100-300ms (compiled to native)
```

**3. Minimize initialization code:**
```python
# BAD: heavy initialization in handler (runs on every cold start AND warm)
def handler(event, context):
    import heavy_library          # Import inside handler
    conn = db_connect()           # New connection every invocation
    return process(event)

# GOOD: initialize once at module level (cached between warm invocations)
import heavy_library              # Module-level import (cached)
conn = db_connect()               # Connection reused across warm invocations

def handler(event, context):
    return process(event, conn)   # Reuse warm connection
```

**4. Keep functions warm with scheduled pings:**
```
EventBridge rule: every 5 minutes, invoke Lambda with a "warmup" event
Handler: if event.type == "warmup": return "ok"
Effect: container never goes cold (tradeoff: cost of 1 invocation/5min)
```

**5. Right-size memory:**
More memory = more CPU = faster initialization. For cold-start-sensitive functions, increasing from 256MB to 1GB can reduce cold start by 40%.

---

### Q9. What are cost optimization strategies for cloud infrastructure?

Cloud costs can spiral without active management. The four main levers are: right instance type, right pricing model, eliminating waste, and tiering storage/compute.

**1. Reserved Instances / Savings Plans (biggest savings: 30–72%):**
```
On-Demand: $0.0832/hour for m5.large (pay-as-you-go)
1-year Reserved (no upfront): $0.0527/hour → 37% savings
3-year Reserved (all upfront): $0.0261/hour → 69% savings

Best for: stable, predictable baseline workloads (DB, app servers)
Rule: Buy reserved for the floor; use on-demand for peak

Savings Plans (more flexible):
  Compute Savings Plan: any instance family, any region
  1-year, 30% savings on committed $/hour spend
```

**2. Spot Instances (60–90% discount for interruptible workloads):**
```
Use for: batch jobs, CI/CD workers, ML training, dev environments
Risk: can be terminated with 2-minute warning when capacity needed

Pattern: use Spot Fleet with multiple instance types for resilience
  m5.large, m4.large, m5a.large (if one Spot pool depleted, use another)

Auto Scaling Group mixed instances policy:
  On-Demand: 20% (for critical baseline)
  Spot: 80% (for cost savings)
```

**3. Right-sizing (eliminate overprovisioning):**
```
AWS Compute Optimizer analyzes CloudWatch metrics:
  Current: db.r5.4xlarge (16 vCPU, 128GB RAM) → CPU 12%, Memory 30%
  Recommendation: db.r5.xlarge (4 vCPU, 32GB RAM) → Save $1,200/month

Tools: AWS Cost Explorer, Trusted Advisor, CloudWatch metrics
Target: CPU utilization 60-80%, Memory 70-80% (headroom for spikes)
```

**4. Storage lifecycle policies (covered in Q6) and intelligent tiering**

**5. Eliminate idle resources:**
```
Common waste:
  - Unattached EBS volumes (snapshot and delete)
  - Unused Elastic IPs ($0.005/hour idle)
  - Old AMI snapshots ($0.05/GB/month)
  - Dev environments running 24/7 (shut down nights/weekends → save 70%)
  - Oversized RDS with tiny load

Schedule Lambda to stop dev RDS at 8pm, start at 8am:
  Saves 16/24 = 67% of dev DB cost
```

**6. Architectural changes:**
- Move appropriate workloads to Lambda (pay per use, not 24/7)
- Use managed services that scale to zero (Aurora Serverless)
- Migrate from data transfer-heavy architecture (avoid cross-AZ data transfer fees: $0.01/GB)

---

### Q10. What are the 12-factor app principles for cloud-native design?

The **12-Factor App** methodology (Heroku, 2011) defines practices for building software-as-a-service applications that are portable, scalable, and maintainable in cloud environments.

```
I.   Codebase        - One codebase, tracked in Git, many deploys (dev/staging/prod)
II.  Dependencies    - Explicitly declare all dependencies (requirements.txt, pom.xml)
III. Config          - Store config in environment variables (not hardcoded, not in code)
IV.  Backing services- Treat DB, cache, queue as attached resources (swappable by config)
V.   Build/Release/Run - Strictly separate build, release, and run stages
VI.  Processes       - Execute as stateless, share-nothing processes
VII. Port binding    - Export services via port binding (app is self-contained)
VIII.Concurrency     - Scale out via the process model (horizontal, not threading)
IX.  Disposability   - Fast startup, graceful shutdown (handle SIGTERM, drain connections)
X.   Dev/prod parity - Keep dev, staging, prod as similar as possible
XI.  Logs            - Treat logs as event streams (stdout/stderr, not files)
XII. Admin processes - Run admin/mgmt tasks as one-off processes (db migrations)
```

**Most impactful factors explained:**

**Factor III — Config in environment variables:**
```python
# BAD (hardcoded):
db_host = "prod-db.internal"  # Different per environment, security risk in Git

# GOOD (12-factor):
import os
db_host = os.environ["DATABASE_HOST"]  # Injected at runtime per environment
```

**Factor VI — Stateless processes:**
```
BAD: Session stored in-process memory
  Request1 → App1 (creates session in memory)
  Request2 → App1 dies, routes to App2 (session lost!)

GOOD: Session in Redis (external backing service)
  Request1 → App1 (session written to Redis)
  Request2 → App2 (reads session from Redis) → works!
```

**Factor IX — Disposability:**
```python
import signal, sys

def graceful_shutdown(signum, frame):
    # Drain current requests, close DB connections, flush buffers
    server.stop(grace_period=30)
    db_pool.closeall()
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
```

Containers and Kubernetes rely on SIGTERM for graceful pod termination — factor IX is non-negotiable for Kubernetes.

---

### Q11. What is SNS + SQS fan-out and how does EventBridge work?

**Fan-out pattern:** One event triggers multiple consumers. SNS (Simple Notification Service) is a pub/sub topic; SQS (Simple Queue Service) provides durable queuing. Combining them: publish once to SNS, multiple SQS queues subscribe and each processes the event independently.

```
Order Placed event:
  → SNS Topic "order-events"
        ├── SQS Queue "inventory-service"    → deduct stock
        ├── SQS Queue "email-service"        → send confirmation
        ├── SQS Queue "analytics-service"    → record conversion
        └── SQS Queue "fraud-service"        → run fraud checks

Benefits:
  - Decoupled: order service doesn't know about consumers
  - Each SQS queue has its own retry/DLQ configuration
  - New consumer: just add another SQS subscription (no order service change)
  - Failed consumer: other consumers unaffected
  - SQS provides buffering if consumer is slow
```

**EventBridge (AWS):**
EventBridge is a serverless event bus that routes events based on content rules — more powerful than SNS for complex routing.

```
Event source → EventBridge Bus → Rules (pattern matching) → Targets

Example rules:
  Rule 1: { "source": ["order-service"], "detail-type": ["OrderPlaced"] }
    → Target: Lambda (send_confirmation_email)
  
  Rule 2: { "source": ["order-service"], "detail-type": ["OrderPlaced"],
            "detail": { "total": [{"numeric": [">=", 1000]}] } }
    → Target: SQS (high-value-order-queue) — only for orders > $1000
  
  Rule 3: { "source": ["payment-service"], "detail-type": ["PaymentFailed"] }
    → Target: SNS (alert-ops-team)
```

**Event payload:**
```json
{
  "source": "order-service",
  "detail-type": "OrderPlaced",
  "detail": {
    "orderId": "ord-12345",
    "userId": "user-789",
    "total": 150.00,
    "items": [{"productId": "p1", "qty": 2}]
  }
}
```

EventBridge also integrates with 200+ SaaS providers (Stripe, GitHub, Zendesk) as event sources, making it a powerful integration hub.

---

### Q12. What are the trade-offs between managed services and self-managed for databases?

Choosing between a managed service (RDS, Cloud SQL) and self-managing PostgreSQL on EC2 involves balancing operational control against engineering overhead.

**Managed service (RDS PostgreSQL):**
```
What the provider handles:
  ✓ OS patching and hardening
  ✓ PostgreSQL minor version upgrades
  ✓ Automated backups (with PITR)
  ✓ Multi-AZ standby + automatic failover (~30s RTO)
  ✓ Read replica management
  ✓ Storage auto-scaling
  ✓ Enhanced Monitoring (per-process metrics)
  ✓ Parameter group management (settings as config, not SSH)

What you give up:
  ✗ No superuser access (no pg_file_settings, limited extensions)
  ✗ Can't run custom OS-level agents
  ✗ Major version upgrades require planning/downtime window
  ✗ Limited to supported PostgreSQL extensions
  ✗ Premium pricing: RDS is 2–3× more expensive than raw EC2
```

**Self-managed on EC2:**
```
What you gain:
  ✓ Full superuser access
  ✓ Any extension (Timescale, Citus, PostGIS full support)
  ✓ Custom kernel tuning (huge pages, scheduler)
  ✓ Lower cost (raw EC2 + EBS cheaper than RDS)
  ✓ Physical replication slots, streaming replication full control

What you take on:
  ✗ OS patching (your team)
  ✗ PostgreSQL upgrades (your team)
  ✗ Backup automation (your responsibility)
  ✗ HA setup (Patroni + etcd + HAProxy — complex)
  ✗ Failover testing and monitoring (your responsibility)
  ✗ On-call for DB incidents 24/7
```

**Decision framework:**

| Factor                      | Prefer Managed (RDS)       | Prefer Self-Managed        |
|-----------------------------|----------------------------|----------------------------|
| Team DB expertise           | Low                        | High                       |
| Need for custom extensions  | No                         | Yes (e.g., Timescale)      |
| Budget priority             | Operations cost reduction  | Infrastructure cost saving |
| Scale                       | < 10TB                     | > 10TB (cost becomes huge) |
| Compliance                  | Standard SOC2/HIPAA        | Air-gapped, custom audit   |

Most startups and mid-size companies benefit from managed services. Large companies with dedicated DBA teams often self-manage at scale where RDS costs become prohibitive.

---

### Q13. How do service quotas and limits affect system design?

Cloud providers impose quotas and limits on resources. Ignoring them leads to unexpected failures at scale. Good architects know the limits and design around them.

**Critical AWS limits to know:**

| Service          | Limit                              | Impact if hit                        |
|------------------|------------------------------------|--------------------------------------|
| Lambda           | 1000 concurrent executions/region  | Throttling → errors                  |
| SQS              | 120,000 in-flight messages (std)   | Messages not dequeued                |
| DynamoDB         | 40,000 RCU/WCU per table (default) | ThrottlingException errors           |
| S3               | 5,500 GET/s per prefix             | 503 SlowDown errors                  |
| EC2              | 96 vCPU on-demand (default)        | Can't launch new instances           |
| RDS              | 40 instances per region (default)  | Can't create new DB                  |
| ALB              | 100 rules per listener             | Can't add more routing rules         |
| API Gateway      | 10,000 RPS per account/region      | 429 Too Many Requests                |

**Design patterns to avoid hitting limits:**

**S3 key prefix sharding (avoid hot partition):**
```
BAD: all objects start with same prefix
  s3://bucket/logs/2025-10-01.log
  s3://bucket/logs/2025-10-02.log
  All keys start with "logs/" → same S3 partition → 5,500 GET/s limit

GOOD: randomize prefix
  s3://bucket/ab12/logs/2025-10-01.log
  s3://bucket/cd34/logs/2025-10-02.log
  → different partitions → 5,500 × many partitions = massive throughput
```

**Lambda concurrency management:**
```python
# Reserve concurrency for critical functions
aws lambda put-function-concurrency \
  --function-name payment-processor \
  --reserved-concurrent-executions 200  # Guarantee 200 slots, protect from others

# Use SQS → Lambda (built-in concurrency management)
# SQS event source mapping handles scaling; Lambda only scales to batch size
```

**Proactive quota management:**
- Request quota increases before you hit them (AWS console → Service Quotas)
- Use multi-account strategy to multiply effective quotas (each account has its own quota)
- Design for graceful degradation when limits are hit (circuit breaker on 429 responses)

---

### Q14. What are the multi-cloud vs multi-region trade-offs for disaster recovery?

**Multi-Region (same cloud provider):** Multiple geographic regions within one provider's infrastructure. Lower operational complexity, native tooling for cross-region replication, but subject to provider-wide incidents.

**Multi-Cloud:** Using two or more cloud providers (e.g., AWS primary + GCP DR). Maximum resilience against a single provider's outage, but dramatically higher complexity and cost.

```
Multi-Region (AWS us-east-1 + eu-west-1):
  Pros:
    ✓ Native replication tooling (S3 CRR, Aurora Global, DynamoDB Global Tables)
    ✓ Same IAM, same monitoring, same tooling
    ✓ Reasonable cost premium (~1.5-2×)
    ✓ Protects against regional datacenter/power failure
    
  Cons:
    ✗ Provider-wide incidents affect both regions (rare but happened: S3 outage 2017)
    ✗ Data sovereignty may require specific regions

Multi-Cloud (AWS primary + GCP standby):
  Pros:
    ✓ Survives a complete cloud provider outage
    ✓ Negotiating leverage with providers
    ✓ Use best-of-breed services per provider
    
  Cons:
    ✗ Two sets of skills, tools, training required
    ✗ No native cross-cloud replication → must build custom pipelines
    ✗ IAM, networking, monitoring all duplicated with different APIs
    ✗ Cost: 2× infrastructure + significant engineering overhead
    ✗ Testing DR across cloud boundaries is complex
```

**Reality check:** True provider-wide cloud outages are rare (< 1 per year, typically brief). For most businesses, the complexity of multi-cloud is not justified by the risk reduction. **Multi-region within one provider** achieves 99.999%+ availability for most threat models.

**When multi-cloud makes sense:**
- Regulatory requirements (EU data in EU providers, US data in US providers)
- Using specialized services (GCP for ML/AI, AWS for general infra)
- Risk management for critical financial systems
- Already have expertise in both clouds from acquisitions

---

### Q15. What is the event-driven serverless architecture pattern?

Event-driven serverless uses Lambda functions triggered by events (HTTP, queue messages, storage changes, scheduled crons) rather than long-running servers. It composes well for microservices with natural event flows.

**Order processing system example:**
```
Customer places order:
  API Gateway (HTTP POST /orders)
       ↓
  Lambda: validate-order
       ↓ (publishes to SQS)
  SQS: order-processing-queue
       ↓ (triggers)
  Lambda: process-payment
       ↓ on success (EventBridge)
  EventBridge: payment-succeeded event
       ├── Lambda: send-confirmation-email
       ├── Lambda: update-inventory
       └── Lambda: notify-warehouse
       ↓ on failure (DLQ)
  SQS: payment-dlq → Lambda: handle-payment-failure → notify-customer

State management:
  DynamoDB: order state machine
  {orderId, status: "pending"|"paid"|"fulfilling"|"shipped"}
```

**Why this architecture scales well:**
```
Each Lambda scales independently:
  Normal day: validate-order → 100 invocations/min
  Flash sale:  validate-order → 5000 invocations/min (auto-scaled)
  
SQS decouples producers from consumers:
  Payment processing slower? SQS buffers the orders → no back-pressure on API
  
Cost: pay only for actual execution, not idle servers
```

**Failure handling:**
```
SQS with DLQ (Dead Letter Queue):
  maxReceiveCount = 3 → try 3 times, then move to DLQ
  
  DLQ trigger: Lambda alerts ops team, writes to error tracking
  Visibility timeout: 30s → if Lambda crashes, message reappears automatically
  
Lambda retry behavior:
  Synchronous (API Gateway): no retry (client retries)
  Asynchronous (EventBridge): 2 retries with exponential backoff, then DLQ
```

---

## Hard (Q16–Q20)

---

### Q16. How do you design for cloud provider failure? Compare multi-cloud vs redundant AZs.

This question tests whether a candidate can make pragmatic architectural decisions about extreme resilience requirements versus operational complexity trade-offs.

**Threat model analysis:**
```
Risk levels for cloud infrastructure failures:
  
  Single instance failure:     High probability  → Mitigate with ASG
  Single AZ failure:           Medium probability → Mitigate with multi-AZ
  Single region failure:       Low probability    → Mitigate with multi-region
  Cloud provider outage:       Very low probability (< once/year, brief)
  → Mitigate with multi-cloud ONLY if business justifies complexity
```

**Multi-AZ design (standard, recommended for most):**
```
AWS us-east-1:
  AZ us-east-1a: App tier (3 pods) + DB Primary + Cache Primary
  AZ us-east-1b: App tier (3 pods) + DB Replica + Cache Replica
  AZ us-east-1c: App tier (3 pods) + DB Replica

  ALB spans all AZs → routes to healthy AZ automatically
  Aurora Multi-AZ → automatic failover to replica (~30s)
  
  Protects against: hardware failure, datacenter power, AZ network
  Composite availability: ~99.99%+
```

**Multi-region design:**
```
us-east-1 (primary, active):
  Full stack deployed, handles all traffic

eu-west-1 (secondary, warm standby):
  Scaled-down stack, ready to accept traffic
  Aurora Global replica (async, < 1s RPO)
  S3 CRR (cross-region replication) for object storage
  
Route 53:
  Primary record: → us-east-1 ALB (health check: /health)
  Failover record: → eu-west-1 ALB (activates if primary unhealthy)
  
RTO: 1-5 minutes (DNS TTL + app warmup)
Cost: ~1.5-2× single region
```

**Multi-cloud design (when justified):**
```
AWS (primary, 100% traffic):
  Full production stack

GCP (DR, warm standby):
  Compute: GKE cluster (scaled down, helm charts maintained in sync)
  Database: Cloud SQL (Postgres), loaded from daily snapshots + WAL streaming
  Storage: GCS with data from AWS S3 via cross-cloud pipeline
  DNS: Route 53 (with GCP endpoint as failover) OR Cloudflare DNS

Challenge: no native cross-cloud replication → custom ETL pipeline needed
  Kafka MirrorMaker 2: stream data from AWS MSK to GCP Pub/Sub (complex)
  
RTO: 15-60 minutes (data sync lag + manual DNS update)
Cost: ~2.5-3× single cloud
Justification: financial institutions, government, $1B+ businesses
```

**Recommendation framework:**
- < $10M ARR: multi-AZ
- $10M–$1B ARR: multi-region same provider
- > $1B ARR or regulated: consider multi-cloud for specific critical paths

---

### Q17. How do reserved instances, spot instances, and savings plans work together for cost optimization?

Sophisticated cloud cost management layers multiple pricing models to reduce costs by 50–70% compared to on-demand while maintaining reliability.

**Three pricing tiers:**
```
On-Demand: Pay full rate, no commitment
  → Use for: unpredictable spikes, experimentation, short-lived resources

Reserved Instances / Savings Plans: Commit to usage, 30-72% discount
  → Use for: baseline stable workload that runs 24/7

Spot Instances: Bid on unused capacity, 60-90% discount, can be terminated
  → Use for: stateless batch processing, CI/CD, fault-tolerant workloads
```

**Optimal allocation strategy:**
```
Actual traffic pattern:
  ┌──────────────────────────────────────────┐
  │ Peak                                     │
  │    ■■■■■■■■■■■■■■ On-Demand (peak burst) │
  │■■■■■■■■■■■■■■■■■■ Spot (variable middle) │
  │■■■■■■■■■■■■■■■■■■ Reserved (floor)      │
  └──────────────────────────────────────────┘
  
  Floor (100% of time): Buy 1-year Reserved
  Variable middle (hours of high usage): Use Spot Fleet
  Unpredictable spikes: On-Demand auto-scaling
```

**Savings Plans (more flexible than RI):**
```
Compute Savings Plan:
  Commit to $10/hour of compute usage (any instance family, any region)
  Save 40% vs On-Demand
  Best for teams that frequently change instance types

EC2 Instance Savings Plan:
  Commit to specific instance family in one region (e.g., m5 in us-east-1)
  Save up to 72% vs On-Demand
  Less flexible but highest discount

Coverage analysis:
  Step 1: Look at last 30 days On-Demand spend
  Step 2: Identify stable baseline (lowest 24h usage in the month)
  Step 3: Buy Savings Plans for that baseline amount
  Step 4: Everything above baseline uses On-Demand or Spot
```

**Spot instance resilience architecture:**
```python
# AWS Auto Scaling Group: mixed instance policy
{
  "MixedInstancesPolicy": {
    "InstancesDistribution": {
      "OnDemandBaseCapacity": 2,          # Always keep 2 On-Demand
      "OnDemandPercentageAboveBaseCapacity": 20,  # 20% On-Demand, 80% Spot
      "SpotAllocationStrategy": "capacity-optimized"  # Pick most available Spot pool
    },
    "LaunchTemplate": {
      "Overrides": [                      # Multiple instance types for resilience
        {"InstanceType": "m5.large"},
        {"InstanceType": "m5a.large"},
        {"InstanceType": "m4.large"},
        {"InstanceType": "m5n.large"}
      ]
    }
  }
}
```

**Real-world savings example:**
```
Before optimization:
  100 m5.large On-Demand 24/7 = $0.096 × 100 × 8760h = $84,096/year

After:
  20 m5.large Reserved (1yr) = $0.062 × 20 × 8760h = $10,870
  60 m5.large Spot (avg 2/3 price) = $0.032 × 60 × 8760h = $16,819
  20 m5.large On-Demand (peak) = $0.096 × 20 × 8760h × 0.3 = $5,047
  Total: $32,736/year → 61% savings
```

---

### Q18. How do Kubernetes and serverless complement each other in cloud architecture?

Kubernetes and serverless are not competing approaches — they are complementary tools optimized for different workload profiles within the same organization.

**Decision matrix:**

| Workload Characteristic         | Use Kubernetes          | Use Serverless (Lambda)       |
|----------------------------------|-------------------------|-------------------------------|
| Always-running (24/7)           | Yes (cost-effective)    | No (pay always = expensive)   |
| Bursty, unpredictable traffic   | Hard (scaling lag)      | Yes (instant auto-scale)      |
| Duration > 15 minutes           | Yes                     | No (Lambda 15min max)         |
| Stateful workloads              | Yes (StatefulSets)      | No                            |
| Cold start sensitive            | Not applicable          | Mitigate with provisioned     |
| Custom runtime / binary deps    | Yes (any container)     | Limited runtime choices       |
| Event-triggered (S3, SQS, etc) | Complex (event sources) | Yes (native integrations)     |
| Cost-sensitive baseline         | Yes (bin packing)       | Yes (pay per use at low traffic)|

**Hybrid architecture pattern:**
```
E-commerce platform:

Kubernetes (EKS):
  - Product catalog API (high traffic, stateless, always on)
  - Order management service (stateful session, always on)
  - PostgreSQL databases (StatefulSets or RDS)
  - Internal microservices with complex networking (Istio service mesh)

Serverless (Lambda):
  - Image resizing (triggered on S3 upload)
  - Email sending (triggered on SQS queue)
  - Cron jobs (report generation, data cleanup)
  - Webhook receivers (Stripe, GitHub)
  - A/B test assignment (triggered on each request via edge Lambda@Edge)
```

**Integration between Kubernetes and Lambda:**
```
Pattern: K8s service calls Lambda via SDK for specialized tasks
  
  Order Service (K8s pod) 
    → AWS SDK → Lambda: calculate-shipping-rate
                         (uses specialized carrier APIs, run event-driven)
  
  Pattern: Lambda writes to SQS, K8s consumer picks up
  
  Lambda: process-webhook (Stripe payment notification)
    → SQS: payment-events
    → K8s consumer pod: update order status, trigger fulfillment
```

Kubernetes excels at long-running services with complex inter-service communication. Lambda excels at event-driven, bursty, short-duration tasks. Using both leverages the strengths of each without forcing mismatches.

---

### Q19. How do you design a cloud-native multi-tenant SaaS platform?

Multi-tenancy means serving multiple customers (tenants) from shared infrastructure while maintaining isolation, fair resource usage, and security boundaries.

**Three tenancy models:**

```
Silo model (complete isolation):
  Tenant A → Dedicated VPC → Dedicated DB → Dedicated K8s namespace
  Tenant B → Dedicated VPC → Dedicated DB → Dedicated K8s namespace
  
  Pros: maximum isolation, easier compliance (HIPAA, FedRAMP)
  Cons: expensive, slow onboarding (provision whole stack per tenant)
  Use: large enterprise customers, regulated industries

Pool model (fully shared):
  All tenants → Shared App Tier → Shared DB (tenant_id column)
  
  Pros: cost efficient, easy to onboard new tenants
  Cons: noisy neighbor, data isolation via application logic only
  Use: small tenants, internal tools

Bridge model (hybrid):
  All tenants → Shared App Tier (with RBAC)
  Small tenants → Shared DB (tenant_id isolated by RLS)
  Large tenants → Dedicated DB (per tenant contract SLA)
```

**Database-level isolation with Row-Level Security:**
```sql
-- PostgreSQL RLS: enforce tenant isolation at DB level
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON orders
  USING (tenant_id = current_setting('app.current_tenant')::UUID);

-- App sets tenant context per request:
SET LOCAL app.current_tenant = '550e8400-e29b-41d4-a716-446655440000';
-- Now: SELECT * FROM orders -- only returns rows for this tenant automatically
```

**Resource quotas per tenant (Kubernetes):**
```yaml
# ResourceQuota per namespace (= per tenant)
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-a-quota
  namespace: tenant-a
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    pods: "20"
```

**Tenant-aware rate limiting:**
```python
# Each tenant has their own rate limit bucket in Redis
def check_rate_limit(tenant_id, api_key, limit=1000, window=60):
    key = f"ratelimit:{tenant_id}:{int(time.time() // window)}"
    current = redis.incr(key)
    redis.expire(key, window)
    
    if current > limit:
        raise RateLimitExceeded(f"Tenant {tenant_id} exceeded {limit} req/{window}s")
    
    # Premium tenants get higher limits based on plan
    plan_limits = {"starter": 1000, "growth": 10000, "enterprise": 100000}
    plan_limit = plan_limits.get(get_tenant_plan(tenant_id), 1000)
```

**Onboarding automation:**
```
New tenant signup:
  1. Create tenant record in control plane DB
  2. Provision tenant resources (Terraform/CDK: namespace, RDS schema, S3 prefix)
  3. Generate API keys, configure DNS subdomain (tenant.app.com)
  4. Seed initial data
  5. Send welcome email with credentials
  
Automate via: Lambda + Step Functions (orchestrate multi-step provisioning)
Target: tenant operational within 60 seconds of signup
```

---

### Q20. How do you architect a global, high-traffic API platform on AWS?

This integrative question tests the ability to combine all cloud architecture patterns into a coherent, production-ready design.

**Requirements:**
- 1 million API requests per day (avg: ~12 RPS, peak: ~500 RPS)
- Global users (US, EU, APAC)
- 99.99% availability
- p99 latency < 200ms globally
- GDPR data residency: EU data stays in EU

**Complete architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│  Global Layer                                               │
│  Route 53 (latency-based routing + health checks)           │
│  CloudFront CDN (edge caching, WAF, DDoS protection)         │
└──────────────┬──────────────────────┬───────────────────────┘
               │ US users             │ EU users
┌──────────────▼──────────┐ ┌─────────▼──────────────────────┐
│  us-east-1              │ │  eu-west-1                     │
│  Multi-AZ               │ │  Multi-AZ (GDPR compliant)     │
│                         │ │                                │
│  ALB (managed, multi-AZ)│ │  ALB (managed, multi-AZ)      │
│  EKS cluster:           │ │  EKS cluster:                  │
│    API pods (HPA)       │ │    API pods (HPA)              │
│    Auth pods            │ │    Auth pods                   │
│  ElastiCache Redis      │ │  ElastiCache Redis             │
│  Aurora Global Primary  │◄►│  Aurora Global Replica        │
│  SQS + Lambda workers   │ │  SQS + Lambda workers          │
└─────────────────────────┘ └────────────────────────────────┘
```

**Layer-by-layer design decisions:**

**1. DNS and Routing:**
```
Route 53 latency routing:
  US users → us-east-1 ALB (health check: /health)
  EU users → eu-west-1 ALB (health check: /health)
  TTL: 60 seconds (balance between failover speed and DNS load)
  
  On region failure: Route 53 stops returning failed endpoint
  APAC users: nearest region wins (either US or EU)
```

**2. CDN Layer (CloudFront):**
```
CloudFront distribution in front of both regions:
  Cache public API responses: GET /products, GET /categories (5-minute TTL)
  Pass-through: POST, PUT, DELETE, authenticated requests
  WAF rules: block SQL injection, XSS, rate limit by IP
  
  Cache-hit ratio target: 60% for public endpoints
  → reduces origin hits from 1M to 400K requests/day
```

**3. Compute (EKS):**
```
EKS cluster per region, multi-AZ:
  Nodes: 3 AZs, node groups with mixed On-Demand (20%) + Spot (80%)
  
  API Deployment:
    replicas: 6 (2 per AZ)
    HPA: scale on CPU > 70% (max 50 pods)
    PodDisruptionBudget: minAvailable=4 (safe rolling updates)
    Resource: requests: 0.25 CPU/256MB, limits: 1 CPU/512MB
  
  Zero-downtime deploys: Helm + ArgoCD (GitOps)
```

**4. Data Layer:**
```
Aurora Global Database:
  Primary: us-east-1 (all writes)
  Replica: eu-west-1 (reads + EU data residency requirement met)
  Replication lag: < 1 second
  Failover: < 1 minute (automated)
  
Cache: ElastiCache Redis Cluster per region
  6 nodes (3 shards × 2 replicas)
  Cache: session data, frequently-read API responses
  
GDPR: EU users' PII data written only to eu-west-1 Aurora
  Sharding key: user.country_code → routes writes to appropriate region
```

**5. Async Processing:**
```
Background jobs via SQS + Lambda:
  - Email notifications
  - Analytics event processing
  - Search index updates (OpenSearch)
  - Report generation
```

**6. Observability:**
```
Metrics: CloudWatch + Datadog (cross-region unified view)
Tracing: AWS X-Ray (distributed tracing across Lambda + EKS)
Logging: CloudWatch Logs → S3 → Athena for query
Alerting: PagerDuty for p99 > 500ms or error rate > 1%

SLO dashboard:
  Availability: 99.99% (rolling 30 days)
  p99 latency: < 200ms globally
  Error budget: 52 min/year remaining
```

---

## Quick Reference

### Shared Responsibility Model
- **Provider:** Physical, hardware, hypervisor, managed service uptime
- **Customer:** Data encryption, IAM, app code, OS patches (IaaS), network config

### Service Type Comparison
| Type | Example      | You Manage                    |
|------|--------------|-------------------------------|
| IaaS | EC2          | OS + runtime + app + data     |
| PaaS | RDS          | App + data                    |
| SaaS | Salesforce   | Configuration + data entry    |

### Auto-Scaling Comparison
- **Horizontal:** Add instances — stateless services, unlimited scale
- **Vertical:** Resize instance — stateful services, limited by max instance size

### Lambda When to Use
- Short-duration (< 15 min), event-driven, bursty, pay-per-use
- Cold start mitigations: Provisioned Concurrency, slim runtimes, module-level init

### Storage Tiers (S3)
`Standard → IA → Glacier Instant → Glacier Flexible → Deep Archive`
Cost range: `$0.023/GB` → `$0.00099/GB`

### Cost Optimization Hierarchy
1. Reserved/Savings Plans (30–72% savings, baseline)
2. Spot Instances (60–90% savings, interruptible workloads)
3. Right-sizing (eliminate waste)
4. Lifecycle policies (cold storage for old data)
5. Serverless for bursty/infrequent workloads

### 12-Factor App Key Points
- Config in env vars, not code
- Stateless processes (sessions in Redis)
- Logs as streams (stdout)
- Graceful shutdown (handle SIGTERM)

### Multi-Tenancy Models
| Model  | Cost      | Isolation | Best For              |
|--------|-----------|-----------|------------------------|
| Silo   | High      | Maximum   | Enterprise, regulated |
| Pool   | Low       | Logical   | SMB, internal tools   |
| Bridge | Medium    | Mixed     | Most SaaS products    |
