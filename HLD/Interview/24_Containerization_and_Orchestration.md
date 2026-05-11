# Containerization and Orchestration — Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. What is Docker and why did containers revolutionize deployment?

**Docker** is a platform for building, shipping, and running applications inside **containers** — lightweight, isolated environments that package an application together with all its dependencies (runtime, libraries, config).

**The problem Docker solved:**
Before containers, the classic complaint was "it works on my machine." Developers had different OS versions, library versions, and environment configurations than staging or production servers. Deployment required complex setup scripts and was error-prone.

```
Before Docker:
  Dev machine:  Python 3.8, Postgres 12, OpenSSL 1.0
  Prod server:  Python 3.6, Postgres 13, OpenSSL 1.1
  Result: import errors, API mismatches, SSL handshake failures
  Fix: hours of debugging environment differences

With Docker:
  Developer builds: docker build -t my-app .
  Container includes: Python 3.8 + Postgres 12 + OpenSSL 1.0 + app code
  Prod runs: docker run my-app
  Result: identical environment everywhere — guaranteed
```

**Key Docker components:**
- **Dockerfile:** Declarative recipe to build an image (FROM, RUN, COPY, CMD)
- **Image:** Immutable snapshot of the application + environment (stored in registry)
- **Container:** Running instance of an image (ephemeral, disposable)
- **Registry:** Repository for images (Docker Hub, AWS ECR, GCR)

**Why containers beat VMs for deployment:**
- Containers start in milliseconds (VMs: minutes to boot OS)
- Containers are megabytes (VMs: gigabytes including full OS)
- Containers share the host OS kernel (VMs need full OS per instance)
- Images are version-controlled: you can roll back to `my-app:v2.3.1` instantly

Docker democratized the "build once, run anywhere" model that transformed CI/CD pipelines, microservices, and cloud deployments.

---

### Q2. How do containers differ from VMs in terms of resource isolation?

Containers and VMs both provide isolation between workloads but achieve it at different layers of the software stack.

```
Virtual Machines:
  ┌────────────┬────────────┬────────────┐
  │  App A     │  App B     │  App C     │
  ├────────────┼────────────┼────────────┤
  │  OS Kernel │  OS Kernel │  OS Kernel │ ← Each VM has full OS (~1-20GB)
  ├────────────┴────────────┴────────────┤
  │           Hypervisor (KVM/VMware)    │
  ├─────────────────────────────────────┤
  │           Host OS Kernel            │
  └─────────────────────────────────────┘

Containers:
  ┌────────────┬────────────┬────────────┐
  │  App A     │  App B     │  App C     │
  ├────────────┼────────────┼────────────┤
  │ Libs/Deps  │ Libs/Deps  │ Libs/Deps  │ ← Each container has own libs (~50-300MB)
  ├────────────┴────────────┴────────────┤
  │        Container Runtime (Docker)   │
  ├─────────────────────────────────────┤
  │      Shared Host OS Kernel          │ ← One kernel for all containers
  └─────────────────────────────────────┘
```

**Isolation mechanisms in containers:**
- **Namespaces:** Isolate PID, network, mount points, users, IPC, hostname per container
- **cgroups:** Limit CPU, memory, disk I/O per container
- **Seccomp/AppArmor:** Restrict system calls a container can make

**Comparison table:**

| Dimension         | Virtual Machine              | Container                         |
|-------------------|------------------------------|-----------------------------------|
| Startup time      | Minutes (full OS boot)       | Milliseconds (process start)      |
| Size              | 1–20 GB (full OS)            | 50–500 MB (app + libs)            |
| Isolation         | Hardware-level (hypervisor)  | OS-level (namespaces + cgroups)   |
| Security          | Stronger (separate kernel)   | Weaker (shared kernel)            |
| Overhead          | ~10–20% CPU/memory overhead  | ~1–5% overhead                    |
| Density           | ~10–50 VMs per host          | 100–1000 containers per host      |
| Use case          | Full OS requirements         | Application deployment            |

VMs are still preferred for security-sensitive multi-tenant environments (cloud providers use VMs as the isolation boundary between customers). Containers are used within a trusted environment for application workloads.

---

### Q3. What are Docker image layers and how do they enable caching and smaller images?

Docker images are built in **layers**. Each instruction in a Dockerfile creates a new layer. Layers are immutable and are cached — if a layer hasn't changed, Docker reuses the cached version instead of rebuilding.

```
Dockerfile:
  FROM python:3.11-slim          # Layer 1: base image (~100MB)
  WORKDIR /app                   # Layer 2: metadata (tiny)
  COPY requirements.txt .        # Layer 3: dependencies file
  RUN pip install -r requirements.txt  # Layer 4: installed packages (~150MB)
  COPY . .                       # Layer 5: application code (~5MB)
  CMD ["python", "app.py"]       # Layer 6: metadata

Build cache behavior:
  First build: all 6 layers built and cached
  
  Code change (only Layer 5 changes):
  Layer 1: CACHED (base image unchanged)
  Layer 2: CACHED
  Layer 3: CACHED
  Layer 4: CACHED (requirements.txt unchanged → packages cached!)
  Layer 5: REBUILT (code changed)
  Layer 6: REBUILT (derived from Layer 5)
  
  Result: build takes seconds instead of minutes (package install skipped)
```

**Key optimization: copy dependencies before code:**
```dockerfile
# BAD order — code change busts package cache
COPY . .                               # If code changes → Layer busted
RUN pip install -r requirements.txt   # Reinstalls ALL packages every build

# GOOD order — maximize cache hits
COPY requirements.txt .               # Only busted when deps change
RUN pip install -r requirements.txt   # Cached unless requirements.txt changes
COPY . .                              # Code changes don't affect dep layer
```

**Minimizing image size:**
```dockerfile
# Multi-stage build — separate build and runtime stages
FROM golang:1.21 AS builder           # Full build environment (1GB)
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o myapp .               # Compiles binary

FROM gcr.io/distroless/static         # Minimal runtime (5MB, no shell!)
COPY --from=builder /app/myapp /      # Only copy the binary
CMD ["/myapp"]
# Final image: ~10MB vs 1GB+
```

---

### Q4. What is Kubernetes and why does it exist?

**Kubernetes (K8s)** is an open-source container orchestration platform originally developed at Google, donated to CNCF in 2014. It automates deployment, scaling, and management of containerized applications across a cluster of machines.

**The problem Kubernetes solves:**
Running a handful of Docker containers manually is manageable. But at scale — hundreds of services, thousands of containers — manual management breaks down:

```
Problems at scale (without Kubernetes):
  - Which server should each container run on? (scheduling)
  - Container crashed on server-3 at 3 AM: who restarts it? (self-healing)
  - Traffic doubled: how do I add more instances? (scaling)
  - Deploy new version without downtime: how? (rolling updates)
  - How do containers on different servers communicate? (networking)
  - Where does each service store its config and secrets? (config management)

Kubernetes solves all of these automatically.
```

**Kubernetes core value proposition:**

```
Desired state: "I want 5 replicas of my-app, always running"
  
  K8s control loop:
  1. Check actual state: 3 replicas running (2 crashed)
  2. Compare to desired state: need 5
  3. Act: schedule 2 new pods
  4. Repeat every few seconds forever
  
  This "reconciliation loop" is the heart of Kubernetes.
  Operators declare *what* they want; K8s figures out *how* to achieve it.
```

**Origins:** Google ran all workloads in containers (Borg system) for a decade before Kubernetes. K8s brought those learnings — bin packing, rolling updates, self-healing — to open source.

Kubernetes became the de facto standard for production container orchestration. All major cloud providers offer managed Kubernetes (EKS, GKE, AKS).

---

### Q5. What are the core Kubernetes concepts: Pod, Deployment, ReplicaSet, Service, and Ingress?

**Pod:** The smallest deployable unit in Kubernetes. A pod runs one or more tightly-coupled containers that share network and storage. Pods are ephemeral — they can die and be replaced.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod
spec:
  containers:
  - name: my-app
    image: my-app:v1.2
    ports:
    - containerPort: 8080
    resources:
      requests: {cpu: "100m", memory: "128Mi"}
      limits:   {cpu: "500m", memory: "256Mi"}
```

**ReplicaSet:** Ensures a specified number of pod replicas are running. If a pod dies, ReplicaSet starts a new one. Rarely used directly — use Deployment instead.

**Deployment:** Manages ReplicaSets and enables declarative updates with rollback. The primary way to deploy stateless applications.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 5
  selector:
    matchLabels: {app: my-app}
  strategy:
    type: RollingUpdate
    rollingUpdate: {maxSurge: 1, maxUnavailable: 0}
  template:
    metadata:
      labels: {app: my-app}
    spec:
      containers:
      - name: my-app
        image: my-app:v1.2
```

**Service:** A stable network endpoint for a set of pods. Pods come and go; Service provides a fixed IP/DNS name for clients.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-svc
spec:
  selector: {app: my-app}          # Routes to pods with this label
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP                   # Internal cluster access only
```

**Ingress:** Routes external HTTP/HTTPS traffic to internal Services based on URL paths or hostnames. Requires an Ingress Controller (nginx, AWS ALB controller).

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /v1
        pathType: Prefix
        backend:
          service: {name: my-app-svc, port: {number: 80}}
```

---

### Q6. What is the difference between liveness, readiness, and startup probes?

Kubernetes uses three types of health probes to manage container lifecycle. Using them correctly is essential for zero-downtime deployments.

**Liveness probe:** Checks if the container is alive. If it fails, Kubernetes **restarts** the container.

```yaml
livenessProbe:
  httpGet:
    path: /healthz    # Should return 200 if app is not deadlocked
    port: 8080
  initialDelaySeconds: 15   # Wait before first probe (app startup time)
  periodSeconds: 10          # Check every 10 seconds
  failureThreshold: 3        # Restart after 3 consecutive failures

# Good liveness check: is the process alive and responding?
# Bad liveness check: dependency checks (DB down → restart loop = bad)
```

**Readiness probe:** Checks if the container is **ready to receive traffic**. If it fails, the pod is removed from Service endpoints (no traffic) but NOT restarted.

```yaml
readinessProbe:
  httpGet:
    path: /ready      # Return 200 only when all dependencies are healthy
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 3

# Good readiness check: can this pod serve requests right now?
# Checks: DB connection available, cache warm, required config loaded
```

**Startup probe:** Used for slow-starting containers. Delays liveness checks until the startup probe succeeds. Prevents liveness probe from killing a container that needs 60+ seconds to initialize.

```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8080
  failureThreshold: 30   # Allow 30 × 10s = 300s for startup
  periodSeconds: 10
```

**Comparison:**

| Probe   | On Failure        | On Success        | Use for                          |
|---------|-------------------|-------------------|----------------------------------|
| Liveness | Restart container | Continue running  | Detecting deadlocks/hung process |
| Readiness| Remove from LB    | Add to LB         | Traffic routing gates            |
| Startup  | Restart container | Switch to liveness| Slow-starting apps (JVM, ML)     |

During a rolling update: new pods must pass readiness before old pods are removed, ensuring zero downtime.

---

### Q7. What are Kubernetes Services: ClusterIP vs NodePort vs LoadBalancer vs ExternalName?

Kubernetes Services abstract access to pods. Different types expose the service at different network levels.

**ClusterIP (default):** Exposes the service on a cluster-internal IP. Only accessible within the cluster. Used for internal service-to-service communication.

```
[Pod A] → ClusterIP: 10.96.0.1:80 → [Pod B, C, D] (round-robin)
External users cannot access this.
```

**NodePort:** Exposes the service on a static port (30000–32767) on every node's external IP. External traffic can reach it via `<NodeIP>:<NodePort>`.

```
External: curl http://node1-ip:30080 → [Pod A, B, C]
Limited to ports 30000-32767; not suitable for production.
```

**LoadBalancer:** Provisions an external cloud load balancer (AWS ALB, GCP LB). Gives the service a public IP. The cloud-native way to expose services to the internet.

```yaml
spec:
  type: LoadBalancer
  # AWS annotation to configure ALB type
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
```

```
Internet → AWS ALB (public IP) → NodePort → [Pod A, B, C]
One LoadBalancer per service = expensive for many services.
Solution: use Ingress (one LB for all services, routing by URL/host).
```

**ExternalName:** Maps a service to an external DNS name. No proxying — just returns a CNAME in DNS. Used to abstract external dependencies.

```yaml
spec:
  type: ExternalName
  externalName: my-database.rds.amazonaws.com
# Pods can use: my-db-service.namespace.svc.cluster.local
# Gets resolved to: my-database.rds.amazonaws.com
# Useful for migrating: gradually move from external to internal DB without code change
```

| Type         | Access scope     | Use case                               |
|--------------|------------------|----------------------------------------|
| ClusterIP    | Cluster internal | Microservice-to-microservice           |
| NodePort     | Node IP + port   | Dev/test, on-premises                  |
| LoadBalancer | Public internet  | Single service internet exposure       |
| ExternalName | DNS alias        | Abstract external dependency (DB, API) |

---

## Medium (Q8–Q15)

---

### Q8. How does Kubernetes scheduling work?

The **Kubernetes scheduler** decides which node a pod runs on. It is a control loop: for every unscheduled pod, it finds the best node based on resources and constraints.

**Scheduling pipeline:**
```
New Pod created (pending) →
  1. Filtering: eliminate nodes that cannot run the pod
     - Not enough CPU/memory (resource fit)
     - Node has taint that pod doesn't tolerate
     - Node not in required zone
     - PodAntiAffinity conflicts with existing pods
  
  2. Scoring: rank remaining feasible nodes
     - LeastRequestedPriority: prefers nodes with more free resources
     - BalancedResourceAllocation: balance CPU/memory usage
     - NodeAffinity: prefer nodes matching affinity rules
     - ImageLocality: prefer nodes that already have the container image
  
  3. Bind: scheduler assigns pod to highest-scored node
     Node kubelet detects assignment → pulls image → starts container
```

**Node selection control mechanisms:**

**Node Affinity (soft/hard preference):**
```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:  # Hard requirement
      nodeSelectorTerms:
      - matchExpressions:
        - {key: topology.kubernetes.io/zone, operator: In, values: [us-east-1a]}
    preferredDuringSchedulingIgnoredDuringExecution:  # Soft preference
    - weight: 100
      preference:
        matchExpressions:
        - {key: node-type, operator: In, values: [high-memory]}
```

**Pod Anti-Affinity (spread across nodes/AZs):**
```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels: {app: my-app}
      topologyKey: kubernetes.io/hostname  # No 2 pods on same node
      # Or: topology.kubernetes.io/zone   # No 2 pods in same AZ
```

**Taints and Tolerations:**
```yaml
# Node taint: only pods that tolerate this can schedule here
kubectl taint nodes gpu-node-1 gpu=true:NoSchedule

# Pod toleration: allow scheduling on tainted node
tolerations:
- key: gpu
  operator: Equal
  value: "true"
  effect: NoSchedule
```

**TopologySpreadConstraints** (preferred over anti-affinity for spreading):
```yaml
topologySpreadConstraints:
- maxSkew: 1                              # Allow at most 1 pod difference between zones
  topologyKey: topology.kubernetes.io/zone
  whenUnsatisfiable: DoNotSchedule
  labelSelector:
    matchLabels: {app: my-app}
```

---

### Q9. How does Horizontal Pod Autoscaler (HPA) work?

The **HPA** automatically adjusts the number of pod replicas based on observed metrics (CPU, memory, or custom metrics) to maintain a target utilization.

**HPA control loop:**
```
Every 15 seconds:
  1. Fetch current metric: current CPU utilization = 80%
  2. Calculate desired replicas:
     desired = ceil(current_replicas × (current_metric / target_metric))
     desired = ceil(4 × (80% / 50%)) = ceil(6.4) = 7
  3. Apply bounds: min(maxReplicas, max(minReplicas, desired)) = 7
  4. Scale deployment to 7 replicas
```

**HPA YAML definition:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60  # Target 60% CPU across all pods
  - type: Resource
    resource:
      name: memory
      target:
        type: AverageValue
        averageValue: 400Mi
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # Wait 60s before scaling up again
      policies:
      - type: Percent
        value: 100                     # Double pods max in one step
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scale-in (prevent flapping)
      policies:
      - type: Pods
        value: 1                       # Remove at most 1 pod per minute
        periodSeconds: 60
```

**Custom metrics HPA (Prometheus):**
```yaml
# Scale on requests per second instead of CPU
metrics:
- type: External
  external:
    metric:
      name: http_requests_per_second
      selector:
        matchLabels: {service: my-app}
    target:
      type: AverageValue
      averageValue: 100  # Target 100 RPS per pod
```

**Prerequisite: metrics-server** must be running in the cluster. For custom metrics, deploy KEDA (Kubernetes Event-Driven Autoscaling) for scaling on queue length, RPS, or any Prometheus metric.

---

### Q10. What is the difference between resource requests and limits and what is CPU throttling?

**Requests:** The minimum resources guaranteed to the container. Used by the scheduler to find a node with sufficient capacity.

**Limits:** The maximum resources the container can use. Enforced by the kubelet via cgroups.

```yaml
resources:
  requests:
    cpu: "250m"     # 0.25 CPU cores guaranteed
    memory: "256Mi" # 256MB RAM guaranteed (for scheduling)
  limits:
    cpu: "1000m"    # 1 CPU core maximum
    memory: "512Mi" # 512MB RAM maximum (OOM kill if exceeded)
```

**CPU throttling — the subtle trap:**
Unlike memory (which causes OOM kill), CPU limits cause **throttling**: the container's CPU is rate-limited to the limit value even if the node has free CPU.

```
Node has 8 CPUs (mostly idle):
  Container limit: 500m (0.5 CPU)
  Container wants: 800m (spiky workload)
  
  Linux cgroup enforces: in any 100ms window, container gets max 50ms of CPU
  If workload needs 80ms of CPU in 100ms → throttled to 50ms → 60% slower!
  
  Symptoms: high p99 latency with low CPU utilization reading
  Cause: CPU throttling — container is starved despite available capacity

Detection:
  kubectl top pods → check CPU usage vs limits
  Prometheus: container_cpu_throttled_seconds_total
```

**Memory over-limit behavior:**
```
Memory limit: 512Mi
Container tries to allocate 600Mi → OOMKilled (immediate restart)

Pod status: OOMKilled
kubectl describe pod → "Last State: OOMKilled, Exit Code: 137"
Fix: increase memory limit or find the memory leak
```

**QoS classes (determines eviction priority):**
```
Guaranteed:  requests == limits for all containers → last to be evicted
Burstable:   requests < limits → evicted after BestEffort
BestEffort:  no requests or limits set → first to be evicted under pressure
```

**Best practice:** Always set requests (required for scheduling). Set limits carefully — CPU limits cause throttling; consider not setting CPU limits in some cases and relying on namespace quotas instead.

---

### Q11. How do ConfigMaps and Secrets work in Kubernetes?

**ConfigMaps** store non-sensitive configuration data as key-value pairs. **Secrets** store sensitive data (passwords, tokens, TLS certs) — base64-encoded by default, with optional encryption at rest.

**ConfigMap:**
```yaml
# Define ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_HOST: "postgres-service"
  LOG_LEVEL: "info"
  config.yaml: |
    server:
      port: 8080
      timeout: 30s
```

**Secret:**
```yaml
# Define Secret (base64-encoded values)
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  DATABASE_PASSWORD: cGFzc3dvcmQxMjM=  # base64("password123")
  API_KEY: c2VjcmV0a2V5            # base64("secretkey")
```

**Consuming in a Pod:**
```yaml
spec:
  containers:
  - name: my-app
    image: my-app:v1
    # Option 1: env vars from ConfigMap/Secret
    envFrom:
    - configMapRef:
        name: app-config
    - secretRef:
        name: app-secrets
    
    # Option 2: mount as files
    volumeMounts:
    - name: config-vol
      mountPath: /etc/config
    - name: secrets-vol
      mountPath: /etc/secrets
      readOnly: true
  
  volumes:
  - name: config-vol
    configMap:
      name: app-config
  - name: secrets-vol
    secret:
      secretName: app-secrets
      defaultMode: 0400  # Read-only by owner only
```

**Volume mounts vs env vars:**
- **Volume mount:** Config changes reflected without pod restart (when using optional watch)
- **Env vars:** Injected at pod start, require restart to pick up changes

**Security best practices for Secrets:**
- Enable etcd encryption at rest (EncryptionConfiguration)
- Use external secret managers: AWS Secrets Manager via External Secrets Operator, Vault via Agent Injector
- Restrict Secret access with RBAC (serviceaccount only gets the secrets it needs)
- Never commit base64-encoded secrets to Git (use sealed-secrets or SOPS)

---

### Q12. How do rolling updates work in Kubernetes? Explain maxSurge and maxUnavailable.

A **rolling update** gradually replaces old pods with new ones, ensuring the service remains available throughout the deployment. Kubernetes handles this automatically when a Deployment's image is updated.

**Update process:**
```
Initial state: 4 pods running v1
  [v1][v1][v1][v1]

Update triggered: kubectl set image deploy/my-app my-app=my-app:v2

Rolling update (maxSurge=1, maxUnavailable=0):
  Step 1: Start 1 new pod (surge): [v1][v1][v1][v1][v2-starting]
  Step 2: v2 passes readiness probe
  Step 3: Terminate 1 old pod:     [v1][v1][v1]    [v2-ready]
  Step 4: Start 1 new pod:         [v1][v1][v1]    [v2][v2-starting]
  Step 5: Repeat until all replaced
  Final:  [v2][v2][v2][v2]
  
  At no point are fewer than 4 pods serving traffic (maxUnavailable=0)
  At most 5 pods exist at once (4 + 1 surge)
```

**maxSurge and maxUnavailable:**
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1        # Allow 1 extra pod above desired count during update
    maxUnavailable: 0  # Never reduce below desired count (zero downtime)

# maxSurge: 25% (default) means scale to 125% of desired during update
# maxUnavailable: 25% (default) allows 75% capacity during update

# For zero downtime: maxUnavailable: 0
# For faster rollout (sacrifice some availability): maxUnavailable: 1
# For resource-constrained clusters: maxSurge: 0, maxUnavailable: 1
#   → terminates old before starting new (no extra capacity needed, brief reduction)
```

**Readiness probe is critical:**
```
Without readiness probe:
  K8s marks v2 pod Running → removes v1 pod → v2 is still initializing
  Users get 503 errors for 10-30 seconds

With readiness probe:
  K8s only routes traffic to v2 pod after /ready returns 200
  v1 pod not removed until v2 is confirmed ready
  → Zero downtime guaranteed
```

**Rollback:**
```bash
# Immediate rollback to previous version
kubectl rollout undo deployment/my-app

# Rollback to specific revision
kubectl rollout undo deployment/my-app --to-revision=3

# Check rollout status
kubectl rollout status deployment/my-app

# View rollout history
kubectl rollout history deployment/my-app
```

---

### Q13. What is the Kubernetes network model?

Kubernetes requires a flat network model: every pod gets a unique IP address and can communicate with any other pod without NAT, regardless of which node they're on.

**Core requirements:**
```
1. Every pod has a unique cluster-wide IP
2. Pods on the same node can communicate directly
3. Pods on different nodes can communicate without NAT
4. Agents on a node can communicate with all pods on that node
```

**How it works:**

```
Node 1 (10.0.1.1):                  Node 2 (10.0.2.1):
  Pod A (10.244.1.1)                  Pod C (10.244.2.1)
  Pod B (10.244.1.2)                  Pod D (10.244.2.2)
  
  veth pair: Pod A ↔ cni0 bridge
  cni0 bridge ↔ eth0 (node NIC)
  
  Pod A (10.244.1.1) → Pod C (10.244.2.1):
  1. Pod A → veth → bridge → node routing table
  2. Node 1 routing: 10.244.2.0/24 → tunnel/overlay to Node 2
  3. Node 2 receives packet → routing table → bridge → Pod C
```

**CNI Plugins (Container Network Interface):**

| Plugin  | Mechanism         | Features                                  |
|---------|-------------------|-------------------------------------------|
| Flannel | VXLAN overlay     | Simple, no encryption, moderate perf      |
| Calico  | BGP routing       | Network policies, high performance, encryption optional |
| Cilium  | eBPF              | Best performance, L7 policies, observability |
| WeaveNet| Mesh overlay      | Easy setup, encryption built-in           |

**Network Policies (firewall rules):**
```yaml
# Allow only pods with label app=frontend to reach backend on port 8080
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
spec:
  podSelector:
    matchLabels: {app: backend}   # Apply to backend pods
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels: {app: frontend}
    ports:
    - protocol: TCP
      port: 8080
# All other ingress traffic to backend is blocked
```

Without network policies, all pods can communicate with all other pods (open by default). Network policies are enforced by the CNI plugin.

---

### Q14. How do you design stateful applications in Kubernetes with StatefulSets?

**StatefulSets** are the Kubernetes workload type for stateful applications (databases, queues, distributed systems) that need stable identity, ordered deployment, and persistent storage.

**Differences from Deployment:**
```
Deployment (stateless):
  Pods: my-app-7f9d8c-xvzk2 (random name)
  Scale up: any order, random names
  Scale down: any pod removed first
  Storage: ephemeral or shared PVC

StatefulSet (stateful):
  Pods: mysql-0, mysql-1, mysql-2 (stable, ordered names)
  Scale up: mysql-0 → mysql-1 → mysql-2 (in order, each must be ready)
  Scale down: mysql-2 → mysql-1 (reverse order, graceful)
  Storage: each pod gets its own PVC (mysql-0 → data-mysql-0)
```

**StatefulSet manifest (Redis Cluster):**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: redis-headless  # Headless service for stable DNS
  replicas: 3
  selector:
    matchLabels: {app: redis}
  template:
    spec:
      containers:
      - name: redis
        image: redis:7.0
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:             # Each pod gets its own PVC
  - metadata:
      name: data
    spec:
      accessModes: [ReadWriteOnce]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 10Gi
```

**Headless Service (stable DNS for StatefulSets):**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-headless
spec:
  clusterIP: None  # Headless: no VIP, returns pod IPs directly
  selector: {app: redis}
  ports:
  - port: 6379

# DNS records created:
# redis-0.redis-headless.default.svc.cluster.local → 10.244.1.5
# redis-1.redis-headless.default.svc.cluster.local → 10.244.2.3
# redis-2.redis-headless.default.svc.cluster.local → 10.244.3.7
#
# Pods know their own identity: redis-0, redis-1, redis-2
# Critical for cluster initialization (each node knows its peers)
```

**Init containers for cluster bootstrap:**
```yaml
initContainers:
- name: init-cluster
  image: redis:7.0
  command: ["sh", "-c", "until redis-cli -h redis-0.redis-headless ping; do sleep 1; done"]
  # Wait for redis-0 (primary) before starting this node
```

---

### Q15. What is a service mesh in Kubernetes and how does Istio work?

A **service mesh** is an infrastructure layer that handles service-to-service communication within Kubernetes — providing observability, security (mTLS), traffic management, and resilience without changing application code.

**Why a service mesh:**
```
Without service mesh:
  Each microservice implements its own:
  - Retry logic
  - Circuit breakers
  - mTLS certificates
  - Distributed tracing
  - Load balancing
  
  Duplicated in every service, in different languages, inconsistently.

With service mesh (Istio):
  All of the above handled uniformly by the mesh layer, transparently.
  Applications just send plain HTTP — mesh handles everything.
```

**Istio architecture:**
```
Control Plane (istiod):
  - Pilot: service discovery, traffic rules
  - Citadel: certificate authority (mTLS certs)
  - Galley: config validation

Data Plane (Envoy sidecar in every pod):
  ┌─────────────────────────────────────────┐
  │  Pod                                    │
  │  ┌──────────────┐  ┌────────────────┐  │
  │  │  App Container│  │ Envoy Sidecar  │  │
  │  │  (port 8080) │  │ (port 15001)   │  │
  │  └──────────────┘  └────────────────┘  │
  │  All traffic intercepted by Envoy via iptables │
  └─────────────────────────────────────────┘
```

**Sidecar injection:**
```yaml
# Auto-inject Envoy sidecar into all pods in namespace
kubectl label namespace production istio-injection=enabled

# Istio then automatically adds Envoy container to every pod
# No application code change required
```

**Traffic splitting for canary deployments:**
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: my-app-canary
spec:
  hosts: [my-app]
  http:
  - match:
    - headers:
        x-canary: {exact: "true"}   # Internal testers get v2
    route:
    - destination: {host: my-app, subset: v2}
  - route:
    - destination: {host: my-app, subset: v1}  # 97% of traffic
      weight: 97
    - destination: {host: my-app, subset: v2}  # 3% canary
      weight: 3
```

**mTLS (mutual TLS) enforcement:**
```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: require-mtls
  namespace: production
spec:
  mtls:
    mode: STRICT  # Only accept mTLS traffic — rejects plain HTTP between services
```

---

## Hard (Q16–Q20)

---

### Q16. What is Kubernetes RBAC and how do you implement least privilege for workloads?

**RBAC (Role-Based Access Control)** governs which users and services can perform which actions on which Kubernetes resources. The principle of least privilege means granting only the minimum permissions necessary.

**RBAC building blocks:**
```
Role/ClusterRole: what actions are allowed on what resources
RoleBinding/ClusterRoleBinding: who gets the role
ServiceAccount: the identity of a workload (pod)

Role (namespace-scoped):
  allows: get/list/watch pods in namespace "production"

ClusterRole (cluster-scoped):
  allows: get/list nodes cluster-wide

RoleBinding: grants Role to a ServiceAccount in a namespace
ClusterRoleBinding: grants ClusterRole to a user/SA cluster-wide
```

**Example: order service should only read order records:**
```yaml
# 1. Create a ServiceAccount for the order service
apiVersion: v1
kind: ServiceAccount
metadata:
  name: order-service-sa
  namespace: production

---
# 2. Create a Role with minimal permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: order-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list"]         # Read config only
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["order-db-secret"]  # Only THIS specific secret
  verbs: ["get"]

---
# 3. Bind the role to the service account
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: order-service-binding
  namespace: production
subjects:
- kind: ServiceAccount
  name: order-service-sa
  namespace: production
roleRef:
  kind: Role
  name: order-reader
  apiGroup: rbac.authorization.k8s.io

---
# 4. Use the ServiceAccount in the Deployment
spec:
  serviceAccountName: order-service-sa  # Pod runs as this identity
```

**Common RBAC mistakes:**
```
1. Using default ServiceAccount (has no extra permissions — fine)
   But: many Helm charts grant cluster-admin to default SA → bad

2. Granting cluster-admin to application pods → any RCE = cluster takeover

3. Wildcard permissions:
   rules:
   - apiGroups: ["*"]    # BAD: all API groups
     resources: ["*"]    # BAD: all resources
     verbs: ["*"]        # BAD: all verbs

4. Not restricting by namespace — ClusterRole gives access everywhere
```

**Audit RBAC:**
```bash
# What can this service account do?
kubectl auth can-i list pods --as=system:serviceaccount:production:order-service-sa

# Show all cluster-admin bindings (potential over-privilege)
kubectl get clusterrolebindings -o json | \
  jq '.items[] | select(.roleRef.name == "cluster-admin") | .metadata.name'
```

---

### Q17. How does GitOps work with ArgoCD or Flux?

**GitOps** is an operational model where Git is the single source of truth for both application code and infrastructure/Kubernetes configuration. Changes to the system are made via Git commits and pull requests — never with `kubectl apply` directly.

**GitOps principles:**
```
1. Declarative: all desired state defined in Git (YAML manifests, Helm charts, Kustomize)
2. Versioned: Git provides history, audit trail, and rollback capability
3. Automatically applied: a controller reconciles actual vs. desired state continuously
4. Pull-based: the cluster pulls from Git (vs. push-based CI/CD that pushes to cluster)
```

**ArgoCD architecture:**
```
Git Repository (desired state):
  /apps/
    production/
      deployment.yaml    ← what should be running
      service.yaml
      ingress.yaml

ArgoCD controller (runs in cluster):
  Watches Git repo for changes
  Compares Git state vs actual cluster state
  If drift detected: applies changes (or alerts if manual sync required)
  
  Dashboard shows: Synced / OutOfSync / Healthy / Degraded per app
```

**Application definition:**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/k8s-configs
    targetRevision: main
    path: apps/production/my-app        # Directory with K8s manifests
    # Or Helm:
    # chart: my-app
    # helm:
    #   values: |
    #     image.tag: v1.2.3
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true       # Delete resources removed from Git
      selfHeal: true    # Re-apply if someone manually changes cluster
    syncOptions:
    - CreateNamespace=true
```

**GitOps workflow:**
```
Developer → PR: update image tag from v1.1 to v1.2
Team reviews → Merge to main
ArgoCD detects change in Git → Syncs to cluster → Rolls out new version

Rollback:
  git revert → new commit → ArgoCD syncs → cluster rolls back
  Clear audit trail: who changed what, when, and why (PR description)
```

**Flux vs ArgoCD:**
| Feature        | ArgoCD                          | Flux                              |
|----------------|---------------------------------|-----------------------------------|
| UI             | Rich web dashboard              | CLI-first (no built-in UI)        |
| Multi-cluster  | Built-in                        | Good support                      |
| Notification   | Via Argo Notifications          | Via Flux Notification controller  |
| Helm support   | Excellent (Helm repos, OCI)     | Excellent                         |
| Image automation| ArgoCD Image Updater             | Flux Image Automation (built-in)  |

---

### Q18. What are Pod Disruption Budgets (PDB) and how do they maintain availability during node drain?

A **Pod Disruption Budget (PDB)** limits the number of pods that can be simultaneously unavailable for a deployment. It protects availability during voluntary disruptions: node drains (upgrades, scaling down), cluster maintenance, and pod evictions.

**The problem without PDB:**
```
Deployment: 4 replicas (minimum needed: 3 for availability)

kubectl drain node-1 node-2 simultaneously (cluster upgrade):
  Node-1 drain: evicts 2 pods → deployment has 2 pods (BELOW minimum!)
  Node-2 drain: evicts 2 more pods → deployment has 0 pods! (outage)
  
Without PDB: Kubernetes allows draining as many nodes simultaneously as you request
```

**PDB definition:**
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: my-app-pdb
spec:
  selector:
    matchLabels: {app: my-app}
  # Option A: minimum available pods
  minAvailable: 3         # Always keep at least 3 pods running
  
  # Option B: max unavailable
  # maxUnavailable: 1     # Allow at most 1 pod disrupted at a time
  # (pick one, not both)
```

**How PDB is enforced:**
```
With PDB (minAvailable: 3), deployment has 4 replicas:

kubectl drain node-1 node-2:
  Node-1 drain → tries to evict pod-1:
    Check PDB: currently 4 running, minAvailable=3, can evict? YES
    Evict pod-1 → 3 pods remaining
  
  Node-2 drain → tries to evict pod-2:
    Check PDB: currently 3 running, minAvailable=3, can evict? NO
    Drain BLOCKED — waits until rescheduled pod starts on another node
    
  Once pod-1 reschedules on node-3: now 4 pods running again
  Node-2 drain resumes: evicts pod-2 → 3 pods remaining
  
  Result: availability never drops below 3 pods
```

**PDB for stateful sets (databases):**
```yaml
# For a 3-node Cassandra ring: never take more than 1 node down at once
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: cassandra-pdb
spec:
  selector:
    matchLabels: {app: cassandra}
  maxUnavailable: 1  # Only 1 Cassandra pod can be unavailable at any time
                     # Prevents cluster going below replication factor - 1
```

**PDB and HPA interaction:**
If PDB minAvailable equals the HPA minimum replicas, scaling down could be blocked. Set PDB minAvailable slightly below HPA minReplicas to allow scale-in.

---

### Q19. How do you architect multi-cluster Kubernetes and when should you split into multiple clusters?

**Single cluster** is the default starting point. Multiple clusters add significant operational complexity and should be justified by concrete requirements.

**Reasons to split into multiple clusters:**

```
1. Blast radius isolation:
   Production cluster ← completely separate from dev/staging
   A bad deploy to staging cannot crash production

2. Compliance / data residency:
   EU cluster (Frankfurt) ← EU customer data stays in EU (GDPR)
   US cluster (Virginia)  ← US customer data

3. Team autonomy:
   Platform team cluster: shared services (monitoring, logging)
   Team A cluster: their microservices
   Team B cluster: their microservices (no blast radius from Team B deploys)

4. Kubernetes version management:
   Cluster A: K8s 1.28 (stable)
   Cluster B: K8s 1.30 (testing new version, rolling upgrade)

5. Scaling limits:
   Single K8s cluster: scales to ~5000 nodes, 150,000 pods
   Beyond this: split to multiple clusters
```

**Multi-cluster architecture:**
```
                    [Fleet Manager: ArgoCD / Flux]
                              │
             ┌────────────────┼────────────────┐
             │                │                │
      [EU Cluster]    [US Cluster]    [APAC Cluster]
         EKS              EKS              EKS
      (eu-west-1)    (us-east-1)      (ap-northeast-1)
         │                │                │
      [EU DB]         [US DB]         [APAC DB]
      (RDS EU)        (RDS US)        (RDS APAC)
```

**Multi-cluster service discovery:**
```
Options:
  1. External DNS: each cluster has its own DNS, cross-cluster via external hostnames
  2. Istio multi-cluster: service mesh spans clusters (complex)
  3. Submariner: connects cluster networks via IPsec tunnels
  4. API Gateway: external clients go through central API GW that routes to clusters

Most common pattern:
  Each cluster is self-contained with its own databases
  Cross-cluster communication via public APIs (not internal mesh)
  ArgoCD ApplicationSet deploys same app to all clusters from one config
```

**ApplicationSet (ArgoCD) for multi-cluster deploy:**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: my-app-all-clusters
spec:
  generators:
  - clusters: {}    # Generates one Application per registered cluster
  template:
    spec:
      source:
        repoURL: https://github.com/myorg/k8s-configs
        path: apps/{{name}}  # Cluster-specific configs
      destination:
        server: {{server}}   # Each cluster's API server
        namespace: production
```

---

### Q20. How do you design a production-grade Kubernetes platform for a microservices architecture?

This integrative question requires synthesizing all Kubernetes concepts into a coherent production design.

**Requirements:**
- 20 microservices
- 99.99% availability
- Zero-downtime deployments
- 5 engineering teams (autonomy + governance)
- Mixed stateful and stateless services

**Platform architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│  EKS Cluster: us-east-1, Multi-AZ (3 AZs)                  │
│                                                             │
│  Node Groups:                                               │
│    system-nodes:    t3.medium × 3 (1 per AZ) [On-Demand]  │
│    app-nodes:       m5.xlarge × 6 (On-Demand: 2 per AZ)   │
│                     m5.xlarge × 12 (Spot: 4 per AZ)        │
│    memory-nodes:    r5.2xlarge × 3 (DB-heavy services)     │
│                                                             │
│  Namespaces (team isolation):                               │
│    kube-system      → K8s system pods                       │
│    monitoring       → Prometheus, Grafana, Jaeger           │
│    team-payments    → Payment microservices                 │
│    team-catalog     → Product catalog services              │
│    team-orders      → Order management services             │
│    team-auth        → Auth/identity services                │
│    team-platform    → Shared infra (Redis, etc.)            │
└─────────────────────────────────────────────────────────────┘
```

**Team isolation via RBAC + ResourceQuota:**
```yaml
# Each team gets their namespace with resource limits
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-quota
  namespace: team-payments
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "3"
```

**Observability stack:**
```
Metrics:    Prometheus (scrapes pods) → Grafana dashboards
Tracing:    Jaeger / Tempo (via Istio sidecar tracing)
Logging:    Fluent Bit (DaemonSet) → Elasticsearch / CloudWatch
Alerting:   Alertmanager → PagerDuty
SLO tracking: Pyrra or Sloth (generates Prometheus recording rules for SLOs)
```

**Deployment pipeline (GitOps):**
```
Developer writes code → PR → Code review → Merge to main
CI (GitHub Actions):
  1. Build image → push to ECR
  2. Update image tag in k8s-configs repo (automated PR)
  
ArgoCD:
  3. Detects config change
  4. Runs health checks: kube-score, Conftest (OPA policy checks)
  5. Syncs to cluster: rolling update
  6. Monitors: if error rate spikes → rollback automatically (Argo Rollouts)
```

**Production readiness checklist:**
```
For each microservice:
  ✓ Deployment with 3+ replicas across AZs (TopologySpreadConstraints)
  ✓ Readiness + liveness probes configured
  ✓ PodDisruptionBudget: minAvailable = replicas - 1
  ✓ HPA: CPU 60%, memory 70%, custom metrics where applicable
  ✓ Resource requests and limits set (right-sized)
  ✓ ServiceAccount with least-privilege RBAC
  ✓ NetworkPolicy: default-deny, explicit allow rules
  ✓ Secrets via External Secrets Operator (from AWS Secrets Manager)
  ✓ Liveness/readiness probes pointing to meaningful health endpoints
  ✓ Graceful shutdown (preStop hook, SIGTERM handling)
  ✓ Horizontal and vertical scaling tested under load
  ✓ ArgoCD application defined, sync policies configured
```

**Cluster upgrades (zero downtime):**
```
Strategy: Rolling node group replacement
1. Provision new node group with new K8s version
2. Taint old nodes: NoSchedule (new pods go to new nodes)
3. kubectl drain old nodes one by one (PDBs protect availability)
4. Old pods reschedule to new nodes
5. Delete old node group

Total downtime: 0 (PDBs ensure minimum pods always running)
Duration: 2-4 hours for large cluster
```

---

## Quick Reference

### Container vs VM
| Dimension    | Container      | VM               |
|--------------|----------------|------------------|
| Startup      | Milliseconds   | Minutes          |
| Size         | MB             | GB               |
| Isolation    | OS-level       | Hardware-level   |
| Density      | 100-1000/host  | 10-50/host       |

### Kubernetes Object Hierarchy
`Deployment → ReplicaSet → Pod → Container`

### Probe Types
- **Liveness:** container alive? Failure = restart
- **Readiness:** ready for traffic? Failure = remove from LB
- **Startup:** allow slow init before liveness kicks in

### Service Types
| Type          | Scope              |
|---------------|-------------------|
| ClusterIP     | Cluster-internal   |
| NodePort      | Node IP + port     |
| LoadBalancer  | Cloud LB (public)  |
| ExternalName  | DNS alias          |

### HPA Formula
`desired = ceil(current_replicas × current_metric / target_metric)`

### Rolling Update Parameters
- `maxSurge: 1` — extra pods during update
- `maxUnavailable: 0` — zero-downtime guarantee

### RBAC Principle
`ServiceAccount → RoleBinding → Role (namespace) / ClusterRole (cluster-wide)`

### StatefulSet vs Deployment
| Feature        | Deployment       | StatefulSet         |
|----------------|------------------|---------------------|
| Pod names      | Random           | Ordered (pod-0, 1)  |
| Scaling order  | Any              | Sequential          |
| Storage        | Shared PVC       | Per-pod PVC         |
| Use case       | Stateless apps   | Databases, queues   |

### GitOps Key Points
- Git = single source of truth
- PRs for all changes (audit trail)
- ArgoCD reconciles desired vs actual state continuously
- Rollback = git revert
