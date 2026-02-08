# Model Deployment and MLOps

## Q1: What are key ML system design principles?

**A1:** ML system design principles include: reliability (handling failures gracefully), scalability (handling increased load), maintainability (easy updates and debugging), monitoring (tracking performance and data quality), versioning (models, data, code), reproducibility (consistent results), security (protecting models and data), and modularity (loose coupling between components). Systems should be designed for change, as models degrade and require updates. Infrastructure should support experimentation and production deployment. Design considers both ML-specific challenges and general software engineering best practices.

## Q2: Compare batch vs real-time model serving.

**A2:** Batch serving processes predictions in scheduled jobs on large datasets, suitable for non-urgent predictions and when throughput matters more than latency. Real-time serving provides predictions on-demand with low latency (milliseconds), necessary for user-facing applications. Batch serving is simpler, more cost-effective, and handles large volumes efficiently. Real-time serving requires infrastructure for low-latency inference, load balancing, and auto-scaling. Hybrid approaches use batch for bulk processing and real-time for urgent requests. The choice depends on latency requirements, prediction frequency, and cost constraints.

## Q3: How do you create REST APIs for model serving?

**A3:** REST APIs for models expose endpoints accepting input data and returning predictions. Frameworks like Flask and FastAPI simplify API creation. APIs should validate inputs, handle errors gracefully, include versioning, and provide health checks. Input validation ensures data format and ranges are correct. Error handling returns meaningful messages without exposing internals. API versioning allows model updates without breaking clients. Health checks enable monitoring and load balancer integration. APIs should be stateless, documented (OpenAPI/Swagger), and include authentication/authorization for production.

## Q4: Explain model serialization formats.

**A4:** Model serialization saves trained models for later loading. Formats include: pickle (Python-native, simple but version-dependent), ONNX (open standard, framework-agnostic, optimized for inference), TorchScript (PyTorch's format, enables C++ deployment), TensorFlow SavedModel (TensorFlow standard), and PMML (XML-based, limited ML library support). Pickle is convenient but ties models to Python versions. ONNX enables cross-framework deployment and hardware optimization. Format choice depends on deployment environment, performance requirements, and framework compatibility. Some formats enable quantization and optimization.

## Q5: How is Docker used for ML deployment?

**A5:** Docker containerizes ML applications with dependencies, ensuring consistent environments across development and production. Dockerfiles specify base images, install dependencies, copy code, and define entry points. Containers package models, inference code, and runtime together. Benefits include reproducibility, isolation, easy scaling, and cloud deployment compatibility. ML-specific considerations: large model files, GPU support (nvidia-docker), and optimized base images. Multi-stage builds reduce image size. Docker Compose orchestrates multi-container applications. Containers enable microservices architecture and simplify deployment pipelines.

## Q6: Explain Kubernetes for ML workloads.

**A6:** Kubernetes orchestrates containerized ML applications, managing deployment, scaling, and resource allocation. It handles rolling updates, health checks, auto-scaling based on load, and service discovery. For ML: supports GPU scheduling, handles large model storage (persistent volumes), enables canary deployments, and manages secrets for API keys. Kubernetes abstractions (Deployments, Services, Ingress) simplify ML service management. Horizontal Pod Autoscaler scales based on CPU/memory/custom metrics. Resource quotas prevent resource contention. Kubernetes is complex but provides production-grade orchestration for ML at scale.

## Q7: What is model monitoring and why is it critical?

**A7:** Model monitoring tracks model performance, data quality, and system health in production. It detects data drift (input distribution changes), concept drift (target relationship changes), performance degradation, and infrastructure issues. Monitoring includes prediction distributions, error rates, latency metrics, and data quality checks. Without monitoring, models silently degrade, causing business impact. Monitoring enables proactive retraining and rollback. Key metrics: prediction distributions, accuracy/error rates, latency percentiles, and data statistics. Alerts trigger when metrics exceed thresholds. Monitoring is essential for maintaining production ML systems.

## Q8: Explain A/B testing in ML production.

**A8:** A/B testing compares model versions by splitting traffic between control (current) and treatment (new) models, measuring business metrics. It provides statistical evidence of improvement before full rollout. Key components: random assignment, sufficient sample size, appropriate metrics, and statistical significance testing. A/B tests should run long enough to capture different conditions but not so long that models drift. Considerations include user experience consistency and ethical implications. A/B testing validates that new models improve outcomes before replacing existing ones, reducing risk of deploying worse models.

## Q9: What are shadow deployment and canary releases?

**A9:** Shadow deployment runs new models alongside production, logging predictions without affecting users, enabling validation without risk. Canary releases gradually roll out new models to increasing percentages of traffic (e.g., 1%, 5%, 50%, 100%), monitoring metrics at each stage. Shadow deployment validates model behavior on real data. Canary releases enable gradual rollout with automatic rollback if metrics degrade. Both reduce deployment risk compared to immediate full replacement. Canary releases provide real user feedback while limiting exposure. They're essential for high-stakes applications.

## Q10: What are feature stores and their benefits?

**A10:** Feature stores centralize feature computation and storage, providing consistent features for training and serving. They compute features once, store them, and serve to both training pipelines and inference services. Benefits include: consistency between training and serving, feature reuse across models, versioning and lineage tracking, and reduced computation duplication. Feature stores handle online features (real-time) and offline features (batch). They enable feature discovery and governance. Popular solutions include Feast, Tecton, and cloud provider offerings. Feature stores are crucial for production ML systems with multiple models.

## Q11: Explain ML pipeline orchestration tools.

**A11:** ML pipeline orchestration tools schedule and coordinate multi-step ML workflows (data ingestion, preprocessing, training, evaluation, deployment). Apache Airflow defines workflows as DAGs (Directed Acyclic Graphs) with Python, providing scheduling, monitoring, and retry logic. Kubeflow runs on Kubernetes, providing ML-specific components and serving capabilities. Other tools include Prefect, Luigi, and cloud-native solutions. Orchestration handles dependencies, parallelization, failure recovery, and scheduling. It enables reproducible, automated ML pipelines. Choice depends on infrastructure (cloud vs on-premise) and requirements (ML-specific vs general workflow).

## Q12: What is experiment tracking and why is it important?

**A12:** Experiment tracking logs hyperparameters, metrics, code versions, and artifacts for each training run, enabling comparison and reproducibility. Tools like MLflow and Weights & Biases provide interfaces for tracking experiments, comparing runs, and organizing projects. Tracking includes: hyperparameters, metrics (training/validation), code version (git commit), data version, model artifacts, and notes. It helps identify best configurations, understand what works, and reproduce results. Without tracking, experiments are lost and insights forgotten. Experiment tracking is essential for systematic ML development and collaboration.

## Q13: Explain model registries and their role.

**A13:** Model registries store trained models with metadata (version, metrics, lineage, stage), enabling model versioning, discovery, and deployment workflows. They track which models are in development, staging, and production. Registries integrate with CI/CD pipelines for automated deployment. They store model artifacts, metadata, and enable model comparison. Registries support approval workflows and access control. They're the source of truth for production models. MLflow Model Registry and cloud provider registries provide these capabilities. Registries enable governance and auditability of model deployments.

## Q14: What is CI/CD for ML?

**A14:** CI/CD for ML automates testing and deployment of ML code and models. Continuous Integration runs tests (unit, integration, data validation) on code changes. Continuous Deployment automatically deploys passing models to production. ML-specific challenges: model testing (performance thresholds), data validation, and longer training times. CI/CD pipelines include: code quality checks, data validation, model training, evaluation against thresholds, and deployment. Automated testing prevents bad models from reaching production. CI/CD enables rapid iteration while maintaining quality. It requires robust testing and monitoring to be effective.

## Q15: How do you optimize GPU serving for models?

**A15:** GPU serving optimization includes: batching requests to utilize GPU parallelism, using optimized inference engines (TensorRT, ONNX Runtime), model quantization (FP16, INT8), dynamic batching, and multi-instance GPU (MIG) for resource isolation. Batching amortizes GPU overhead across multiple requests. Optimized engines use kernel fusion and specialized operations. Quantization reduces memory and increases throughput with minimal accuracy loss. Dynamic batching groups requests arriving within a time window. GPU serving requires balancing latency and throughput, as batching increases latency but improves utilization.

## Q16: What is model compression for deployment?

**A16:** Model compression reduces model size and inference cost while maintaining acceptable accuracy. Techniques include: quantization (reducing precision from FP32 to FP16/INT8), pruning (removing unimportant weights), knowledge distillation (training smaller student model from larger teacher), and architecture search (finding efficient architectures). Quantization is most common, providing 2-4x speedup with minimal accuracy loss. Pruning can achieve high sparsity. Compression enables edge deployment and reduces serving costs. Trade-offs exist between compression ratio, accuracy loss, and implementation complexity. Compression is essential for resource-constrained deployments.

## Q17: What are considerations for edge deployment?

**A17:** Edge deployment considerations include: limited compute (CPU/memory constraints), power consumption (battery-powered devices), network connectivity (offline capability), model size (storage limits), latency requirements (real-time inference), and security (on-device processing). Edge deployment requires compressed models, optimized inference engines, and potentially specialized hardware (NPUs, TPUs). Models must handle device-specific constraints and variations. Updates are challenging due to limited connectivity. Edge deployment enables low-latency inference, privacy (data stays on device), and reduced cloud costs. It's essential for IoT, mobile, and real-time applications.

## Q18: What should you monitor in production ML systems?

**A18:** Production monitoring should track: prediction distributions (detect data drift), model performance metrics (accuracy, error rates), prediction latency (p50, p95, p99), system metrics (CPU, memory, GPU utilization), data quality (missing values, ranges, schema), error rates and types, and business metrics (if available). Monitoring requires establishing baselines and setting alerts for anomalies. Dashboards visualize trends over time. Monitoring should cover both ML-specific metrics (drift, performance) and infrastructure metrics (latency, errors). Effective monitoring enables proactive issue detection and resolution.

## Q19: Explain responsible AI considerations in production.

**A19:** Responsible AI in production includes: fairness (detecting and mitigating bias across demographic groups), explainability (providing model explanations for decisions), privacy (protecting user data, differential privacy), security (adversarial robustness, model theft prevention), and transparency (documenting model limitations and use cases). Monitoring should track fairness metrics across groups. Models should provide explanations when possible. Privacy-preserving techniques protect sensitive data. Security measures prevent adversarial attacks and unauthorized access. Responsible AI requires ongoing evaluation and mitigation of risks throughout the ML lifecycle.

## Q20: What are key challenges in production ML systems?

**A20:** Key challenges include: data drift and concept drift requiring model updates, maintaining consistency between training and serving, scaling to handle variable load, debugging model failures (black-box nature), managing multiple model versions, ensuring low latency while maintaining accuracy, handling edge cases and adversarial inputs, and maintaining data quality pipelines. ML systems are more complex than traditional software due to data dependencies, non-deterministic behavior, and continuous degradation. Addressing these requires robust infrastructure, monitoring, and processes. Success requires both ML expertise and software engineering practices.
