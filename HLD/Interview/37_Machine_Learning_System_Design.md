# 37 — Machine Learning System Design

## Easy (Q1–Q7)

---

### Q1. How does ML system design differ from traditional system design?

Traditional software systems are **deterministic**: given the same input, they always produce the same output. ML systems are **probabilistic**: outputs are predictions with confidence scores, and the system's behavior changes over time as data distributions shift.

```
TRADITIONAL SYSTEM                 ML SYSTEM
──────────────────────────────────────────────────────
Input → Logic → Output             Input → Model → Prediction
Logic written by humans            Logic learned from data
Bugs fixed by changing code        Bugs fixed by retraining
Same input = same output           Same input = possibly different output
No data flywheel                   More data → better model
Correctness is binary              Correctness is a distribution
No data management needed          Data pipeline is critical infra
Deployment = code release          Deployment = code + model artifact
```

**Key additional concerns in ML systems:**

| Concern | Description |
|---|---|
| Data quality | Garbage in → garbage predictions |
| Training-serving skew | Features computed differently at training vs serving time |
| Model drift | Production data changes; model becomes stale |
| Reproducibility | Can you retrain and get the same model? |
| Experiment tracking | Which hyperparameters/data produced which model? |
| Model monitoring | Is the model's accuracy degrading in production? |
| Feature store | Consistent feature computation across training and serving |

**Interviewer focus:** In an ML system design interview, you must address **all** of: data ingestion, feature engineering, training pipeline, model evaluation, serving infrastructure, and monitoring. Traditional system design only covers the serving/storage layer.

---

### Q2. What are the core components of an ML system?

An ML system is not just a model. It is a complete data-to-prediction pipeline with many interconnected components.

```
ML SYSTEM COMPONENTS
──────────────────────────────────────────────────────────────────
                        ┌─────────────────┐
Raw Data Sources ──────►│  Data Ingestion  │ (Kafka, batch ETL)
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │Feature Engineering│ (Spark, Flink, dbt)
                        │  Feature Store   │ (Feast, Tecton)
                        └────────┬────────┘
                          ┌──────┴──────┐
                          │             │
                   ┌──────▼──┐   ┌──────▼──┐
                   │Training │   │ Serving  │
                   │Pipeline │   │  Path    │
                   │(offline)│   │(online)  │
                   └──────┬──┘   └──────┬──┘
                          │             │
                   ┌──────▼──┐   ┌──────▼──┐
                   │ Model   │   │Prediction│
                   │Registry │   │  API     │
                   │(MLflow) │   │          │
                   └─────────┘   └──────────┘
                          │
                   ┌──────▼──────────────┐
                   │ Monitoring & Alerts  │
                   │(accuracy, drift,     │
                   │ latency, skew)       │
                   └─────────────────────┘
```

**Component Responsibilities:**

**Data Ingestion:** Collect raw events, logs, user interactions from databases, Kafka streams, data lakes (S3/GCS). Handles schema evolution, data validation, late arrivals.

**Feature Engineering:** Transform raw data into ML features. Normalization, encoding, aggregation, embeddings. Critical concern: use the same transformation logic in training AND serving.

**Feature Store:** Centralized repository for computed features. Provides offline features (for training) and online features (for serving) with a unified API.

**Training Pipeline:** Orchestrated workflow (Airflow, Kubeflow Pipelines, Metaflow) that reads features, trains model, evaluates, and pushes to Model Registry.

**Model Registry:** Versioned storage for trained model artifacts with metadata (metrics, hyperparameters, training data version).

**Serving Path:** Real-time inference API or batch scoring job. Loads model from registry, fetches features from feature store, returns predictions.

**Monitoring:** Tracks prediction accuracy, feature distribution drift, latency, error rate, and business metrics.

---

### Q3. What is a feature store and why is it needed?

A **feature store** is a centralized platform for storing, computing, sharing, and serving machine learning features. It is the most important piece of infrastructure for preventing training-serving skew.

**The Problem Without a Feature Store:**
```
Data Scientist (training):
  user_avg_purchase_last_30d = df.groupby('user_id')['amount']
                                  .rolling('30D').mean()

Engineer (serving, written 3 months later):
  user_avg_purchase_last_30d = sum(purchases[-30:]) / len(purchases[-30:])
                                ↑ Different time window definition!
                                ↑ Different filtering of refunds!
                                ↑ Training and serving give DIFFERENT values!
                                → Model makes worse predictions in production
```

**Feature Store Architecture:**
```
                    ┌──────────────────────────┐
                    │       Feature Store        │
                    │                            │
Batch Pipeline ────►│  Offline Store (S3/HDFS)  │◄──── Training Jobs
(daily/hourly)      │  Historical features       │      read historical
                    │                            │      features for
Streaming ─────────►│  Online Store (Redis/DDB)  │      training
Pipeline            │  Latest feature values     │
(real-time)         └──────────────┬─────────────┘
                                   │
                            Serving API
                                   │
                         Model Inference Service
                         fetches latest features
                         at inference time
```

**Key Benefits:**
| Benefit | Description |
|---|---|
| No training-serving skew | Same feature computation code for both |
| Feature reuse | Feature computed once, used by many models |
| Point-in-time correctness | Historical features correct as of label timestamp |
| Consistency | All models use the same `user_avg_purchase_30d` definition |

**Popular implementations:** Feast (open-source), Tecton (managed), Hopsworks, AWS SageMaker Feature Store, Google Vertex AI Feature Store.

---

### Q4. What is the difference between online and offline feature computation?

**Offline features** are computed over historical data in batch, stored in a data warehouse, and used to build training datasets.
**Online features** are computed in real-time (or near-real-time), stored in a low-latency store (Redis), and used during model inference.

```
OFFLINE vs ONLINE FEATURE COMPUTATION
──────────────────────────────────────────────────────────────
                    OFFLINE                    ONLINE
──────────────────────────────────────────────────────────────
When computed    Daily/hourly batch          Per-event or at request time
Storage          S3, BigQuery, Hive          Redis, DynamoDB, Cassandra
Latency          Minutes–hours              < 5 ms
Used for         Training datasets           Real-time inference
Scale            Petabytes (Spark)           Low-latency reads
Example          "All user clicks last 90d"  "User's last 5 actions (live)"
Staleness        High (batch lag)            Low (< 1 minute)
```

**Dual Pipeline Pattern:**
```python
# Offline pipeline (Spark, runs daily)
def compute_offline_features(spark, date):
    purchases = spark.read.parquet(f"s3://data/purchases/{date}")
    user_features = purchases.groupBy("user_id").agg(
        F.sum("amount").alias("total_spend_30d"),
        F.count("*").alias("purchase_count_30d")
    )
    user_features.write.parquet(f"s3://features/user/{date}")

# Online pipeline (Flink/Kafka, runs continuously)
def on_purchase_event(event):
    user_id = event["user_id"]
    redis.incrbyfloat(f"user:{user_id}:total_spend_30d", event["amount"])
    redis.incr(f"user:{user_id}:purchase_count_30d")
    redis.expire(f"user:{user_id}:total_spend_30d", 30*24*3600)  # 30d TTL
```

**Trade-offs:**
- **Staleness vs Latency:** Online features are fresh but expensive to compute in real-time; offline features are cheap but stale.
- **Complexity:** Maintaining both pipelines with identical logic is hard. Feature stores manage this duplication.

---

### Q5. What are the model serving options and when do you choose each?

**Three primary serving patterns:**

**1. Real-Time Inference API (synchronous)**
```
User Request ──► Inference Service ──► Model ──► Prediction (< 100ms)
```
Use when: predictions must be fresh, latency matters, user is waiting.
Examples: fraud detection, search ranking, ad serving.

**2. Batch Scoring (asynchronous, pre-computed)**
```
Scheduled job ──► All users ──► Model ──► Predictions stored in DB
User request  ──► DB lookup   ──► Return pre-computed prediction
```
Use when: predictions can be computed ahead of time, latency requirements are < 10ms, user universe is known.
Examples: email campaign targeting, weekly product recommendations.

**3. Edge Inference (on-device)**
```
Model deployed to mobile device / browser
No network call needed; inference runs locally
```
Use when: latency < 10ms required, offline capability needed, privacy (data never leaves device).
Examples: face detection in camera app, keyboard autocorrect.

**Decision Matrix:**
```
Is the user waiting for the prediction?
      YES → Real-time inference API
      NO  →  Can we pre-compute for all users?
               YES → Batch scoring (cheaper, faster at request time)
               NO  → Real-time inference API
                      (e.g., predictions depend on live context)

Is latency < 10ms required AND privacy matters?
      YES → Edge inference
```

**Hybrid Pattern (most production systems):**
```
Pre-compute top-N recommendations in batch (most of the serving)
At request time, apply real-time re-ranking with live context
(e.g., recent clicks in this session, current inventory)
```

---

### Q6. What is training-serving skew and how do you prevent it?

**Training-serving skew** occurs when the features fed to the model during training are computed differently from the features fed to the model during inference. The model was trained on data it has never seen in the exact same form as production provides.

```
TRAINING-SERVING SKEW EXAMPLE
──────────────────────────────────────────────────────────────────
Training time (offline):
  age_normalized = (age - dataset.mean()) / dataset.std()
  # mean=35.2, std=12.1 from training dataset

Serving time (online):
  age_normalized = (age - 30) / 15
  # Hardcoded constants from memory, slightly wrong!
  # Result: model input is different from what it was trained on
  # Model accuracy degrades silently
```

**Causes:**
| Cause | Example |
|---|---|
| Feature recomputed with different logic | Different aggregation window |
| Different null handling | NaN → 0 in training, NaN → -1 in serving |
| Different data sources | Training used DB, serving uses cache with stale values |
| Time-leaking features | Future data accidentally included in training features |
| Preprocessing inconsistency | Tokenizer version mismatch |

**Prevention Strategies:**

**1. Feature Store (single source of truth):**
Both training and serving read from the same feature store using the same feature computation code.

**2. Shared Pipeline Code:**
```python
# Same transformation function used in BOTH training and serving
def compute_user_features(user_id: str, as_of_date: date) -> dict:
    purchases = get_purchases(user_id, end_date=as_of_date, days=30)
    return {
        "total_spend_30d": sum(p.amount for p in purchases),
        "purchase_count_30d": len(purchases),
        "avg_spend": sum(p.amount for p in purchases) / max(len(purchases), 1)
    }
# Training: calls this function for each (user, label_date) pair
# Serving:  calls this function with as_of_date=today
```

**3. Monitoring Skew in Production:**
```python
# Log training distribution statistics with the model
model.metadata["feature_stats"] = {
    "age": {"mean": 35.2, "std": 12.1, "min": 18, "max": 90}
}
# At serving time, compare incoming features to expected distribution
# Alert if serving distribution shifts significantly from training distribution
```

---

### Q7. What is model drift and data drift — how do you detect and handle them?

**Data drift** (covariate shift) means the distribution of input features has changed since the model was trained. **Model drift** (concept drift) means the relationship between features and labels has changed — what was predictive before is no longer predictive.

```
DATA DRIFT EXAMPLE:
  Training data (2023): users aged 25-35 dominant (social media app)
  Production data (2024): user base aged 45-60 (app went mainstream)
  → Feature distributions shifted; model performs poorly for new demographic

CONCEPT DRIFT EXAMPLE:
  Fraud model trained pre-COVID: travel transactions = low risk
  Post-COVID: travel transactions = high risk (new fraud patterns)
  → The relationship between features and fraud label changed
```

**Detection Methods:**

```python
# Statistical tests for data drift
from scipy import stats

def detect_drift(training_distribution: list, serving_distribution: list,
                 threshold: float = 0.05) -> bool:
    """KS test: p-value < threshold means distributions are different."""
    ks_statistic, p_value = stats.ks_2samp(training_distribution,
                                            serving_distribution)
    return p_value < threshold

# Population Stability Index (PSI)
def compute_psi(expected: np.array, actual: np.array, bins=10) -> float:
    """PSI > 0.2 indicates significant drift."""
    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=bins)[0] / len(actual)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

**Handling Drift:**
| Strategy | When to Use |
|---|---|
| Scheduled retraining | Mild drift; retrain weekly/monthly |
| Triggered retraining | Drift detected above threshold |
| Online learning | Continuous model update with new data |
| Model ensembling | Blend old and new model outputs |
| Feature engineering update | New features that capture new patterns |

---

## Medium (Q8–Q15)

---

### Q8. How do you design a recommendation system?

A production recommendation system uses a **two-stage architecture**: fast candidate retrieval (ANN search over embedding space) followed by expensive ranking (full ML model with many features).

```
TWO-STAGE RECOMMENDATION ARCHITECTURE
──────────────────────────────────────────────────────────────
User Request (user_id, context)
      │
      ▼
┌─────────────────────────────┐
│  STAGE 1: CANDIDATE RETRIEVAL│  Goal: 100K items → 1000 candidates
│                              │  Speed: < 20ms
│  Methods:                    │  Methods:
│  • Collaborative Filtering   │  • ANN vector search (FAISS, Pinecone)
│  • Content-based filtering   │  • User-item matrix factorization
│  • Popularity baseline       │  • Two-tower model embeddings
└─────────────────────────────┘
      │ 1000 candidates
      ▼
┌─────────────────────────────┐
│  STAGE 2: RANKING           │  Goal: 1000 → top 10
│                              │  Speed: < 80ms
│  Full feature set:           │  Model: LightGBM, DLRM, DeepFM
│  • User features (history)   │  Features: all available signals
│  • Item features (metadata)  │
│  • Context (time, device)    │
│  • Cross features            │
└─────────────────────────────┘
      │ Top 10
      ▼
     Result
```

**Collaborative Filtering (User-Item Matrix Factorization):**
```python
# Matrix Factorization: learn user embeddings U and item embeddings V
# such that U[user] · V[item] ≈ rating(user, item)
import implicit

model = implicit.als.AlternatingLeastSquares(factors=128, iterations=20)
model.fit(user_item_matrix)  # sparse matrix

# At serving time: get user embedding, ANN search for closest items
user_embedding = model.user_factors[user_id]
similar_items = ann_index.search(user_embedding, k=1000)
```

**Cold Start Problem:**
- New user: use popularity-based recommendations + ask for preferences onboarding
- New item: use content-based features (genre, category, keywords) until interaction data builds up

---

### Q9. How do you design a fraud detection system?

Fraud detection is a real-time ML system with extremely tight latency requirements (< 100ms from transaction to decision) and highly imbalanced training data (< 0.1% of transactions are fraudulent).

```
FRAUD DETECTION ARCHITECTURE
──────────────────────────────────────────────────────────────
Transaction ──► Kafka Topic ──► Feature Engineering Service
                                         │
                               ┌─────────▼─────────┐
                               │   Online Feature   │
                               │   Computation      │
                               │                    │
                               │  • txn velocity    │ Redis: sliding window
                               │  • merchant risk   │ counters
                               │  • geo anomaly     │
                               │  • device patterns │
                               └─────────┬──────────┘
                                         │
                               ┌─────────▼──────────┐
                               │  Model Inference   │ < 50ms
                               │  (fraud score 0-1) │
                               └─────────┬──────────┘
                                         │
                               ┌─────────▼──────────┐
                               │  Decision Engine   │
                               │  score > 0.8 → DENY│
                               │  0.5-0.8 → REVIEW  │
                               │  < 0.5   → APPROVE │
                               └────────────────────┘
```

**Feature Engineering for Fraud:**
```python
def compute_fraud_features(transaction: dict) -> dict:
    user_id    = transaction["user_id"]
    merchant   = transaction["merchant_id"]
    amount     = transaction["amount"]
    now        = transaction["timestamp"]

    # Velocity features (Redis sliding window counters)
    txn_count_1h   = redis.zcount(f"txn:{user_id}", now-3600, now)
    txn_count_24h  = redis.zcount(f"txn:{user_id}", now-86400, now)
    spend_1h       = redis.get(f"spend:{user_id}:1h")

    # Historical features (offline feature store)
    avg_txn_amount = feature_store.get(user_id, "avg_txn_amount_30d")
    merchant_risk  = feature_store.get(merchant, "merchant_risk_score")

    return {
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "amount_vs_avg_ratio": amount / max(avg_txn_amount, 1),
        "merchant_risk_score": merchant_risk,
        "is_new_merchant": txn_count_24h == 0,
    }
```

**Handling Class Imbalance:**
```python
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight('balanced', classes=[0, 1], y=labels)
# Or: use SMOTE for oversampling, or undersample majority class
```

**Key Design Decisions:**
- Feature freshness: transaction velocity MUST be real-time (Redis), not from last night's batch
- Model: Gradient Boosted Trees (XGBoost/LightGBM) for tabular data; fast inference, handles imbalance well
- Threshold: tune precision/recall trade-off based on business cost of false positive vs false negative

---

### Q10. How do you design the training pipeline for reproducibility?

A reproducible training pipeline guarantees that given the same code, the same data, and the same hyperparameters, you always produce the same model. Without reproducibility, debugging production model degradations is nearly impossible.

```
REPRODUCIBLE TRAINING PIPELINE
──────────────────────────────────────────────────────────────
┌──────────────────────────────────────────────────────────┐
│                  MLflow / Kubeflow Pipeline              │
│                                                          │
│  Step 1: Data Versioning (DVC)                           │
│    data.dvc → points to S3 path + SHA256 hash            │
│    git tag: training-run-2024-01-15                      │
│                                                          │
│  Step 2: Feature Pipeline                                │
│    Input: versioned raw data                             │
│    Output: feature matrix (versioned in DVC)             │
│                                                          │
│  Step 3: Train                                           │
│    mlflow.log_param("learning_rate", 0.01)               │
│    mlflow.log_param("data_version", "v2.3.1")            │
│    mlflow.log_metric("auc", 0.94)                        │
│    mlflow.sklearn.log_model(model, "model")              │
│                                                          │
│  Step 4: Evaluate                                        │
│    Compare against champion model; promote if better     │
│                                                          │
│  Step 5: Register                                        │
│    mlflow.register_model(model_uri, "fraud_detector")    │
└──────────────────────────────────────────────────────────┘
```

**MLflow Experiment Tracking:**
```python
import mlflow

with mlflow.start_run(run_name=f"fraud_v2_{date.today()}"):
    mlflow.log_params({
        "model_type": "xgboost",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "data_version": data_version,    # DVC data hash
        "feature_set_version": "v3.2",
    })

    model = train_model(X_train, y_train, params)

    metrics = evaluate(model, X_test, y_test)
    mlflow.log_metrics({
        "auc": metrics.auc,
        "precision_at_80_recall": metrics.precision_at_80_recall,
        "f1": metrics.f1
    })

    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("feature_importance.png")
```

**Data Versioning with DVC:**
```bash
dvc add data/training_set_2024_01_15.parquet
git add data/training_set_2024_01_15.parquet.dvc
git commit -m "Add training dataset v2024-01-15"
# Anyone can reproduce: dvc pull → gets exact same dataset
```

---

### Q11. How do you design a vector database system for similarity search?

Vector databases store high-dimensional embeddings and support **approximate nearest neighbor (ANN) search** — find the K vectors most similar to a query vector. This is the foundation of semantic search, recommendation systems, and RAG (Retrieval-Augmented Generation).

```
VECTOR DATABASE ARCHITECTURE
──────────────────────────────────────────────────────────────
Documents/Items
      │
      ▼ Embedding Model (BERT, OpenAI, custom)
Dense Vectors (e.g., 1536 dimensions)
      │
      ▼ Indexing (HNSW / IVF-PQ)
Vector Index (in-memory / on-disk)
      │
      ▼ Query time:
User Query → Embed query → ANN Search → Top-K results
```

**HNSW (Hierarchical Navigable Small World) Index:**
```
Layer 2 (coarse): ●─────────────────────●
                        few connections
Layer 1 (mid):   ●──●──────●────●──────●
Layer 0 (fine):  ●─●─●─●─●─●─●─●─●─●─●
                     all nodes connected locally
Query: start at top layer, greedily navigate to query region,
       descend to fine layer for exact search within neighborhood.
       O(log N) complexity vs O(N) for brute force.
```

**pgvector Example (PostgreSQL extension):**
```sql
CREATE EXTENSION vector;
CREATE TABLE items (
    id       BIGSERIAL PRIMARY KEY,
    content  TEXT,
    embedding VECTOR(1536)
);
-- IVFFlat index: approximate search, fast queries
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Query: find 10 most similar items to a query embedding
SELECT id, content,
       1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM items
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

**ANN Algorithm Trade-offs:**
```
HNSW    → Best recall, high memory (O(N × dims × 4 bytes)), fast build
IVF-PQ  → Good recall, low memory (product quantization compresses), slower build
ScaNN   → Google's; best latency/recall trade-off at scale
```

---

### Q12. How do you design the two-tower model architecture for recommendation systems at scale?

The two-tower model (also called dual encoder) is the dominant architecture for large-scale retrieval. It produces separate embeddings for users and items that can be searched efficiently with ANN.

```
TWO-TOWER ARCHITECTURE
──────────────────────────────────────────────────────────────
User Features                          Item Features
(user_id, history, demographics)       (item_id, category, text, image)
      │                                       │
      ▼                                       ▼
┌───────────┐                         ┌───────────┐
│  User     │                         │  Item     │
│  Tower    │                         │  Tower    │
│(deep NN)  │                         │(deep NN)  │
└─────┬─────┘                         └─────┬─────┘
      │  user_embedding (128-dim)            │  item_embedding (128-dim)
      └──────────────┬────────────────────┘
                     │
              dot product / cosine similarity
                     │
              training: positive pair (user, item they interacted with)
                        negative pair (user, random item)
                     │
              loss: softmax loss or in-batch negatives
```

**Training:**
```python
class TwoTowerModel(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size, embedding_dim=128):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Embedding(user_vocab_size, 64),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Embedding(item_vocab_size, 64),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, user_ids, item_ids):
        user_emb = F.normalize(self.user_tower(user_ids), dim=1)
        item_emb = F.normalize(self.item_tower(item_ids), dim=1)
        return torch.sum(user_emb * item_emb, dim=1)  # dot product
```

**Offline (pre-computed) item index:**
All item embeddings computed once and loaded into ANN index (FAISS).
At serving time: compute user embedding → ANN search in item index → get top 1000 candidates in < 20ms.

---

### Q13. How do you handle the cold start problem in recommendation systems?

**Cold start** is the inability to make good recommendations when there is insufficient interaction data — either for a new user or a new item.

```
COLD START SCENARIOS
──────────────────────────────────────────────────────────────
New User (no history)
  → No click history → collaborative filtering fails
  → Solutions:
    1. Popularity-based: recommend top items globally
    2. Onboarding questionnaire: ask for preferences
    3. Context signals: device, location, referral source, time-of-day
    4. Demographic fallback: aggregate behavior for similar users

New Item (no interactions)
  → No one has clicked it → collaborative filtering fails
  → Solutions:
    1. Content-based features: use item metadata (text, tags, category)
    2. Item embedding from description text (BERT/sentence-transformers)
    3. Exploration: show new items to a small random% of users
    4. Publisher signals: use author/creator historical performance
```

**Hybrid Approach for Cold Start:**
```python
def get_recommendations(user_id: str, n: int = 10) -> list:
    user_interaction_count = get_interaction_count(user_id)

    if user_interaction_count < 5:
        # Cold start: content + popularity hybrid
        content_recs  = content_based_recommendations(user_id, n=n*2)
        popular_recs  = get_popular_items(n=n*2)
        return interleave(content_recs, popular_recs)[:n]
    elif user_interaction_count < 50:
        # Warm start: blend collaborative + content
        cf_recs      = collaborative_filter_recs(user_id, n=n)
        content_recs = content_based_recommendations(user_id, n=n)
        return blend(cf_recs, 0.6, content_recs, 0.4)[:n]
    else:
        # Full collaborative + ranking
        return two_tower_recommendations(user_id, n=n)
```

**New Item Bootstrap:**
```python
# Use text embedding from item description for new items
def get_item_embedding(item: Item) -> np.array:
    if item.interaction_count > 100:
        return item.learned_embedding  # from matrix factorization
    # Cold start: embed item description with text model
    return sentence_transformer.encode(item.description)
```

---

### Q14. What is ML model monitoring — what metrics should you track?

Model monitoring is the practice of continuously observing a deployed model's behavior to detect degradation before it causes business impact. Models silently degrade — no runtime exception is thrown when accuracy drops.

**Four Pillars of ML Monitoring:**

```
┌─────────────────────────────────────────────────────────────┐
│                    ML MONITORING PILLARS                    │
│                                                             │
│  1. OPERATIONAL      2. DATA DRIFT        3. MODEL PERF    │
│  Latency p50/p99     Feature dist shift   Accuracy         │
│  Throughput          Null rate change     Precision/Recall  │
│  Error rate          Schema violations   AUC / NDCG        │
│  Memory/CPU          Outlier rate         Business KPIs     │
│                                                             │
│  4. DATA QUALITY                                           │
│  Missing features    Training-serving skew                  │
│  Out-of-range values Feature coverage                       │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
class ModelMonitor:
    def log_prediction(self, request_features: dict, prediction: float,
                       model_version: str):
        # 1. Operational metrics
        metrics.histogram("model.latency_ms",   self.last_latency)
        metrics.histogram("model.prediction",   prediction)

        # 2. Feature distribution monitoring
        for feature_name, value in request_features.items():
            metrics.histogram(f"feature.{feature_name}.value", value)
            if value is None:
                metrics.increment(f"feature.{feature_name}.null_rate")

        # 3. Drift detection (compare to training baseline)
        for feature_name, value in request_features.items():
            baseline = self.training_stats[feature_name]
            z_score = (value - baseline["mean"]) / baseline["std"]
            if abs(z_score) > 4:  # extreme outlier
                metrics.increment(f"feature.{feature_name}.outliers")

    def log_label(self, request_id: str, actual_label: int):
        # When ground truth label arrives (delayed), compute accuracy
        predicted = self.prediction_log[request_id]
        correct = (predicted > 0.5) == actual_label
        metrics.increment("model.correct" if correct else "model.incorrect")
```

**Alerting Thresholds:**
- Feature PSI > 0.2: investigate drift
- Prediction distribution shifts by > 15%: investigate concept drift
- Accuracy drops by > 5% absolute: trigger retraining

---

### Q15. How do you design a real-time personalization system?

A real-time personalization system takes the user's current context (what they're doing right now) and returns the most relevant content in < 100ms.

```
REAL-TIME PERSONALIZATION PIPELINE
──────────────────────────────────────────────────────────────
User Context
{user_id, current_page, recent_actions, device, time}
      │
      ▼ Feature Lookup (< 10ms)
┌──────────────────────────────────────┐
│  Feature Assembly                    │
│  • User profile (Redis): 2ms         │
│  • Recent behavior (Redis stream): 3ms│
│  • Current session (in-memory): 0ms  │
│  • Item candidate metadata: 2ms      │
└──────────────────────────────────────┘
      │ feature vector
      ▼ Model Inference (< 20ms)
┌──────────────────────────────────────┐
│  Ranking Model                       │
│  • LightGBM (fast) or TorchScript    │
│  • Score each candidate              │
│  • Return top-N ranked items         │
└──────────────────────────────────────┘
      │ ranked results
      ▼ Business Rules (< 5ms)
┌──────────────────────────────────────┐
│  Post-processing                     │
│  • Diversity injection               │
│  • Business rules (promoted items)   │
│  • A/B test variant assignment       │
└──────────────────────────────────────┘
      │ final results
      ▼ Response to client (< 100ms total)
```

**Latency Budget:**
```
Total budget: 100ms
  Network (client to API): 20ms
  Feature lookup (Redis):   10ms
  Candidate retrieval (ANN): 15ms
  Ranking model:             20ms
  Post-processing:            5ms
  Network (API to client):  20ms
  Buffer:                   10ms
```

---

## Hard (Q16–Q20)

---

### Q16. How do you design a feature store that handles both batch and streaming features with point-in-time correctness?

**Point-in-time correctness** means that when creating a training dataset, the feature value used for each training example reflects what was known at the time the label was generated — not any future information. Violating this creates **data leakage** and leads to inflated offline metrics.

```
POINT-IN-TIME CORRECTNESS PROBLEM
──────────────────────────────────────────────────────────────
User submitted loan application at T=100
Model should use features computed AT T=100

If we join features at T=now (later):
  Feature "user_credit_events_30d" includes events AFTER T=100
  This is data leakage — the model uses future information

Correct: features as of T=100
  credit_events_30d = events from T=70 to T=100 only
```

**Feature Store with Time-Travel:**
```sql
-- Store feature values with timestamps
CREATE TABLE feature_values (
    entity_id     VARCHAR(100) NOT NULL,
    feature_name  VARCHAR(100) NOT NULL,
    feature_value DOUBLE PRECISION,
    event_time    TIMESTAMP NOT NULL,
    created_time  TIMESTAMP DEFAULT NOW()  -- when record was written
);
CREATE INDEX ON feature_values (entity_id, feature_name, event_time DESC);

-- Point-in-time join: for each (entity, label_timestamp),
-- get the feature value that was valid AT label_timestamp
SELECT
    labels.entity_id,
    labels.label,
    labels.event_time AS label_time,
    fv.feature_value  AS credit_score
FROM labels
LEFT JOIN LATERAL (
    SELECT feature_value
    FROM   feature_values
    WHERE  entity_id    = labels.entity_id
      AND  feature_name = 'credit_score'
      AND  event_time   <= labels.event_time   -- no future leakage!
    ORDER BY event_time DESC
    LIMIT 1
) fv ON TRUE;
```

**Batch + Streaming Unified Architecture:**
```
                    OFFLINE (Spark)              ONLINE (Flink)
Raw events ────────────────────────────────────────────────────
                         │                           │
                  batch aggregate                stream aggregate
                  (daily job)                   (per event)
                         │                           │
                   Offline Store              Online Store
                   (S3/BigQuery)              (Redis / DynamoDB)
                   [all history]              [latest value only]
                         │                           │
                    Training                    Inference
                  dataset joins               feature lookup
                  (point-in-time)             (< 5ms)
```

**Feast Implementation (open-source feature store):**
```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Offline: point-in-time correct historical features for training
training_df = store.get_historical_features(
    entity_df=entity_df_with_timestamps,
    features=["user_features:total_spend_30d",
              "user_features:purchase_count_7d"]
).to_df()

# Online: real-time feature lookup for serving
online_features = store.get_online_features(
    features=["user_features:total_spend_30d"],
    entity_rows=[{"user_id": "user_123"}]
).to_dict()
```

This guarantees the same feature definitions in both training and serving — eliminating training-serving skew while maintaining correctness.

---

### Q17. How do you design a scalable ML training pipeline that handles 100TB of training data?

At 100TB, training data does not fit on a single machine. You need a distributed data pipeline, distributed training, and careful management of I/O bottlenecks (I/O is typically the bottleneck, not compute).

```
LARGE-SCALE TRAINING PIPELINE
──────────────────────────────────────────────────────────────
Data Lake (100TB, S3/GCS, Parquet format)
      │
      ▼ Data Preprocessing (Spark cluster, 100 workers)
  Distributed feature computation, filtering, joins
  Output: sharded TFRecord/Parquet (e.g., 10,000 files × 10GB)
      │
      ▼ Data Loading
  tf.data pipeline / PyTorch DataLoader (worker processes)
  Prefetch from object storage in parallel
  Cache hot shards in local SSD
      │
      ▼ Distributed Training
  Strategy: data parallelism (each GPU sees different mini-batch)
  Framework: PyTorch DDP / TensorFlow MirroredStrategy / Horovod
  N GPU workers, each with replica of model
  Gradients synchronized via AllReduce (NCCL)
      │
      ▼ Model Artifacts + Metrics → MLflow
      ▼ Model Registry → Staging → Production
```

**Distributed Training with PyTorch DDP:**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])  # wraps model for gradient sync

    # Each rank reads different data shards
    dataset = ShardedDataset(shard_id=rank, num_shards=world_size)
    dataloader = DataLoader(dataset, batch_size=256)

    for batch in dataloader:
        outputs = model(batch["features"])
        loss = criterion(outputs, batch["labels"])
        loss.backward()     # gradients computed locally
        optimizer.step()    # AllReduce synchronizes gradients across GPUs
        # All GPUs now have identical model weights
```

**Handling 100TB I/O Efficiently:**
```
Storage format: Parquet (columnar, compressed) → 3-5x smaller than CSV
Sharding: split into 10K files for parallel read (avoid single-file bottleneck)
Caching: warm data in local SSD before training epoch starts (pre-fetch)
Shuffling: shuffle within shard (not globally — global shuffle of 100TB is impractical)
Mixed precision: FP16 training (2x memory efficiency, 2x throughput on modern GPUs)
```

**Linear Scaling Rule:**
When scaling from 1 GPU to N GPUs, scale learning rate by N (linear scaling rule) and warm up for the first few epochs.

---

### Q18. How do you detect and prevent training-serving skew at scale?

At scale (millions of features, hundreds of models), training-serving skew is a silent killer of ML model performance. A systematic approach is needed.

```
SKEW DETECTION ARCHITECTURE
──────────────────────────────────────────────────────────────
Training Pipeline                     Serving Pipeline
      │                                      │
      ▼ Log feature statistics              ▼ Log serving feature stats
┌────────────────────┐              ┌──────────────────────┐
│ Training Stats     │              │  Serving Stats       │
│ (per feature):     │              │  (per feature):      │
│  mean, std, min,   │    compare   │   mean, std, min,    │
│  max, null_rate,   │ ◄──────────► │   max, null_rate,    │
│  histogram         │              │   histogram          │
└────────────────────┘              └──────────────────────┘
          │                                   │
          └───────────────┬───────────────────┘
                          ▼
                   Skew Detector
                   KS test, PSI, Jensen-Shannon divergence
                          │
                  Skew > threshold?
                     YES → Alert + investigation
```

**Automated Skew Report:**
```python
def generate_skew_report(model_name: str, model_version: str) -> SkewReport:
    training_stats = load_training_stats(model_name, model_version)
    serving_stats  = load_serving_stats(model_name, window="7d")

    report = SkewReport()
    for feature_name in training_stats.features:
        train_dist = training_stats.get_distribution(feature_name)
        serve_dist = serving_stats.get_distribution(feature_name)

        psi  = compute_psi(train_dist, serve_dist)
        ks_p = scipy.stats.ks_2samp(train_dist.values,
                                     serve_dist.values).pvalue

        report.add(SkewResult(
            feature=feature_name,
            psi=psi,
            ks_pvalue=ks_p,
            severity="HIGH" if psi > 0.2 else ("MEDIUM" if psi > 0.1 else "LOW"),
            training_mean=train_dist.mean,
            serving_mean=serve_dist.mean
        ))

    return report.sort_by_severity()
```

**Prevention via Shared Transform Library:**
```python
# transforms.py — shared between training pipeline and serving
class UserFeatureTransformer:
    def transform(self, raw_features: dict) -> np.array:
        # EXACT same logic in training and serving
        age_normalized = (raw_features["age"] - 35.2) / 12.1
        spend_log      = np.log1p(raw_features.get("spend_30d", 0))
        return np.array([age_normalized, spend_log, ...])

# Training: from transforms import UserFeatureTransformer
# Serving:  from transforms import UserFeatureTransformer (same import!)
```

If the transformer is in a shared Python package, version-pinned in both training and serving `requirements.txt`, skew from diverging code is eliminated.

---

### Q19. How do you design a model rollback strategy — shadow mode, champion-challenger, and canary?

Production ML model deployment needs progressive validation at each stage. The strategies form a hierarchy from zero-risk (shadow) to full-risk (champion).

```
ML MODEL DEPLOYMENT LADDER
──────────────────────────────────────────────────────────────
Stage 1: SHADOW MODE (zero risk)
  New model runs in production, gets real features
  Its predictions are LOGGED but not used
  Compare new model vs champion on real traffic
  Duration: 1–7 days

Stage 2: CHAMPION-CHALLENGER (low risk)
  New model (challenger) serves 5–10% of traffic
  Champion serves 90–95%
  Statistical test: is challenger better? (p < 0.05)
  Duration: 1–4 weeks

Stage 3: CANARY (medium risk)
  Challenger expanded to 20–30% of traffic
  Monitor business metrics closely
  Duration: 3–7 days

Stage 4: FULL ROLLOUT (challenger becomes new champion)
  Old champion retained for N days as rollback option
  Duration: permanent
```

**Shadow Mode Implementation:**
```python
class ModelRouter:
    def __init__(self, champion: Model, challenger: Model,
                 challenger_pct: float = 0.0, shadow: bool = False):
        self.champion   = champion
        self.challenger = challenger
        self.shadow     = shadow

    def predict(self, features: dict) -> float:
        # Always get champion prediction
        champion_score = self.champion.predict(features)

        if self.shadow:
            # Shadow: run challenger but don't use result
            try:
                challenger_score = self.challenger.predict(features)
                metrics.histogram("challenger.score", challenger_score)
                metrics.histogram("champion.score",   champion_score)
                metrics.histogram("score.delta",
                                  challenger_score - champion_score)
            except Exception as e:
                metrics.increment("challenger.errors")
            return champion_score  # Always return champion

        if random.random() < self.challenger_pct:
            # Champion-challenger: small % get challenger result
            metrics.increment("challenger.served")
            return self.challenger.predict(features)

        return champion_score
```

**Automated Promotion Criteria:**
```python
def should_promote_challenger(experiment_id: str) -> PromotionDecision:
    results = experiment_db.get(experiment_id)

    # Statistical significance
    p_value = ttest_ind(results.champion_scores,
                        results.challenger_scores).pvalue
    if p_value >= 0.05:
        return PromotionDecision.INSUFFICIENT_DATA

    # Effect size (challenger must be meaningfully better, not just statistically)
    effect_size = (results.challenger_mean - results.champion_mean) / results.champion_std
    if effect_size < 0.05:    # less than 5% improvement (Cohen's d)
        return PromotionDecision.NOT_MEANINGFUL

    # No regression on safety metrics
    if results.challenger_error_rate > results.champion_error_rate * 1.01:
        return PromotionDecision.SAFETY_REGRESSION

    return PromotionDecision.PROMOTE
```

---

### Q20. How do you design a real-time ML pipeline that ingests user events and makes predictions available in under 5 minutes?

This is a **lambda + streaming** architecture: a streaming pipeline computes near-real-time features from events as they occur, while a batch pipeline provides historical context. The model uses both layers.

```
NEAR-REAL-TIME ML PIPELINE (< 5 minute lag)
──────────────────────────────────────────────────────────────
User Events (clicks, purchases, searches)
      │
      ▼ Kafka Topic: user_events
      │  Partitioned by user_id (ordered per user)
      │
      ▼ Flink Streaming Job (3-minute windows + event triggers)
┌─────────────────────────────────────────────────────────────┐
│  Per-User Stream Processing:                                │
│  • Sliding window aggregates (clicks_last_5min)             │
│  • Session detection (session_length, page_sequence)        │
│  • State management (Flink stateful operators)              │
└──────────────────────┬──────────────────────────────────────┘
                       │ Feature updates emitted to Kafka
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Feature Writer                                             │
│  Upsert to Redis (online store): user feature values        │
│  Upsert to Cassandra (durable store): point-in-time log     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Inference Service                                    │
│  • Reads user features from Redis (< 2ms)                   │
│  • Reads historical features from offline store             │
│  • Assembles feature vector                                 │
│  • Runs model inference (< 20ms)                            │
│  • Returns personalized result                              │
└─────────────────────────────────────────────────────────────┘
```

**Flink Streaming Feature Computation:**
```python
# PySpark Structured Streaming equivalent (simpler to illustrate)
def compute_streaming_features():
    events = (spark.readStream
                   .format("kafka")
                   .option("subscribe", "user_events")
                   .load())

    user_features = (events
        .withWatermark("event_time", "2 minutes")  # handle late events
        .groupBy(
            F.window("event_time", "5 minutes", "1 minute"),  # sliding window
            "user_id"
        )
        .agg(
            F.count("*").alias("event_count_5min"),
            F.sum(F.when(F.col("event_type") == "click", 1).otherwise(0))
              .alias("click_count_5min"),
            F.countDistinct("item_id").alias("unique_items_5min")
        )
    )

    (user_features.writeStream
        .foreachBatch(write_to_redis)   # upsert to Redis online store
        .trigger(processingTime="30 seconds")  # micro-batch every 30s
        .start())
```

**End-to-End Latency Budget:**
```
Event occurs              T=0
Kafka ingestion           T + 50ms   (producer → broker → partition)
Flink processes event     T + 500ms  (trigger-based)
Flink writes to Redis     T + 800ms
Feature available in store T + 1s
Model uses updated feature T + next_request (up to 5 min lag acceptable)
```

**Key Design Decisions:**
- **Watermarks:** Flink watermarks handle late-arriving events (mobile apps often batch events)
- **Idempotent writes:** Event replay must not double-count; include event_id in state
- **State management:** Flink's RocksDB state backend for large per-user state
- **Backpressure:** If Redis write is slow, Flink applies backpressure; no data loss

---

## Quick Reference

```
ML SYSTEM COMPONENTS
──────────────────────────────────────────────────────────────
Data Ingestion → Feature Engineering → Feature Store
→ Training Pipeline → Model Registry → Serving → Monitoring

FEATURE STORE PURPOSE
──────────────────────────────────────────────────────────────
Offline store (S3/BigQuery): historical features for training
Online store (Redis/DynamoDB): latest features for serving (<5ms)
Same feature code → eliminates training-serving skew

TWO-STAGE RECOMMENDATION
──────────────────────────────────────────────────────────────
Stage 1: Retrieval   100K items → 1000 candidates  (ANN, <20ms)
Stage 2: Ranking     1000 → top 10                 (ML model, <80ms)

COLD START STRATEGIES
──────────────────────────────────────────────────────────────
New user   → popularity + onboarding questionnaire
New item   → content-based features (text embedding)

DRIFT TYPES
──────────────────────────────────────────────────────────────
Data drift     → feature distribution changed (PSI > 0.2 = alert)
Concept drift  → feature→label relationship changed
Detection      → KS test, PSI, Jensen-Shannon divergence

MODEL DEPLOYMENT LADDER
──────────────────────────────────────────────────────────────
Shadow mode (0% served) → Champion-Challenger (5-10%)
→ Canary (20-30%) → Full rollout (100%)

VECTOR SEARCH COMPLEXITY
──────────────────────────────────────────────────────────────
Brute force     O(N × d)   exact, too slow at N=1M
HNSW index      O(log N)   approximate, high recall
IVF-PQ          O(√N)      approximate, low memory

LATENCY BUDGETS
──────────────────────────────────────────────────────────────
Fraud detection           < 100ms
Search ranking            < 200ms
Recommendation            < 200ms
Batch scoring             hours (pre-computed)
Near-real-time features   < 5 minutes

TRAINING PIPELINE TOOLS
──────────────────────────────────────────────────────────────
Experiment tracking  → MLflow, Weights & Biases
Data versioning      → DVC, Delta Lake, Iceberg
Orchestration        → Airflow, Kubeflow, Metaflow
Distributed training → PyTorch DDP, Horovod
Feature store        → Feast, Tecton, Hopsworks

FRAUD DETECTION FEATURES
──────────────────────────────────────────────────────────────
Velocity (Redis counters): txn_count_1h, spend_1h
Historical (offline): avg_txn_amount, merchant_risk
Behavioral: is_new_merchant, geo_anomaly, device_fingerprint
```
