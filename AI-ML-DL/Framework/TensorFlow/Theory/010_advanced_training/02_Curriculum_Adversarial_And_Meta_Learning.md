# Curriculum, Adversarial, and Meta Learning

## Table of Contents

1. [Curriculum Learning](#1-curriculum-learning)
2. [Adversarial Training](#2-adversarial-training)
3. [Meta-Learning](#3-meta-learning)
4. [Few-Shot Learning](#4-few-shot-learning)

---

## 1. Curriculum Learning

**Curriculum learning** trains a model by gradually increasing task difficulty, mimicking how humans learn. The idea is to order training examples from easy to hard.

### Difficulty Scoring

Assign a difficulty score to each sample. Common approaches:

| Method | Description |
|--------|-------------|
| Loss-based | Use model's loss on sample as difficulty |
| Uncertainty | Use entropy or variance of predictions |
| Heuristic | Domain-specific (e.g., sentence length, image complexity) |
| Oracle | Human or auxiliary model labels difficulty |

```python
def compute_difficulty(model, x, y):
    """Higher loss = harder sample"""
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)
    return tf.reduce_mean(loss)
```

### Pacing Functions

A **pacing function** determines what fraction of data (by difficulty) to use at each training step or epoch.

```python
def linear_pacing(epoch, total_epochs):
    """Linearly increase from 0.2 to 1.0"""
    return 0.2 + 0.8 * (epoch / total_epochs)

def root_pacing(epoch, total_epochs):
    """Square root schedule: slower start"""
    return (epoch / total_epochs) ** 0.5

def step_pacing(epoch, milestones=[10, 20, 30], fractions=[0.3, 0.6, 1.0]):
    """Step-wise increase at milestones"""
    for m, f in zip(milestones, fractions):
        if epoch < m:
            return f
    return 1.0
```

### Data Scheduling

Sort or sample data by difficulty and expose only the easiest fraction initially.

```python
def curriculum_dataloader(dataset, difficulties, pacing_fn, epoch):
    """Filter dataset to include only easiest fraction"""
    fraction = pacing_fn(epoch)
    n_keep = int(len(dataset) * fraction)
    sorted_indices = tf.argsort(difficulties)
    easy_indices = sorted_indices[:n_keep]
    return tf.gather(dataset, easy_indices)
```

### Implementation Pattern

```python
# 1. Compute difficulties (e.g., once at start or periodically)
difficulties = []
for x, y in train_ds:
    d = compute_difficulty(model, x, y)
    difficulties.append(d)
difficulties = tf.concat(difficulties, axis=0)

# 2. Sort and create curriculum indices
order = tf.argsort(difficulties)

# 3. Each epoch, select subset based on pacing
for epoch in range(num_epochs):
    frac = pacing_fn(epoch, num_epochs)
    n = int(len(order) * frac)
    curr_indices = order[:n]
    curr_ds = train_ds.unbatch().take(n)  # Simplified
    for x, y in curr_ds:
        train_step(x, y)
```

---

## 2. Adversarial Training

**Adversarial training** improves robustness by training on adversarial examples: inputs perturbed to fool the model.

### Fast Gradient Sign Method (FGSM)

**FGSM** creates adversarial examples by moving the input in the direction of the loss gradient.

```python
def fgsm_attack(model, x, y, epsilon=0.1):
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)
        loss = tf.reduce_mean(loss)
    grad = tape.gradient(loss, x)
    # Perturbation: sign of gradient
    perturbation = epsilon * tf.sign(grad)
    x_adv = x + perturbation
    # Clip to valid range
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv
```

### Projected Gradient Descent (PGD)

**PGD** is an iterative version of FGSM. Multiple steps with projection onto an epsilon-ball.

```python
def pgd_attack(model, x, y, epsilon=0.1, alpha=0.01, steps=10):
    x_adv = tf.identity(x)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            pred = model(x_adv, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)
            loss = tf.reduce_mean(loss)
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        # Project back to epsilon ball around x
        perturbation = tf.clip_by_value(x_adv - x, -epsilon, epsilon)
        x_adv = x + perturbation
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv
```

### Adversarial Loss

Train on both clean and adversarial examples. The **adversarial loss** can be the standard loss on adversarial inputs, or a combined objective.

```python
def adversarial_loss(model, x, y, epsilon=0.1):
    x_adv = fgsm_attack(model, x, y, epsilon)
    pred_clean = model(x, training=True)
    pred_adv = model(x_adv, training=True)
    loss_clean = tf.keras.losses.sparse_categorical_crossentropy(y, pred_clean)
    loss_adv = tf.keras.losses.sparse_categorical_crossentropy(y, pred_adv)
    return tf.reduce_mean(loss_clean) + tf.reduce_mean(loss_adv)
```

### Robust Training Loop

```python
@tf.function
def train_step_robust(model, optimizer, x, y, epsilon=0.1):
    with tf.GradientTape() as tape:
        x_adv = fgsm_attack(model, x, y, epsilon)
        pred = model(x_adv, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

## 3. Meta-Learning

**Meta-learning** (learning to learn) trains a model to quickly adapt to new tasks with few examples. The outer loop updates meta-parameters; the inner loop adapts to a task.

### MAML: Model-Agnostic Meta-Learning

**MAML** finds initialization parameters that can be fine-tuned quickly. Inner loop: task-specific adaptation. Outer loop: meta-update.

### Inner Loop

For each task, compute adapted parameters using a few gradient steps on the task's support set.

```python
def inner_loop(model, support_x, support_y, inner_lr, num_steps=5):
    """Adapt model to task using support set"""
    weights = [tf.identity(w) for w in model.trainable_variables]
    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            pred = forward(model, support_x, weights)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(support_y, pred)
            )
        grads = tape.gradient(loss, weights)
        weights = [w - inner_lr * g for w, g in zip(weights, grads)]
    return weights
```

### Outer Loop

Compute meta-gradient on query set using the adapted parameters. Update the initialization.

```python
def maml_step(model, task_batch, inner_lr, outer_lr):
    """One MAML meta-update"""
    meta_grads = [tf.zeros_like(w) for w in model.trainable_variables]
    for (support_x, support_y), (query_x, query_y) in task_batch:
        adapted = inner_loop(model, support_x, support_y, inner_lr)
        with tf.GradientTape() as tape:
            pred = forward(model, query_x, adapted)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(query_y, pred)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        meta_grads = [mg + g for mg, g in zip(meta_grads, grads)]
    # Average and apply
    n_tasks = len(task_batch)
    meta_grads = [mg / n_tasks for mg in meta_grads]
    optimizer.apply_gradients(zip(meta_grads, model.trainable_variables))
```

### Task Distribution

Sample tasks from a distribution (e.g., different classes per task). Each task has a support set and query set.

```python
def sample_task(dataset, n_way=5, k_shot=5, q_query=15):
    """N-way K-shot: N classes, K support examples per class, Q query examples"""
    classes = tf.random.shuffle(tf.unique(dataset['class'])[0])[:n_way]
    support_x, support_y = [], []
    query_x, query_y = [], []
    for c in classes:
        class_data = dataset[dataset['class'] == c]
        indices = tf.random.shuffle(tf.range(len(class_data)))
        support_idx = indices[:k_shot]
        query_idx = indices[k_shot:k_shot + q_query]
        support_x.append(tf.gather(class_data['x'], support_idx))
        support_y.append(tf.fill([k_shot], c))
        query_x.append(tf.gather(class_data['x'], query_idx))
        query_y.append(tf.fill([q_query], c))
    return (tf.concat(support_x, 0), tf.concat(support_y, 0)),
           (tf.concat(query_x, 0), tf.concat(query_y, 0))
```

### First-Order MAML

**First-order MAML** (FOMAML) ignores second-order derivatives. Use the adapted parameters for the query loss but approximate the meta-gradient. Simpler and faster.

```python
# In first-order MAML, we treat the inner-loop adaptation as a fixed
# transformation and only backprop through the query loss w.r.t. the
# pre-adaptation parameters. Implementation often uses tf.stop_gradient
# or simply differentiates the query loss directly.
```

---

## 4. Few-Shot Learning

**Few-shot learning** aims to recognize new classes from very few examples. Common setup: **N-way K-shot** (N classes, K examples per class).

### Support and Query Sets

| Set | Purpose |
|-----|---------|
| **Support set** | Few labeled examples per class for adaptation |
| **Query set** | Examples to classify (evaluation) |

### Prototypical Networks

**Prototypical networks** compute a prototype (centroid) per class from the support set. Query examples are classified by nearest prototype.

```python
def compute_prototypes(support_x, support_y, n_way):
    """support_x: [N*K, D], support_y: [N*K] with values 0..n_way-1"""
    prototypes = []
    for c in range(n_way):
        mask = tf.equal(support_y, c)
        class_features = tf.boolean_mask(support_x, mask)
        proto = tf.reduce_mean(class_features, axis=0)
        prototypes.append(proto)
    return tf.stack(prototypes)  # [n_way, D]

def prototypical_loss(support_x, support_y, query_x, query_y, encoder, n_way):
    support_emb = encoder(support_x)
    query_emb = encoder(query_x)
    prototypes = compute_prototypes(support_emb, support_y, n_way)
    # Euclidean distance: [n_query, n_way]
    dists = tf.reduce_sum(
        tf.square(query_emb[:, None, :] - prototypes[None, :, :]),
        axis=2
    )
    # Negative distance as logits (closer = higher score)
    logits = -dists
    loss = tf.keras.losses.sparse_categorical_crossentropy(query_y, logits)
    return tf.reduce_mean(loss)
```

### N-Way K-Shot

- **5-way 1-shot**: 5 classes, 1 support example per class
- **5-way 5-shot**: 5 classes, 5 support examples per class

```python
def create_episode(dataset, n_way=5, k_shot=5, n_query=15):
    """Create one few-shot episode"""
    classes = random.sample(dataset.classes, n_way)
    support, query = [], []
    for c in classes:
        samples = dataset.get_class_samples(c, k_shot + n_query)
        support.append(samples[:k_shot])
        query.append(samples[k_shot:])
    return (concat(support), concat(query))
```

### Training Prototypical Networks

```python
for episode in range(num_episodes):
    (support_x, support_y), (query_x, query_y) = sample_episode(
        dataset, n_way=5, k_shot=5
    )
    with tf.GradientTape() as tape:
        loss = prototypical_loss(
            support_x, support_y, query_x, query_y, encoder, n_way=5
        )
    grads = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
```
