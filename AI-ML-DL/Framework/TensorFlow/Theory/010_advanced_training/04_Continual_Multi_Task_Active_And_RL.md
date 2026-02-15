# Continual, Multi-Task, Active Learning, and Reinforcement Learning

## Table of Contents

1. [Continual Learning](#1-continual-learning)
2. [Multi-Task Learning](#2-multi-task-learning)
3. [Active Learning](#3-active-learning)
4. [Reinforcement Learning Basics](#4-reinforcement-learning-basics)

---

## 1. Continual Learning

**Continual learning** (lifelong learning) trains on a sequence of tasks without forgetting previous ones. The main challenge is **catastrophic forgetting**: new task training overwrites knowledge from old tasks.

### Elastic Weight Consolidation (EWC)

EWC penalizes changes to parameters important for previous tasks. Importance is approximated by the **Fisher information matrix** (diagonal).

```python
# After training on task 1, compute Fisher
star_params = [tf.identity(v) for v in model.trainable_variables]
fisher = []
for x, y in task1_ds:
    with tf.GradientTape() as tape:
        pred = model(x, training=False)
        log_prob = tf.reduce_sum(tf.math.log(pred + 1e-8) * tf.one_hot(y, n_classes), axis=1)
        loss = -tf.reduce_mean(log_prob)
    grads = tape.gradient(loss, model.trainable_variables)
    fisher.append([tf.square(g) for g in grads])
fisher_diag = [tf.reduce_mean(tf.stack([f[i] for f in fisher]), axis=0)
               for i in range(len(model.trainable_variables))]

# When training on task 2, add EWC penalty
loss_ewc = 0.5 * ewc_lambda * sum(
    tf.reduce_sum(f * tf.square(p - s))
    for f, p, s in zip(fisher_diag, model.trainable_variables, star_params)
)
loss = loss_task2 + loss_ewc
```

### Replay Buffers

Store a subset of old task data and replay it when training on new tasks:

```python
replay_buffer_x, replay_buffer_y = [], []
# After task 1, add samples to buffer
replay_buffer_x.append(task1_x[:100])
replay_buffer_y.append(task1_y[:100])
# During task 2 training
loss_replay = loss_fn(replay_buffer_y, model(replay_buffer_x))
loss = loss_task2 + 0.5 * loss_replay
```

### Other Approaches

- **Progressive Neural Networks**: New capacity per task, lateral connections to previous networks.
- **PackNet**: Prune and freeze subsets of weights per task.
- **iCaRL**: Replay + knowledge distillation from previous model.

---

## 2. Multi-Task Learning

**Multi-task learning (MTL)** trains a single model on multiple related tasks, sharing representations and potentially improving generalization.

### Shared Encoder and Task Heads

```python
shared = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(32, activation='relu')
])
head_a = tf.keras.layers.Dense(5, activation='softmax', name='task_a')
head_b = tf.keras.layers.Dense(1, activation='sigmoid', name='task_b')

def forward(x):
    feat = shared(x, training=True)
    return head_a(feat), head_b(feat)
```

### Loss Weighting

Combine task losses with weights:

```python
loss = w_a * loss_a + w_b * loss_b
```

**Uncertainty weighting** (Kendall et al.): learn task-dependent weights via homoscedastic uncertainty. **GradNorm**: balance gradient magnitudes across tasks.

### Task Relationships

- **Positive transfer**: Shared features help all tasks.
- **Negative transfer**: Conflicting objectives hurt some tasks.
- **Task-specific layers**: Allow task-specific adaptation while sharing low-level features.

---

## 3. Active Learning

**Active learning** selects which unlabeled samples to label next, aiming to maximize model improvement with minimal labeling cost.

### Pool-Based Setting

- **Pool**: Large set of unlabeled data.
- **Labeled set**: Initially small; grows by querying the oracle.
- **Query strategy**: Decides which pool samples to label.

### Uncertainty Sampling

Query samples where the model is most uncertain:

- **Entropy**: `H(p) = -sum p_c log p_c`; high entropy means uncertain.
- **Least confidence**: `1 - max_c p_c`; low max probability means uncertain.
- **Margin**: Difference between top two probabilities; small margin means uncertain.

```python
def uncertainty_sampling(model, x, n_query=10):
    pred = model(x, training=False)
    probs = pred.numpy()
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    return np.argsort(entropy)[-n_query:]
```

### Query-By-Committee

Train multiple models; query samples where they disagree.

### Expected Model Change

Query samples that would change the model the most if labeled (expensive to compute).

### Active Learning Loop

```python
for round in range(num_rounds):
    model.fit(labeled_x, labeled_y, epochs=5)
    query_idx = uncertainty_sampling(model, pool_x, n_query=20)
    new_labels = oracle.label(pool_x[query_idx])
    labeled_x = tf.concat([labeled_x, pool_x[query_idx]], axis=0)
    labeled_y = tf.concat([labeled_y, new_labels], axis=0)
    pool_x = remove(pool_x, query_idx)
```

---

## 4. Reinforcement Learning Basics

**Reinforcement learning (RL)** learns a policy to maximize cumulative reward through interaction with an environment.

### Policy Gradient (REINFORCE)

**REINFORCE** is a Monte Carlo policy gradient method. The gradient of the expected return with respect to policy parameters is:

```
nabla J = E[nabla log pi(a|s) * G_t]
```

where `G_t` is the return (sum of future rewards) from time t.

### Implementation with TensorFlow

```python
def reinforce_update(policy, states, actions, returns, optimizer):
    with tf.GradientTape() as tape:
        probs = policy(states, training=True)
        action_probs = tf.reduce_sum(probs * tf.one_hot(actions, n_actions), axis=1)
        log_probs = tf.math.log(action_probs + 1e-8)
        loss = -tf.reduce_mean(log_probs * returns)
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
```

### Returns and Baseline

Use **baseline subtraction** to reduce variance: `returns = returns - baseline`. The baseline (e.g., value function or running mean) does not change the gradient expectation but reduces variance.

### Sampling Actions

```python
def sample_action(probs):
    action = np.random.choice(n_actions, p=probs.numpy())
    return action
```

### Extensions

- **Actor-Critic**: Use a value function as baseline; train both policy and value.
- **PPO, A2C, A3C**: More stable policy gradient algorithms.
- **DQN**: Value-based; learn Q-function instead of policy.
