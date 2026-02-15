# Training and Experiment Tools: Keras Tuner, TF Probability, TF-Agents, TF Recommenders

## Table of Contents

1. [Keras Tuner](#1-keras-tuner)
2. [TensorFlow Probability](#2-tensorflow-probability)
3. [TF-Agents](#3-tf-agents)
4. [TensorFlow Recommenders](#4-tensorflow-recommenders)

---

## 1. Keras Tuner

**Keras Tuner** automates hyperparameter search for Keras models. It supports multiple search algorithms and integrates with TensorBoard.

### HyperModel and build_model

Define a **HyperModel** by creating a function that takes a `HyperParameters` object and returns a compiled model:

```python
import keras_tuner as kt

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f"units_{i}", 32, 256, step=32),
            activation="relu"
        ))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

### kt.RandomSearch

**RandomSearch** samples hyperparameters randomly for a fixed number of trials:

```python
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=2,
    directory="my_dir",
    project_name="mnist"
)
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_hps = tuner.get_best_hyperparameters(1)[0]
```

### kt.Hyperband

**Hyperband** uses early stopping and resource allocation. It trains many models for few epochs and promotes promising ones:

```python
tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    hyperband_iterations=2,
    directory="my_dir",
    project_name="mnist"
)
```

### kt.BayesianOptimization

**BayesianOptimization** models the objective as a Gaussian process and selects hyperparameters to optimize expected improvement:

```python
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=15,
    directory="my_dir",
    project_name="mnist"
)
```

### Hyperparameter Types

| Type | Example |
|------|---------|
| Int | hp.Int("units", 32, 256, step=32) |
| Float | hp.Float("lr", 1e-4, 1e-2, sampling="log") |
| Choice | hp.Choice("activation", ["relu", "gelu"]) |
| Boolean | hp.Boolean("use_bn") |

---

## 2. TensorFlow Probability

**TensorFlow Probability (TFP)** adds probabilistic reasoning to TensorFlow: distributions, probabilistic layers, and variational inference.

### Distributions: tfd.Normal, tfd.MultivariateNormalDiag

**tfd.Normal** represents a univariate normal distribution:

```python
import tensorflow_probability as tfp
tfd = tfp.distributions

normal = tfd.Normal(loc=0.0, scale=1.0)
samples = normal.sample(10)
log_prob = normal.log_prob(samples)
```

**tfd.MultivariateNormalDiag** is a multivariate normal with diagonal covariance:

```python
mvn = tfd.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[1.0, 2.0])
samples = mvn.sample(5)
```

### tfp.layers

**DistributionLambda** wraps a distribution as a layer. The layer outputs a distribution object:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2),
    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-5))
])
dist = model(x)
samples = dist.sample()
```

**IndependentBernoulli**, **OneHotCategorical**, and **MixtureSameFamily** support discrete and mixture outputs.

### Probabilistic Models

For regression with uncertainty, output a distribution and use negative log-likelihood as the loss:

```python
def negloglik(y_true, y_pred):
    return -y_pred.log_prob(y_true)

# Model outputs mean and scale; wrap in Normal
dist_layer = tfp.layers.DistributionLambda(
    lambda t: tfd.Normal(loc=t[..., :1], scale=tf.nn.softplus(t[..., 1:]) + 1e-5)
)
model.compile(optimizer="adam", loss=negloglik)
```

### Variational Inference

TFP provides **tfp.distributions** for priors and **tfp.vi** for variational inference, including ELBO computation.

---

## 3. TF-Agents

**TF-Agents** is a library for reinforcement learning (RL) in TensorFlow. It provides environments, agents, policies, and replay buffers.

### Environment

Environments follow the **PyEnvironment** interface. Use **suite_gym** for OpenAI Gym compatibility:

```python
from tf_agents.environments import suite_gym, tf_py_environment

env = suite_gym.load("CartPole-v1")
tf_env = tf_py_environment.TFPyEnvironment(env)
```

**TimeStep** contains observation, reward, discount, and step type (first, mid, last).

### Agent

An **agent** implements a learning algorithm (e.g., DQN, PPO, SAC). It uses a **QNetwork** or **ActorCriticNetwork**:

```python
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network

q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=(100, 50)
)
agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    td_errors_loss_fn=tf.keras.losses.Huber()
)
```

### Policy

A **policy** maps observations to actions. Agents expose a policy for collection and evaluation:

```python
policy = agent.policy
action_step = policy.action(time_step)
```

**RandomTFPolicy** and **PyPolicy** wrap Python policies for random or custom behavior.

### Replay Buffer

**Replay buffers** store experience tuples for off-policy learning:

```python
from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000
)
```

**Drivers** (e.g., **dynamic_step_driver**) collect experience from the environment and store it in the replay buffer.

---

## 4. TensorFlow Recommenders

**TensorFlow Recommenders (TFRS)** simplifies building recommendation systems with retrieval and ranking models.

### tfrs.Model

**tfrs.Model** extends Keras Model with a `compute_loss` method. Subclass it to define retrieval or ranking models:

```python
import tensorflow_recommenders as tfrs

class RetrievalModel(tfrs.Model):
    def __init__(self, user_model, item_model):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(candidates=...)
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        item_embeddings = self.item_model(features["item_id"])
        return self.task(user_embeddings, item_embeddings)
```

### Retrieval Task

**Retrieval** learns to rank items for a user. **FactorizedTopK** uses in-batch negatives and computes metrics like recall@K:

```python
task = tfrs.tasks.Retrieval(
    metrics=tfrs.metrics.FactorizedTopK(
        candidates=item_dataset.batch(128).map(item_model)
    )
)
```

### Ranking Task

**Ranking** predicts a score (e.g., rating) for user-item pairs:

```python
task = tfrs.tasks.Ranking(
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

### Two-Tower Architecture

A common pattern is a **two-tower** model: one tower for users, one for items. Embeddings are computed separately and combined (e.g., dot product for retrieval, concatenation for ranking).

---

## Summary Table

| Tool | Primary Use |
|------|-------------|
| Keras Tuner | Hyperparameter search (RandomSearch, Hyperband, BayesianOptimization) |
| TF Probability | Distributions, probabilistic layers, uncertainty quantification |
| TF-Agents | RL environments, agents, policies, replay buffers |
| TF Recommenders | Retrieval and ranking models, two-tower architectures |
