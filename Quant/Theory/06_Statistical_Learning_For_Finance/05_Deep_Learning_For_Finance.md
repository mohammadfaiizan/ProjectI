# Deep Learning for Finance

## Neural Networks for Return Prediction

Neural networks can capture complex non-linear relationships in financial data.

### Feedforward Neural Network

**Architecture:**
- **Input layer:** Features $\mathbf{x}$
- **Hidden layers:** Multiple layers of neurons
- **Output layer:** Prediction $\hat{y}$

**Single neuron:**
$$z = \sigma(\mathbf{w}^T\mathbf{x} + b)$$

where $\sigma$ is activation function.

**Common activations:**
- **ReLU:** $\sigma(x) = \max(0, x)$
- **Sigmoid:** $\sigma(x) = 1/(1+e^{-x})$
- **Tanh:** $\sigma(x) = \tanh(x)$

### Multi-Layer Network

**Forward pass:**
$$\mathbf{h}_1 = \sigma_1(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \sigma_2(\mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2)$$
$$\hat{y} = \sigma_{out}(\mathbf{W}_3\mathbf{h}_2 + \mathbf{b}_3)$$

**Backpropagation:** Compute gradients via chain rule, update weights via gradient descent.

### Applications

**Return prediction:** Predict next period return
**Direction prediction:** Classify up/down movement
**Volatility prediction:** Forecast volatility
**Factor models:** Extract non-linear factors

### Challenges

**Overfitting:** Many parameters, limited data
**Non-stationarity:** Relationships change over time
**Low signal-to-noise:** Hard to learn meaningful patterns

## RNN/LSTM for Time Series Forecasting

Recurrent Neural Networks process sequences by maintaining hidden state.

### RNN

**Hidden state update:**
$$\mathbf{h}_t = \sigma(\mathbf{W}_h\mathbf{h}_{t-1} + \mathbf{W}_x\mathbf{x}_t + \mathbf{b})$$

**Output:**
$$\hat{y}_t = \mathbf{W}_o\mathbf{h}_t + \mathbf{b}_o$$

**Problem:** Vanishing/exploding gradients in long sequences.

### LSTM

Long Short-Term Memory networks solve gradient problems with gating mechanisms.

**Cell state:** $\mathbf{c}_t$ (carries information)
**Hidden state:** $\mathbf{h}_t$ (output)

**Gates:**

**Forget gate:**
$$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input gate:**
$$\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Update cell state:**
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output gate:**
$$\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

**Advantages:**
- Handles long sequences
- Learns long-term dependencies
- Prevents vanishing gradients

### Applications

**Return forecasting:** Use past returns to predict future
**Volatility forecasting:** Model volatility dynamics
**High-frequency trading:** Process tick-by-tick data
**Portfolio optimization:** Dynamic allocation based on predictions

### GRU

Gated Recurrent Unit is simpler than LSTM:
- Two gates (reset, update) instead of three
- Often similar performance
- Faster training

## Attention and Transformers

Attention mechanisms allow models to focus on relevant parts of input.

### Attention Mechanism

**Query, Key, Value:** For each position, compute attention weights:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**Interpretation:** Weighted combination of values, where weights depend on query-key similarity.

### Self-Attention

**Query, Key, Value from same sequence:**
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

**Multi-head attention:** Multiple attention heads in parallel, concatenate outputs.

### Transformer Architecture

**Encoder-decoder:** 
- **Encoder:** Processes input sequence
- **Decoder:** Generates output sequence

**Key components:**
- Multi-head self-attention
- Position encoding
- Feedforward networks
- Residual connections
- Layer normalization

### Applications for Financial Data

**Return prediction:** Attend to relevant past periods
**Factor extraction:** Identify important features
**Anomaly detection:** Focus on unusual patterns
**News sentiment:** Process financial text

### Challenges

**Computational cost:** Quadratic in sequence length
**Data requirements:** Need large datasets
**Interpretability:** Attention weights help but still complex

## Autoencoders

Autoencoders learn compressed representations of data.

### Architecture

**Encoder:** Maps input to latent representation
$$\mathbf{z} = f_{enc}(\mathbf{x})$$

**Decoder:** Reconstructs input from latent
$$\hat{\mathbf{x}} = f_{dec}(\mathbf{z})$$

**Loss:** Reconstruction error
$$L = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

### Variational Autoencoder (VAE)

**Latent distribution:** $\mathbf{z} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

**Loss:** Reconstruction + KL divergence
$$L = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + KL(q(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

**Advantage:** Smooth latent space, can generate new samples.

### Applications

**Anomaly detection:** High reconstruction error → anomaly
**Feature learning:** Use encoder as feature extractor
**Dimensionality reduction:** Latent representation
**Denoising:** Remove noise from data

**Financial applications:**
- Detect fraudulent transactions
- Identify market anomalies
- Extract risk factors
- Generate synthetic data

## Reinforcement Learning for Trading

Reinforcement learning learns optimal actions through trial and error.

### Framework

**State $s_t$:** Market conditions, portfolio state
**Action $a_t$:** Trading decisions (buy/sell/hold, position sizes)
**Reward $r_t$:** Profit, Sharpe ratio, or other objective
**Policy $\pi(a|s)$:** Probability of action given state

### Q-Learning

**Q-function:** Expected future reward from state-action pair:
$$Q(s, a) = \mathbb{E}[R_t | s_t = s, a_t = a]$$

**Bellman equation:**
$$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

**Update:**
$$Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

### Deep Q-Network (DQN)

Use neural network to approximate Q-function:
$$Q(s, a; \theta) \approx Q(s, a)$$

**Training:** Minimize TD error:
$$L = (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2$$

**Tricks:**
- Experience replay: Store transitions, sample batches
- Target network: Separate network for target values
- Epsilon-greedy: Balance exploration/exploitation

### Policy Gradient Methods

**Policy gradient:** Directly optimize policy:
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \ln \pi_\theta(a|s) R]$$

**Actor-critic:** Combine policy gradient with value function:
- **Actor:** Policy $\pi_\theta(a|s)$
- **Critic:** Value function $V_\phi(s)$

### Applications

**Algorithmic trading:** Learn optimal execution
**Portfolio optimization:** Dynamic allocation
**Market making:** Provide liquidity
**Risk management:** Optimal hedging

### Challenges

**Non-stationarity:** Market dynamics change
**Delayed rewards:** Profits realized later
**Exploration:** Need to try different strategies
**Simulation:** Realistic market simulation needed

## Challenges: Non-Stationarity

Financial data is non-stationary - relationships change over time.

### Problem

**Distribution shift:** $P(Y|X)$ changes over time
**Regime changes:** Different market regimes
**Structural breaks:** Sudden changes in relationships

**Impact:** Model performance degrades over time.

### Solutions

**Online learning:** Update model as new data arrives
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t; \mathbf{x}_{t+1}, y_{t+1})$$

**Rolling windows:** Retrain on recent data periodically

**Exponential weighting:** Give more weight to recent observations

**Regime-switching:** Allow parameters to change with regime

**Transfer learning:** Pre-train on historical data, fine-tune on recent

**Domain adaptation:** Adapt model to new distribution

## Challenges: Low Signal-to-Noise

Financial returns have low predictability.

### Problem

**Noise dominates:** Random fluctuations larger than signal
**Overfitting risk:** Model fits noise, not signal
**Poor generalization:** Works in-sample, fails out-of-sample

### Solutions

**Regularization:** L1/L2 penalties, dropout, early stopping

**Simpler models:** Prefer interpretable models when possible

**Feature engineering:** Use domain knowledge

**Ensemble methods:** Combine multiple models

**Robust loss functions:** Less sensitive to outliers

**Data augmentation:** Create synthetic examples (carefully)

## Challenges: Overfitting

Deep learning models have many parameters, risk overfitting.

### Regularization Techniques

**L2 regularization:** Penalize large weights
$$L = L_{data} + \lambda \|\mathbf{W}\|_2^2$$

**Dropout:** Randomly set neurons to zero during training
- Prevents co-adaptation
- Acts as ensemble
- Use during training, not inference

**Early stopping:** Stop when validation error increases

**Batch normalization:** Normalize activations
- Stabilizes training
- Acts as regularization
- Allows higher learning rates

**Data augmentation:** Increase effective dataset size

### Validation

**Time-series cross-validation:** Forward chaining (no future data)

**Walk-forward analysis:** Retrain periodically, test on future

**Out-of-sample:** Always test on unseen future data

## Interpretability and Explainability

Regulators and stakeholders require model explanations.

### Methods

**Feature importance:** Which features matter most

**SHAP values:** Shapley Additive Explanations
- Game-theoretic approach
- Allocates prediction to features
- Satisfies desirable properties

**LIME:** Local Interpretable Model-agnostic Explanations
- Approximate model locally with interpretable model
- Explains individual predictions

**Attention weights:** For attention-based models
- Which parts of input model focuses on

**Gradient-based:** Saliency maps, integrated gradients

### Requirements

**Regulatory:** May need to explain decisions
**Risk management:** Understand model behavior
**Debugging:** Identify model failures
**Trust:** Build confidence in model

### Trade-offs

**Interpretability vs accuracy:** More interpretable models may be less accurate

**Global vs local:** Explain overall model vs individual predictions

**Post-hoc vs intrinsic:** Explain after training vs built-in interpretability

## Practical Considerations

### Data Preprocessing

**Normalization:** Scale features (important for neural networks)

**Handling missing data:** Imputation, indicator variables

**Feature engineering:** Domain knowledge crucial

**Stationarity:** May need to difference or detrend

### Architecture Design

**Network depth:** Deeper networks can learn more complex patterns, but harder to train

**Width:** More neurons per layer increases capacity

**Activation functions:** ReLU common, others for specific cases

**Initialization:** Proper initialization crucial (Xavier, He)

### Training

**Optimization:** Adam, RMSprop often better than SGD for finance

**Learning rate:** Critical hyperparameter, use learning rate scheduling

**Batch size:** Affects gradient estimates, generalization

**Monitoring:** Track training/validation loss, early stopping

### Evaluation

**Time-series aware:** Proper train/test splits (no future data)

**Multiple metrics:** Returns, Sharpe ratio, max drawdown

**Transaction costs:** Include in evaluation

**Robustness:** Test across different market conditions

### Production

**Latency:** Fast inference for real-time trading

**Model serving:** Efficient deployment

**Monitoring:** Track performance, detect drift

**Retraining:** Update model periodically

**Version control:** Track model versions
