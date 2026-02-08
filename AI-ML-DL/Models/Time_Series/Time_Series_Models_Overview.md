# Time Series Models -- Historical Evolution Applied to Bitcoin Prediction

## Overview

This document chronicles 12 implementations tracing the historical evolution of time series forecasting from ARIMA (1970s) to modern foundation models (2020+). All implementations address a unified problem: Bitcoin price prediction across multiple forecasting horizons (1-day, 7-day, 30-day). Each model uses Bitcoin OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance and is evaluated using consistent financial metrics to enable fair comparison across different eras of time series methodology.

The progression demonstrates how forecasting techniques evolved from statistical foundations requiring manual parameter tuning to modern deep learning approaches that automatically learn complex temporal patterns. Bitcoin's high volatility, non-stationarity, and regime changes make it an ideal testbed for evaluating how each generation of models handles real-world financial time series challenges.

Each implementation follows a consistent structure: data loading from Yahoo Finance, feature engineering (where applicable), model training with temporal cross-validation, evaluation using both forecasting and financial metrics, and trading strategy backtesting. This standardization enables meaningful comparison across different methodological eras, revealing how advances in machine learning and deep learning address fundamental challenges in financial time series prediction.

## Evolution Timeline

| Era | Period | Key Characteristics | Representative Models |
|-----|--------|---------------------|---------------------|
| **Era 1: Traditional Statistical Methods** | 1970s-2000s | Manual model identification, statistical diagnostics, linear assumptions | ARIMA, Exponential Smoothing, Prophet |
| **Era 2: Machine Learning Transition** | 2000s-2010s | Feature engineering, non-linear patterns, ensemble methods | Random Forest, Gradient Boosting, SVR |
| **Era 3: Deep Learning Revolution** | 2010s-2020 | Automatic feature learning, sequence modeling, attention mechanisms | LSTM, GRU, Transformer, LSTM Autoencoder |
| **Era 4: Modern Foundation Models** | 2020-Present | Probabilistic forecasting, interpretability, long-range dependencies | TCN, N-BEATS, DeepAR |

## Implementations

### Era 1: Traditional Statistical Methods (1970s-2000s)

#### 001_arima_foundations.py

**Year:** 1970-1976 (Box-Jenkins methodology)

**Innovation:** Systematic approach to ARIMA modeling with diagnostic checking and statistical rigor

**Previous Limitation:** Ad-hoc time series modeling without statistical foundation or systematic methodology

**What Code Implements:**

This implementation demonstrates the classical Box-Jenkins approach to ARIMA modeling applied to Bitcoin price prediction. The code showcases the power and limitations of traditional statistical methods when applied to highly volatile cryptocurrency markets.

Key features include:
- **Model Identification:** ACF/PACF analysis for determining ARIMA(p,d,q) parameters
- **Stationarity Testing:** Augmented Dickey-Fuller (ADF) and KPSS tests for unit root detection
- **Differencing:** Automatic differencing to achieve stationarity
- **Parameter Estimation:** Maximum Likelihood Estimation (MLE) for ARIMA coefficients
- **Model Selection:** AIC/BIC criteria for optimal parameter selection
- **Diagnostic Checking:** Ljung-Box test for residual autocorrelation, Q-Q plots for normality
- **Forecasting:** Multi-step ahead predictions with confidence intervals
- **Seasonal Patterns:** Detection and modeling of Bitcoin seasonal patterns

The implementation reveals ARIMA's strengths in capturing linear trends and autocorrelations, while exposing limitations in handling Bitcoin's non-linear volatility and regime changes. The Box-Jenkins methodology's systematic approach provides a solid foundation for understanding Bitcoin's temporal structure, but the model's linear assumptions struggle with cryptocurrency markets' inherent non-linearity and structural breaks. Results demonstrate ARIMA's effectiveness for short-term Bitcoin forecasting when market conditions remain stable, but performance degrades during volatile periods or regime transitions.

#### 002_exponential_smoothing_trends.py

**Year:** 1950s-1960s (foundational), 2000s (ETS framework)

**Innovation:** Exponential smoothing methods for trend and seasonality decomposition without requiring stationarity

**Previous Limitation:** ARIMA required differencing and struggled with multiplicative seasonality and changing trends

**What Code Implements:**

This implementation demonstrates exponential smoothing methods for Bitcoin price forecasting, focusing on trend and seasonality decomposition without the strict stationarity requirements of ARIMA.

Key features include:
- **Simple Exponential Smoothing:** Single smoothing parameter for level estimation
- **Double Exponential Smoothing (Holt's Method):** Separate smoothing for level and trend components
- **Triple Exponential Smoothing (Holt-Winters):** Adds seasonal component for periodic patterns
- **ETS Models:** Error-Trend-Seasonal framework with additive and multiplicative variants
- **Trend Decomposition:** Automatic extraction of trend, seasonal, and residual components
- **Adaptive Smoothing:** Parameters that adapt to changing Bitcoin market regimes
- **Multiplicative Seasonality:** Handling of Bitcoin's volatility-dependent seasonal patterns

The code illustrates how exponential smoothing provides intuitive trend and seasonality decomposition, making it valuable for understanding Bitcoin's cyclical behavior despite its limitations with non-linear patterns. The ETS framework's flexibility in handling both additive and multiplicative seasonality proves particularly useful for Bitcoin, where volatility often scales with price levels. However, exponential smoothing's reliance on exponential weighting means it responds slowly to sudden Bitcoin market shifts, limiting effectiveness during rapid price movements or market crashes.

#### 003_prophet_business_cycles.py

**Year:** 2017 (Facebook Prophet)

**Innovation:** Business-focused time series forecasting with automatic changepoint detection and holiday effects

**Previous Limitation:** Traditional methods required manual changepoint specification and couldn't incorporate domain knowledge like holidays

**What Code Implements:**

This implementation demonstrates Facebook Prophet applied to Bitcoin price prediction, showcasing how business-focused forecasting tools handle cryptocurrency market dynamics.

Key features include:
- **Additive Model:** Decomposes Bitcoin price into trend, seasonality, and holiday components
- **Changepoint Detection:** Automatic identification of Bitcoin market regime changes and structural breaks
- **Holiday Effects:** Incorporation of cryptocurrency-specific events (halvings, regulatory announcements)
- **Uncertainty Intervals:** Probabilistic forecasts with confidence bounds
- **Cross-Validation:** Time series cross-validation for hyperparameter tuning
- **Flexible Seasonality:** Daily, weekly, and yearly seasonality patterns
- **Robust to Missing Data:** Handles gaps and outliers common in cryptocurrency data

The implementation shows Prophet's strength in interpretability and automatic changepoint detection, valuable for understanding Bitcoin's evolving market structure while maintaining forecasting accuracy. Prophet's ability to incorporate cryptocurrency-specific events like halvings and major regulatory announcements provides domain knowledge integration that purely data-driven methods lack. The automatic changepoint detection successfully identifies Bitcoin market regime transitions, though the additive model assumption may limit performance during periods of multiplicative volatility scaling.

### Era 2: Machine Learning Transition (2000s-2010s)

#### 004_ensemble_classical.py

**Year:** 2001 (Random Forest), 2001-2010 (Gradient Boosting evolution)

**Innovation:** Ensemble methods leveraging multiple decision trees for non-linear pattern recognition in time series

**Previous Limitation:** Statistical methods assumed linear relationships and couldn't capture complex feature interactions

**What Code Implements:**

This implementation demonstrates ensemble machine learning methods (Random Forest and Gradient Boosting) applied to Bitcoin price prediction, marking the transition from statistical to machine learning approaches.

Key features include:
- **Random Forest:** Ensemble of decision trees with bootstrap aggregation for Bitcoin forecasting
- **Gradient Boosting:** Sequential tree building minimizing prediction errors
- **Technical Indicator Features:** SMA, EMA, RSI, MACD, Bollinger Bands as engineered features
- **Lag Features:** Historical price and volume features at multiple time horizons
- **Feature Importance:** Identification of most predictive indicators for Bitcoin
- **Non-Linear Patterns:** Capturing complex relationships between technical indicators and price movements
- **Multi-Horizon Forecasting:** Separate models for 1-day, 7-day, and 30-day predictions

The code demonstrates how feature engineering combined with ensemble methods can capture non-linear Bitcoin patterns, though requiring domain expertise for indicator selection. Random Forest's ability to handle high-dimensional feature spaces with technical indicators proves valuable, while Gradient Boosting's sequential error correction captures complex Bitcoin price dynamics. Feature importance analysis reveals which technical indicators (RSI, MACD, Bollinger Bands) contribute most to Bitcoin prediction, providing interpretability absent in deep learning approaches. However, the need for manual feature engineering represents a limitation compared to end-to-end learning approaches.

#### 005_svm_regression.py

**Year:** 1995 (Support Vector Machines), 2000s (SVR for time series)

**Innovation:** Support Vector Regression with kernel methods for non-linear time series patterns

**Previous Limitation:** Linear models couldn't capture non-linear relationships in Bitcoin's price dynamics

**What Code Implements:**

This implementation demonstrates Support Vector Regression (SVR) applied to Bitcoin price prediction, showcasing kernel methods for non-linear pattern recognition.

Key features include:
- **Kernel Methods:** RBF, polynomial, and sigmoid kernels for non-linear transformations
- **Feature Scaling:** Standardization critical for SVR performance
- **Hyperparameter Optimization:** Grid search for C, epsilon, and kernel parameters
- **Non-Linear Patterns:** Capturing complex relationships between Bitcoin features and price
- **Sparse Solution:** Support vectors identifying critical data points
- **Regularization:** C parameter controlling bias-variance tradeoff
- **Multi-Feature Input:** OHLCV data and technical indicators as input features

The implementation illustrates SVR's ability to find non-linear decision boundaries in Bitcoin data, though computational complexity limits scalability compared to tree-based methods. The RBF kernel's ability to capture complex non-linear relationships proves effective for Bitcoin's price dynamics, while the sparse solution (support vectors) identifies critical historical patterns influencing predictions. However, SVR's O(n²) training complexity and sensitivity to hyperparameters make it less practical for large-scale Bitcoin datasets compared to ensemble methods or deep learning approaches.

### Era 3: Deep Learning Revolution (2010s-2020)

#### 006_lstm_breakthrough.py

**Year:** 1997 (Hochreiter & Schmidhuber foundational), 2011-2017 (popularity in finance)

**Innovation:** Long Short-Term Memory networks with gating mechanisms solving vanishing gradients for long sequence modeling

**Previous Limitation:** Traditional RNNs couldn't learn long-term dependencies in financial time series due to vanishing gradients

**What Code Implements:**

This implementation demonstrates LSTM applied to Bitcoin price prediction with multi-variate time series, showcasing deep learning's automatic feature learning capabilities.

Key features include:
- **Multi-Variate Input:** OHLCV data plus technical indicators as sequential features
- **Sequence-to-Sequence Architecture:** Encoder-decoder LSTM for multi-step forecasting
- **Attention Mechanisms:** Temporal attention for focusing on relevant historical patterns
- **Volatility Modeling:** Separate LSTM branches for price and volatility prediction
- **Gating Mechanisms:** Forget, input, and output gates controlling information flow
- **Long-Range Dependencies:** 60-day lookback windows capturing extended Bitcoin patterns
- **Multi-Horizon Forecasting:** Simultaneous prediction of 1-day, 7-day, and 30-day horizons

The code demonstrates LSTM's breakthrough in learning complex temporal patterns automatically, revolutionizing time series forecasting by eliminating manual feature engineering requirements. The gating mechanisms successfully prevent vanishing gradients, enabling the model to learn long-term Bitcoin dependencies spanning weeks or months. Attention mechanisms further enhance performance by focusing on relevant historical patterns, while the multi-variate architecture captures interactions between price, volume, and technical indicators. Results show significant improvements over statistical and classical ML methods, particularly for multi-horizon Bitcoin forecasting.

#### 007_gru_efficiency.py

**Year:** 2014 (Cho et al., GRU introduction)

**Innovation:** Gated Recurrent Unit with simplified gating for computational efficiency while maintaining LSTM-like performance

**Previous Limitation:** LSTM's three gates (forget, input, output) created computational overhead

**What Code Implements:**

This implementation demonstrates GRU applied to Bitcoin price prediction, comparing efficiency and performance against LSTM.

Key features include:
- **Simplified Gating:** Reset and update gates reducing parameters compared to LSTM
- **Computational Efficiency:** Faster training and inference with fewer parameters
- **Multi-Step Forecasting:** Direct multi-horizon prediction without separate models
- **Gating Comparison:** Analysis of reset vs update gate contributions to Bitcoin forecasting
- **Memory Efficiency:** Lower memory footprint enabling longer sequences
- **Performance Benchmarking:** Direct comparison with LSTM on Bitcoin data

The implementation shows GRU achieving comparable Bitcoin forecasting performance to LSTM with reduced computational cost, making it attractive for production systems. The simplified gating mechanism (reset and update gates versus LSTM's three gates) reduces parameters by approximately 33% while maintaining similar predictive accuracy on Bitcoin data. This efficiency gain enables faster training iterations and real-time inference, critical for cryptocurrency trading applications. Benchmarking reveals GRU's performance is particularly competitive for Bitcoin's shorter-term forecasting horizons (1-day, 7-day), with minimal degradation compared to LSTM.

#### 008_transformer_attention.py

**Year:** 2017 (Vaswani et al., Transformer architecture), 2019-2020 (time series adaptation)

**Innovation:** Transformer architecture with self-attention mechanisms for parallelizable sequence modeling

**Previous Limitation:** RNNs (LSTM/GRU) required sequential processing limiting parallelization

**What Code Implements:**

This implementation demonstrates Transformer architecture applied to Bitcoin time series forecasting, showcasing attention mechanisms for temporal pattern recognition.

Key features include:
- **Temporal Attention:** Self-attention identifying relevant historical Bitcoin patterns
- **Position Encoding:** Sinusoidal and learned positional encodings for temporal order
- **Multi-Head Attention:** Multiple attention heads capturing different temporal relationships
- **Parallel Processing:** Full sequence processing enabling efficient GPU utilization
- **Multi-Horizon Forecasting:** Simultaneous 1-day, 7-day, and 30-day predictions
- **Encoder-Decoder Architecture:** Separate encoding of historical patterns and decoding for forecasts
- **Long-Range Dependencies:** Attention mechanism capturing relationships across entire sequence

The code illustrates Transformer's ability to model long-range Bitcoin dependencies with parallel processing, though requiring more data than RNNs for effective training. Self-attention mechanisms successfully identify relevant historical Bitcoin patterns across the entire sequence, enabling the model to capture relationships spanning the full 60-day lookback window simultaneously. Multi-head attention captures different types of temporal relationships (trends, cycles, volatility patterns), while positional encodings maintain temporal order information. The parallelizable architecture enables efficient GPU utilization, though the quadratic attention complexity limits sequence length scalability compared to linear RNNs.

#### 009_lstm_autoencoder_anomalies.py

**Year:** 2015-2017 (LSTM Autoencoders for anomaly detection)

**Innovation:** LSTM Autoencoder for unsupervised anomaly detection and regime change identification

**Previous Limitation:** Supervised methods required labeled anomalies and couldn't detect novel market regimes

**What Code Implements:**

This implementation demonstrates LSTM Autoencoder applied to Bitcoin anomaly detection and market crash prediction, showcasing unsupervised learning for financial time series.

Key features include:
- **Encoder-Decoder Architecture:** LSTM encoder compressing Bitcoin patterns, decoder reconstructing
- **Anomaly Detection:** Reconstruction error identifying unusual Bitcoin market behavior
- **Market Crash Detection:** Identifying Bitcoin price crashes and extreme volatility events
- **Regime Change Identification:** Detecting transitions between Bitcoin market regimes
- **Unsupervised Learning:** No labeled anomalies required, learns normal Bitcoin patterns
- **Reconstruction Error:** Threshold-based anomaly scoring from prediction residuals
- **Feature Learning:** Automatic extraction of Bitcoin patterns without manual engineering

The implementation demonstrates how autoencoders can identify Bitcoin anomalies and regime changes without supervision, valuable for risk management and market monitoring. The encoder-decoder architecture learns a compressed representation of normal Bitcoin market behavior, with reconstruction error serving as an anomaly score. High reconstruction errors successfully flag Bitcoin market crashes, extreme volatility events, and structural breaks. The unsupervised approach eliminates the need for labeled anomaly data, which is particularly valuable for cryptocurrency markets where anomaly definitions evolve with market maturity. Results show the autoencoder effectively identifies Bitcoin regime transitions and extreme events that supervised methods might miss.

### Era 4: Modern Foundation Models (2020-Present)

#### 010_tcn_modern.py

**Year:** 2018 (Bai et al., Temporal Convolutional Networks)

**Innovation:** Temporal Convolutional Networks with dilated causal convolutions for long-range dependencies

**Previous Limitation:** RNNs required sequential processing, Transformers needed large datasets

**What Code Implements:**

This implementation demonstrates TCN applied to Bitcoin price prediction, showcasing modern convolutional approaches to time series forecasting.

Key features include:
- **Dilated Causal Convolutions:** Exponentially increasing receptive fields capturing long Bitcoin patterns
- **Long-Range Dependencies:** Efficient modeling of extended temporal relationships
- **Parallelizable:** Convolutional operations enabling full parallelization
- **Residual Connections:** Skip connections preventing gradient degradation in deep networks
- **Weight Normalization:** Stabilizing training for Bitcoin's volatile patterns
- **Multi-Scale Features:** Different dilation rates capturing Bitcoin patterns at various time scales
- **Computational Efficiency:** Faster than RNNs while maintaining long-range modeling

The code demonstrates TCN's combination of RNN-like sequence modeling with CNN's parallelization benefits, achieving efficient long-range Bitcoin pattern recognition. Dilated convolutions with exponentially increasing dilation rates create receptive fields spanning the entire sequence while maintaining computational efficiency through parallel processing. Residual connections prevent gradient degradation in deep networks, enabling effective training on Bitcoin's complex patterns. Weight normalization stabilizes training dynamics for volatile cryptocurrency data. Results show TCN achieves comparable or superior performance to LSTM/GRU with significantly faster training times, making it attractive for production Bitcoin forecasting systems requiring both accuracy and efficiency.

#### 011_nbeats_interpretable.py

**Year:** 2020 (Oreshkin et al., N-BEATS)

**Innovation:** Pure deep learning approach with interpretable trend and seasonality decomposition without feature engineering

**Previous Limitation:** Deep learning models lacked interpretability and required manual feature engineering

**What Code Implements:**

This implementation demonstrates N-BEATS applied to Bitcoin price prediction, showcasing interpretable deep learning for time series forecasting.

Key features include:
- **Interpretable Deep Learning:** Trend and seasonality blocks with explicit decomposition
- **No Feature Engineering:** Pure deep learning using only raw Bitcoin OHLCV data
- **Backcast/Forecast:** Each block produces both reconstruction (backcast) and prediction (forecast)
- **Basis Expansion:** Polynomial basis for trends, Fourier basis for seasonality
- **Stacked Architecture:** Multiple blocks refining predictions iteratively
- **Trend Decomposition:** Explicit extraction of Bitcoin's underlying trend component
- **Seasonality Decomposition:** Identification of periodic Bitcoin patterns
- **State-of-the-Art Performance:** Achieving top results on time series benchmarks

The implementation shows how N-BEATS achieves interpretability through explicit decomposition while maintaining deep learning's automatic pattern learning, providing insights into Bitcoin's trend and seasonal components. The stacked architecture with trend and seasonality blocks enables explicit extraction of Bitcoin's underlying trend and periodic patterns, providing interpretability absent in black-box deep learning models. Each block's backcast (reconstruction) and forecast (prediction) outputs enable analysis of how different components contribute to Bitcoin predictions. The pure deep learning approach eliminates feature engineering requirements while achieving state-of-the-art performance on time series benchmarks. Results demonstrate N-BEATS successfully decomposes Bitcoin price into interpretable trend and seasonality components while maintaining forecasting accuracy competitive with or superior to other deep learning approaches.

#### 012_deepar_probabilistic.py

**Year:** 2020 (Salinas et al., DeepAR)

**Innovation:** Probabilistic forecasting with autoregressive RNNs and learned likelihood distributions for uncertainty quantification

**Previous Limitation:** Point forecasts without uncertainty quantification inadequate for financial risk management

**What Code Implements:**

This implementation demonstrates DeepAR applied to Bitcoin price prediction, showcasing probabilistic forecasting for risk-aware decision making.

Key features include:
- **Probabilistic Forecasting:** Gaussian likelihood producing full predictive distributions
- **Autoregressive RNN:** LSTM/GRU generating parameters of probability distributions
- **Uncertainty Quantification:** Prediction intervals and quantiles for risk assessment
- **Quantile Predictions:** 10th, 50th, 90th percentiles for Bitcoin price ranges
- **Multiple Likelihoods:** Gaussian, Student-t distributions for different Bitcoin volatility regimes
- **Scalable Architecture:** Amazon's production forecasting approach handling multiple time series
- **Risk Management:** Uncertainty bounds enabling informed Bitcoin trading decisions

The code demonstrates DeepAR's ability to provide probabilistic Bitcoin forecasts with uncertainty quantification, critical for financial applications requiring risk assessment beyond point predictions. The autoregressive RNN architecture generates parameters of probability distributions (mean and variance for Gaussian likelihood), enabling full predictive distributions rather than single point estimates. Quantile predictions (10th, 50th, 90th percentiles) provide risk managers with price range estimates essential for position sizing and risk management. Multiple likelihood distributions (Gaussian for normal periods, Student-t for high volatility) adapt to different Bitcoin market regimes. Results show DeepAR's probabilistic forecasts enable more informed trading decisions compared to point forecasts, with uncertainty bounds accurately reflecting Bitcoin's inherent volatility and prediction difficulty.

## Comparison Table

| File | Model | Year | Type | Key Innovation | Primary Metric |
|------|-------|------|------|----------------|----------------|
| 001_arima_foundations.py | ARIMA | 1970-1976 | Statistical | Box-Jenkins systematic methodology | AIC/BIC, RMSE |
| 002_exponential_smoothing_trends.py | ETS | 1950s-2000s | Statistical | Trend/seasonality decomposition | MAPE, AIC |
| 003_prophet_business_cycles.py | Prophet | 2017 | Statistical | Changepoint detection, holiday effects | MAPE, Cross-validation |
| 004_ensemble_classical.py | Random Forest/GBM | 2001-2010 | ML | Non-linear ensemble methods | RMSE, Feature Importance |
| 005_svm_regression.py | SVR | 1995-2000s | ML | Kernel methods for non-linearity | RMSE, R² |
| 006_lstm_breakthrough.py | LSTM | 1997-2017 | DL | Gating mechanisms, long dependencies | RMSE, MAE |
| 007_gru_efficiency.py | GRU | 2014 | DL | Simplified gating, efficiency | RMSE, Training Time |
| 008_transformer_attention.py | Transformer | 2017-2020 | DL | Self-attention, parallelization | RMSE, Attention Weights |
| 009_lstm_autoencoder_anomalies.py | LSTM Autoencoder | 2015-2017 | DL | Unsupervised anomaly detection | Reconstruction Error |
| 010_tcn_modern.py | TCN | 2018 | Modern | Dilated convolutions, long-range | RMSE, Receptive Field |
| 011_nbeats_interpretable.py | N-BEATS | 2020 | Modern | Interpretable DL decomposition | RMSE, Trend/Seasonality |
| 012_deepar_probabilistic.py | DeepAR | 2020 | Modern | Probabilistic forecasting | CRPS, Quantile Loss |

## Dataset and Evaluation Framework

### Dataset

All implementations use Bitcoin OHLCV data with the following characteristics:

- **Source:** Yahoo Finance (yfinance library)
- **Time Period:** 2010-present (Bitcoin's full trading history)
- **Frequency:** Daily OHLCV (Open, High, Low, Close, Volume)
- **Technical Indicators:** Computed features including:
  - Simple Moving Average (SMA) at multiple windows (7, 14, 30, 50, 200 days)
  - Exponential Moving Average (EMA) with various decay factors
  - Relative Strength Index (RSI) for momentum analysis
  - Moving Average Convergence Divergence (MACD) with signal line
  - Bollinger Bands (upper, middle, lower) for volatility bands
  - Volume-based indicators (On-Balance Volume, Volume Rate of Change)
  - Price-based indicators (Average True Range, Commodity Channel Index)

### Forecasting Metrics

Standard time series evaluation metrics applied consistently:

- **MAE (Mean Absolute Error):** Average absolute prediction error
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily
- **MAPE (Mean Absolute Percentage Error):** Percentage-based error for scale-independent comparison
- **Directional Accuracy:** Percentage of correct price direction predictions (up/down)

### Financial Metrics

Domain-specific metrics for Bitcoin trading evaluation:

- **Sharpe Ratio:** Risk-adjusted returns measuring excess return per unit volatility
- **Max Drawdown:** Maximum peak-to-trough decline during trading period
- **Calmar Ratio:** Annualized return divided by maximum drawdown
- **Win Rate:** Percentage of profitable trades
- **Total Return:** Cumulative return from trading strategy

### Trading Framework

Consistent trading evaluation across all models:

- **Signals:** Buy/sell signals generated from model predictions
- **Transaction Costs:** 0.1% per trade (realistic cryptocurrency exchange fees)
- **Benchmark:** Comparison against buy-and-hold Bitcoin strategy
- **Risk Management:** Position sizing and stop-loss considerations

### Training Methodology

Standardized training approach for fair comparison:

- **Temporal Split:** 70% training, 15% validation, 15% test (chronological order, respecting Bitcoin's temporal structure)
- **Lookback Window:** 60 days of historical data for predictions (consistent across all models for fair comparison)
- **Walk-Forward Validation:** Time series cross-validation respecting temporal order, preventing data leakage
- **Hyperparameter Tuning:** Grid search or Bayesian optimization on validation set, avoiding test set contamination
- **Early Stopping:** Preventing overfitting with validation loss monitoring and patience mechanisms
- **Data Preprocessing:** Standardization/normalization applied consistently, with handling of missing values and outliers
- **Feature Scaling:** Min-Max scaling or standardization depending on model requirements (critical for SVR, less important for tree-based methods)

## Key Takeaways

1. **Evolution from Manual to Automatic:** The progression from ARIMA's manual parameter selection to deep learning's automatic feature learning demonstrates how time series forecasting has become increasingly automated. Modern models like N-BEATS and DeepAR require minimal domain expertise while achieving superior performance.

2. **Non-Linearity and Regime Changes:** Bitcoin's high volatility and frequent regime changes expose limitations of linear statistical methods (ARIMA, exponential smoothing). Machine learning (ensemble methods) and deep learning (LSTM, Transformer) models better capture non-linear patterns and adapt to changing market conditions.

3. **Uncertainty Quantification Critical:** Point forecasts alone are insufficient for Bitcoin trading. Probabilistic models like DeepAR provide uncertainty bounds essential for risk management. The evolution toward probabilistic forecasting reflects the financial industry's need for risk-aware predictions.

4. **Interpretability vs Performance Trade-off:** Early statistical methods (ARIMA, Prophet) offer interpretability but limited performance. Deep learning models (LSTM, Transformer) achieve higher accuracy but lack interpretability. Modern approaches like N-BEATS bridge this gap by providing interpretable decompositions within deep learning frameworks.

5. **Computational Efficiency Matters:** The transition from sequential RNNs (LSTM, GRU) to parallelizable architectures (Transformer, TCN) reflects the importance of computational efficiency for production systems. GRU's efficiency gains over LSTM, and TCN's combination of long-range modeling with parallelization, demonstrate ongoing optimization for real-world deployment.
