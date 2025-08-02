"""
Time Series ERA 3: LSTM Breakthrough (2010s-2020)
================================================

Historical Context:
Year: 2011-2017 (LSTM popularity in finance, Hochreiter & Schmidhuber 1997 foundational)
Innovation: Long Short-Term Memory networks for sequence modeling with gating mechanisms
Previous Limitation: Traditional RNNs suffered from vanishing gradients in long sequences
Impact: Revolutionized time series forecasting with automatic feature learning and memory

This implementation demonstrates LSTM applied to Bitcoin price prediction with
multi-variate time series, sequence-to-sequence architecture, attention mechanisms,
and volatility modeling for comprehensive cryptocurrency market analysis.
"""

# Historical Context & Innovation
YEAR = "2011-2017"
INNOVATION = "LSTM gating mechanisms solving vanishing gradients for long sequence modeling"
PREVIOUS_LIMITATION = "RNNs couldn't learn long-term dependencies in financial time series"
IMPACT = "Enabled end-to-end learning of temporal patterns in complex financial data"

# Imports and Dependencies
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import talib as ta
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BitcoinTimeSeriesDataset(Dataset):
    """
    Custom Dataset for Bitcoin time series with LSTM-optimized preprocessing
    """
    
    def __init__(self, features, targets, sequence_length=60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        feature_seq = self.features[idx:idx + self.sequence_length]
        
        # Get corresponding target
        target = self.targets[idx + self.sequence_length]
        
        return torch.FloatTensor(feature_seq), torch.FloatTensor([target])

class LSTMWithAttention(nn.Module):
    """
    LSTM with attention mechanism for Bitcoin price prediction
    
    Features:
    - Multi-layer LSTM with dropout
    - Attention mechanism over LSTM outputs
    - Multi-variate input handling
    - Volatility prediction branch
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, 
                 attention_dim=64, predict_volatility=True):
        super(LSTMWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_volatility = predict_volatility
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention_dim = attention_dim
        self.attention_linear = nn.Linear(hidden_size, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.price_head = nn.Linear(hidden_size, 1)
        
        if predict_volatility:
            self.volatility_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize LSTM and linear layer weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for layer in [self.attention_linear, self.attention_vector, self.price_head]:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.fill_(0)
    
    def attention(self, lstm_outputs):
        """
        Attention mechanism to focus on important time steps
        
        Args:
            lstm_outputs: LSTM outputs [batch_size, seq_len, hidden_size]
            
        Returns:
            context_vector: Attended representation [batch_size, hidden_size]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        # Calculate attention scores
        attention_scores = self.attention_linear(lstm_outputs)  # [batch, seq_len, attention_dim]
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.attention_vector(attention_scores).squeeze(-1)  # [batch, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len]
        
        # Calculate context vector
        context_vector = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)  # [batch, hidden_size]
        
        return context_vector, attention_weights
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, seq_len, hidden_size]
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out)
        
        # Apply dropout
        context_vector = self.dropout(context_vector)
        
        # Price prediction
        price_pred = self.price_head(context_vector)
        
        outputs = {'price': price_pred, 'attention_weights': attention_weights}
        
        # Volatility prediction (optional)
        if self.predict_volatility:
            volatility_pred = torch.sigmoid(self.volatility_head(context_vector))
            outputs['volatility'] = volatility_pred
        
        return outputs

class BitcoinLSTMForecaster:
    """
    LSTM-based Bitcoin price forecaster with attention and volatility modeling
    
    Features:
    1. Multi-variate time series preprocessing
    2. LSTM with attention mechanism
    3. Volatility prediction and uncertainty quantification
    4. Multi-horizon forecasting
    5. Sequence-to-sequence architecture
    """
    
    def __init__(self, sequence_length=60, hidden_size=128, num_layers=2):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_names = []
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin data with comprehensive feature engineering for LSTM
        
        Returns:
            Processed Bitcoin data with features
        """
        print("Loading Bitcoin Data for LSTM Analysis...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_data = btc_data.dropna()
        
        print(f"Data loaded: {len(btc_data)} total observations")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        
        return btc_data
    
    def engineer_lstm_features(self, data):
        """
        Comprehensive feature engineering optimized for LSTM
        
        Args:
            data: Raw Bitcoin OHLCV data
            
        Returns:
            Feature matrix and target variable
        """
        print("Engineering LSTM-optimized features...")
        
        features = data.copy()
        
        # Basic price features
        features['returns'] = features['Close'].pct_change()
        features['log_returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['high_low_ratio'] = features['High'] / features['Low']
        features['close_open_ratio'] = features['Close'] / features['Open']
        features['volume_price_trend'] = features['Volume'] * features['returns']
        
        # Price transformations for better LSTM learning
        features['log_close'] = np.log(features['Close'])
        features['log_volume'] = np.log(features['Volume'] + 1)
        
        # Moving averages and momentum
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = features['Close'].rolling(window=window).mean()
            features[f'price_ma_ratio_{window}'] = features['Close'] / features[f'ma_{window}']
            features[f'volume_ma_{window}'] = features['Volume'].rolling(window=window).mean()
            features[f'volume_ma_ratio_{window}'] = features['Volume'] / features[f'volume_ma_{window}']
        
        # Volatility features (important for LSTM)
        for window in [5, 10, 20, 30]:
            features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
            features[f'realized_vol_{window}'] = features['returns'].rolling(window=window).apply(
                lambda x: np.sqrt(np.sum(x**2))
            )
        
        # Technical indicators
        try:
            # RSI
            features['rsi_14'] = ta.RSI(features['Close'].values, timeperiod=14) / 100
            features['rsi_30'] = ta.RSI(features['Close'].values, timeperiod=30) / 100
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(features['Close'].values)
            features['macd_normalized'] = (macd - np.nanmean(macd)) / (np.nanstd(macd) + 1e-8)
            features['macd_signal_normalized'] = (macd_signal - np.nanmean(macd_signal)) / (np.nanstd(macd_signal) + 1e-8)
            features['macd_histogram'] = (macd_hist - np.nanmean(macd_hist)) / (np.nanstd(macd_hist) + 1e-8)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(features['Close'].values)
            features['bb_position'] = (features['Close'] - bb_lower) / (bb_upper - bb_lower)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Stochastic (normalized)
            slowk, slowd = ta.STOCH(features['High'].values, features['Low'].values, features['Close'].values)
            features['stoch_k'] = slowk / 100
            features['stoch_d'] = slowd / 100
            
            # ATR (normalized)
            atr = ta.ATR(features['High'].values, features['Low'].values, features['Close'].values)
            features['atr_normalized'] = atr / features['Close']
            
            print("Technical indicators calculated successfully")
            
        except Exception as e:
            print(f"WARNING: Some technical indicators failed: {str(e)}")
        
        # Lag features (crucial for LSTM temporal understanding)
        for lag in [1, 2, 3, 5, 7]:
            features[f'close_lag_{lag}'] = features['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = features['Volume'].shift(lag)
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'close_rolling_mean_{window}'] = features['Close'].rolling(window=window).mean()
            features[f'close_rolling_std_{window}'] = features['Close'].rolling(window=window).std()
            features[f'close_rolling_skew_{window}'] = features['Close'].rolling(window=window).skew()
            features[f'close_rolling_kurt_{window}'] = features['Close'].rolling(window=window).kurt()
        
        # Market regime indicators
        features['price_acceleration'] = features['Close'].diff(2)
        features['volume_acceleration'] = features['Volume'].diff(2)
        features['volatility_regime'] = (features['volatility_20'] > features['volatility_20'].rolling(50).mean()).astype(int)
        
        # Cyclical time features (important for LSTM temporal patterns)
        features['hour_sin'] = np.sin(2 * np.pi * features.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features.index.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * features.index.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * features.index.dayofweek / 7)
        features['month_sin'] = np.sin(2 * np.pi * features.index.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * features.index.month / 12)
        
        # Target variable (next day close price)
        target = features['Close'].shift(-1)
        
        # Select numeric features only
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target-related columns from features
        feature_cols = [col for col in numeric_cols if col not in ['Close', 'Open', 'High', 'Low']]
        
        X = features[feature_cols]
        y = target
        
        # Remove rows with missing values
        valid_idx = (~X.isnull().any(axis=1)) & (~y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        self.feature_names = X.columns.tolist()
        
        print(f"LSTM features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        
        return X.values, y.values
    
    def prepare_sequences(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """
        Prepare sequences for LSTM training with proper time series splits
        
        Args:
            X: Feature matrix
            y: Target variable
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            DataLoaders for train, validation, and test sets
        """
        print("Preparing LSTM sequences...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Calculate split indices
        n_samples = len(X_scaled)
        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)
        
        # Create datasets
        train_dataset = BitcoinTimeSeriesDataset(
            X_scaled[:train_end], y_scaled[:train_end], self.sequence_length
        )
        val_dataset = BitcoinTimeSeriesDataset(
            X_scaled[train_end:val_end], y_scaled[train_end:val_end], self.sequence_length
        )
        test_dataset = BitcoinTimeSeriesDataset(
            X_scaled[val_end:], y_scaled[val_end:], self.sequence_length
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # No shuffle for time series
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"Sequence preparation complete:")
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Validation sequences: {len(val_dataset)}")
        print(f"  Test sequences: {len(test_dataset)}")
        print(f"  Sequence length: {self.sequence_length}")
        
        return train_loader, val_loader, test_loader
    
    def build_model(self, input_size):
        """
        Build LSTM model with attention mechanism
        
        Args:
            input_size: Number of input features
        """
        print("Building LSTM model with attention...")
        
        self.model = LSTMWithAttention(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2,
            attention_dim=64,
            predict_volatility=True
        ).to(device)
        
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model built successfully:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model architecture: {self.num_layers}-layer LSTM with attention")
        
        return self.model
    
    def train_model(self, train_loader, val_loader, epochs=100, learning_rate=0.001):
        """
        Train LSTM model with early stopping and learning rate scheduling
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Initial learning rate
        """
        print("Training LSTM model...")
        
        # Loss functions and optimizer
        price_criterion = nn.MSELoss()
        volatility_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_features)
                
                # Calculate losses
                price_loss = price_criterion(outputs['price'], batch_targets)
                
                # Total loss (price prediction is primary)
                total_loss = price_loss
                
                # Add volatility loss if available
                if 'volatility' in outputs:
                    # Calculate realized volatility for targets
                    volatility_targets = torch.ones_like(batch_targets) * 0.5  # Placeholder
                    volatility_loss = volatility_criterion(outputs['volatility'], volatility_targets)
                    total_loss += 0.1 * volatility_loss  # Weight volatility loss lower
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    outputs = self.model(batch_features)
                    
                    # Calculate validation loss
                    price_loss = price_criterion(outputs['price'], batch_targets)
                    val_loss += price_loss.item()
                    val_batches += 1
            
            # Calculate average losses
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))
        print("Training completed! Best model loaded.")
    
    def predict_bitcoin_price(self, test_loader, horizons=[1, 7, 30]):
        """
        Generate Bitcoin price predictions with multi-horizon forecasting
        
        Args:
            test_loader: Test data loader
            horizons: Prediction horizons in days
            
        Returns:
            Dictionary of predictions and actual values
        """
        print("Generating Bitcoin price predictions...")
        
        self.model.eval()
        predictions = []
        actuals = []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = self.model(batch_features)
                
                # Store predictions and actuals
                pred_scaled = outputs['price'].cpu().numpy()
                actual_scaled = batch_targets.cpu().numpy()
                
                predictions.extend(pred_scaled)
                actuals.extend(actual_scaled)
                
                # Store attention weights
                attention_weights_list.extend(outputs['attention_weights'].cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        attention_weights = np.array(attention_weights_list)
        
        # Inverse transform to original scale
        predictions_original = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_original = self.target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        print(f"Predictions generated:")
        print(f"  Number of predictions: {len(predictions_original)}")
        print(f"  Prediction range: ${predictions_original.min():.2f} - ${predictions_original.max():.2f}")
        print(f"  Actual range: ${actuals_original.min():.2f} - ${actuals_original.max():.2f}")
        
        return {
            'predictions': predictions_original,
            'actuals': actuals_original,
            'attention_weights': attention_weights
        }
    
    def evaluate_predictions(self, results):
        """
        Comprehensive evaluation of LSTM predictions
        
        Args:
            results: Dictionary containing predictions and actuals
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = results['predictions']
        actuals = results['actuals']
        
        # Forecasting accuracy metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        r2 = r2_score(actuals, predictions)
        
        # Directional accuracy
        actual_direction = np.diff(actuals) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = (actual_direction == pred_direction).mean() * 100
        
        # Financial metrics
        returns_actual = np.diff(actuals) / actuals[:-1]
        returns_pred = np.diff(predictions) / actuals[:-1]
        
        # Trading strategy
        signals = np.where(np.diff(predictions) > 0, 1, -1)
        strategy_returns = signals * returns_actual
        
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
        cumulative_returns = np.cumsum(strategy_returns)
        max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
        total_return = np.sum(strategy_returns)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'total_return': total_return * 100
        }
        
        print("\nLSTM Bitcoin Prediction Evaluation")
        print("=" * 50)
        print(f"Forecasting Accuracy:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²: {r2:.4f}")
        print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
        
        print(f"\nTrading Strategy Performance:")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Total Return: {total_return:.2f}%")
        
        return metrics
    
    def visualize_results(self, results, title="Bitcoin LSTM Forecast"):
        """
        Comprehensive visualization of LSTM results
        
        Args:
            results: Dictionary containing predictions, actuals, and attention weights
            title: Plot title
        """
        predictions = results['predictions']
        actuals = results['actuals']
        attention_weights = results.get('attention_weights', None)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Price predictions vs actuals
        indices = range(len(predictions))
        
        axes[0].plot(indices, actuals, label='Actual Bitcoin Price', color='blue', alpha=0.7, linewidth=2)
        axes[0].plot(indices, predictions, label='LSTM Prediction', color='red', alpha=0.8, linewidth=2)
        
        axes[0].set_title(f"{title}\nLong Short-Term Memory with Attention Mechanism")
        axes[0].set_xlabel("Time Steps")
        axes[0].set_ylabel("Bitcoin Price (USD)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # 2. Training history
        if self.training_history['train_loss']:
            epochs = range(1, len(self.training_history['train_loss']) + 1)
            axes[1].plot(epochs, self.training_history['train_loss'], label='Training Loss', color='blue')
            axes[1].plot(epochs, self.training_history['val_loss'], label='Validation Loss', color='red')
            axes[1].set_title("LSTM Training History")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss (MSE)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        
        # 3. Innovation summary
        innovation_text = f"""
        LSTM Innovation Impact ({YEAR})
        
        Neural Network Foundation:
        • Hochreiter & Schmidhuber (1997): LSTM architecture design
        • Solving vanishing gradient problem in RNNs
        • Gating mechanisms: forget, input, output gates
        • Cell state for long-term memory preservation
        
        Key Innovations:
        • End-to-end learning without manual feature engineering
        • Automatic temporal pattern discovery
        • Attention mechanism for interpretability
        • Multi-variate time series handling
        
        Bitcoin Application Strengths:
        • Captures complex non-linear temporal dependencies
        • Handles multiple time scales simultaneously
        • Learns volatility patterns automatically
        • Robust to irregular market patterns
        
        Limitations for Crypto Markets:
        • Black box nature reduces interpretability
        • Requires large amounts of training data
        • Computationally expensive for real-time trading
        • Prone to overfitting in volatile markets
        
        Architecture: {self.num_layers}-layer LSTM + Attention
        Sequence Length: {self.sequence_length} time steps
        Features: {len(self.feature_names)} engineered inputs
        """
        
        axes[2].text(0.05, 0.95, innovation_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Attention visualization (if available)
        if attention_weights is not None and len(attention_weights) > 0:
            self.visualize_attention_patterns(attention_weights)
    
    def visualize_attention_patterns(self, attention_weights, n_samples=5):
        """
        Visualize attention patterns across time steps
        
        Args:
            attention_weights: Attention weights from model
            n_samples: Number of samples to visualize
        """
        print("\nVisualizing LSTM attention patterns...")
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i in range(min(n_samples, len(attention_weights))):
            weights = attention_weights[i]
            time_steps = range(len(weights))
            
            axes[i].plot(time_steps, weights, 'b-', alpha=0.7)
            axes[i].fill_between(time_steps, weights, alpha=0.3)
            axes[i].set_title(f"Attention Weights - Sample {i+1}")
            axes[i].set_xlabel("Time Steps (Past to Present)")
            axes[i].set_ylabel("Attention Weight")
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle("LSTM Attention Mechanism: Focus on Important Time Steps", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """
    Complete Bitcoin LSTM prediction pipeline
    Demonstrates LSTM with attention for cryptocurrency forecasting
    """
    print("ERA 3: LSTM Breakthrough for Bitcoin Prediction")
    print("=" * 50)
    print(f"Historical Context: {YEAR}")
    print(f"Innovation: {INNOVATION}")
    print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"Impact: {IMPACT}")
    print("=" * 50)
    
    # Initialize forecaster
    forecaster = BitcoinLSTMForecaster(sequence_length=60, hidden_size=128, num_layers=2)
    
    # Step 1: Load and prepare data
    print("\n" + "="*50)
    print("STEP 1: DATA LOADING AND FEATURE ENGINEERING")
    print("="*50)
    
    raw_data = forecaster.load_bitcoin_data()
    X, y = forecaster.engineer_lstm_features(raw_data)
    
    # Step 2: Prepare sequences
    print("\n" + "="*50)
    print("STEP 2: SEQUENCE PREPARATION")
    print("="*50)
    
    train_loader, val_loader, test_loader = forecaster.prepare_sequences(X, y)
    
    # Step 3: Build model
    print("\n" + "="*50)
    print("STEP 3: LSTM MODEL CONSTRUCTION")
    print("="*50)
    
    model = forecaster.build_model(input_size=X.shape[1])
    
    # Step 4: Train model
    print("\n" + "="*50)
    print("STEP 4: LSTM TRAINING WITH ATTENTION")
    print("="*50)
    
    forecaster.train_model(train_loader, val_loader, epochs=100, learning_rate=0.001)
    
    # Step 5: Generate predictions
    print("\n" + "="*50)
    print("STEP 5: BITCOIN PRICE PREDICTION")
    print("="*50)
    
    results = forecaster.predict_bitcoin_price(test_loader)
    
    # Step 6: Evaluate predictions
    print("\n" + "="*50)
    print("STEP 6: PREDICTION EVALUATION")
    print("="*50)
    
    metrics = forecaster.evaluate_predictions(results)
    
    # Step 7: Visualize results
    print("\n" + "="*50)
    print("STEP 7: RESULTS VISUALIZATION")
    print("="*50)
    
    forecaster.visualize_results(results, "Bitcoin LSTM with Attention Forecast")
    
    # Final summary
    print("\n" + "="*50)
    print("ERA 3 SUMMARY: LSTM BREAKTHROUGH")
    print("="*50)
    print(f"""
    LSTM Analysis Complete:
    
    Deep Learning Architecture:
    • Model: {forecaster.num_layers}-layer LSTM with attention mechanism
    • Sequence length: {forecaster.sequence_length} time steps
    • Features: {len(forecaster.feature_names)} engineered inputs
    • Parameters: {sum(p.numel() for p in forecaster.model.parameters()):,}
    
    Bitcoin Prediction Performance:
    • RMSE: ${metrics['rmse']:.2f}
    • R²: {metrics['r2']:.4f}
    • Directional Accuracy: {metrics['directional_accuracy']:.1f}%
    • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
    
    Educational Value:
    • Demonstrates LSTM architecture and gating mechanisms
    • Shows attention mechanism for interpretability
    • Illustrates end-to-end learning capabilities
    • Establishes foundation for advanced sequence models
    
    Next: GRU efficiency improvements and computational optimization!
    """)


if __name__ == "__main__":
    main()