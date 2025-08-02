"""
Time Series ERA 3: Transformer Attention (2017-2020)
====================================================

Historical Context:
Year: 2017 (Vaswani et al., "Attention Is All You Need")
Innovation: Self-attention mechanism replacing recurrence for parallel processing
Previous Limitation: RNNs and LSTMs required sequential processing limiting parallelization
Impact: Revolutionized sequence modeling with parallelizable attention and global context

This implementation demonstrates Transformer applied to Bitcoin time series with
temporal attention mechanisms, positional encoding for time sequences,
multi-horizon forecasting, and self-attention visualization for interpretability.
"""

# Historical Context & Innovation
YEAR = "2017"
INNOVATION = "Self-attention mechanism with parallel processing and global temporal context"
PREVIOUS_LIMITATION = "RNNs processed sequences sequentially limiting training speed and global context"
IMPACT = "Enabled parallel training and global attention for superior sequence modeling"

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
import math
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
    """
    Temporal positional encoding for time series Transformer
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for time series
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention with optional masking
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention and feed-forward
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attention_weights = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for Bitcoin time series forecasting
    
    Features:
    - Multi-head self-attention
    - Positional encoding for temporal sequences
    - Multi-horizon prediction capability
    - Attention weight extraction for interpretability
    """
    
    def __init__(self, input_size, d_model=128, num_heads=8, num_layers=6, 
                 d_ff=512, max_seq_len=1000, dropout=0.1, num_horizons=1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_horizons = num_horizons
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, num_horizons)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize transformer weights"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
    
    def forward(self, x, mask=None):
        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch, d_model] for positional encoding
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # Store attention weights for visualization
        all_attention_weights = []
        
        # Pass through transformer layers
        for transformer_layer in self.transformer_layers:
            x, attention_weights = transformer_layer(x, mask)
            all_attention_weights.append(attention_weights)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Use the last time step for prediction
        last_output = x[:, -1, :]  # [batch, d_model]
        
        # Output projection
        output = self.output_projection(last_output)  # [batch, num_horizons]
        
        return output, all_attention_weights

class BitcoinTimeSeriesDataset(Dataset):
    """
    Dataset for Bitcoin time series with Transformer-optimized preprocessing
    """
    
    def __init__(self, features, targets, sequence_length=100, num_horizons=1):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.num_horizons = num_horizons
        
    def __len__(self):
        return len(self.features) - self.sequence_length - self.num_horizons + 1
    
    def __getitem__(self, idx):
        # Input sequence
        feature_seq = self.features[idx:idx + self.sequence_length]
        
        # Multi-horizon targets
        if self.num_horizons == 1:
            target = self.targets[idx + self.sequence_length]
        else:
            target = self.targets[idx + self.sequence_length:idx + self.sequence_length + self.num_horizons]
        
        return torch.FloatTensor(feature_seq), torch.FloatTensor(target)

class BitcoinTransformerForecaster:
    """
    Transformer-based Bitcoin price forecaster with attention visualization
    
    Features:
    1. Multi-head self-attention for temporal patterns
    2. Positional encoding for time series
    3. Multi-horizon forecasting (1d, 7d, 30d)
    4. Attention weight visualization and interpretation
    5. Parallel training efficiency
    """
    
    def __init__(self, sequence_length=100, d_model=128, num_heads=8, num_layers=6):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = None
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_names = []
        self.attention_weights_history = []
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin data with Transformer-optimized feature engineering
        
        Returns:
            Processed Bitcoin data
        """
        print("Loading Bitcoin Data for Transformer Analysis...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_data = btc_data.dropna()
        
        print(f"Data loaded: {len(btc_data)} total observations")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        
        return btc_data
    
    def engineer_transformer_features(self, data):
        """
        Feature engineering optimized for Transformer attention
        
        Args:
            data: Raw Bitcoin OHLCV data
            
        Returns:
            Feature matrix and target variable
        """
        print("Engineering Transformer-optimized features...")
        
        features = data.copy()
        
        # Core price and volume features
        features['returns'] = features['Close'].pct_change()
        features['log_returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['volatility'] = features['returns'].rolling(window=20).std()
        features['volume_change'] = features['Volume'].pct_change()
        
        # Price ratios and trends
        features['high_low_ratio'] = features['High'] / features['Low']
        features['close_open_ratio'] = features['Close'] / features['Open']
        features['hl2'] = (features['High'] + features['Low']) / 2
        features['hlc3'] = (features['High'] + features['Low'] + features['Close']) / 3
        features['ohlc4'] = (features['Open'] + features['High'] + features['Low'] + features['Close']) / 4
        
        # Multi-timeframe moving averages
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            features[f'sma_{period}'] = features['Close'].rolling(window=period).mean()
            features[f'ema_{period}'] = features['Close'].ewm(span=period).mean()
            features[f'price_sma_{period}_ratio'] = features['Close'] / features[f'sma_{period}']
            features[f'price_ema_{period}_ratio'] = features['Close'] / features[f'ema_{period}']
        
        # Volatility indicators
        volatility_periods = [10, 20, 30, 50]
        for period in volatility_periods:
            features[f'volatility_{period}'] = features['returns'].rolling(window=period).std()
            features[f'volatility_{period}_norm'] = features[f'volatility_{period}'] / features['volatility'].mean()
        
        # Technical indicators
        try:
            # RSI with multiple periods
            for period in [14, 30]:
                features[f'rsi_{period}'] = ta.RSI(features['Close'].values, timeperiod=period) / 100
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(features['Close'].values)
            features['macd'] = (macd - np.nanmean(macd)) / (np.nanstd(macd) + 1e-8)
            features['macd_signal'] = (macd_signal - np.nanmean(macd_signal)) / (np.nanstd(macd_signal) + 1e-8)
            features['macd_histogram'] = (macd_hist - np.nanmean(macd_hist)) / (np.nanstd(macd_hist) + 1e-8)
            
            # Bollinger Bands
            for period in [20, 50]:
                bb_upper, bb_middle, bb_lower = ta.BBANDS(features['Close'].values, timeperiod=period)
                features[f'bb_upper_{period}'] = bb_upper
                features[f'bb_lower_{period}'] = bb_lower
                features[f'bb_position_{period}'] = (features['Close'] - bb_lower) / (bb_upper - bb_lower)
                features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
            
            # Stochastic oscillator
            for k_period, d_period in [(14, 3), (21, 5)]:
                slowk, slowd = ta.STOCH(features['High'].values, features['Low'].values, 
                                       features['Close'].values, fastk_period=k_period, slowd_period=d_period)
                features[f'stoch_k_{k_period}_{d_period}'] = slowk / 100
                features[f'stoch_d_{k_period}_{d_period}'] = slowd / 100
            
            # Average True Range
            for period in [14, 30]:
                atr = ta.ATR(features['High'].values, features['Low'].values, features['Close'].values, timeperiod=period)
                features[f'atr_{period}'] = atr / features['Close']  # Normalized ATR
            
            # Commodity Channel Index
            features['cci'] = ta.CCI(features['High'].values, features['Low'].values, features['Close'].values) / 100
            
            # Williams %R
            features['williams_r'] = (ta.WILLR(features['High'].values, features['Low'].values, features['Close'].values) + 100) / 100
            
            print("Technical indicators calculated successfully")
            
        except Exception as e:
            print(f"WARNING: Some technical indicators failed: {str(e)}")
        
        # Lag features (important for attention mechanism)
        lag_periods = [1, 2, 3, 5, 7, 10, 14, 21]
        for lag in lag_periods:
            features[f'close_lag_{lag}'] = features['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = features['Volume'].shift(lag)
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        # Rolling statistics for different windows
        for window in [5, 10, 20, 30, 50]:
            features[f'close_rolling_mean_{window}'] = features['Close'].rolling(window=window).mean()
            features[f'close_rolling_std_{window}'] = features['Close'].rolling(window=window).std()
            features[f'close_rolling_min_{window}'] = features['Close'].rolling(window=window).min()
            features[f'close_rolling_max_{window}'] = features['Close'].rolling(window=window).max()
            features[f'close_rolling_skew_{window}'] = features['Close'].rolling(window=window).skew()
            features[f'close_rolling_kurt_{window}'] = features['Close'].rolling(window=window).kurt()
            
            # Volume statistics
            features[f'volume_rolling_mean_{window}'] = features['Volume'].rolling(window=window).mean()
            features[f'volume_rolling_std_{window}'] = features['Volume'].rolling(window=window).std()
        
        # Market microstructure
        features['spread'] = features['High'] - features['Low']
        features['spread_norm'] = features['spread'] / features['Close']
        features['body'] = abs(features['Close'] - features['Open'])
        features['body_norm'] = features['body'] / features['Close']
        features['upper_shadow'] = features['High'] - np.maximum(features['Close'], features['Open'])
        features['lower_shadow'] = np.minimum(features['Close'], features['Open']) - features['Low']
        features['shadow_ratio'] = features['upper_shadow'] / (features['lower_shadow'] + 1e-8)
        
        # Market regime and momentum
        features['momentum_10'] = features['Close'] / features['Close'].shift(10) - 1
        features['momentum_20'] = features['Close'] / features['Close'].shift(20) - 1
        features['momentum_50'] = features['Close'] / features['Close'].shift(50) - 1
        
        # Price acceleration
        features['price_acceleration'] = features['Close'].diff(2)
        features['volume_acceleration'] = features['Volume'].diff(2)
        
        # Cyclical time features (important for transformers)
        features['hour_sin'] = np.sin(2 * np.pi * features.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features.index.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * features.index.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * features.index.dayofweek / 7)
        features['month_sin'] = np.sin(2 * np.pi * features.index.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * features.index.month / 12)
        features['year_progress'] = features.index.dayofyear / 365.25
        
        # Target variables for multi-horizon prediction
        target_1d = features['Close'].shift(-1)
        target_7d = features['Close'].shift(-7)
        target_30d = features['Close'].shift(-30)
        
        # Select numeric features
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
        
        X = features[feature_cols]
        y = target_1d  # Use 1-day target for now
        
        # Remove rows with missing values
        valid_idx = (~X.isnull().any(axis=1)) & (~y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        self.feature_names = X.columns.tolist()
        
        print(f"Transformer features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"Feature types: Price ratios, Technical indicators, Lag features, Rolling stats, Time features")
        
        return X.values, y.values
    
    def prepare_transformer_sequences(self, X, y, train_ratio=0.7, val_ratio=0.15, 
                                    num_horizons=1, batch_size=32):
        """
        Prepare sequences optimized for Transformer training
        
        Args:
            X: Feature matrix
            y: Target variable
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            num_horizons: Number of prediction horizons
            batch_size: Batch size for training
            
        Returns:
            DataLoaders for train, validation, and test sets
        """
        print(f"Preparing Transformer sequences (horizons: {num_horizons})...")
        
        # Scale features and targets
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Calculate split indices
        n_samples = len(X_scaled)
        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)
        
        # Create datasets
        train_dataset = BitcoinTimeSeriesDataset(
            X_scaled[:train_end], y_scaled[:train_end], 
            self.sequence_length, num_horizons
        )
        val_dataset = BitcoinTimeSeriesDataset(
            X_scaled[train_end:val_end], y_scaled[train_end:val_end], 
            self.sequence_length, num_horizons
        )
        test_dataset = BitcoinTimeSeriesDataset(
            X_scaled[val_end:], y_scaled[val_end:], 
            self.sequence_length, num_horizons
        )
        
        # Create data loaders with optimizations for Transformer
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,  # Shuffle for Transformer
            num_workers=4, pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True if device.type == 'cuda' else False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"Transformer sequence preparation complete:")
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Validation sequences: {len(val_dataset)}")
        print(f"  Test sequences: {len(test_dataset)}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Batch size: {batch_size}")
        
        return train_loader, val_loader, test_loader, num_horizons
    
    def build_transformer_model(self, input_size, num_horizons=1):
        """
        Build Transformer model for time series forecasting
        
        Args:
            input_size: Number of input features
            num_horizons: Number of prediction horizons
        """
        print("Building Transformer model for time series...")
        
        self.model = TimeSeriesTransformer(
            input_size=input_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_model * 4,
            max_seq_len=self.sequence_length * 2,
            dropout=0.1,
            num_horizons=num_horizons
        ).to(device)
        
        # Model analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Transformer model built:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model dimensions: d_model={self.d_model}, heads={self.num_heads}, layers={self.num_layers}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Attention heads: {self.num_heads}")
        
        return self.model
    
    def train_transformer_model(self, train_loader, val_loader, epochs=100, learning_rate=0.0001):
        """
        Train Transformer model with attention monitoring
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Initial learning rate
        """
        print("Training Transformer model...")
        
        # Optimizer and loss function
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, 
                             betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
        
        # Learning rate scheduler (Transformer-specific)
        def lr_lambda(epoch):
            # Warmup for first 10 epochs, then decay
            if epoch < 10:
                return epoch / 10.0
            else:
                return 0.5 ** ((epoch - 10) // 20)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25
        
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs, attention_weights = self.model(batch_features)
                
                # Handle target dimensions
                if batch_targets.dim() == 1:
                    batch_targets = batch_targets.unsqueeze(1)
                
                loss = criterion(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for Transformer stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device, non_blocking=True)
                    batch_targets = batch_targets.to(device, non_blocking=True)
                    
                    outputs, attention_weights = self.model(batch_features)
                    
                    if batch_targets.dim() == 1:
                        batch_targets = batch_targets.unsqueeze(1)
                    
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
                    val_batches += 1
            
            # Calculate averages
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_transformer_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_transformer_model.pth'))
        print("Training completed! Best model loaded.")
    
    def predict_with_attention(self, test_loader, horizons=[1, 7, 30]):
        """
        Generate predictions with attention weight extraction
        
        Args:
            test_loader: Test data loader
            horizons: Prediction horizons
            
        Returns:
            Dictionary of predictions and attention weights
        """
        print("Generating predictions with attention analysis...")
        
        self.model.eval()
        all_predictions = []
        all_actuals = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                
                outputs, attention_weights = self.model(batch_features)
                
                # Store results
                pred_scaled = outputs.cpu().numpy()
                actual_scaled = batch_targets.cpu().numpy()
                
                all_predictions.extend(pred_scaled)
                all_actuals.extend(actual_scaled)
                
                # Store attention weights (from last layer, first head)
                if attention_weights:
                    last_layer_attention = attention_weights[-1]  # Last layer
                    first_head_attention = last_layer_attention[:, 0, :, :].cpu().numpy()  # First head
                    all_attention_weights.extend(first_head_attention)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions).squeeze()
        actuals = np.array(all_actuals).squeeze()
        attention_weights = np.array(all_attention_weights)
        
        # Inverse transform
        predictions_original = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_original = self.target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        print(f"Predictions with attention complete:")
        print(f"  Predictions: {len(predictions_original)}")
        print(f"  Attention matrices: {attention_weights.shape}")
        print(f"  Prediction range: ${predictions_original.min():.2f} - ${predictions_original.max():.2f}")
        
        return {
            'predictions': predictions_original,
            'actuals': actuals_original,
            'attention_weights': attention_weights
        }
    
    def evaluate_transformer_predictions(self, results):
        """
        Comprehensive evaluation of Transformer predictions
        
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
        
        print("\nTransformer Bitcoin Prediction Evaluation")
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
    
    def visualize_attention_patterns(self, attention_weights, n_samples=3):
        """
        Visualize self-attention patterns for interpretability
        
        Args:
            attention_weights: Attention weight matrices
            n_samples: Number of samples to visualize
        """
        print("Visualizing Transformer attention patterns...")
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i in range(min(n_samples, len(attention_weights))):
            attention_matrix = attention_weights[i]
            
            # Plot attention heatmap
            im = axes[i].imshow(attention_matrix, cmap='Blues', aspect='auto')
            axes[i].set_title(f"Self-Attention Pattern - Sample {i+1}")
            axes[i].set_xlabel("Key Positions (Past Time Steps)")
            axes[i].set_ylabel("Query Positions (Time Steps)")
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
        
        plt.suptitle("Transformer Self-Attention: Temporal Dependencies", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def visualize_transformer_results(self, results, title="Bitcoin Transformer Forecast"):
        """
        Comprehensive visualization of Transformer results
        
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
        axes[0].plot(indices, predictions, label='Transformer Prediction', color='red', alpha=0.8, linewidth=2)
        
        axes[0].set_title(f"{title}\nTransformer with Multi-Head Self-Attention")
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
            axes[1].set_title("Transformer Training History")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss (MSE)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        
        # 3. Innovation summary
        innovation_text = f"""
        Transformer Innovation Impact ({YEAR})
        
        Attention Mechanism Revolution:
        • Vaswani et al. (2017): "Attention Is All You Need"
        • Self-attention replaces recurrence for sequence modeling
        • Parallel processing enables faster training
        • Global context through attention across all time steps
        
        Key Innovations:
        • Multi-head attention for different representation subspaces
        • Positional encoding for temporal sequence understanding
        • Layer normalization and residual connections
        • Scaled dot-product attention for computational efficiency
        
        Bitcoin Application Strengths:
        • Captures long-range temporal dependencies
        • Parallel training significantly faster than RNNs
        • Attention weights provide interpretability
        • Scales well with longer sequences
        
        Limitations for Time Series:
        • Requires large amounts of training data
        • High memory requirements for long sequences
        • Position encoding may not capture time irregularities
        • Attention can be noisy for very long sequences
        
        Architecture: {self.num_layers} layers, {self.num_heads} heads, {self.d_model} dimensions
        Sequence Length: {self.sequence_length} time steps
        Features: {len(self.feature_names)} multi-timeframe inputs
        Parameters: {sum(p.numel() for p in self.model.parameters()):,}
        """
        
        axes[2].text(0.05, 0.95, innovation_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Attention visualization
        if attention_weights is not None and len(attention_weights) > 0:
            self.visualize_attention_patterns(attention_weights, n_samples=3)


def main():
    """
    Complete Bitcoin Transformer prediction pipeline
    Demonstrates Transformer architecture for time series forecasting
    """
    print("ERA 3: Transformer Attention for Bitcoin Prediction")
    print("=" * 55)
    print(f"Historical Context: {YEAR}")
    print(f"Innovation: {INNOVATION}")
    print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"Impact: {IMPACT}")
    print("=" * 55)
    
    # Initialize forecaster
    forecaster = BitcoinTransformerForecaster(
        sequence_length=100, d_model=128, num_heads=8, num_layers=6
    )
    
    # Step 1: Load and prepare data
    print("\n" + "="*55)
    print("STEP 1: DATA LOADING AND FEATURE ENGINEERING")
    print("="*55)
    
    raw_data = forecaster.load_bitcoin_data()
    X, y = forecaster.engineer_transformer_features(raw_data)
    
    # Step 2: Prepare sequences
    print("\n" + "="*55)
    print("STEP 2: TRANSFORMER SEQUENCE PREPARATION")
    print("="*55)
    
    train_loader, val_loader, test_loader, num_horizons = forecaster.prepare_transformer_sequences(
        X, y, num_horizons=1, batch_size=32
    )
    
    # Step 3: Build model
    print("\n" + "="*55)
    print("STEP 3: TRANSFORMER MODEL CONSTRUCTION")
    print("="*55)
    
    model = forecaster.build_transformer_model(input_size=X.shape[1], num_horizons=num_horizons)
    
    # Step 4: Train model
    print("\n" + "="*55)
    print("STEP 4: TRANSFORMER TRAINING WITH ATTENTION")
    print("="*55)
    
    forecaster.train_transformer_model(train_loader, val_loader, epochs=100, learning_rate=0.0001)
    
    # Step 5: Generate predictions
    print("\n" + "="*55)
    print("STEP 5: BITCOIN PREDICTION WITH ATTENTION")
    print("="*55)
    
    results = forecaster.predict_with_attention(test_loader)
    
    # Step 6: Evaluate predictions
    print("\n" + "="*55)
    print("STEP 6: TRANSFORMER EVALUATION")
    print("="*55)
    
    metrics = forecaster.evaluate_transformer_predictions(results)
    
    # Step 7: Visualize results
    print("\n" + "="*55)
    print("STEP 7: ATTENTION VISUALIZATION")
    print("="*55)
    
    forecaster.visualize_transformer_results(results, "Bitcoin Transformer with Self-Attention")
    
    # Final summary
    print("\n" + "="*55)
    print("ERA 3 SUMMARY: TRANSFORMER ATTENTION REVOLUTION")
    print("="*55)
    print(f"""
    Transformer Analysis Complete:
    
    Attention Architecture:
    • Model: {forecaster.num_layers}-layer Transformer with {forecaster.num_heads} attention heads
    • Dimensions: {forecaster.d_model} model size, {forecaster.sequence_length} sequence length
    • Features: {len(forecaster.feature_names)} multi-timeframe inputs
    • Parameters: {sum(p.numel() for p in forecaster.model.parameters()):,}
    
    Bitcoin Prediction Performance:
    • RMSE: ${metrics['rmse']:.2f}
    • R²: {metrics['r2']:.4f}
    • Directional Accuracy: {metrics['directional_accuracy']:.1f}%
    • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
    
    Educational Value:
    • Demonstrates self-attention mechanism revolution
    • Shows parallel processing advantages over RNNs
    • Illustrates attention weight interpretability
    • Establishes foundation for modern NLP/vision models
    
    Next: LSTM Autoencoder for anomaly detection and unsupervised learning!
    """)


if __name__ == "__main__":
    main()