"""
Time Series ERA 3: LSTM Autoencoder Anomalies (2015-2020)
=========================================================

Historical Context:
Year: 2015-2020 (Autoencoder applications to time series, anomaly detection surge)
Innovation: LSTM Autoencoders for unsupervised anomaly detection in time series
Previous Limitation: Supervised methods required labeled anomalies, difficult to obtain in finance
Impact: Enabled unsupervised detection of market crashes, regime changes, and unusual patterns

This implementation demonstrates LSTM Autoencoder applied to Bitcoin anomaly detection
with unsupervised market crash detection, reconstruction error analysis,
regime change identification, and portfolio risk management applications.
"""

# Historical Context & Innovation
YEAR = "2015-2020"
INNOVATION = "LSTM Autoencoder for unsupervised anomaly detection in financial time series"
PREVIOUS_LIMITATION = "Supervised anomaly detection required labeled data unavailable in finance"
IMPACT = "Enabled automatic detection of market crashes, bubbles, and regime changes"

# Imports and Dependencies
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import talib as ta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BitcoinAnomalyDataset(Dataset):
    """
    Dataset for Bitcoin anomaly detection with sequence preprocessing
    """
    
    def __init__(self, features, sequence_length=60):
        self.features = features
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequence
        sequence = self.features[idx:idx + self.sequence_length]
        
        # For autoencoder, input and target are the same
        return torch.FloatTensor(sequence), torch.FloatTensor(sequence)

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for Bitcoin anomaly detection
    
    Features:
    - Encoder-Decoder architecture with LSTM
    - Bottleneck for dimensionality reduction
    - Reconstruction loss for anomaly scoring
    - Attention mechanism for interpretability
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, latent_size=32, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_size = latent_size
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Bottleneck
        self.encoder_to_latent = nn.Linear(hidden_size, latent_size)
        self.latent_to_decoder = nn.Linear(latent_size, hidden_size)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize LSTM and linear layer weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and param.dim() == 2:
                nn.init.xavier_uniform_(param.data)
    
    def encode(self, x):
        """
        Encode input sequence to latent representation
        
        Args:
            x: Input sequence [batch, seq_len, input_size]
            
        Returns:
            Latent representation [batch, latent_size]
        """
        # LSTM encoding
        lstm_out, (hidden, cell) = self.encoder_lstm(x)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # [batch, hidden_size]
        
        # Map to latent space
        latent = self.encoder_to_latent(last_hidden)  # [batch, latent_size]
        latent = torch.tanh(latent)  # Apply activation
        
        return latent
    
    def decode(self, latent, sequence_length):
        """
        Decode latent representation to sequence
        
        Args:
            latent: Latent representation [batch, latent_size]
            sequence_length: Length of sequence to decode
            
        Returns:
            Reconstructed sequence [batch, seq_len, input_size]
        """
        batch_size = latent.size(0)
        
        # Map latent to decoder hidden size
        decoder_input = self.latent_to_decoder(latent)  # [batch, hidden_size]
        
        # Prepare initial hidden state for decoder
        h_0 = decoder_input.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch, hidden_size]
        c_0 = torch.zeros_like(h_0)
        
        # Initialize decoder input
        decoder_inputs = decoder_input.unsqueeze(1).repeat(1, sequence_length, 1)  # [batch, seq_len, hidden_size]
        
        # LSTM decoding
        lstm_out, _ = self.decoder_lstm(decoder_inputs, (h_0, c_0))
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        output = self.output_layer(lstm_out)  # [batch, seq_len, input_size]
        
        return output
    
    def forward(self, x):
        """
        Forward pass through autoencoder
        
        Args:
            x: Input sequence [batch, seq_len, input_size]
            
        Returns:
            Reconstructed sequence and latent representation
        """
        sequence_length = x.size(1)
        
        # Encode
        latent = self.encode(x)
        
        # Decode
        reconstructed = self.decode(latent, sequence_length)
        
        return reconstructed, latent

class BitcoinAnomalyDetector:
    """
    LSTM Autoencoder-based Bitcoin anomaly detector
    
    Features:
    1. Unsupervised anomaly detection using reconstruction error
    2. Market crash and bubble detection
    3. Regime change identification
    4. Risk management and portfolio applications
    5. Threshold optimization for anomaly detection
    """
    
    def __init__(self, sequence_length=60, hidden_size=64, latent_size=32):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = []
        self.reconstruction_errors = []
        self.anomaly_threshold = None
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin data for anomaly detection analysis
        
        Returns:
            Processed Bitcoin data
        """
        print("Loading Bitcoin Data for Anomaly Detection...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_data = btc_data.dropna()
        
        print(f"Data loaded: {len(btc_data)} total observations")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        
        return btc_data
    
    def engineer_anomaly_features(self, data):
        """
        Feature engineering optimized for anomaly detection
        
        Args:
            data: Raw Bitcoin OHLCV data
            
        Returns:
            Feature matrix optimized for anomaly detection
        """
        print("Engineering features for anomaly detection...")
        
        features = data.copy()
        
        # Core price and volatility features
        features['returns'] = features['Close'].pct_change()
        features['log_returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['abs_returns'] = np.abs(features['returns'])
        features['squared_returns'] = features['returns'] ** 2
        
        # Volatility indicators (crucial for anomaly detection)
        for window in [5, 10, 20, 30]:
            features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
            features[f'realized_vol_{window}'] = features['returns'].rolling(window=window).apply(
                lambda x: np.sqrt(np.sum(x**2))
            )
        
        # Price momentum and acceleration
        for lag in [1, 2, 5, 10, 20]:
            features[f'momentum_{lag}'] = features['Close'] / features['Close'].shift(lag) - 1
            features[f'log_momentum_{lag}'] = np.log(features['Close'] / features['Close'].shift(lag))
        
        # Volume anomalies
        features['volume_change'] = features['Volume'].pct_change()
        features['volume_log_change'] = np.log(features['Volume'] / features['Volume'].shift(1))
        features['volume_sma_20'] = features['Volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['Volume'] / features['volume_sma_20']
        features['volume_zscore'] = (features['Volume'] - features['volume_sma_20']) / features['Volume'].rolling(window=20).std()
        
        # Price level indicators
        features['high_low_ratio'] = features['High'] / features['Low']
        features['close_open_ratio'] = features['Close'] / features['Open']
        features['intraday_return'] = (features['Close'] - features['Open']) / features['Open']
        features['gap'] = (features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1)
        
        # Technical indicators for anomaly detection
        try:
            # RSI extremes
            features['rsi_14'] = ta.RSI(features['Close'].values, timeperiod=14)
            features['rsi_extreme'] = ((features['rsi_14'] > 80) | (features['rsi_14'] < 20)).astype(int)
            
            # Bollinger Band position (anomalies when outside bands)
            bb_upper, bb_middle, bb_lower = ta.BBANDS(features['Close'].values)
            features['bb_position'] = (features['Close'] - bb_lower) / (bb_upper - bb_lower)
            features['bb_outside'] = ((features['Close'] > bb_upper) | (features['Close'] < bb_lower)).astype(int)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # MACD divergence
            macd, macd_signal, macd_hist = ta.MACD(features['Close'].values)
            features['macd_hist'] = macd_hist
            features['macd_divergence'] = np.abs(macd - macd_signal)
            
            # Average True Range (volatility)
            features['atr_14'] = ta.ATR(features['High'].values, features['Low'].values, features['Close'].values)
            features['atr_normalized'] = features['atr_14'] / features['Close']
            
            print("Technical indicators for anomaly detection calculated")
            
        except Exception as e:
            print(f"WARNING: Some technical indicators failed: {str(e)}")
        
        # Market microstructure anomalies
        features['spread'] = features['High'] - features['Low']
        features['spread_normalized'] = features['spread'] / features['Close']
        features['body'] = abs(features['Close'] - features['Open'])
        features['body_normalized'] = features['body'] / features['Close']
        features['upper_shadow'] = features['High'] - np.maximum(features['Close'], features['Open'])
        features['lower_shadow'] = np.minimum(features['Close'], features['Open']) - features['Low']
        features['shadow_ratio'] = features['upper_shadow'] / (features['lower_shadow'] + 1e-8)
        
        # Statistical anomaly indicators
        for window in [10, 20, 50]:
            # Z-scores
            features[f'price_zscore_{window}'] = (features['Close'] - features['Close'].rolling(window=window).mean()) / features['Close'].rolling(window=window).std()
            features[f'volume_zscore_{window}'] = (features['Volume'] - features['Volume'].rolling(window=window).mean()) / features['Volume'].rolling(window=window).std()
            features[f'returns_zscore_{window}'] = (features['returns'] - features['returns'].rolling(window=window).mean()) / features['returns'].rolling(window=window).std()
            
            # Rolling statistics
            features[f'price_skew_{window}'] = features['Close'].rolling(window=window).skew()
            features[f'price_kurt_{window}'] = features['Close'].rolling(window=window).kurt()
            features[f'returns_skew_{window}'] = features['returns'].rolling(window=window).skew()
            features[f'returns_kurt_{window}'] = features['returns'].rolling(window=window).kurt()
        
        # Regime change indicators
        features['price_acceleration'] = features['Close'].diff(2)
        features['volatility_acceleration'] = features['volatility_20'].diff(2)
        features['trend_change'] = (features['returns'].shift(1) * features['returns'] < 0).astype(int)
        
        # Select numeric features
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
        
        X = features[feature_cols]
        
        # Remove rows with missing values
        valid_idx = ~X.isnull().any(axis=1)
        X = X[valid_idx]
        original_dates = features.index[valid_idx]
        
        self.feature_names = X.columns.tolist()
        
        print(f"Anomaly detection features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"Feature categories: Price/Volume ratios, Volatility, Technical indicators, Statistical measures")
        
        return X.values, original_dates
    
    def prepare_anomaly_sequences(self, X, train_ratio=0.8, batch_size=32):
        """
        Prepare sequences for autoencoder training
        
        Args:
            X: Feature matrix
            train_ratio: Ratio of data for training
            batch_size: Batch size
            
        Returns:
            DataLoaders for train and test sets
        """
        print("Preparing sequences for anomaly detection...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate split index
        n_samples = len(X_scaled)
        train_end = int(train_ratio * n_samples)
        
        # Create datasets
        train_dataset = BitcoinAnomalyDataset(X_scaled[:train_end], self.sequence_length)
        test_dataset = BitcoinAnomalyDataset(X_scaled, self.sequence_length)  # Use all data for testing
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Anomaly sequence preparation complete:")
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Test sequences: {len(test_dataset)}")
        print(f"  Sequence length: {self.sequence_length}")
        
        return train_loader, test_loader
    
    def build_autoencoder_model(self, input_size):
        """
        Build LSTM Autoencoder model
        
        Args:
            input_size: Number of input features
        """
        print("Building LSTM Autoencoder for anomaly detection...")
        
        self.model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            latent_size=self.latent_size,
            dropout=0.2
        ).to(device)
        
        # Model analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"LSTM Autoencoder built:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Latent size: {self.latent_size}")
        print(f"  Compression ratio: {input_size * self.sequence_length / self.latent_size:.1f}:1")
        
        return self.model
    
    def train_autoencoder(self, train_loader, val_loader=None, epochs=100, learning_rate=0.001):
        """
        Train LSTM Autoencoder for anomaly detection
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        print("Training LSTM Autoencoder...")
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_data, batch_target in train_loader:
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, latent = self.model(batch_data)
                
                # Reconstruction loss
                loss = criterion(reconstructed, batch_target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            # Validation phase (if available)
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                val_batches = 0
                
                with torch.no_grad():
                    for batch_data, batch_target in val_loader:
                        batch_data = batch_data.to(device)
                        batch_target = batch_target.to(device)
                        
                        reconstructed, _ = self.model(batch_data)
                        loss = criterion(reconstructed, batch_target)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            # Calculate averages
            avg_train_loss = train_loss / train_batches
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            if val_loader is not None:
                self.training_history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss if val_loader is not None else avg_train_loss)
            
            # Early stopping
            current_loss = val_loss if val_loader is not None else avg_train_loss
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_autoencoder_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_autoencoder_model.pth'))
        print("Training completed! Best model loaded.")
    
    def detect_anomalies(self, test_loader, dates):
        """
        Detect anomalies using reconstruction error
        
        Args:
            test_loader: Test data loader
            dates: Corresponding dates
            
        Returns:
            Dictionary containing reconstruction errors and anomaly scores
        """
        print("Detecting anomalies using reconstruction error...")
        
        self.model.eval()
        all_reconstruction_errors = []
        all_latent_representations = []
        
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                
                # Forward pass
                reconstructed, latent = self.model(batch_data)
                
                # Calculate reconstruction error
                mse_error = torch.mean((reconstructed - batch_target) ** 2, dim=(1, 2))
                mae_error = torch.mean(torch.abs(reconstructed - batch_target), dim=(1, 2))
                
                # Store errors and latent representations
                all_reconstruction_errors.extend(mse_error.cpu().numpy())
                all_latent_representations.extend(latent.cpu().numpy())
        
        # Convert to numpy arrays
        reconstruction_errors = np.array(all_reconstruction_errors)
        latent_representations = np.array(all_latent_representations)
        
        # Calculate anomaly threshold using statistical methods
        threshold_percentile = 95  # Top 5% as anomalies
        threshold_zscore = 3  # 3 standard deviations
        
        threshold_p95 = np.percentile(reconstruction_errors, threshold_percentile)
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        threshold_z3 = mean_error + threshold_zscore * std_error
        
        # Use the more conservative threshold
        self.anomaly_threshold = max(threshold_p95, threshold_z3)
        
        # Identify anomalies
        anomalies = reconstruction_errors > self.anomaly_threshold
        
        # Align with dates
        valid_dates = dates[self.sequence_length-1:]  # Account for sequence length
        
        results = {
            'reconstruction_errors': reconstruction_errors,
            'latent_representations': latent_representations,
            'anomalies': anomalies,
            'threshold': self.anomaly_threshold,
            'dates': valid_dates[:len(reconstruction_errors)],
            'anomaly_dates': valid_dates[:len(reconstruction_errors)][anomalies]
        }
        
        print(f"Anomaly detection complete:")
        print(f"  Total sequences analyzed: {len(reconstruction_errors)}")
        print(f"  Anomalies detected: {np.sum(anomalies)} ({np.mean(anomalies)*100:.2f}%)")
        print(f"  Anomaly threshold: {self.anomaly_threshold:.6f}")
        print(f"  Mean reconstruction error: {mean_error:.6f}")
        print(f"  Std reconstruction error: {std_error:.6f}")
        
        return results
    
    def analyze_market_events(self, results, bitcoin_data):
        """
        Analyze detected anomalies in context of known market events
        
        Args:
            results: Anomaly detection results
            bitcoin_data: Original Bitcoin price data
            
        Returns:
            Analysis of market events and anomalies
        """
        print("\nAnalyzing detected anomalies vs market events...")
        
        anomaly_dates = results['anomaly_dates']
        
        # Define known Bitcoin market events
        known_events = {
            '2017-12-17': 'Bitcoin ATH ~$20k',
            '2018-01-01': 'Crypto market crash begins',
            '2020-03-12': 'COVID-19 market crash',
            '2021-04-14': 'Bitcoin ATH ~$65k',
            '2021-05-19': 'China mining ban announcement',
            '2022-05-09': 'Terra Luna collapse',
            '2022-11-09': 'FTX collapse',
            '2023-03-10': 'Silicon Valley Bank collapse'
        }
        
        # Find anomalies near known events
        event_matches = []
        for event_date, event_desc in known_events.items():
            event_timestamp = pd.Timestamp(event_date)
            
            # Find anomalies within 7 days of event
            for anomaly_date in anomaly_dates:
                if abs((anomaly_date - event_timestamp).days) <= 7:
                    event_matches.append({
                        'event_date': event_date,
                        'event_description': event_desc,
                        'anomaly_date': anomaly_date,
                        'days_difference': (anomaly_date - event_timestamp).days
                    })
                    break
        
        # Calculate price movements around anomalies
        anomaly_analysis = []
        for anomaly_date in anomaly_dates:
            try:
                # Find nearest price data
                price_data = bitcoin_data.loc[bitcoin_data.index <= anomaly_date].iloc[-1]
                
                # Calculate price changes
                price_1d_before = bitcoin_data.loc[bitcoin_data.index <= anomaly_date - pd.Timedelta(days=1)].iloc[-1]['Close']
                price_1d_after = bitcoin_data.loc[bitcoin_data.index >= anomaly_date + pd.Timedelta(days=1)].iloc[0]['Close']
                price_7d_before = bitcoin_data.loc[bitcoin_data.index <= anomaly_date - pd.Timedelta(days=7)].iloc[-1]['Close']
                price_7d_after = bitcoin_data.loc[bitcoin_data.index >= anomaly_date + pd.Timedelta(days=7)].iloc[0]['Close']
                
                change_1d = (price_1d_after - price_1d_before) / price_1d_before * 100
                change_7d = (price_7d_after - price_7d_before) / price_7d_before * 100
                
                anomaly_analysis.append({
                    'date': anomaly_date,
                    'price': price_data['Close'],
                    'volume': price_data['Volume'],
                    'change_1d': change_1d,
                    'change_7d': change_7d
                })
                
            except (IndexError, KeyError):
                continue
        
        print(f"Market Event Analysis:")
        print(f"  Known events detected: {len(event_matches)}")
        print(f"  Detection rate: {len(event_matches)/len(known_events)*100:.1f}%")
        
        if event_matches:
            print(f"\n  Detected Events:")
            for match in event_matches:
                print(f"    {match['event_description']} ({match['event_date']}) - "
                      f"Detected on {match['anomaly_date'].strftime('%Y-%m-%d')} "
                      f"({match['days_difference']:+d} days)")
        
        return {
            'event_matches': event_matches,
            'anomaly_analysis': anomaly_analysis,
            'detection_rate': len(event_matches)/len(known_events)*100
        }
    
    def visualize_anomaly_results(self, results, bitcoin_data, market_analysis, 
                                title="Bitcoin LSTM Autoencoder Anomaly Detection"):
        """
        Comprehensive visualization of anomaly detection results
        
        Args:
            results: Anomaly detection results
            bitcoin_data: Original Bitcoin data
            market_analysis: Market event analysis
            title: Plot title
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        
        # 1. Bitcoin price with anomalies
        axes[0].plot(bitcoin_data.index, bitcoin_data['Close'], 
                    label='Bitcoin Price', color='blue', alpha=0.7, linewidth=1)
        
        # Mark anomalies
        for anomaly_date in results['anomaly_dates']:
            try:
                anomaly_price = bitcoin_data.loc[anomaly_date, 'Close']
                axes[0].scatter(anomaly_date, anomaly_price, color='red', s=50, alpha=0.8, zorder=5)
            except KeyError:
                continue
        
        # Mark known events
        known_events = {
            '2017-12-17': 'ATH $20k',
            '2018-01-01': 'Crash 2018',
            '2020-03-12': 'COVID Crash',
            '2021-04-14': 'ATH $65k',
            '2022-05-09': 'Terra Luna',
            '2022-11-09': 'FTX Collapse'
        }
        
        for event_date, event_label in known_events.items():
            try:
                event_price = bitcoin_data.loc[event_date, 'Close']
                axes[0].axvline(x=pd.Timestamp(event_date), color='green', linestyle='--', alpha=0.7)
                axes[0].text(pd.Timestamp(event_date), event_price, event_label, 
                           rotation=90, fontsize=8, ha='right')
            except KeyError:
                continue
        
        axes[0].set_title(f"{title}\nBitcoin Price with Detected Anomalies (Red) and Known Events (Green)")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Bitcoin Price (USD)")
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. Reconstruction error over time
        axes[1].plot(results['dates'], results['reconstruction_errors'], 
                    color='purple', alpha=0.7, linewidth=1)
        axes[1].axhline(y=results['threshold'], color='red', linestyle='--', 
                       label=f'Anomaly Threshold ({results["threshold"]:.6f})')
        
        # Highlight anomalies
        anomaly_errors = results['reconstruction_errors'][results['anomalies']]
        anomaly_dates_filtered = results['dates'][results['anomalies']]
        axes[1].scatter(anomaly_dates_filtered, anomaly_errors, color='red', s=30, alpha=0.8, zorder=5)
        
        axes[1].set_title("Reconstruction Error Over Time")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Reconstruction Error (MSE)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Training history
        if self.training_history['train_loss']:
            epochs = range(1, len(self.training_history['train_loss']) + 1)
            axes[2].plot(epochs, self.training_history['train_loss'], label='Training Loss', color='blue')
            if self.training_history['val_loss']:
                axes[2].plot(epochs, self.training_history['val_loss'], label='Validation Loss', color='red')
            
            axes[2].set_title("LSTM Autoencoder Training History")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Reconstruction Loss (MSE)")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_yscale('log')
        
        # 4. Innovation summary
        innovation_text = f"""
        LSTM Autoencoder Innovation Impact ({YEAR})
        
        Unsupervised Learning Foundation:
        • Encoder-Decoder architecture for sequence reconstruction
        • Bottleneck latent representation for dimensionality reduction
        • Reconstruction error as anomaly score
        • No labeled anomaly data required for training
        
        Key Innovations:
        • Automatic feature learning from raw time series
        • Temporal pattern compression in latent space
        • Threshold-based anomaly detection using statistical methods
        • Scalable to any time series without domain expertise
        
        Bitcoin Application Strengths:
        • Detects market crashes and bubbles automatically
        • Identifies regime changes and structural breaks
        • Early warning system for portfolio risk management
        • Captures complex multivariate anomaly patterns
        
        Anomaly Detection Results:
        • Total anomalies detected: {np.sum(results['anomalies'])} ({np.mean(results['anomalies'])*100:.1f}%)
        • Known events detected: {len(market_analysis['event_matches'])}/{len(known_events)} ({market_analysis['detection_rate']:.1f}%)
        • Compression ratio: {len(self.feature_names) * self.sequence_length / self.latent_size:.1f}:1
        • Model parameters: {sum(p.numel() for p in self.model.parameters()):,}
        
        Limitations:
        • Requires sufficient normal data for training
        • Threshold selection affects false positive rate
        • May miss gradual regime changes
        • Reconstruction quality depends on model capacity
        """
        
        axes[3].text(0.05, 0.95, innovation_text, transform=axes[3].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        axes[3].set_xlim(0, 1)
        axes[3].set_ylim(0, 1)
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Complete Bitcoin LSTM Autoencoder anomaly detection pipeline
    Demonstrates unsupervised anomaly detection for financial time series
    """
    print("ERA 3: LSTM Autoencoder Anomaly Detection for Bitcoin")
    print("=" * 60)
    print(f"Historical Context: {YEAR}")
    print(f"Innovation: {INNOVATION}")
    print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"Impact: {IMPACT}")
    print("=" * 60)
    
    # Initialize detector
    detector = BitcoinAnomalyDetector(sequence_length=60, hidden_size=64, latent_size=32)
    
    # Step 1: Load and prepare data
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING AND FEATURE ENGINEERING")
    print("="*60)
    
    raw_data = detector.load_bitcoin_data()
    X, dates = detector.engineer_anomaly_features(raw_data)
    
    # Step 2: Prepare sequences
    print("\n" + "="*60)
    print("STEP 2: AUTOENCODER SEQUENCE PREPARATION")
    print("="*60)
    
    train_loader, test_loader = detector.prepare_anomaly_sequences(X, train_ratio=0.8, batch_size=32)
    
    # Step 3: Build model
    print("\n" + "="*60)
    print("STEP 3: LSTM AUTOENCODER CONSTRUCTION")
    print("="*60)
    
    model = detector.build_autoencoder_model(input_size=X.shape[1])
    
    # Step 4: Train model
    print("\n" + "="*60)
    print("STEP 4: UNSUPERVISED AUTOENCODER TRAINING")
    print("="*60)
    
    detector.train_autoencoder(train_loader, epochs=100, learning_rate=0.001)
    
    # Step 5: Detect anomalies
    print("\n" + "="*60)
    print("STEP 5: BITCOIN ANOMALY DETECTION")
    print("="*60)
    
    results = detector.detect_anomalies(test_loader, dates)
    
    # Step 6: Analyze market events
    print("\n" + "="*60)
    print("STEP 6: MARKET EVENT ANALYSIS")
    print("="*60)
    
    market_analysis = detector.analyze_market_events(results, raw_data)
    
    # Step 7: Visualize results
    print("\n" + "="*60)
    print("STEP 7: ANOMALY VISUALIZATION")
    print("="*60)
    
    detector.visualize_anomaly_results(results, raw_data, market_analysis, 
                                     "Bitcoin LSTM Autoencoder Anomaly Detection")
    
    # Final summary
    print("\n" + "="*60)
    print("ERA 3 SUMMARY: LSTM AUTOENCODER ANOMALY DETECTION")
    print("="*60)
    print(f"""
    Anomaly Detection Analysis Complete:
    
    Unsupervised Learning Architecture:
    • Model: LSTM Autoencoder with {detector.latent_size}-dimensional latent space
    • Compression ratio: {len(detector.feature_names) * detector.sequence_length / detector.latent_size:.1f}:1
    • Parameters: {sum(p.numel() for p in detector.model.parameters()):,}
    • Features: {len(detector.feature_names)} anomaly-focused indicators
    
    Bitcoin Anomaly Detection Performance:
    • Total anomalies detected: {np.sum(results['anomalies'])} ({np.mean(results['anomalies'])*100:.1f}%)
    • Known events detected: {len(market_analysis['event_matches'])}/8 ({market_analysis['detection_rate']:.1f}%)
    • Anomaly threshold: {results['threshold']:.6f}
    • Mean reconstruction error: {np.mean(results['reconstruction_errors']):.6f}
    
    Educational Value:
    • Demonstrates unsupervised anomaly detection principles
    • Shows encoder-decoder architecture for sequence compression
    • Illustrates reconstruction error as anomaly measure
    • Establishes foundation for modern anomaly detection systems
    
    ERA 3 Complete! Deep Learning Revolution demonstrated across:
    • LSTM: Sequential modeling with gating mechanisms
    • GRU: Computational efficiency improvements
    • Transformer: Attention mechanism revolution
    • Autoencoder: Unsupervised anomaly detection
    
    Ready for ERA 4: Modern Deep Learning and Advanced Architectures!
    """)


if __name__ == "__main__":
    main()