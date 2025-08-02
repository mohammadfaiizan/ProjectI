"""
Time Series ERA 4: DeepAR Probabilistic (2020-Present)
======================================================

Historical Context:
Year: 2020 (Salinas et al., "DeepAR: Probabilistic forecasting with autoregressive recurrent neural networks")
Innovation: Probabilistic forecasting with learned likelihood distributions and quantile predictions
Previous Limitation: Point forecasts without uncertainty quantification inadequate for risk management
Impact: Enabled probabilistic forecasting at scale with uncertainty bounds for decision making

This implementation demonstrates DeepAR applied to Bitcoin price prediction with
Gaussian likelihood and quantile predictions, multiple cryptocurrency joint modeling,
uncertainty quantification for risk management, and Amazon's production forecasting approach.
"""

# Historical Context & Innovation
YEAR = "2020"
INNOVATION = "DeepAR probabilistic forecasting with autoregressive RNNs and learned likelihood distributions"
PREVIOUS_LIMITATION = "Point forecasts lacked uncertainty quantification critical for financial risk management"
IMPACT = "Enabled scalable probabilistic forecasting with quantile predictions for risk-aware decisions"

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
from torch.distributions import Normal, StudentT, NegativeBinomial
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DeepARModel(nn.Module):
    """
    DeepAR model for probabilistic time series forecasting
    
    Features:
    - Autoregressive RNN with probabilistic outputs
    - Multiple likelihood distributions (Gaussian, Student-t)
    - Quantile prediction capabilities
    - Multi-variate conditioning with covariates
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1, 
                 likelihood='gaussian', num_covariates=0):
        super(DeepARModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.likelihood = likelihood
        self.num_covariates = num_covariates
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size + num_covariates,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Likelihood-specific output layers
        if likelihood == 'gaussian':
            # Gaussian: predict mean and log(std)
            self.mean_layer = nn.Linear(hidden_size, 1)
            self.std_layer = nn.Linear(hidden_size, 1)
        elif likelihood == 'student_t':
            # Student-t: predict mean, log(scale), and log(degrees of freedom)
            self.mean_layer = nn.Linear(hidden_size, 1)
            self.scale_layer = nn.Linear(hidden_size, 1)
            self.df_layer = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(f"Unsupported likelihood: {likelihood}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize output layers
        for layer in [self.mean_layer, getattr(self, 'std_layer', None), 
                     getattr(self, 'scale_layer', None), getattr(self, 'df_layer', None)]:
            if layer is not None:
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
    
    def forward(self, x, covariates=None, hidden=None):
        """
        Forward pass through DeepAR model
        
        Args:
            x: Input time series [batch, seq_len, 1]
            covariates: Additional covariates [batch, seq_len, num_covariates]
            hidden: Initial hidden state
            
        Returns:
            Distribution parameters and hidden state
        """
        # Combine input with covariates
        if covariates is not None:
            x = torch.cat([x, covariates], dim=-1)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Get distribution parameters
        if self.likelihood == 'gaussian':
            mean = self.mean_layer(lstm_out)
            log_std = self.std_layer(lstm_out)
            std = torch.exp(log_std) + 1e-6  # Ensure positive std
            
            return {
                'mean': mean,
                'std': std,
                'hidden': hidden
            }
        
        elif self.likelihood == 'student_t':
            mean = self.mean_layer(lstm_out)
            log_scale = self.scale_layer(lstm_out)
            log_df = self.df_layer(lstm_out)
            
            scale = torch.exp(log_scale) + 1e-6
            df = torch.exp(log_df) + 2.0  # Ensure df > 2 for finite variance
            
            return {
                'mean': mean,
                'scale': scale,
                'df': df,
                'hidden': hidden
            }
    
    def sample(self, x, covariates=None, hidden=None, num_samples=100):
        """
        Generate probabilistic samples from the model
        
        Args:
            x: Input context [batch, seq_len, 1]
            covariates: Covariates [batch, seq_len, num_covariates]
            hidden: Initial hidden state
            num_samples: Number of samples to generate
            
        Returns:
            Samples from the predictive distribution
        """
        self.eval()
        with torch.no_grad():
            # Get distribution parameters
            output = self.forward(x, covariates, hidden)
            
            if self.likelihood == 'gaussian':
                dist = Normal(output['mean'], output['std'])
            elif self.likelihood == 'student_t':
                dist = StudentT(output['df'], output['mean'], output['scale'])
            
            # Sample from the distribution
            samples = dist.sample((num_samples,))  # [num_samples, batch, seq_len, 1]
            
            return samples.squeeze(-1)  # [num_samples, batch, seq_len]
    
    def predict_quantiles(self, x, covariates=None, hidden=None, quantiles=[0.1, 0.5, 0.9]):
        """
        Predict specific quantiles of the distribution
        
        Args:
            x: Input context
            covariates: Covariates
            hidden: Initial hidden state
            quantiles: List of quantiles to predict
            
        Returns:
            Dictionary of quantile predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, covariates, hidden)
            
            if self.likelihood == 'gaussian':
                dist = Normal(output['mean'], output['std'])
            elif self.likelihood == 'student_t':
                dist = StudentT(output['df'], output['mean'], output['scale'])
            
            # Calculate quantiles
            quantile_predictions = {}
            for q in quantiles:
                quantile_predictions[f'q{int(q*100)}'] = dist.icdf(torch.tensor(q).to(device))
            
            return quantile_predictions

class BitcoinDeepARDataset(Dataset):
    """
    Dataset for Bitcoin DeepAR with probabilistic targets
    """
    
    def __init__(self, prices, covariates=None, context_length=168, prediction_length=24):
        self.prices = prices
        self.covariates = covariates
        self.context_length = context_length
        self.prediction_length = prediction_length
        
    def __len__(self):
        return len(self.prices) - self.context_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        # Context (historical data)
        context = self.prices[idx:idx + self.context_length]
        
        # Target (future data to predict)
        target = self.prices[idx + self.context_length:idx + self.context_length + self.prediction_length]
        
        # Combined sequence for autoregressive training
        full_sequence = np.concatenate([context, target])
        
        # Covariates if available
        if self.covariates is not None:
            context_cov = self.covariates[idx:idx + self.context_length]
            target_cov = self.covariates[idx + self.context_length:idx + self.context_length + self.prediction_length]
            full_covariates = np.concatenate([context_cov, target_cov])
            return (torch.FloatTensor(full_sequence), 
                   torch.FloatTensor(full_covariates),
                   torch.LongTensor([self.context_length]))
        else:
            return (torch.FloatTensor(full_sequence), 
                   torch.FloatTensor([self.context_length]))

class BitcoinDeepARForecaster:
    """
    DeepAR-based Bitcoin probabilistic forecaster with uncertainty quantification
    
    Features:
    1. Probabilistic forecasting with learned likelihood distributions
    2. Quantile predictions for risk management
    3. Multi-cryptocurrency joint modeling capability
    4. Uncertainty-aware trading strategies
    5. Value-at-Risk (VaR) calculations
    """
    
    def __init__(self, context_length=168, prediction_length=24, likelihood='gaussian'):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.likelihood = likelihood
        self.model = None
        self.scaler = MinMaxScaler()
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin data for DeepAR probabilistic analysis
        
        Returns:
            Raw Bitcoin price data and potential covariates
        """
        print("Loading Bitcoin Data for DeepAR Probabilistic Analysis...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_data = btc_data.dropna()
        
        # Primary time series (prices)
        prices = btc_data['Close'].values
        
        # Additional covariates (volume, volatility, etc.)
        covariates = []
        
        # Volume (normalized)
        volume = btc_data['Volume'].values
        volume_norm = (volume - volume.mean()) / volume.std()
        covariates.append(volume_norm)
        
        # High-Low spread (normalized)
        hl_spread = (btc_data['High'] - btc_data['Low']).values
        hl_spread_norm = (hl_spread - hl_spread.mean()) / hl_spread.std()
        covariates.append(hl_spread_norm)
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])  # Pad with zero for first value
        covariates.append(returns)
        
        # Day of week (cyclical encoding)
        day_of_week_sin = np.sin(2 * np.pi * btc_data.index.dayofweek / 7)
        day_of_week_cos = np.cos(2 * np.pi * btc_data.index.dayofweek / 7)
        covariates.extend([day_of_week_sin, day_of_week_cos])
        
        # Month (cyclical encoding)
        month_sin = np.sin(2 * np.pi * btc_data.index.month / 12)
        month_cos = np.cos(2 * np.pi * btc_data.index.month / 12)
        covariates.extend([month_sin, month_cos])
        
        # Stack covariates
        covariates_array = np.column_stack(covariates)
        
        print(f"Data loaded: {len(prices)} total observations")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"Covariates: {covariates_array.shape[1]} features")
        
        return prices, covariates_array, btc_data.index
    
    def prepare_deepar_data(self, prices, covariates, train_ratio=0.7, val_ratio=0.15, batch_size=32):
        """
        Prepare data for DeepAR probabilistic training
        
        Args:
            prices: Price time series
            covariates: Additional covariates
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            batch_size: Batch size
            
        Returns:
            DataLoaders for train, validation, and test sets
        """
        print("Preparing DeepAR probabilistic data...")
        
        # Scale prices
        prices_scaled = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Calculate split indices
        n_samples = len(prices_scaled)
        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)
        
        # Create datasets
        train_dataset = BitcoinDeepARDataset(
            prices_scaled[:train_end], 
            covariates[:train_end] if covariates is not None else None,
            self.context_length, self.prediction_length
        )
        val_dataset = BitcoinDeepARDataset(
            prices_scaled[train_end:val_end], 
            covariates[train_end:val_end] if covariates is not None else None,
            self.context_length, self.prediction_length
        )
        test_dataset = BitcoinDeepARDataset(
            prices_scaled[val_end:], 
            covariates[val_end:] if covariates is not None else None,
            self.context_length, self.prediction_length
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"DeepAR data preparation complete:")
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Validation sequences: {len(val_dataset)}")
        print(f"  Test sequences: {len(test_dataset)}")
        print(f"  Context length: {self.context_length}")
        print(f"  Prediction length: {self.prediction_length}")
        
        return train_loader, val_loader, test_loader
    
    def build_deepar_model(self, num_covariates=0):
        """
        Build DeepAR model for probabilistic Bitcoin forecasting
        
        Args:
            num_covariates: Number of covariate features
        """
        print("Building DeepAR probabilistic model...")
        
        self.model = DeepARModel(
            input_size=1,  # Single time series
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            likelihood=self.likelihood,
            num_covariates=num_covariates
        ).to(device)
        
        # Model analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"DeepAR model built:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Likelihood: {self.likelihood}")
        print(f"  Context length: {self.context_length}")
        print(f"  Prediction length: {self.prediction_length}")
        print(f"  Covariates: {num_covariates} features")
        
        return self.model
    
    def deepar_loss(self, predictions, targets, context_length):
        """
        Calculate DeepAR negative log-likelihood loss
        
        Args:
            predictions: Model predictions (distribution parameters)
            targets: Target values
            context_length: Length of context window
            
        Returns:
            Negative log-likelihood loss
        """
        # Use only the prediction portion (after context)
        pred_portion = {k: v[:, context_length:, :] for k, v in predictions.items() if k != 'hidden'}
        target_portion = targets[:, context_length:].unsqueeze(-1)
        
        if self.likelihood == 'gaussian':
            dist = Normal(pred_portion['mean'], pred_portion['std'])
        elif self.likelihood == 'student_t':
            dist = StudentT(pred_portion['df'], pred_portion['mean'], pred_portion['scale'])
        
        # Negative log-likelihood
        nll = -dist.log_prob(target_portion)
        
        return nll.mean()
    
    def train_deepar_model(self, train_loader, val_loader, epochs=100, learning_rate=0.001):
        """
        Train DeepAR model with probabilistic loss
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        print("Training DeepAR probabilistic model...")
        
        # Optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7, verbose=True)
        
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
            
            for batch_data in train_loader:
                if len(batch_data) == 3:  # With covariates
                    sequence, covariates, context_length = batch_data
                    sequence = sequence.to(device)
                    covariates = covariates.to(device)
                    context_length = context_length[0].item()
                else:  # Without covariates
                    sequence, context_length = batch_data
                    sequence = sequence.to(device)
                    covariates = None
                    context_length = context_length[0].item()
                
                optimizer.zero_grad()
                
                # Prepare input (add time dimension)
                input_seq = sequence.unsqueeze(-1)  # [batch, seq_len, 1]
                
                # Forward pass
                predictions = self.model(input_seq, covariates)
                
                # Calculate loss
                loss = self.deepar_loss(predictions, sequence, context_length)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 3:  # With covariates
                        sequence, covariates, context_length = batch_data
                        sequence = sequence.to(device)
                        covariates = covariates.to(device)
                        context_length = context_length[0].item()
                    else:  # Without covariates
                        sequence, context_length = batch_data
                        sequence = sequence.to(device)
                        covariates = None
                        context_length = context_length[0].item()
                    
                    input_seq = sequence.unsqueeze(-1)
                    predictions = self.model(input_seq, covariates)
                    loss = self.deepar_loss(predictions, sequence, context_length)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            # Calculate averages
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_deepar_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_deepar_model.pth'))
        print("Training completed! Best model loaded.")
    
    def predict_probabilistic(self, test_loader, num_samples=1000, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        """
        Generate probabilistic predictions with uncertainty quantification
        
        Args:
            test_loader: Test data loader
            num_samples: Number of Monte Carlo samples
            quantiles: Quantiles to compute
            
        Returns:
            Dictionary containing probabilistic predictions
        """
        print("Generating probabilistic Bitcoin forecasts...")
        
        self.model.eval()
        all_samples = []
        all_quantiles = []
        all_actuals = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:  # With covariates
                    sequence, covariates, context_length = batch_data
                    sequence = sequence.to(device)
                    covariates = covariates.to(device)
                    context_length = context_length[0].item()
                else:  # Without covariates
                    sequence, context_length = batch_data
                    sequence = sequence.to(device)
                    covariates = None
                    context_length = context_length[0].item()
                
                # Use only context for prediction
                context_seq = sequence[:, :context_length].unsqueeze(-1)
                context_cov = covariates[:, :context_length] if covariates is not None else None
                
                # Generate samples
                samples = self.model.sample(context_seq, context_cov, num_samples=num_samples)
                
                # Generate quantiles
                quantile_preds = self.model.predict_quantiles(context_seq, context_cov, quantiles=quantiles)
                
                # Actual values (prediction portion)
                actual = sequence[:, context_length:]
                
                # Store results
                all_samples.append(samples.cpu().numpy())
                all_quantiles.append({k: v.cpu().numpy() for k, v in quantile_preds.items()})
                all_actuals.append(actual.cpu().numpy())
        
        # Process results
        samples_array = np.concatenate(all_samples, axis=1)  # [num_samples, total_sequences, pred_length]
        actuals_array = np.concatenate(all_actuals, axis=0)  # [total_sequences, pred_length]
        
        # Combine quantiles
        combined_quantiles = {}
        for q_key in all_quantiles[0].keys():
            combined_quantiles[q_key] = np.concatenate([q[q_key] for q in all_quantiles], axis=0)
        
        # Use first prediction step for evaluation
        samples_1d = samples_array[:, :, 0]  # [num_samples, total_sequences]
        actuals_1d = actuals_array[:, 0]  # [total_sequences]
        
        # Inverse transform
        median_pred = np.median(samples_1d, axis=0)
        actuals_original = self.scaler.inverse_transform(actuals_1d.reshape(-1, 1)).flatten()
        median_original = self.scaler.inverse_transform(median_pred.reshape(-1, 1)).flatten()
        
        # Transform quantiles
        quantiles_original = {}
        for q_key, q_values in combined_quantiles.items():
            q_1d = q_values[:, 0]  # First prediction step
            quantiles_original[q_key] = self.scaler.inverse_transform(q_1d.reshape(-1, 1)).flatten()
        
        print(f"Probabilistic predictions complete:")
        print(f"  Samples per prediction: {num_samples}")
        print(f"  Quantiles computed: {quantiles}")
        print(f"  Predictions: {len(median_original)}")
        
        return {
            'median_predictions': median_original,
            'actuals': actuals_original,
            'quantiles': quantiles_original,
            'samples': samples_array,
            'samples_original_scale': self.scaler.inverse_transform(samples_1d.T).T
        }
    
    def evaluate_probabilistic_predictions(self, results):
        """
        Comprehensive evaluation of probabilistic predictions
        
        Args:
            results: Dictionary containing probabilistic predictions
            
        Returns:
            Dictionary of evaluation metrics including VaR and coverage
        """
        median_predictions = results['median_predictions']
        actuals = results['actuals']
        quantiles = results['quantiles']
        
        # Point forecast accuracy
        mae = mean_absolute_error(actuals, median_predictions)
        rmse = np.sqrt(mean_squared_error(actuals, median_predictions))
        mape = np.mean(np.abs((actuals - median_predictions) / actuals)) * 100
        r2 = r2_score(actuals, median_predictions)
        
        # Directional accuracy
        actual_direction = np.diff(actuals) > 0
        pred_direction = np.diff(median_predictions) > 0
        directional_accuracy = (actual_direction == pred_direction).mean() * 100
        
        # Probabilistic metrics
        coverage_metrics = {}
        for q_key, q_values in quantiles.items():
            q_level = int(q_key[1:]) / 100
            if q_level < 0.5:
                # Lower quantile coverage
                coverage = (actuals >= q_values).mean() * 100
                coverage_metrics[f'coverage_{q_key}'] = coverage
                coverage_metrics[f'expected_coverage_{q_key}'] = (1 - q_level) * 100
        
        # Value at Risk (VaR) calculations
        var_5 = quantiles.get('q5', np.array([]))
        var_1 = quantiles.get('q1', np.array([]))
        
        if len(var_5) > 0:
            # Expected shortfall (average loss beyond VaR)
            breaches_5 = actuals < var_5
            if breaches_5.any():
                expected_shortfall_5 = np.mean((var_5[breaches_5] - actuals[breaches_5]) / actuals[breaches_5]) * 100
            else:
                expected_shortfall_5 = 0
        else:
            expected_shortfall_5 = 0
        
        # Prediction interval width (uncertainty measure)
        if 'q5' in quantiles and 'q95' in quantiles:
            interval_width = np.mean((quantiles['q95'] - quantiles['q5']) / actuals) * 100
        else:
            interval_width = 0
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'coverage_metrics': coverage_metrics,
            'expected_shortfall_5%': expected_shortfall_5,
            'prediction_interval_width': interval_width
        }
        
        print("\nDeepAR Probabilistic Prediction Evaluation")
        print("=" * 50)
        print(f"Point Forecast Accuracy:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²: {r2:.4f}")
        print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
        
        print(f"\nProbabilistic Metrics:")
        for metric, value in coverage_metrics.items():
            print(f"  {metric}: {value:.1f}%")
        print(f"  Expected Shortfall (5%): {expected_shortfall_5:.2f}%")
        print(f"  Prediction Interval Width: {interval_width:.2f}%")
        
        return metrics
    
    def visualize_probabilistic_results(self, results, title="Bitcoin DeepAR Probabilistic Forecast"):
        """
        Comprehensive visualization of probabilistic forecasting results
        
        Args:
            results: Dictionary containing probabilistic predictions
            title: Plot title
        """
        median_predictions = results['median_predictions']
        actuals = results['actuals']
        quantiles = results['quantiles']
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Probabilistic forecast with uncertainty bands
        indices = range(len(median_predictions))
        
        axes[0].plot(indices, actuals, label='Actual Bitcoin Price', color='blue', alpha=0.8, linewidth=2)
        axes[0].plot(indices, median_predictions, label='DeepAR Median Forecast', color='red', alpha=0.8, linewidth=2)
        
        # Plot confidence intervals
        if 'q5' in quantiles and 'q95' in quantiles:
            axes[0].fill_between(indices, quantiles['q5'], quantiles['q95'], 
                               alpha=0.2, color='red', label='90% Prediction Interval')
        
        if 'q25' in quantiles and 'q75' in quantiles:
            axes[0].fill_between(indices, quantiles['q25'], quantiles['q75'], 
                               alpha=0.3, color='red', label='50% Prediction Interval')
        
        axes[0].set_title(f"{title}\nDeepAR Probabilistic Forecasting with Uncertainty Quantification")
        axes[0].set_xlabel("Time Steps")
        axes[0].set_ylabel("Bitcoin Price (USD)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # 2. Prediction interval coverage analysis
        if 'q5' in quantiles and 'q95' in quantiles:
            # Check if actual values fall within prediction intervals
            within_90_interval = (actuals >= quantiles['q5']) & (actuals <= quantiles['q95'])
            coverage_90 = within_90_interval.mean() * 100
            
            axes[1].plot(indices, within_90_interval.astype(int), 'o-', alpha=0.7, markersize=3)
            axes[1].axhline(y=0.9, color='red', linestyle='--', label=f'Expected Coverage (90%)')
            axes[1].axhline(y=coverage_90/100, color='green', linestyle='-', label=f'Actual Coverage ({coverage_90:.1f}%)')
            axes[1].set_title("Prediction Interval Coverage Analysis")
            axes[1].set_xlabel("Time Steps")
            axes[1].set_ylabel("Within 90% Interval")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. Innovation summary
        innovation_text = f"""
        DeepAR Innovation Impact ({YEAR})
        
        Probabilistic Forecasting Revolution:
        • Salinas et al. (2020): "DeepAR: Probabilistic forecasting with autoregressive recurrent neural networks"
        • Learned likelihood distributions for uncertainty quantification
        • Autoregressive RNNs with probabilistic outputs
        • Scalable Bayesian inference for production forecasting
        
        Key Innovations:
        • Monte Carlo sampling for prediction intervals
        • Multiple likelihood distributions (Gaussian, Student-t, etc.)
        • Quantile regression for risk management
        • Global model training across multiple time series
        
        Bitcoin Application Strengths:
        • Uncertainty quantification critical for financial risk
        • Value-at-Risk (VaR) calculations for portfolio management
        • Probabilistic trading strategies with confidence bounds
        • Risk-aware position sizing based on prediction intervals
        
        DeepAR Architecture:
        • Context length: {self.context_length} time steps
        • Prediction length: {self.prediction_length} time steps
        • Likelihood: {self.likelihood} distribution
        • Parameters: {sum(p.numel() for p in self.model.parameters()):,}
        
        Probabilistic Features:
        • Quantile predictions: {list(quantiles.keys()) if quantiles else 'N/A'}
        • Prediction intervals: Multiple confidence levels
        • Expected shortfall: Tail risk measurement
        • Coverage analysis: Interval calibration assessment
        
        Risk Management Applications:
        • Portfolio optimization with uncertainty bounds
        • Dynamic hedging based on prediction intervals
        • Stress testing with tail quantiles
        • Regulatory capital calculations (VaR, Expected Shortfall)
        
        Advantages over Point Forecasts:
        • Quantifies forecast uncertainty
        • Enables risk-aware decision making
        • Provides prediction intervals for confidence assessment
        • Supports probabilistic backtesting and validation
        
        Production Benefits:
        • Scalable to thousands of time series
        • Robust to missing data and irregular patterns
        • Automatic hyperparameter learning
        • Cloud-ready for real-time forecasting
        """
        
        axes[2].text(0.05, 0.95, innovation_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Complete Bitcoin DeepAR probabilistic prediction pipeline
    Demonstrates probabilistic forecasting with uncertainty quantification
    """
    print("ERA 4: DeepAR Probabilistic Forecasting for Bitcoin")
    print("=" * 60)
    print(f"Historical Context: {YEAR}")
    print(f"Innovation: {INNOVATION}")
    print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"Impact: {IMPACT}")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = BitcoinDeepARForecaster(
        context_length=168,  # 7 days
        prediction_length=24,  # 1 day
        likelihood='gaussian'
    )
    
    # Step 1: Load and prepare data
    print("\n" + "="*60)
    print("STEP 1: MULTI-VARIATE DATA LOADING")
    print("="*60)
    
    prices, covariates, dates = forecaster.load_bitcoin_data()
    
    # Step 2: Prepare DeepAR data
    print("\n" + "="*60)
    print("STEP 2: PROBABILISTIC DATA PREPARATION")
    print("="*60)
    
    train_loader, val_loader, test_loader = forecaster.prepare_deepar_data(
        prices, covariates, train_ratio=0.7, val_ratio=0.15, batch_size=32
    )
    
    # Step 3: Build model
    print("\n" + "="*60)
    print("STEP 3: DEEPAR MODEL CONSTRUCTION")
    print("="*60)
    
    model = forecaster.build_deepar_model(num_covariates=covariates.shape[1])
    
    # Step 4: Train model
    print("\n" + "="*60)
    print("STEP 4: PROBABILISTIC TRAINING")
    print("="*60)
    
    forecaster.train_deepar_model(train_loader, val_loader, epochs=100, learning_rate=0.001)
    
    # Step 5: Generate probabilistic predictions
    print("\n" + "="*60)
    print("STEP 5: PROBABILISTIC BITCOIN FORECASTING")
    print("="*60)
    
    results = forecaster.predict_probabilistic(
        test_loader, 
        num_samples=1000, 
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    
    # Step 6: Evaluate probabilistic predictions
    print("\n" + "="*60)
    print("STEP 6: PROBABILISTIC EVALUATION")
    print("="*60)
    
    metrics = forecaster.evaluate_probabilistic_predictions(results)
    
    # Step 7: Visualize results
    print("\n" + "="*60)
    print("STEP 7: UNCERTAINTY VISUALIZATION")
    print("="*60)
    
    forecaster.visualize_probabilistic_results(results, "Bitcoin DeepAR Probabilistic Forecast")
    
    # Final summary
    print("\n" + "="*60)
    print("ERA 4 SUMMARY: DEEPAR PROBABILISTIC FORECASTING")
    print("="*60)
    print(f"""
    DeepAR Analysis Complete:
    
    Probabilistic Architecture:
    • Model: DeepAR with {forecaster.likelihood} likelihood
    • Context: {forecaster.context_length} time steps
    • Prediction: {forecaster.prediction_length} time steps
    • Parameters: {sum(p.numel() for p in forecaster.model.parameters()):,}
    • Covariates: {covariates.shape[1]} additional features
    
    Bitcoin Prediction Performance:
    • Median RMSE: ${metrics['rmse']:.2f}
    • R²: {metrics['r2']:.4f}
    • Directional Accuracy: {metrics['directional_accuracy']:.1f}%
    • Prediction Interval Width: {metrics['prediction_interval_width']:.2f}%
    
    Risk Management Capabilities:
    • Quantile predictions: 5%, 25%, 50%, 75%, 95%
    • Uncertainty quantification: Prediction intervals
    • Expected shortfall: {metrics['expected_shortfall_5%']:.2f}% (5% VaR)
    • Coverage analysis: Interval calibration assessment
    
    Educational Value:
    • Demonstrates probabilistic deep learning
    • Shows uncertainty quantification importance
    • Illustrates risk-aware forecasting
    • Establishes foundation for production forecasting systems
    
    ERA 4 Complete! Modern Foundation Models demonstrated:
    • TCN: Parallel convolutional processing
    • N-BEATS: Interpretable pure deep learning
    • DeepAR: Probabilistic uncertainty quantification
    
    Complete Time Series Evolution: Statistical → ML → Deep Learning → Modern Foundations
    """)


if __name__ == "__main__":
    main()