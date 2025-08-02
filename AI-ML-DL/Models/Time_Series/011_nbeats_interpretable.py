"""
Time Series ERA 4: N-BEATS Interpretable (2020-Present)
======================================================

Historical Context:
Year: 2020 (Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting")
Innovation: Pure deep learning approach with interpretable trend and seasonality decomposition
Previous Limitation: Deep learning models lacked interpretability and required feature engineering
Impact: Achieved state-of-the-art results with interpretable components and no feature engineering

This implementation demonstrates N-BEATS applied to Bitcoin price prediction with
interpretable deep learning, trend and seasonality decomposition,
pure deep learning without hand-crafted features, and backcast/forecast interpretation.
"""

# Historical Context & Innovation
YEAR = "2020"
INNOVATION = "N-BEATS pure deep learning with interpretable trend/seasonality decomposition"
PREVIOUS_LIMITATION = "Deep learning time series models lacked interpretability and required manual feature engineering"
IMPACT = "Achieved SOTA performance with interpretable components using only raw time series data"

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
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class NBeatsBlock(nn.Module):
    """
    N-BEATS block with trend or seasonality interpretation
    
    Each block produces both backcast (reconstruction) and forecast predictions
    along with interpretable basis functions for trend or seasonality
    """
    
    def __init__(self, input_size, theta_size, basis_size, num_layers=4, layer_size=512, 
                 share_weights=False, block_type='generic'):
        super(NBeatsBlock, self).__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_size = basis_size
        self.block_type = block_type
        self.share_weights = share_weights
        
        # Fully connected stack
        layers = []
        layers.append(nn.Linear(input_size, layer_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta layers (for basis function parameters)
        self.theta_b_fc = nn.Linear(layer_size, theta_size)  # Backcast theta
        self.theta_f_fc = nn.Linear(layer_size, theta_size)  # Forecast theta
        
        # Basis function parameters
        if block_type == 'trend':
            # Polynomial basis for trend
            self.register_buffer('T_b', self._polynomial_basis(input_size))
            self.register_buffer('T_f', self._polynomial_basis(basis_size))
        elif block_type == 'seasonality':
            # Fourier basis for seasonality
            self.register_buffer('T_b', self._fourier_basis(input_size))
            self.register_buffer('T_f', self._fourier_basis(basis_size))
        else:
            # Generic learnable basis
            self.T_b = nn.Parameter(torch.randn(theta_size, input_size) * 0.02)
            self.T_f = nn.Parameter(torch.randn(theta_size, basis_size) * 0.02)
    
    def _polynomial_basis(self, size):
        """Generate polynomial basis functions for trend modeling"""
        t = torch.arange(0, size, dtype=torch.float32) / size
        basis = torch.stack([t**i for i in range(self.theta_size)], dim=0)
        return basis
    
    def _fourier_basis(self, size):
        """Generate Fourier basis functions for seasonality modeling"""
        t = torch.arange(0, size, dtype=torch.float32) / size
        basis = []
        
        # Add constant term
        basis.append(torch.ones(size))
        
        # Add sine and cosine terms
        for i in range(1, (self.theta_size // 2) + 1):
            basis.append(torch.cos(2 * np.pi * i * t))
            if len(basis) < self.theta_size:
                basis.append(torch.sin(2 * np.pi * i * t))
        
        return torch.stack(basis[:self.theta_size], dim=0)
    
    def forward(self, x):
        # Pass through fully connected stack
        hidden = self.fc_stack(x)  # [batch_size, layer_size]
        
        # Generate theta parameters
        theta_b = self.theta_b_fc(hidden)  # [batch_size, theta_size]
        theta_f = self.theta_f_fc(hidden)  # [batch_size, theta_size]
        
        # Generate backcast and forecast using basis functions
        backcast = torch.matmul(theta_b, self.T_b)  # [batch_size, input_size]
        forecast = torch.matmul(theta_f, self.T_f)  # [batch_size, basis_size]
        
        return backcast, forecast, theta_b, theta_f

class NBeatsStack(nn.Module):
    """
    N-BEATS stack consisting of multiple blocks of the same type
    """
    
    def __init__(self, input_size, forecast_size, num_blocks=3, block_type='generic',
                 theta_size=None, share_weights=False, num_layers=4, layer_size=512):
        super(NBeatsStack, self).__init__()
        
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.num_blocks = num_blocks
        self.block_type = block_type
        
        # Set theta size based on block type
        if theta_size is None:
            if block_type == 'trend':
                theta_size = 3  # Polynomial degree + 1
            elif block_type == 'seasonality':
                theta_size = min(input_size // 2, 10) * 2 + 1  # Fourier harmonics
            else:
                theta_size = input_size  # Generic
        
        # Create blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = NBeatsBlock(
                input_size=input_size,
                theta_size=theta_size,
                basis_size=forecast_size,
                num_layers=num_layers,
                layer_size=layer_size,
                share_weights=share_weights,
                block_type=block_type
            )
            self.blocks.append(block)
    
    def forward(self, x):
        residual = x
        forecast_sum = 0
        
        # Store intermediate results for interpretation
        backcast_components = []
        forecast_components = []
        theta_b_list = []
        theta_f_list = []
        
        for block in self.blocks:
            backcast, forecast, theta_b, theta_f = block(residual)
            
            # Store for interpretation
            backcast_components.append(backcast)
            forecast_components.append(forecast)
            theta_b_list.append(theta_b)
            theta_f_list.append(theta_f)
            
            # Update residual and forecast sum
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        
        return forecast_sum, {
            'backcast_components': backcast_components,
            'forecast_components': forecast_components,
            'theta_b': theta_b_list,
            'theta_f': theta_f_list,
            'residual': residual
        }

class NBeatsModel(nn.Module):
    """
    Complete N-BEATS model with multiple stacks for interpretable forecasting
    
    Features:
    - Trend stack for modeling long-term trends
    - Seasonality stack for modeling periodic patterns
    - Generic stack for modeling residual patterns
    - Hierarchical decomposition with interpretable components
    """
    
    def __init__(self, input_size, forecast_size, stack_types=['trend', 'seasonality', 'generic'],
                 num_blocks_per_stack=3, share_weights_in_stack=False, 
                 num_layers=4, layer_size=512):
        super(NBeatsModel, self).__init__()
        
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.stack_types = stack_types
        
        # Create stacks
        self.stacks = nn.ModuleList()
        for stack_type in stack_types:
            stack = NBeatsStack(
                input_size=input_size,
                forecast_size=forecast_size,
                num_blocks=num_blocks_per_stack,
                block_type=stack_type,
                share_weights=share_weights_in_stack,
                num_layers=num_layers,
                layer_size=layer_size
            )
            self.stacks.append(stack)
    
    def forward(self, x):
        residual = x
        forecast_sum = 0
        
        # Store results from each stack for interpretation
        stack_outputs = {}
        
        for i, (stack, stack_type) in enumerate(zip(self.stacks, self.stack_types)):
            stack_forecast, stack_info = stack(residual)
            
            # Store stack information
            stack_outputs[f'{stack_type}_stack'] = {
                'forecast': stack_forecast,
                'info': stack_info
            }
            
            # Update residual and total forecast
            residual = stack_info['residual']
            forecast_sum = forecast_sum + stack_forecast
        
        return forecast_sum, stack_outputs

class BitcoinNBeatsDataset(Dataset):
    """
    Dataset for Bitcoin N-BEATS with lookback and forecast horizons
    """
    
    def __init__(self, data, lookback_size=168, forecast_size=24):  # 7 days lookback, 1 day forecast
        self.data = data
        self.lookback_size = lookback_size
        self.forecast_size = forecast_size
        
    def __len__(self):
        return len(self.data) - self.lookback_size - self.forecast_size + 1
    
    def __getitem__(self, idx):
        # Lookback window
        lookback = self.data[idx:idx + self.lookback_size]
        
        # Forecast target
        forecast = self.data[idx + self.lookback_size:idx + self.lookback_size + self.forecast_size]
        
        return torch.FloatTensor(lookback), torch.FloatTensor(forecast)

class BitcoinNBeatsForecaster:
    """
    N-BEATS based Bitcoin price forecaster with interpretable components
    
    Features:
    1. Pure deep learning without feature engineering
    2. Interpretable trend and seasonality decomposition
    3. Hierarchical residual learning
    4. Multi-horizon forecasting
    5. Component analysis and visualization
    """
    
    def __init__(self, lookback_size=168, forecast_size=24, stack_types=['trend', 'seasonality', 'generic']):
        self.lookback_size = lookback_size  # 7 days * 24 hours
        self.forecast_size = forecast_size  # 1 day * 24 hours
        self.stack_types = stack_types
        self.model = None
        self.scaler = MinMaxScaler()
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin data for N-BEATS analysis (using hourly data for better resolution)
        
        Returns:
            Raw Bitcoin price data
        """
        print("Loading Bitcoin Data for N-BEATS Analysis...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
        btc_data = btc_data.dropna()
        
        # For demonstration, we'll use daily data but treat it as if it's hourly
        # In practice, you would use actual hourly data
        prices = btc_data['Close'].values
        
        print(f"Data loaded: {len(prices)} total observations")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        
        return prices, btc_data.index
    
    def prepare_nbeats_data(self, prices, train_ratio=0.7, val_ratio=0.15, batch_size=32):
        """
        Prepare data for N-BEATS training (pure time series, no features)
        
        Args:
            prices: Raw price time series
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            batch_size: Batch size
            
        Returns:
            DataLoaders for train, validation, and test sets
        """
        print("Preparing N-BEATS data (pure time series)...")
        
        # Scale prices
        prices_scaled = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Calculate split indices
        n_samples = len(prices_scaled)
        train_end = int(train_ratio * n_samples)
        val_end = int((train_ratio + val_ratio) * n_samples)
        
        # Create datasets
        train_dataset = BitcoinNBeatsDataset(
            prices_scaled[:train_end], self.lookback_size, self.forecast_size
        )
        val_dataset = BitcoinNBeatsDataset(
            prices_scaled[train_end:val_end], self.lookback_size, self.forecast_size
        )
        test_dataset = BitcoinNBeatsDataset(
            prices_scaled[val_end:], self.lookback_size, self.forecast_size
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"N-BEATS data preparation complete:")
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Validation sequences: {len(val_dataset)}")
        print(f"  Test sequences: {len(test_dataset)}")
        print(f"  Lookback size: {self.lookback_size}")
        print(f"  Forecast size: {self.forecast_size}")
        
        return train_loader, val_loader, test_loader
    
    def build_nbeats_model(self):
        """
        Build N-BEATS model for Bitcoin forecasting
        """
        print("Building N-BEATS model...")
        
        self.model = NBeatsModel(
            input_size=self.lookback_size,
            forecast_size=self.forecast_size,
            stack_types=self.stack_types,
            num_blocks_per_stack=3,
            share_weights_in_stack=False,
            num_layers=4,
            layer_size=512
        ).to(device)
        
        # Model analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"N-BEATS model built:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Stacks: {self.stack_types}")
        print(f"  Lookback size: {self.lookback_size}")
        print(f"  Forecast size: {self.forecast_size}")
        print(f"  Pure deep learning: No feature engineering required")
        
        return self.model
    
    def train_nbeats_model(self, train_loader, val_loader, epochs=100, learning_rate=0.001):
        """
        Train N-BEATS model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        print("Training N-BEATS model...")
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
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
            
            for batch_lookback, batch_forecast in train_loader:
                batch_lookback = batch_lookback.to(device)
                batch_forecast = batch_forecast.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predicted_forecast, stack_outputs = self.model(batch_lookback)
                
                # Calculate loss
                loss = criterion(predicted_forecast, batch_forecast)
                
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
                for batch_lookback, batch_forecast in val_loader:
                    batch_lookback = batch_lookback.to(device)
                    batch_forecast = batch_forecast.to(device)
                    
                    predicted_forecast, _ = self.model(batch_lookback)
                    loss = criterion(predicted_forecast, batch_forecast)
                    
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
                torch.save(self.model.state_dict(), 'best_nbeats_model.pth')
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
        self.model.load_state_dict(torch.load('best_nbeats_model.pth'))
        print("Training completed! Best model loaded.")
    
    def predict_with_interpretation(self, test_loader):
        """
        Generate predictions with interpretable component analysis
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary containing predictions and interpretable components
        """
        print("Generating interpretable Bitcoin forecasts...")
        
        self.model.eval()
        all_predictions = []
        all_actuals = []
        all_stack_outputs = []
        
        with torch.no_grad():
            for batch_lookback, batch_forecast in test_loader:
                batch_lookback = batch_lookback.to(device)
                batch_forecast = batch_forecast.to(device)
                
                # Forward pass with interpretation
                predicted_forecast, stack_outputs = self.model(batch_lookback)
                
                # Store results
                pred_scaled = predicted_forecast.cpu().numpy()
                actual_scaled = batch_forecast.cpu().numpy()
                
                all_predictions.extend(pred_scaled)
                all_actuals.extend(actual_scaled)
                all_stack_outputs.append(stack_outputs)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        
        # Inverse transform (only for 1-day ahead, take first forecast step)
        if predictions.ndim > 1:
            predictions_1d = predictions[:, 0]  # First forecast step
            actuals_1d = actuals[:, 0]
        else:
            predictions_1d = predictions
            actuals_1d = actuals
        
        predictions_original = self.scaler.inverse_transform(predictions_1d.reshape(-1, 1)).flatten()
        actuals_original = self.scaler.inverse_transform(actuals_1d.reshape(-1, 1)).flatten()
        
        print(f"N-BEATS predictions with interpretation complete:")
        print(f"  Predictions: {len(predictions_original)}")
        print(f"  Stack components: {len(self.stack_types)} interpretable stacks")
        print(f"  Prediction range: ${predictions_original.min():.2f} - ${predictions_original.max():.2f}")
        
        return {
            'predictions': predictions_original,
            'actuals': actuals_original,
            'stack_outputs': all_stack_outputs,
            'predictions_multi': predictions,
            'actuals_multi': actuals
        }
    
    def evaluate_nbeats_predictions(self, results):
        """
        Comprehensive evaluation of N-BEATS predictions
        
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
        
        print("\nN-BEATS Bitcoin Prediction Evaluation")
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
    
    def analyze_interpretable_components(self, results, n_samples=3):
        """
        Analyze and visualize interpretable N-BEATS components
        
        Args:
            results: Prediction results with stack outputs
            n_samples: Number of samples to analyze
        """
        print("Analyzing N-BEATS interpretable components...")
        
        if not results['stack_outputs']:
            print("No stack outputs available for analysis")
            return
        
        # Analyze first batch of stack outputs
        stack_outputs = results['stack_outputs'][0]
        
        fig, axes = plt.subplots(len(self.stack_types), 2, figsize=(15, 4*len(self.stack_types)))
        if len(self.stack_types) == 1:
            axes = axes.reshape(1, -1)
        
        for i, stack_type in enumerate(self.stack_types):
            if f'{stack_type}_stack' in stack_outputs:
                stack_data = stack_outputs[f'{stack_type}_stack']
                
                # Get first sample for visualization
                forecast_components = stack_data['info']['forecast_components']
                backcast_components = stack_data['info']['backcast_components']
                
                # Plot forecast components
                for j, forecast_comp in enumerate(forecast_components):
                    if j < n_samples:
                        component = forecast_comp[0].cpu().numpy()  # First sample
                        axes[i, 0].plot(component, alpha=0.7, label=f'Block {j+1}')
                
                axes[i, 0].set_title(f'{stack_type.capitalize()} Stack - Forecast Components')
                axes[i, 0].set_xlabel('Forecast Horizon')
                axes[i, 0].set_ylabel('Component Value')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
                
                # Plot backcast components
                for j, backcast_comp in enumerate(backcast_components):
                    if j < n_samples:
                        component = backcast_comp[0].cpu().numpy()  # First sample
                        axes[i, 1].plot(component, alpha=0.7, label=f'Block {j+1}')
                
                axes[i, 1].set_title(f'{stack_type.capitalize()} Stack - Backcast Components')
                axes[i, 1].set_xlabel('Lookback Time Steps')
                axes[i, 1].set_ylabel('Component Value')
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
        
        plt.suptitle('N-BEATS Interpretable Component Decomposition', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def visualize_nbeats_results(self, results, title="Bitcoin N-BEATS Interpretable Forecast"):
        """
        Comprehensive visualization of N-BEATS results
        
        Args:
            results: Dictionary containing predictions and interpretable components
            title: Plot title
        """
        predictions = results['predictions']
        actuals = results['actuals']
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Price predictions vs actuals
        indices = range(len(predictions))
        
        axes[0].plot(indices, actuals, label='Actual Bitcoin Price', color='blue', alpha=0.7, linewidth=2)
        axes[0].plot(indices, predictions, label='N-BEATS Prediction', color='red', alpha=0.8, linewidth=2)
        
        axes[0].set_title(f"{title}\nNeural Basis Expansion Analysis with Interpretable Components")
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
            axes[1].set_title("N-BEATS Training History")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss (MSE)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        
        # 3. Innovation summary
        innovation_text = f"""
        N-BEATS Innovation Impact ({YEAR})
        
        Pure Deep Learning Revolution:
        • Oreshkin et al. (2020): "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
        • No feature engineering required - works with raw time series data
        • Interpretable trend and seasonality decomposition
        • Hierarchical doubly residual stacking architecture
        
        Key Innovations:
        • Basis expansion with learnable or fixed basis functions
        • Backcast and forecast branches for residual learning
        • Stack-specific interpretability (trend, seasonality, generic)
        • State-of-the-art performance without external features
        
        Bitcoin Application Strengths:
        • Pure price signal analysis without technical indicators
        • Interpretable trend and cycle identification
        • Multi-horizon forecasting capability
        • Automatic pattern discovery in price movements
        
        N-BEATS Architecture:
        • Stacks: {len(self.stack_types)} ({', '.join(self.stack_types)})
        • Lookback window: {self.lookback_size} time steps
        • Forecast horizon: {self.forecast_size} time steps
        • Parameters: {sum(p.numel() for p in self.model.parameters()):,}
        
        Interpretability Features:
        • Trend stack: Polynomial basis functions for long-term trends
        • Seasonality stack: Fourier basis functions for periodic patterns
        • Generic stack: Learnable basis for complex residual patterns
        • Component decomposition: Clear attribution of forecast sources
        
        Advantages over Traditional Deep Learning:
        • No manual feature engineering required
        • Interpretable components aid decision making
        • Robust to different time series characteristics
        • Hierarchical learning captures multiple patterns
        
        Limitations:
        • Fixed architecture may not suit all domains
        • Computational overhead from multiple stacks
        • Limited incorporation of external variables
        • Basis function choice affects interpretability
        """
        
        axes[2].text(0.05, 0.95, innovation_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgoldenrodyellow", alpha=0.8))
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze interpretable components
        self.analyze_interpretable_components(results)


def main():
    """
    Complete Bitcoin N-BEATS prediction pipeline
    Demonstrates interpretable deep learning for time series forecasting
    """
    print("ERA 4: N-BEATS Interpretable Forecasting for Bitcoin")
    print("=" * 60)
    print(f"Historical Context: {YEAR}")
    print(f"Innovation: {INNOVATION}")
    print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"Impact: {IMPACT}")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = BitcoinNBeatsForecaster(
        lookback_size=168,  # 7 days
        forecast_size=24,   # 1 day
        stack_types=['trend', 'seasonality', 'generic']
    )
    
    # Step 1: Load and prepare data
    print("\n" + "="*60)
    print("STEP 1: RAW TIME SERIES DATA LOADING")
    print("="*60)
    
    prices, dates = forecaster.load_bitcoin_data()
    
    # Step 2: Prepare N-BEATS data (no feature engineering)
    print("\n" + "="*60)
    print("STEP 2: PURE TIME SERIES PREPARATION")
    print("="*60)
    
    train_loader, val_loader, test_loader = forecaster.prepare_nbeats_data(
        prices, train_ratio=0.7, val_ratio=0.15, batch_size=32
    )
    
    # Step 3: Build model
    print("\n" + "="*60)
    print("STEP 3: N-BEATS MODEL CONSTRUCTION")
    print("="*60)
    
    model = forecaster.build_nbeats_model()
    
    # Step 4: Train model
    print("\n" + "="*60)
    print("STEP 4: INTERPRETABLE DEEP LEARNING TRAINING")
    print("="*60)
    
    forecaster.train_nbeats_model(train_loader, val_loader, epochs=100, learning_rate=0.001)
    
    # Step 5: Generate interpretable predictions
    print("\n" + "="*60)
    print("STEP 5: INTERPRETABLE BITCOIN FORECASTING")
    print("="*60)
    
    results = forecaster.predict_with_interpretation(test_loader)
    
    # Step 6: Evaluate predictions
    print("\n" + "="*60)
    print("STEP 6: N-BEATS EVALUATION")
    print("="*60)
    
    metrics = forecaster.evaluate_nbeats_predictions(results)
    
    # Step 7: Visualize results and components
    print("\n" + "="*60)
    print("STEP 7: INTERPRETABILITY VISUALIZATION")
    print("="*60)
    
    forecaster.visualize_nbeats_results(results, "Bitcoin N-BEATS Interpretable Forecast")
    
    # Final summary
    print("\n" + "="*60)
    print("ERA 4 SUMMARY: N-BEATS INTERPRETABLE FORECASTING")
    print("="*60)
    print(f"""
    N-BEATS Analysis Complete:
    
    Pure Deep Learning Architecture:
    • Model: N-BEATS with {len(forecaster.stack_types)} interpretable stacks
    • Stacks: {', '.join(forecaster.stack_types)}
    • Parameters: {sum(p.numel() for p in forecaster.model.parameters()):,}
    • No feature engineering: Pure time series input
    
    Bitcoin Prediction Performance:
    • RMSE: ${metrics['rmse']:.2f}
    • R²: {metrics['r2']:.4f}
    • Directional Accuracy: {metrics['directional_accuracy']:.1f}%
    • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
    
    Interpretability Achievements:
    • Trend decomposition: Polynomial basis functions
    • Seasonality analysis: Fourier basis functions
    • Residual patterns: Learnable generic basis
    • Component attribution: Clear forecast source identification
    
    Educational Value:
    • Demonstrates pure deep learning approach
    • Shows interpretable neural network design
    • Illustrates hierarchical residual learning
    • Establishes foundation for modern interpretable AI
    
    Next: DeepAR for probabilistic forecasting with uncertainty quantification!
    """)


if __name__ == "__main__":
    main()