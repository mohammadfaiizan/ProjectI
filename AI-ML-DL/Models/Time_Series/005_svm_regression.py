"""
Time Series ERA 2: Support Vector Regression (1990s-2010s)
==========================================================

Historical Context:
Year: 1995-2010 (SVM development by Vapnik, SVR extension)
Innovation: Kernel methods for non-linear regression with structural risk minimization
Previous Limitation: Linear methods couldn't capture complex non-linear patterns efficiently
Impact: Introduced kernel trick and robust non-linear modeling for time series forecasting

This implementation demonstrates Support Vector Regression applied to Bitcoin price
prediction, showcasing kernel methods for capturing non-linear patterns in
cryptocurrency markets with proper feature scaling and hyperparameter optimization.
"""

# Historical Context & Innovation
YEAR = "1995-2010"
INNOVATION = "Support Vector Regression with kernel methods and structural risk minimization"
PREVIOUS_LIMITATION = "Linear methods insufficient for complex non-linear financial patterns"
IMPACT = "Enabled robust non-linear modeling with theoretical foundations and kernel flexibility"

# Imports and Dependencies
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.pipeline import Pipeline
import talib as ta
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class BitcoinSVRForecaster:
    """
    Support Vector Regression for Bitcoin price prediction with kernel methods
    
    Features:
    1. Multiple kernel functions (RBF, Polynomial, Sigmoid)
    2. Feature scaling and preprocessing pipelines
    3. Hyperparameter optimization with time series validation
    4. Feature selection and engineering
    5. Kernel parameter analysis and visualization
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.data = None
        self.best_features = None
        self.kernel_performance = {}
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin OHLCV data for SVR analysis
        
        Returns:
            Bitcoin OHLCV DataFrame
        """
        print("Loading Bitcoin Data for SVR Analysis...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_data = btc_data.dropna()
        
        # Store raw data
        self.data = btc_data.copy()
        
        print(f"Data loaded: {len(btc_data)} total observations")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        print(f"Price range: ${btc_data['Close'].min():.2f} - ${btc_data['Close'].max():.2f}")
        
        return btc_data
    
    def engineer_svm_features(self, data):
        """
        Feature engineering optimized for SVM regression
        
        Args:
            data: Bitcoin OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        print("\nEngineering Features for SVM...")
        
        features = data.copy()
        
        # Basic price features
        features['returns'] = features['Close'].pct_change()
        features['log_returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['price_momentum'] = features['Close'] / features['Close'].shift(5) - 1
        features['volume_momentum'] = features['Volume'] / features['Volume'].shift(5) - 1
        
        # Price transformations
        features['log_price'] = np.log(features['Close'])
        features['sqrt_price'] = np.sqrt(features['Close'])
        features['price_squared'] = features['Close'] ** 2
        
        # Moving averages and ratios
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = features['Close'].rolling(window=window).mean()
            features[f'price_ma_ratio_{window}'] = features['Close'] / features[f'ma_{window}']
            features[f'ma_slope_{window}'] = features[f'ma_{window}'].diff(5)
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
            features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features['returns'].std()
        
        # Technical indicators (with error handling)
        try:
            # RSI
            features['rsi_14'] = ta.RSI(features['Close'].values, timeperiod=14)
            features['rsi_normalized'] = (features['rsi_14'] - 50) / 50
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(features['Close'].values)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            features['macd_ratio'] = macd / macd_signal
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(features['Close'].values)
            features['bb_position'] = (features['Close'] - bb_lower) / (bb_upper - bb_lower)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Stochastic
            slowk, slowd = ta.STOCH(features['High'].values, features['Low'].values, features['Close'].values)
            features['stoch_k'] = slowk / 100  # Normalize to [0,1]
            features['stoch_d'] = slowd / 100
            
            # Williams %R
            features['williams_r'] = ta.WILLR(features['High'].values, features['Low'].values, features['Close'].values)
            features['williams_r_norm'] = (features['williams_r'] + 100) / 100  # Normalize to [0,1]
            
            print("Technical indicators calculated successfully")
            
        except Exception as e:
            print(f"WARNING: Some technical indicators failed: {str(e)}")
        
        # Lag features (important for SVM temporal patterns)
        for lag in [1, 2, 3, 5, 7]:
            features[f'price_lag_{lag}'] = features['Close'].shift(lag)
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = features['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'price_rolling_std_{window}'] = features['Close'].rolling(window=window).std()
            features[f'price_rolling_skew_{window}'] = features['Close'].rolling(window=window).skew()
            features[f'price_rolling_kurt_{window}'] = features['Close'].rolling(window=window).kurt()
        
        # Market microstructure
        features['spread'] = features['High'] - features['Low']
        features['spread_normalized'] = features['spread'] / features['Close']
        features['body'] = abs(features['Close'] - features['Open'])
        features['body_normalized'] = features['body'] / features['Close']
        features['upper_shadow'] = features['High'] - np.maximum(features['Close'], features['Open'])
        features['lower_shadow'] = np.minimum(features['Close'], features['Open']) - features['Low']
        
        # Time-based features (cyclical encoding for SVM)
        features['day_of_week_sin'] = np.sin(2 * np.pi * features.index.dayofweek / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features.index.dayofweek / 7)
        features['month_sin'] = np.sin(2 * np.pi * features.index.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * features.index.month / 12)
        features['day_of_year_sin'] = np.sin(2 * np.pi * features.index.dayofyear / 365)
        features['day_of_year_cos'] = np.cos(2 * np.pi * features.index.dayofyear / 365)
        
        # Interaction features (important for kernel methods)
        features['price_volume_interaction'] = features['Close'] * features['Volume']
        features['volatility_momentum_interaction'] = features['volatility_10'] * features['price_momentum']
        features['rsi_price_interaction'] = features.get('rsi_normalized', 0) * features['price_ma_ratio_20']
        
        print(f"Total features engineered: {len(features.columns)}")
        return features
    
    def select_optimal_features(self, X, y, n_features=30):
        """
        Feature selection optimized for SVM regression
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            Selected feature indices and names
        """
        print(f"\nSelecting optimal features for SVM (target: {n_features} features)...")
        
        # Remove features with too many missing values
        missing_threshold = 0.1
        valid_features = X.columns[X.isnull().mean() < missing_threshold]
        X_clean = X[valid_features]
        
        # Fill remaining missing values
        X_clean = X_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Combine multiple feature selection methods
        
        # 1. Statistical significance (F-test)
        f_selector = SelectKBest(score_func=f_regression, k=min(50, len(X_clean.columns)))
        X_f_selected = f_selector.fit_transform(X_clean, y)
        f_selected_features = X_clean.columns[f_selector.get_support()]
        
        # 2. Recursive Feature Elimination with SVR
        svr_selector = SVR(kernel='rbf', C=1.0, gamma='scale')
        rfe_selector = RFE(estimator=svr_selector, n_features_to_select=n_features)
        X_rfe_selected = rfe_selector.fit_transform(X_clean, y)
        rfe_selected_features = X_clean.columns[rfe_selector.get_support()]
        
        # 3. Combine selections (intersection for robustness)
        combined_features = list(set(f_selected_features) & set(rfe_selected_features))
        
        # If intersection is too small, take union and select top by F-scores
        if len(combined_features) < n_features:
            combined_features = list(set(f_selected_features) | set(rfe_selected_features))
            
            # Score remaining features
            remaining_selector = SelectKBest(score_func=f_regression, k=n_features)
            remaining_selector.fit(X_clean[combined_features], y)
            feature_scores = remaining_selector.scores_
            
            # Select top N features
            top_indices = np.argsort(feature_scores)[-n_features:]
            final_features = [combined_features[i] for i in top_indices]
        else:
            final_features = combined_features[:n_features]
        
        self.best_features = final_features
        
        print(f"Feature selection complete:")
        print(f"  F-test selected: {len(f_selected_features)} features")
        print(f"  RFE selected: {len(rfe_selected_features)} features")
        print(f"  Final selection: {len(final_features)} features")
        
        return final_features
    
    def analyze_scaling_methods(self, X_train, y_train, X_val, y_val):
        """
        Compare different scaling methods for SVM performance
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Best scaler and performance comparison
        """
        print("\nAnalyzing Feature Scaling Methods for SVM...")
        
        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }
        
        scaling_results = {}
        
        for scaler_name, scaler in scalers.items():
            try:
                # Fit scaler on training data
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Quick SVR model for comparison
                svr = SVR(kernel='rbf', C=1.0, gamma='scale')
                svr.fit(X_train_scaled, y_train)
                
                # Validation performance
                val_pred = svr.predict(X_val_scaled)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                val_r2 = r2_score(y_val, val_pred)
                
                scaling_results[scaler_name] = {
                    'rmse': val_rmse,
                    'r2': val_r2,
                    'scaler': scaler
                }
                
                print(f"  {scaler_name}: RMSE=${val_rmse:.2f}, R²={val_r2:.4f}")
                
            except Exception as e:
                print(f"  {scaler_name}: FAILED - {str(e)}")
                continue
        
        # Select best scaler
        if scaling_results:
            best_scaler_name = min(scaling_results.keys(), key=lambda x: scaling_results[x]['rmse'])
            best_scaler = scaling_results[best_scaler_name]['scaler']
            
            print(f"Best scaling method: {best_scaler_name}")
            self.scalers['best'] = best_scaler
            
            return best_scaler, scaling_results
        else:
            print("WARNING: All scaling methods failed, using StandardScaler as fallback")
            fallback_scaler = StandardScaler()
            self.scalers['best'] = fallback_scaler
            return fallback_scaler, {}
    
    def fit_svr_kernels(self, X_train, y_train, X_val, y_val):
        """
        Fit SVR with different kernel functions and hyperparameter optimization
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        print("\nFitting SVR with Multiple Kernels...")
        
        # Define kernel configurations
        kernel_configs = {
            'RBF': {
                'kernel': ['rbf'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'epsilon': [0.01, 0.1, 0.2]
            },
            'Polynomial': {
                'kernel': ['poly'],
                'C': [0.1, 1, 10, 100],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'epsilon': [0.01, 0.1, 0.2]
            },
            'Sigmoid': {
                'kernel': ['sigmoid'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        best_overall_score = -np.inf
        best_overall_model = None
        best_kernel_name = None
        
        for kernel_name, param_grid in kernel_configs.items():
            print(f"\nOptimizing {kernel_name} kernel...")
            
            try:
                # Grid search for this kernel
                svr = SVR()
                grid_search = GridSearchCV(
                    svr, param_grid, cv=tscv, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Best model for this kernel
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_
                
                # Validation performance
                val_pred = best_model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                val_mae = mean_absolute_error(y_val, val_pred)
                val_r2 = r2_score(y_val, val_pred)
                
                # Store results
                self.models[kernel_name] = best_model
                self.kernel_performance[kernel_name] = {
                    'cv_score': best_score,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_r2': val_r2,
                    'best_params': grid_search.best_params_
                }
                
                print(f"  Best CV score: {best_score:.4f}")
                print(f"  Validation RMSE: ${val_rmse:.2f}")
                print(f"  Validation R²: {val_r2:.4f}")
                print(f"  Best params: {grid_search.best_params_}")
                
                # Track overall best
                if best_score > best_overall_score:
                    best_overall_score = best_score
                    best_overall_model = best_model
                    best_kernel_name = kernel_name
                    
            except Exception as e:
                print(f"  {kernel_name} kernel optimization failed: {str(e)}")
                continue
        
        if best_overall_model is not None:
            self.models['best'] = best_overall_model
            print(f"\nBest overall kernel: {best_kernel_name}")
            print(f"Best overall CV score: {best_overall_score:.4f}")
        else:
            print("\nWARNING: All kernel optimizations failed!")
        
        return best_overall_model
    
    def analyze_kernel_complexity(self, X_train, y_train):
        """
        Analyze model complexity vs performance for different kernels
        
        Args:
            X_train, y_train: Training data
        """
        print("\nAnalyzing Kernel Complexity vs Performance...")
        
        # RBF kernel complexity analysis
        if 'RBF' in self.models:
            # C parameter validation curve
            C_range = np.logspace(-2, 2, 10)
            train_scores, val_scores = validation_curve(
                SVR(kernel='rbf', gamma='scale'), X_train, y_train,
                param_name='C', param_range=C_range,
                cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error'
            )
            
            # Gamma parameter validation curve
            gamma_range = np.logspace(-4, 0, 10)
            gamma_train_scores, gamma_val_scores = validation_curve(
                SVR(kernel='rbf', C=1.0), X_train, y_train,
                param_name='gamma', param_range=gamma_range,
                cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error'
            )
            
            # Plot complexity analysis
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # C parameter curve
            axes[0, 0].semilogx(C_range, -train_scores.mean(axis=1), 'o-', label='Training RMSE')
            axes[0, 0].semilogx(C_range, -val_scores.mean(axis=1), 'o-', label='Validation RMSE')
            axes[0, 0].set_xlabel('C Parameter')
            axes[0, 0].set_ylabel('RMSE')
            axes[0, 0].set_title('RBF Kernel: C Parameter Validation Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Gamma parameter curve
            axes[0, 1].semilogx(gamma_range, -gamma_train_scores.mean(axis=1), 'o-', label='Training RMSE')
            axes[0, 1].semilogx(gamma_range, -gamma_val_scores.mean(axis=1), 'o-', label='Validation RMSE')
            axes[0, 1].set_xlabel('Gamma Parameter')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].set_title('RBF Kernel: Gamma Parameter Validation Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Kernel performance comparison
            if len(self.kernel_performance) > 1:
                kernels = list(self.kernel_performance.keys())
                rmse_scores = [self.kernel_performance[k]['val_rmse'] for k in kernels]
                r2_scores = [self.kernel_performance[k]['val_r2'] for k in kernels]
                
                axes[1, 0].bar(kernels, rmse_scores, alpha=0.7, color=['red', 'green', 'blue'])
                axes[1, 0].set_ylabel('Validation RMSE')
                axes[1, 0].set_title('Kernel Performance Comparison (RMSE)')
                axes[1, 0].grid(True, alpha=0.3)
                
                axes[1, 1].bar(kernels, r2_scores, alpha=0.7, color=['red', 'green', 'blue'])
                axes[1, 1].set_ylabel('Validation R²')
                axes[1, 1].set_title('Kernel Performance Comparison (R²)')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle('SVM Kernel Analysis for Bitcoin Prediction', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    def predict_bitcoin_price(self, model_name, X_test):
        """
        Generate Bitcoin price predictions using SVR
        
        Args:
            model_name: Name of SVR model to use
            X_test: Test features
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            print(f"ERROR: Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        print(f"\nBitcoin Price Forecasting - SVR {model_name}")
        print("=" * 50)
        
        try:
            # Generate predictions
            predictions = model.predict(X_test)
            
            print(f"Generated predictions for {len(predictions)} observations")
            print(f"Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}")
            
            return predictions
            
        except Exception as e:
            print(f"ERROR: Prediction failed - {str(e)}")
            return None
    
    def evaluate_bitcoin_predictions(self, actual_prices, predictions, model_name):
        """
        Comprehensive evaluation of SVR predictions
        
        Args:
            actual_prices: True Bitcoin prices
            predictions: Model predictions
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        if predictions is None or len(actual_prices) == 0:
            print("WARNING: Insufficient data for evaluation")
            return {}
        
        # Forecasting accuracy metrics
        mae = mean_absolute_error(actual_prices, predictions)
        rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
        mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
        r2 = r2_score(actual_prices, predictions)
        
        # Directional accuracy
        if len(actual_prices) > 1:
            actual_direction = np.diff(actual_prices) > 0
            pred_direction = np.diff(predictions) > 0
            
            if len(actual_direction) > 0:
                directional_accuracy = (actual_direction == pred_direction).mean() * 100
            else:
                directional_accuracy = 0
        else:
            directional_accuracy = 0
        
        # Financial metrics
        returns_actual = np.diff(actual_prices) / actual_prices[:-1]
        returns_pred = np.diff(predictions) / actual_prices[:-1]
        
        if len(returns_actual) > 1:
            # Trading strategy based on SVR predictions
            signals = np.where(np.diff(predictions) > 0, 1, -1)
            strategy_returns = signals * returns_actual
            
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
            cumulative_returns = np.cumsum(strategy_returns)
            max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
            total_return = np.sum(strategy_returns)
        else:
            sharpe_ratio = max_drawdown = total_return = 0
        
        metrics = {
            'model_name': model_name,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'total_return': total_return * 100
        }
        
        print(f"\nBitcoin SVR Prediction Evaluation - {model_name}")
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
    
    def visualize_results(self, actual_data, predictions_dict, title="Bitcoin SVR Forecast"):
        """
        Comprehensive visualization of SVR results
        
        Args:
            actual_data: Historical Bitcoin prices
            predictions_dict: Dictionary of model predictions
            title: Plot title
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Price predictions comparison
        axes[0].plot(actual_data.index, actual_data.values, 
                    label='Actual Bitcoin Price', color='blue', alpha=0.7, linewidth=2)
        
        colors = ['red', 'green', 'orange', 'purple']
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            if predictions is not None:
                pred_index = actual_data.index[-len(predictions):]
                axes[0].plot(pred_index, predictions, 
                           label=f'SVR {model_name}', 
                           color=colors[i % len(colors)], linewidth=2, alpha=0.8)
        
        axes[0].set_title(f"{title}\nSupport Vector Regression with Kernel Methods")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Bitcoin Price (USD)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # 2. Kernel performance comparison
        if len(self.kernel_performance) > 0:
            kernels = list(self.kernel_performance.keys())
            rmse_scores = [self.kernel_performance[k]['val_rmse'] for k in kernels]
            r2_scores = [self.kernel_performance[k]['val_r2'] for k in kernels]
            
            x_pos = np.arange(len(kernels))
            width = 0.35
            
            ax2_twin = axes[1].twinx()
            
            bars1 = axes[1].bar(x_pos - width/2, rmse_scores, width, 
                              label='RMSE', color='red', alpha=0.7)
            bars2 = ax2_twin.bar(x_pos + width/2, r2_scores, width, 
                               label='R²', color='blue', alpha=0.7)
            
            axes[1].set_xlabel('Kernel Type')
            axes[1].set_ylabel('RMSE ($)', color='red')
            ax2_twin.set_ylabel('R² Score', color='blue')
            axes[1].set_title('SVR Kernel Performance Comparison')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(kernels)
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars1, rmse_scores):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'${score:.0f}', ha='center', va='bottom')
            
            for bar, score in zip(bars2, r2_scores):
                height = bar.get_height()
                ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                            f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Innovation summary
        innovation_text = f"""
        Support Vector Regression Innovation Impact ({YEAR})
        
        Theoretical Foundation:
        • Vapnik (1995): Statistical Learning Theory and VC dimension
        • Structural Risk Minimization principle
        • Kernel trick for non-linear mapping to high-dimensional space
        • Support vector concept for robust regression
        
        Key Innovations:
        • Non-parametric approach with kernel flexibility
        • Robust to outliers through epsilon-insensitive loss
        • Global optimum guaranteed (convex optimization)
        • Automatic feature space transformation via kernels
        
        Bitcoin Application Strengths:
        • Captures complex non-linear price patterns
        • Robust to market outliers and volatility spikes
        • Kernel flexibility for different market regimes
        • Strong theoretical foundation for financial modeling
        
        Limitations for Crypto Markets:
        • Computationally expensive for large datasets
        • Hyperparameter sensitivity requiring careful tuning
        • Limited interpretability compared to linear methods
        • Memory requirements scale with number of support vectors
        
        Feature Engineering: {len(self.best_features) if self.best_features else 'N/A'} optimal features selected
        Best Kernel: {max(self.kernel_performance, key=lambda x: self.kernel_performance[x]['val_r2']) if self.kernel_performance else 'N/A'}
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
    Complete Bitcoin SVR prediction pipeline
    Demonstrates Support Vector Regression with kernel methods on cryptocurrency data
    """
    print("ERA 2: Support Vector Regression for Bitcoin Prediction")
    print("=" * 60)
    print(f"Historical Context: {YEAR}")
    print(f"Innovation: {INNOVATION}")
    print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"Impact: {IMPACT}")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = BitcoinSVRForecaster()
    
    # Step 1: Load and analyze data
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("="*60)
    
    raw_data = forecaster.load_bitcoin_data()
    
    # Step 2: Feature engineering
    print("\n" + "="*60)
    print("STEP 2: SVR-OPTIMIZED FEATURE ENGINEERING")
    print("="*60)
    
    engineered_data = forecaster.engineer_svm_features(raw_data)
    
    # Prepare target variable (next day price)
    target = engineered_data['Close'].shift(-1)
    
    # Step 3: Feature selection
    print("\n" + "="*60)
    print("STEP 3: OPTIMAL FEATURE SELECTION")
    print("="*60)
    
    # Remove non-numeric columns and handle missing values
    numeric_cols = engineered_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
    
    X = engineered_data[feature_cols].copy()
    y = target.copy()
    
    # Remove rows with missing values
    valid_idx = (~X.isnull().any(axis=1)) & (~y.isnull())
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Feature selection
    selected_features = forecaster.select_optimal_features(X, y, n_features=25)
    X_selected = X[selected_features]
    
    # Step 4: Train-validation split (time series aware)
    print("\n" + "="*60)
    print("STEP 4: TIME SERIES TRAIN-VALIDATION SPLIT")
    print("="*60)
    
    train_size = int(0.8 * len(X_selected))
    X_train, X_val = X_selected[:train_size], X_selected[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Selected features: {len(selected_features)}")
    
    # Step 5: Feature scaling analysis
    print("\n" + "="*60)
    print("STEP 5: FEATURE SCALING ANALYSIS")
    print("="*60)
    
    best_scaler, scaling_results = forecaster.analyze_scaling_methods(X_train, y_train, X_val, y_val)
    
    # Apply best scaling
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_val_scaled = best_scaler.transform(X_val)
    
    # Step 6: SVR kernel optimization
    print("\n" + "="*60)
    print("STEP 6: SVR KERNEL OPTIMIZATION")
    print("="*60)
    
    best_model = forecaster.fit_svr_kernels(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Step 7: Kernel complexity analysis
    print("\n" + "="*60)
    print("STEP 7: KERNEL COMPLEXITY ANALYSIS")
    print("="*60)
    
    forecaster.analyze_kernel_complexity(X_train_scaled, y_train)
    
    # Step 8: Predictions and evaluation
    print("\n" + "="*60)
    print("STEP 8: BITCOIN PRICE PREDICTION")
    print("="*60)
    
    # Generate predictions for all kernels
    predictions_dict = {}
    for model_name in forecaster.models.keys():
        predictions = forecaster.predict_bitcoin_price(model_name, X_val_scaled)
        predictions_dict[model_name] = predictions
        
        # Evaluate predictions
        if predictions is not None:
            metrics = forecaster.evaluate_bitcoin_predictions(y_val.values, predictions, model_name)
    
    # Step 9: Results visualization
    print("\n" + "="*60)
    print("STEP 9: RESULTS VISUALIZATION")
    print("="*60)
    
    # Use actual prices for visualization
    historical_prices = engineered_data['Close'][valid_idx]
    forecaster.visualize_results(historical_prices, predictions_dict, 
                               "Bitcoin SVR Forecast")
    
    # Step 10: Model comparison summary
    print("\n" + "="*60)
    print("STEP 10: KERNEL PERFORMANCE SUMMARY")
    print("="*60)
    
    if forecaster.kernel_performance:
        print("\nKernel Performance Summary:")
        print("Kernel      | RMSE     | R²      | Best Parameters")
        print("-" * 60)
        for kernel, metrics in forecaster.kernel_performance.items():
            print(f"{kernel:11} | ${metrics['val_rmse']:7.2f} | {metrics['val_r2']:6.4f} | {str(metrics['best_params'])[:30]}...")
    
    # Final summary
    print("\n" + "="*60)
    print("ERA 2 SUMMARY: SUPPORT VECTOR REGRESSION")
    print("="*60)
    print(f"""
    SVR Analysis Complete:
    
    Kernel Methods Applied:
    • Kernels tested: {len(forecaster.kernel_performance)} variants
    • Features selected: {len(selected_features)} optimal features
    • Best kernel: {max(forecaster.kernel_performance, key=lambda x: forecaster.kernel_performance[x]['val_r2']) if forecaster.kernel_performance else 'N/A'}
    • Scaling method: {type(best_scaler).__name__}
    
    Bitcoin Prediction Performance:
    • Non-linear pattern capture via kernel trick
    • Robust regression with epsilon-insensitive loss
    • Hyperparameter optimization with time series CV
    • Feature scaling and selection optimization
    
    Educational Value:
    • Demonstrates kernel methods and SVM theory
    • Shows feature scaling importance for SVMs
    • Illustrates structural risk minimization
    • Bridges classical ML and modern approaches
    
    ERA 2 Complete! Ready for ERA 3: Deep Learning Revolution!
    """)


if __name__ == "__main__":
    main()