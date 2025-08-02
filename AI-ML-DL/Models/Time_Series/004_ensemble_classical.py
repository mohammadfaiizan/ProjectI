"""
Time Series ERA 2: Classical Ensemble Methods (2000s-2010s)
===========================================================

Historical Context:
Year: 2001-2010 (Random Forest 2001, Gradient Boosting 2001-2008)
Innovation: Ensemble methods for non-linear time series patterns
Previous Limitation: Traditional methods assumed linear relationships and limited feature handling
Impact: Enabled complex feature engineering and non-linear pattern recognition in forecasting

This implementation demonstrates ensemble methods (Random Forest, Gradient Boosting)
applied to Bitcoin price prediction with extensive feature engineering including
technical indicators, market sentiment, and regime detection.
"""

# Historical Context & Innovation
YEAR = "2001-2010"
INNOVATION = "Ensemble methods with non-linear feature relationships and robust prediction"
PREVIOUS_LIMITATION = "Traditional methods limited to linear patterns and simple features"
IMPACT = "Enabled complex feature engineering and non-linear time series modeling"

# Imports and Dependencies
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import talib as ta
import warnings
warnings.filterwarnings('ignore')

class BitcoinEnsembleForecaster:
    """
    Ensemble methods for Bitcoin price prediction with extensive feature engineering
    
    Features:
    1. Technical indicator engineering
    2. Lag features and rolling statistics
    3. Market regime classification
    4. Random Forest and Gradient Boosting ensembles
    5. Feature importance analysis
    """
    
    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.data = None
        self.feature_importance = {}
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin OHLCV data with extended market information
        
        Returns:
            train_data, val_data, test_data: Feature engineered datasets
        """
        print("Loading Bitcoin Data for Ensemble Analysis...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_data = btc_data.dropna()
        
        # Store raw data
        self.data = btc_data.copy()
        
        print(f"Data loaded: {len(btc_data)} total observations")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        print(f"OHLCV columns: {list(btc_data.columns)}")
        
        return btc_data
    
    def engineer_technical_indicators(self, data):
        """
        Comprehensive technical indicator feature engineering
        
        Args:
            data: Bitcoin OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        print("\nEngineering Technical Indicators...")
        
        features = data.copy()
        
        # Price-based indicators
        features['returns'] = features['Close'].pct_change()
        features['log_returns'] = np.log(features['Close'] / features['Close'].shift(1))
        features['price_change'] = features['Close'] - features['Open']
        features['high_low_ratio'] = features['High'] / features['Low']
        features['volume_price_trend'] = features['Volume'] * features['returns']
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            features[f'ma_{window}'] = features['Close'].rolling(window=window).mean()
            features[f'price_ma_{window}_ratio'] = features['Close'] / features[f'ma_{window}']
            features[f'volume_ma_{window}'] = features['Volume'].rolling(window=window).mean()
        
        # Exponential moving averages
        for span in [12, 26, 50]:
            features[f'ema_{span}'] = features['Close'].ewm(span=span).mean()
            features[f'price_ema_{span}_ratio'] = features['Close'] / features[f'ema_{span}']
        
        # Volatility indicators
        for window in [10, 20, 30]:
            features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
            features[f'volatility_{window}_norm'] = features[f'volatility_{window}'] / features['returns'].std()
        
        # Technical indicators using TA-Lib (with error handling)
        try:
            # RSI
            features['rsi_14'] = ta.RSI(features['Close'].values, timeperiod=14)
            features['rsi_30'] = ta.RSI(features['Close'].values, timeperiod=30)
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(features['Close'].values)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(features['Close'].values)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (features['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic oscillator
            slowk, slowd = ta.STOCH(features['High'].values, features['Low'].values, features['Close'].values)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            
            # Average True Range
            features['atr_14'] = ta.ATR(features['High'].values, features['Low'].values, features['Close'].values)
            
            # Commodity Channel Index
            features['cci_14'] = ta.CCI(features['High'].values, features['Low'].values, features['Close'].values)
            
            # Williams %R
            features['williams_r'] = ta.WILLR(features['High'].values, features['Low'].values, features['Close'].values)
            
            print("Technical indicators calculated successfully")
            
        except Exception as e:
            print(f"WARNING: Some technical indicators failed: {str(e)}")
        
        # Lag features
        for lag in [1, 2, 3, 5, 7, 14]:
            features[f'close_lag_{lag}'] = features['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = features['Volume'].shift(lag)
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            features[f'close_rolling_mean_{window}'] = features['Close'].rolling(window=window).mean()
            features[f'close_rolling_std_{window}'] = features['Close'].rolling(window=window).std()
            features[f'close_rolling_min_{window}'] = features['Close'].rolling(window=window).min()
            features[f'close_rolling_max_{window}'] = features['Close'].rolling(window=window).max()
            features[f'volume_rolling_mean_{window}'] = features['Volume'].rolling(window=window).mean()
        
        # Market microstructure indicators
        features['spread'] = features['High'] - features['Low']
        features['body'] = abs(features['Close'] - features['Open'])
        features['upper_shadow'] = features['High'] - np.maximum(features['Close'], features['Open'])
        features['lower_shadow'] = np.minimum(features['Close'], features['Open']) - features['Low']
        features['doji'] = (features['body'] / features['spread']).fillna(0)
        
        # Time-based features
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter
        features['day_of_month'] = features.index.day
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        print(f"Total features engineered: {len(features.columns)}")
        return features
    
    def detect_market_regimes(self, data, lookback=30):
        """
        Market regime classification using volatility and trend analysis
        
        Args:
            data: Feature engineered data
            lookback: Lookback period for regime detection
        """
        print("\nDetecting Market Regimes...")
        
        regimes = data.copy()
        
        # Volatility regimes
        vol_rolling = regimes['returns'].rolling(window=lookback).std()
        vol_quantiles = vol_rolling.quantile([0.33, 0.67])
        
        regimes['volatility_regime'] = 1  # Medium volatility
        regimes.loc[vol_rolling <= vol_quantiles.iloc[0], 'volatility_regime'] = 0  # Low volatility
        regimes.loc[vol_rolling >= vol_quantiles.iloc[1], 'volatility_regime'] = 2  # High volatility
        
        # Trend regimes
        ma_short = regimes['Close'].rolling(window=10).mean()
        ma_long = regimes['Close'].rolling(window=50).mean()
        
        regimes['trend_regime'] = 1  # Sideways
        regimes.loc[ma_short > ma_long, 'trend_regime'] = 2  # Uptrend
        regimes.loc[ma_short < ma_long, 'trend_regime'] = 0  # Downtrend
        
        # Market phase (combining trend and volatility)
        regimes['market_phase'] = regimes['trend_regime'] * 3 + regimes['volatility_regime']
        
        # Momentum regimes
        momentum = regimes['Close'] / regimes['Close'].shift(lookback) - 1
        momentum_quantiles = momentum.quantile([0.33, 0.67])
        
        regimes['momentum_regime'] = 1  # Medium momentum
        regimes.loc[momentum <= momentum_quantiles.iloc[0], 'momentum_regime'] = 0  # Low momentum
        regimes.loc[momentum >= momentum_quantiles.iloc[1], 'momentum_regime'] = 2  # High momentum
        
        print("Market regimes detected:")
        print(f"Volatility regimes: {regimes['volatility_regime'].value_counts().to_dict()}")
        print(f"Trend regimes: {regimes['trend_regime'].value_counts().to_dict()}")
        print(f"Momentum regimes: {regimes['momentum_regime'].value_counts().to_dict()}")
        
        return regimes
    
    def prepare_ml_features(self, data, target_horizon=1):
        """
        Prepare features and targets for machine learning
        
        Args:
            data: Feature engineered data
            target_horizon: Days ahead to predict
            
        Returns:
            X, y: Features and targets
        """
        print(f"\nPreparing ML Features (target horizon: {target_horizon} days)...")
        
        # Create target variable (future returns)
        data['target'] = data['Close'].shift(-target_horizon)
        
        # Remove non-numeric columns and handle missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target and future-looking variables
        feature_cols = [col for col in numeric_cols if col not in ['target', 'Close', 'Open', 'High', 'Low', 'Volume']]
        
        # Create feature matrix
        X = data[feature_cols].copy()
        y = data['target'].copy()
        
        # Remove rows with missing values
        valid_idx = (~X.isnull().any(axis=1)) & (~y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"Target variable: Bitcoin price {target_horizon} days ahead")
        
        return X, y
    
    def fit_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit Random Forest model with hyperparameter optimization
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data for evaluation
        """
        print("\nFitting Random Forest Model...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        self.models['RandomForest'] = best_rf
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['RandomForest'] = feature_importance
        
        print("Random Forest fitted successfully!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        # Validation performance
        if X_val is not None and y_val is not None:
            val_pred = best_rf.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            print(f"Validation Performance:")
            print(f"  RMSE: ${val_rmse:.2f}")
            print(f"  MAE: ${val_mae:.2f}")
            print(f"  R²: {val_r2:.4f}")
        
        return best_rf
    
    def fit_gradient_boosting(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit Gradient Boosting model with hyperparameter optimization
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data for evaluation
        """
        print("\nFitting Gradient Boosting Model...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(
            gb, param_grid, cv=tscv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_gb = grid_search.best_estimator_
        self.models['GradientBoosting'] = best_gb
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best_gb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['GradientBoosting'] = feature_importance
        
        print("Gradient Boosting fitted successfully!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        # Validation performance
        if X_val is not None and y_val is not None:
            val_pred = best_gb.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            print(f"Validation Performance:")
            print(f"  RMSE: ${val_rmse:.2f}")
            print(f"  MAE: ${val_mae:.2f}")
            print(f"  R²: {val_r2:.4f}")
        
        return best_gb
    
    def predict_bitcoin_price(self, model_name, X_test, steps=30):
        """
        Generate Bitcoin price predictions using ensemble models
        
        Args:
            model_name: Name of model to use
            X_test: Test features
            steps: Number of steps ahead to predict
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            print(f"ERROR: Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        print(f"\nBitcoin Price Forecasting - {model_name}")
        print("=" * 50)
        
        # Generate predictions
        predictions = model.predict(X_test)
        
        print(f"Generated predictions for {len(predictions)} observations")
        print(f"Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        
        return predictions
    
    def evaluate_bitcoin_predictions(self, actual_prices, predictions, model_name):
        """
        Comprehensive evaluation of ensemble model predictions
        
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
            # Simple trading strategy based on predictions
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
        
        print(f"\nBitcoin Prediction Evaluation - {model_name}")
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
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance across models
        """
        print("\nAnalyzing Feature Importance...")
        
        if len(self.feature_importance) == 0:
            print("No feature importance data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, (model_name, importance_df) in enumerate(self.feature_importance.items()):
            if i < len(axes):
                top_features = importance_df.head(15)
                
                axes[i].barh(range(len(top_features)), top_features['importance'])
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(top_features['feature'])
                axes[i].set_title(f"Top 15 Features - {model_name}")
                axes[i].set_xlabel("Feature Importance")
                axes[i].grid(True, alpha=0.3)
        
        # Feature importance comparison
        if len(self.feature_importance) > 1:
            # Common top features
            all_features = set()
            for importance_df in self.feature_importance.values():
                all_features.update(importance_df['feature'].head(20))
            
            comparison_data = {}
            for model_name, importance_df in self.feature_importance.items():
                feature_dict = dict(zip(importance_df['feature'], importance_df['importance']))
                comparison_data[model_name] = [feature_dict.get(feat, 0) for feat in sorted(all_features)]
            
            comparison_df = pd.DataFrame(comparison_data, index=sorted(all_features))
            
            if len(axes) > len(self.feature_importance):
                ax_idx = len(self.feature_importance)
                comparison_df.head(15).plot(kind='bar', ax=axes[ax_idx])
                axes[ax_idx].set_title("Feature Importance Comparison")
                axes[ax_idx].set_xlabel("Features")
                axes[ax_idx].set_ylabel("Importance")
                axes[ax_idx].legend()
                axes[ax_idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle("Ensemble Model Feature Importance Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print top features for each model
        for model_name, importance_df in self.feature_importance.items():
            print(f"\nTop 10 Features - {model_name}:")
            for i, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']:30} {row['importance']:.4f}")
    
    def visualize_results(self, actual_data, predictions_dict, title="Bitcoin Ensemble Forecast"):
        """
        Comprehensive visualization of ensemble model results
        
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
                           label=f'{model_name} Prediction', 
                           color=colors[i % len(colors)], linewidth=2, alpha=0.8)
        
        axes[0].set_title(f"{title}\nEnsemble Methods for Bitcoin Price Prediction")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Bitcoin Price (USD)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # 2. Feature importance comparison
        if len(self.feature_importance) > 0:
            importance_data = {}
            for model_name, importance_df in self.feature_importance.items():
                top_features = importance_df.head(10)
                importance_data[model_name] = dict(zip(top_features['feature'], top_features['importance']))
            
            if importance_data:
                # Get union of all top features
                all_features = set()
                for features in importance_data.values():
                    all_features.update(features.keys())
                
                # Create comparison matrix
                comparison_matrix = []
                feature_names = sorted(list(all_features))[:15]  # Top 15 features
                
                for feature in feature_names:
                    row = [importance_data.get(model, {}).get(feature, 0) for model in importance_data.keys()]
                    comparison_matrix.append(row)
                
                im = axes[1].imshow(comparison_matrix, aspect='auto', cmap='viridis')
                axes[1].set_yticks(range(len(feature_names)))
                axes[1].set_yticklabels(feature_names)
                axes[1].set_xticks(range(len(importance_data)))
                axes[1].set_xticklabels(importance_data.keys())
                axes[1].set_title("Feature Importance Heatmap")
                plt.colorbar(im, ax=axes[1])
        
        # 3. Innovation summary
        innovation_text = f"""
        Ensemble Methods Innovation Impact ({YEAR})
        
        Machine Learning Foundation:
        • Random Forest (Breiman 2001): Bootstrap aggregating with random features
        • Gradient Boosting (Friedman 2001): Sequential weak learner combination
        • Ensemble diversity through bagging and boosting
        
        Key Innovations:
        • Non-linear feature interactions automatically captured
        • Robust to outliers and noise in financial data
        • Feature importance ranking for interpretability
        • Overfitting reduction through ensemble averaging
        
        Bitcoin Application Strengths:
        • Handles complex technical indicator relationships
        • Robust to market regime changes
        • Automatic feature selection and interaction detection
        • Ensemble uncertainty quantification
        
        Limitations for Crypto Markets:
        • Still assumes feature stationarity
        • Limited temporal sequence modeling
        • Requires extensive feature engineering
        • No direct handling of time dependencies
        
        Feature Engineering Highlights:
        • {len(self.feature_names)} total engineered features
        • Technical indicators, market regimes, lag features
        • Time-based and microstructure features
        """
        
        axes[2].text(0.05, 0.95, innovation_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Complete Bitcoin ensemble prediction pipeline
    Demonstrates classical machine learning ensemble methods on cryptocurrency data
    """
    print("ERA 2: Classical Ensemble Methods for Bitcoin Prediction")
    print("=" * 60)
    print(f"Historical Context: {YEAR}")
    print(f"Innovation: {INNOVATION}")
    print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"Impact: {IMPACT}")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = BitcoinEnsembleForecaster()
    
    # Step 1: Load and analyze data
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("="*60)
    
    raw_data = forecaster.load_bitcoin_data()
    
    # Step 2: Feature engineering
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    # Technical indicators
    engineered_data = forecaster.engineer_technical_indicators(raw_data)
    
    # Market regime detection
    regime_data = forecaster.detect_market_regimes(engineered_data)
    
    # Prepare ML features
    X, y = forecaster.prepare_ml_features(regime_data, target_horizon=1)
    
    # Step 3: Train-validation split (time series aware)
    print("\n" + "="*60)
    print("STEP 3: TIME SERIES TRAIN-VALIDATION SPLIT")
    print("="*60)
    
    # Use temporal split to avoid look-ahead bias
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Step 4: Model training
    print("\n" + "="*60)
    print("STEP 4: ENSEMBLE MODEL TRAINING")
    print("="*60)
    
    # Fit Random Forest
    rf_model = forecaster.fit_random_forest(X_train, y_train, X_val, y_val)
    
    # Fit Gradient Boosting
    gb_model = forecaster.fit_gradient_boosting(X_train, y_train, X_val, y_val)
    
    # Step 5: Predictions and evaluation
    print("\n" + "="*60)
    print("STEP 5: BITCOIN PRICE PREDICTION")
    print("="*60)
    
    # Generate predictions
    predictions_dict = {}
    for model_name in ['RandomForest', 'GradientBoosting']:
        predictions = forecaster.predict_bitcoin_price(model_name, X_val)
        predictions_dict[model_name] = predictions
        
        # Evaluate predictions
        if predictions is not None:
            metrics = forecaster.evaluate_bitcoin_predictions(y_val.values, predictions, model_name)
    
    # Step 6: Feature importance analysis
    print("\n" + "="*60)
    print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    forecaster.analyze_feature_importance()
    
    # Step 7: Results visualization
    print("\n" + "="*60)
    print("STEP 7: RESULTS VISUALIZATION")
    print("="*60)
    
    # Combine historical data for visualization
    historical_prices = regime_data['Close']
    forecaster.visualize_results(historical_prices, predictions_dict, 
                               "Bitcoin Ensemble Forecast")
    
    # Final summary
    print("\n" + "="*60)
    print("ERA 2 SUMMARY: CLASSICAL ENSEMBLE METHODS")
    print("="*60)
    print(f"""
    Ensemble Analysis Complete:
    
    Machine Learning Foundation:
    • Models: Random Forest + Gradient Boosting
    • Features: {len(forecaster.feature_names)} engineered features
    • Market regimes incorporated
    • Time-series aware validation
    
    Bitcoin Prediction Performance:
    • Non-linear pattern capture
    • Feature importance analysis
    • Robust ensemble predictions
    • Market regime adaptability
    
    Educational Value:
    • Demonstrates ensemble learning principles
    • Shows comprehensive feature engineering
    • Bridges traditional and modern ML approaches
    • Establishes foundation for deep learning
    
    Ready for ERA 3: Deep Learning Revolution!
    """)


if __name__ == "__main__":
    main()