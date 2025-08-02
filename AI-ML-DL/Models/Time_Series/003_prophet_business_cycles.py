"""
Time Series ERA 1: Prophet Business Cycles (2017-present)
=========================================================

Historical Context:
Year: 2017 (Facebook's Prophet release)
Innovation: Decomposable time series model with business-focused interpretability
Previous Limitation: Traditional methods couldn't handle holidays, changepoints, and complex seasonality
Impact: Made sophisticated time series forecasting accessible to business analysts

This implementation demonstrates Facebook Prophet applied to Bitcoin price prediction,
showcasing how modern business-oriented forecasting handles cryptocurrency market
patterns including regime changes, volatility cycles, and external events.
"""

# Historical Context & Innovation
YEAR = "2017"
INNOVATION = "Decomposable time series with automatic changepoint detection and holiday effects"
PREVIOUS_LIMITATION = "Classical methods struggled with complex seasonality and structural breaks"
IMPACT = "Democratized advanced time series forecasting for business applications"

# Imports and Dependencies
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet with fallback handling
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
    print("Prophet library available")
except ImportError:
    print("WARNING: Prophet library not available. Install with: pip install prophet")
    PROPHET_AVAILABLE = False


class BitcoinProphetForecaster:
    """
    Facebook Prophet implementation for Bitcoin price prediction
    
    Features Prophet-specific capabilities:
    1. Automatic changepoint detection
    2. Holiday and event modeling
    3. Multiple seasonality patterns
    4. Uncertainty quantification
    5. Business-friendly interpretability
    """
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.data = None
        self.prophet_data = None
        self.predictions = None
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin OHLCV data and prepare for Prophet modeling
        
        Returns:
            train_data, val_data, test_data: Temporal splits
        """
        print("Loading Bitcoin Data for Prophet Analysis...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_data = btc_data.dropna()
        
        # Use closing prices
        prices = btc_data['Close'].copy()
        
        # Create temporal splits (no data leakage)
        train_size = int(0.7 * len(prices))
        val_size = int(0.15 * len(prices))
        
        train_data = prices[:train_size]
        val_data = prices[train_size:train_size + val_size]
        test_data = prices[train_size + val_size:]
        
        # Prepare Prophet format (ds, y columns)
        prophet_train = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
        print(f"Data loaded: {len(prices)} total observations")
        print(f"Training: {len(train_data)} observations")
        print(f"Validation: {len(val_data)} observations")
        print(f"🧪 Test: {len(test_data)} observations")
        print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        
        self.data = train_data
        self.prophet_data = prophet_train
        return train_data, val_data, test_data, prophet_train
    
    def create_bitcoin_holidays(self):
        """
        Create Bitcoin-specific holiday/event dataset
        
        Returns:
            DataFrame with Bitcoin market events
        """
        print("\nCreating Bitcoin Market Events Calendar...")
        
        # Major Bitcoin events that could affect price
        bitcoin_events = [
            # Early adoption events
            {'holiday': 'Pizza_Day', 'ds': '2010-05-22', 'lower_window': -1, 'upper_window': 1},
            
            # First major exchange launches
            {'holiday': 'MtGox_Launch', 'ds': '2010-07-17', 'lower_window': -2, 'upper_window': 2},
            
            # First major crash
            {'holiday': 'First_Bubble_Burst', 'ds': '2011-06-08', 'lower_window': -3, 'upper_window': 3},
            
            # Silk Road shutdown
            {'holiday': 'Silk_Road_Shutdown', 'ds': '2013-10-02', 'lower_window': -2, 'upper_window': 2},
            
            # MtGox collapse
            {'holiday': 'MtGox_Collapse', 'ds': '2014-02-24', 'lower_window': -5, 'upper_window': 5},
            
            # First ETF rejection
            {'holiday': 'ETF_Rejection_2017', 'ds': '2017-03-10', 'lower_window': -2, 'upper_window': 2},
            
            # Bitcoin futures launch
            {'holiday': 'Futures_Launch', 'ds': '2017-12-10', 'lower_window': -3, 'upper_window': 3},
            
            # 2017 peak
            {'holiday': 'ATH_2017', 'ds': '2017-12-17', 'lower_window': -5, 'upper_window': 5},
            
            # COVID crash
            {'holiday': 'COVID_Crash', 'ds': '2020-03-12', 'lower_window': -3, 'upper_window': 3},
            
            # Tesla announcement
            {'holiday': 'Tesla_Announcement', 'ds': '2021-02-08', 'lower_window': -2, 'upper_window': 2},
            
            # 2021 peak
            {'holiday': 'ATH_2021', 'ds': '2021-11-10', 'lower_window': -5, 'upper_window': 5},
            
            # FTX collapse
            {'holiday': 'FTX_Collapse', 'ds': '2022-11-11', 'lower_window': -5, 'upper_window': 5},
        ]
        
        holidays_df = pd.DataFrame(bitcoin_events)
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        
        print(f"Created {len(holidays_df)} Bitcoin market events")
        print("Event types: Market crashes, regulatory changes, institutional adoption")
        
        return holidays_df
    
    def analyze_changepoints(self, series):
        """
        Analyze Bitcoin price changepoints and trend changes
        
        Args:
            series: Bitcoin price time series
        """
        print("\nBitcoin Changepoint Analysis")
        print("=" * 40)
        
        # Calculate price changes and volatility
        price_changes = series.pct_change().abs()
        volatility = price_changes.rolling(window=30).std()
        
        # Identify periods of high volatility (potential changepoints)
        volatility_threshold = volatility.quantile(0.9)
        high_vol_periods = volatility[volatility > volatility_threshold]
        
        print(f"Volatility Analysis:")
        print(f"  Average daily change: {price_changes.mean()*100:.2f}%")
        print(f"  Maximum daily change: {price_changes.max()*100:.2f}%")
        print(f"  High volatility periods: {len(high_vol_periods)}")
        print(f"  Volatility threshold (90th percentile): {volatility_threshold*100:.2f}%")
        
        # Plot volatility over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Price chart
        ax1.plot(series.index, series, alpha=0.8, color='blue')
        ax1.set_title("Bitcoin Price with High Volatility Periods", fontweight='bold')
        ax1.set_ylabel("Price (USD)")
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Highlight high volatility periods
        for date in high_vol_periods.index:
            ax1.axvline(x=date, color='red', alpha=0.3, linewidth=0.5)
        
        # Volatility chart
        ax2.plot(volatility.index, volatility*100, alpha=0.8, color='orange')
        ax2.axhline(y=volatility_threshold*100, color='red', linestyle='--', 
                   label=f'90th percentile ({volatility_threshold*100:.1f}%)')
        ax2.set_title("Bitcoin Daily Volatility (30-day rolling)", fontweight='bold')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volatility (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle("Bitcoin Changepoint Analysis\nIdentifying Trend Changes and Market Regimes", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return high_vol_periods
    
    def fit_prophet_model(self, prophet_data, holidays=None):
        """
        Fit Prophet model with Bitcoin-specific configurations
        
        Args:
            prophet_data: DataFrame with ds, y columns
            holidays: Holiday/event DataFrame
        """
        if not PROPHET_AVAILABLE:
            print("ERROR: Prophet library not available")
            return None
        
        print("\nFitting Facebook Prophet Model...")
        
        # Configure Prophet with Bitcoin-specific parameters
        self.model = Prophet(
            # Growth and trend
            growth='linear',  # Bitcoin shows linear growth in log space
            changepoint_prior_scale=0.05,  # Allow for trend changes
            changepoint_range=0.8,  # Consider changepoints in first 80% of data
            
            # Seasonality
            yearly_seasonality=True,  # Annual patterns
            weekly_seasonality=True,  # Weekly trading patterns
            daily_seasonality=False,  # Not relevant for daily data
            
            # Holidays and events
            holidays=holidays if holidays is not None else None,
            holidays_prior_scale=10.0,  # Strong holiday effects
            
            # Uncertainty
            mcmc_samples=0,  # Use MAP estimation (faster)
            interval_width=0.95,  # 95% confidence intervals
            
            # Seasonality mode
            seasonality_mode='multiplicative',  # Bitcoin seasonality scales with price
        )
        
        # Add custom seasonalities
        # Monthly seasonality (some evidence of monthly patterns in crypto)
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Quarterly seasonality (business cycles)
        self.model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=3)
        
                    print("Prophet Configuration:")
        print(f"  Growth model: Linear")
        print(f"  Changepoint prior scale: 0.05 (moderate flexibility)")
        print(f"  Seasonality mode: Multiplicative")
        print(f"  Custom seasonalities: Monthly, Quarterly")
                    print(f"  Holidays: {'Included' if holidays is not None else 'None'}")
        
        # Fit the model
        try:
            self.fitted_model = self.model.fit(prophet_data)
            print("Prophet model fitted successfully!")
            
            # Print changepoint information
            changepoints = self.fitted_model.changepoints
            print(f"Detected changepoints: {len(changepoints)}")
            if len(changepoints) > 0:
                print(f"  First changepoint: {changepoints[0].strftime('%Y-%m-%d')}")
                print(f"  Last changepoint: {changepoints[-1].strftime('%Y-%m-%d')}")
            
            return self.fitted_model
            
        except Exception as e:
            print(f"ERROR: Prophet fitting failed: {str(e)}")
            return None
    
    def predict_bitcoin_price(self, periods=30, freq='D'):
        """
        Generate Bitcoin price forecasts using Prophet
        
        Args:
            periods: Number of periods ahead to forecast
            freq: Frequency ('D' for daily)
            
        Returns:
            Prophet predictions DataFrame
        """
        if self.fitted_model is None:
            print("ERROR: No fitted Prophet model available")
            return None
        
        print(f"\nBitcoin Price Forecasting with Prophet ({periods} days ahead)")
        print("=" * 65)
        
        try:
            # Create future dataframe
            future = self.fitted_model.make_future_dataframe(periods=periods, freq=freq)
            
            # Generate predictions
            predictions = self.fitted_model.predict(future)
            
            # Extract forecast portion
            forecast_start_idx = len(self.prophet_data)
            forecast_predictions = predictions.iloc[forecast_start_idx:].copy()
            
            print(f"Generated {periods}-day Bitcoin price forecast")
            print(f"Forecast range: ${forecast_predictions['yhat'].min():.2f} - ${forecast_predictions['yhat'].max():.2f}")
            print(f"Uncertainty range: ${forecast_predictions['yhat_lower'].min():.2f} - ${forecast_predictions['yhat_upper'].max():.2f}")
            
            # Store predictions for later use
            self.predictions = predictions
            
            return forecast_predictions
            
        except Exception as e:
            print(f"ERROR: Forecasting failed: {str(e)}")
            return None
    
    def cross_validate_prophet(self, initial='730 days', period='180 days', horizon='30 days'):
        """
        Perform time series cross-validation on Prophet model
        
        Args:
            initial: Initial training period
            period: Gap between cutoff dates
            horizon: Forecast horizon
        """
        if not PROPHET_AVAILABLE or self.fitted_model is None:
            print("ERROR: Prophet model not available for cross-validation")
            return None
        
        print(f"\nProphet Cross-Validation")
        print("=" * 40)
        print(f"Configuration:")
        print(f"  Initial period: {initial}")
        print(f"  Period gap: {period}")
        print(f"  Forecast horizon: {horizon}")
        
        try:
            # Perform cross-validation
            cv_results = cross_validation(
                self.fitted_model, 
                initial=initial, 
                period=period, 
                horizon=horizon
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            print(f"\nCross-Validation Results:")
            print(f"  Number of folds: {len(cv_results['cutoff'].unique())}")
            print(f"  Average MAE: ${metrics['mae'].mean():.2f}")
            print(f"  Average RMSE: ${metrics['rmse'].mean():.2f}")
            print(f"  Average MAPE: {metrics['mape'].mean()*100:.2f}%")
            
            return cv_results, metrics
            
        except Exception as e:
            print(f"ERROR: Cross-validation failed: {str(e)}")
            return None, None
    
    def evaluate_bitcoin_predictions(self, actual_prices, forecast_predictions):
        """
        Comprehensive evaluation of Bitcoin Prophet predictions
        
        Args:
            actual_prices: True Bitcoin prices
            forecast_predictions: Prophet forecast DataFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        if forecast_predictions is None or len(actual_prices) == 0:
            print("WARNING: Insufficient data for evaluation")
            return {}
        
        # Convert forecast dates to align with actual data
        forecast_df = forecast_predictions.set_index('ds')
        
        # Align data
        common_index = actual_prices.index.intersection(forecast_df.index)
        if len(common_index) == 0:
            print("WARNING: No overlapping dates for evaluation")
            return {}
        
        actual = actual_prices.loc[common_index]
        pred = forecast_df.loc[common_index, 'yhat']
        pred_lower = forecast_df.loc[common_index, 'yhat_lower']
        pred_upper = forecast_df.loc[common_index, 'yhat_upper']
        
        # Forecasting accuracy metrics
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        # Coverage of prediction intervals
        coverage = ((actual >= pred_lower) & (actual <= pred_upper)).mean() * 100
        
        # Directional accuracy
        if len(actual) > 1:
            actual_direction = (actual.shift(-1) > actual).astype(int)[:-1]
            pred_direction = (pred.shift(-1) > pred).astype(int)[:-1]
            directional_accuracy = (actual_direction == pred_direction).mean() * 100
        else:
            directional_accuracy = 0
        
        # Financial metrics
        returns_actual = actual.pct_change().dropna()
        returns_pred = pred.pct_change().dropna()
        
        # Trading strategy based on Prophet predictions
        if len(returns_actual) > 1:
            # Use prediction confidence for position sizing
            pred_strength = np.abs(pred - (pred_lower + pred_upper) / 2) / ((pred_upper - pred_lower) / 2)
            signals = np.where(pred.shift(1) < pred, 1, -1)  # Long if price expected to rise
            
            # Weight signals by prediction confidence
            weighted_signals = signals[1:len(returns_actual)+1] * pred_strength[1:len(returns_actual)+1]
            strategy_returns = weighted_signals * returns_actual
            
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
            cumulative_returns = strategy_returns.cumsum()
            max_drawdown = (cumulative_returns - cumulative_returns.expanding().max()).min()
            total_return = strategy_returns.sum()
        else:
            sharpe_ratio = max_drawdown = total_return = 0
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'coverage': coverage,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'total_return': total_return * 100
        }
        
        print(f"\nBitcoin Prophet Prediction Evaluation")
        print("=" * 50)
        print(f"Forecasting Accuracy:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Interval Coverage: {coverage:.1f}%")
        print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
        
        print(f"\nTrading Strategy Performance:")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Total Return: {total_return:.2f}%")
        
        return metrics
    
    def visualize_prophet_components(self):
        """
        Visualize Prophet model components
        """
        if not PROPHET_AVAILABLE or self.fitted_model is None or self.predictions is None:
            print("ERROR: Prophet model or predictions not available")
            return
        
        print("\nVisualizing Prophet Components...")
        
        # Plot Prophet components
        fig = self.fitted_model.plot_components(self.predictions, figsize=(15, 12))
        plt.suptitle("Bitcoin Prophet Model Components\nTrend, Seasonality, and Holiday Effects", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def visualize_results(self, actual_data, forecast_predictions, title="Bitcoin Prophet Forecast"):
        """
        Comprehensive visualization of Prophet results
        
        Args:
            actual_data: Historical Bitcoin prices
            forecast_predictions: Prophet forecast DataFrame
            title: Plot title
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Price forecast plot
        axes[0].plot(actual_data.index, actual_data.values, 
                    label='Historical Bitcoin Price', color='blue', alpha=0.7)
        
        if forecast_predictions is not None and not forecast_predictions.empty:
            axes[0].plot(forecast_predictions['ds'], forecast_predictions['yhat'], 
                        label='Prophet Forecast', color='red', linewidth=2)
            axes[0].fill_between(forecast_predictions['ds'], 
                               forecast_predictions['yhat_lower'], 
                               forecast_predictions['yhat_upper'],
                               alpha=0.3, color='red', label='95% Confidence Interval')
        
        axes[0].set_title(f"{title}\nFacebook Prophet with Changepoints and Seasonality")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Bitcoin Price (USD)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # 2. Changepoints visualization
        if self.fitted_model is not None:
            changepoints = self.fitted_model.changepoints
            for cp in changepoints:
                axes[0].axvline(x=cp, color='green', alpha=0.5, linestyle='--', linewidth=1)
            
            # Plot trend component separately
            if self.predictions is not None:
                trend_data = self.predictions[self.predictions['ds'] <= actual_data.index[-1]]
                axes[1].plot(trend_data['ds'], trend_data['trend'], 
                           color='green', linewidth=2, label='Trend Component')
                
                # Highlight changepoints
                for cp in changepoints:
                    axes[1].axvline(x=cp, color='red', alpha=0.7, linestyle='--', linewidth=1)
                
                axes[1].set_title("Prophet Trend Component with Changepoints")
                axes[1].set_xlabel("Date")
                axes[1].set_ylabel("Trend")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        
        # 3. Innovation comparison
        innovation_text = f"""
        Prophet Innovation Impact ({YEAR})
        
        Business-Oriented Foundation:
        • Facebook's internal forecasting needs
        • Designed for business analysts, not just data scientists
        • Handles missing data and outliers gracefully
        
        Key Innovations:
        • Automatic changepoint detection
        • Holiday and event modeling
        • Multiple seasonality patterns
        • Intuitive parameter interpretation
        • Uncertainty quantification
        
        Bitcoin Application Strengths:
        • Captures market regime changes automatically
        • Handles Bitcoin's complex seasonality
        • Models major market events as holidays
        • Provides interpretable trend decomposition
        • Robust to missing data and outliers
        
        Limitations for Crypto Markets:
        • Assumes additive/multiplicative decomposition
        • May not capture non-linear relationships
        • Limited multivariate capabilities
        • Struggles with very high frequency patterns
        
        Bitcoin-Specific Features Used:
        • Market events as holidays
        • Monthly/quarterly seasonality
        • Multiplicative seasonality mode
        • Flexible changepoint detection
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
    Complete Bitcoin Prophet prediction pipeline
    Demonstrates business-oriented forecasting on cryptocurrency data
    """
    print("ERA 1: Facebook Prophet for Bitcoin Prediction")
    print("=" * 60)
    print(f"Historical Context: {YEAR}")
    print(f"Innovation: {INNOVATION}")
    print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"Impact: {IMPACT}")
    print("=" * 60)
    
    if not PROPHET_AVAILABLE:
        print("ERROR: Prophet library not available. Please install: pip install prophet")
        print("🔗 Alternative: Use conda install -c conda-forge prophet")
        return
    
    # Initialize forecaster
    forecaster = BitcoinProphetForecaster()
    
    # Load Bitcoin data
    train_data, val_data, test_data, prophet_data = forecaster.load_bitcoin_data()
    
    # Step 1: Create Bitcoin market events
    print("\n" + "="*60)
    print("STEP 1: BITCOIN MARKET EVENTS & HOLIDAYS")
    print("="*60)
    
    bitcoin_holidays = forecaster.create_bitcoin_holidays()
    
    # Step 2: Changepoint analysis
    print("\n" + "="*60)
    print("STEP 2: CHANGEPOINT ANALYSIS")
    print("="*60)
    
    volatility_periods = forecaster.analyze_changepoints(train_data)
    
    # Step 3: Fit Prophet model
    print("\n" + "="*60)
    print("STEP 3: PROPHET MODEL FITTING")
    print("="*60)
    
    fitted_model = forecaster.fit_prophet_model(prophet_data, holidays=bitcoin_holidays)
    
    if fitted_model is None:
        print("ERROR: Prophet model fitting failed!")
        return
    
    # Step 4: Cross-validation
    print("\n" + "="*60)
    print("STEP 4: TIME SERIES CROSS-VALIDATION")
    print("="*60)
    
    cv_results, cv_metrics = forecaster.cross_validate_prophet()
    
    # Step 5: Forecasting
    print("\n" + "="*60)
    print("STEP 5: BITCOIN PRICE FORECASTING")
    print("="*60)
    
    # Generate predictions
    forecast_predictions = forecaster.predict_bitcoin_price(periods=30)
    
    # Evaluate on validation data
    if len(val_data) > 0 and forecast_predictions is not None:
        val_forecast = forecaster.predict_bitcoin_price(periods=len(val_data))
        metrics = forecaster.evaluate_bitcoin_predictions(val_data, val_forecast)
    
    # Step 6: Component visualization
    print("\n" + "="*60)
    print("STEP 6: PROPHET COMPONENT ANALYSIS")
    print("="*60)
    
    forecaster.visualize_prophet_components()
    
    # Step 7: Results visualization
    print("\n" + "="*60)
    print("STEP 7: RESULTS VISUALIZATION")
    print("="*60)
    
    # Combine historical and forecast data for visualization
    full_data = pd.concat([train_data, val_data, test_data])
    forecaster.visualize_results(full_data, forecast_predictions, 
                               "Bitcoin Prophet Forecast")
    
    # Final summary
    print("\n" + "="*62)
    print("ERA 1 SUMMARY: PROPHET BUSINESS FORECASTING")
    print("="*62)
    print(f"""
    Prophet Analysis Complete:
    
    Business-Oriented Features:
    • Market events modeled as holidays
    • Automatic changepoint detection
    • {len(bitcoin_holidays)} Bitcoin-specific events included
    • Multiple seasonality patterns captured
    
    Bitcoin Prediction Performance:
    • 30-day forecast with uncertainty intervals
    • Cross-validation performed for robustness
    • Trend decomposition provides interpretability
    
    Educational Value:
    • Demonstrates modern business forecasting
    • Shows automated pattern detection
    • Provides uncertainty quantification
    • Bridges traditional and modern approaches
    
    ERA 1 Complete! Ready for ERA 2: Machine Learning Transition!
    """)


if __name__ == "__main__":
    main()