"""
Time Series ERA 1: Exponential Smoothing Trends (1950s-1980s)
============================================================

Historical Context:
Year: 1957-1980 (Holt-Winters development)
Innovation: Exponential smoothing with trend and seasonal components
Previous Limitation: Simple moving averages couldn't handle trends or seasonality
Impact: Practical forecasting method widely adopted in business and industry

This implementation demonstrates exponential smoothing methods applied to Bitcoin
price prediction, showcasing how classical trend and seasonal modeling approaches
perform on modern cryptocurrency data with extreme volatility patterns.
"""

# Historical Context & Innovation
YEAR = "1957-1980"
INNOVATION = "Exponential smoothing with trend and seasonal decomposition"
PREVIOUS_LIMITATION = "Simple averages couldn't capture trends or seasonal patterns"
IMPACT = "Practical business forecasting tool with automatic parameter adaptation"

# Imports and Dependencies
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothingResults
import warnings
warnings.filterwarnings('ignore')

class BitcoinExponentialSmoothingForecaster:
    """
    Comprehensive Exponential Smoothing implementation for Bitcoin prediction
    
    Features multiple smoothing variants:
    1. Simple Exponential Smoothing (SES)
    2. Holt's Linear Trend (Double Exponential)
    3. Holt-Winters Seasonal (Triple Exponential)
    4. Automatic parameter optimization
    """
    
    def __init__(self):
        self.models = {}
        self.fitted_models = {}
        self.data = None
        self.seasonal_periods = 7  # Weekly seasonality for Bitcoin
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin OHLCV data and prepare for exponential smoothing
        
        Returns:
            train_data, val_data, test_data: Temporal splits
        """
        print("📊 Loading Bitcoin Data for Exponential Smoothing...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_data = btc_data.dropna()
        
        # Use closing prices
        prices = btc_data['Close'].copy()
        
        # Handle missing values with forward fill
        prices = prices.ffill()
        
        # Create temporal splits (no data leakage)
        train_size = int(0.7 * len(prices))
        val_size = int(0.15 * len(prices))
        
        train_data = prices[:train_size]
        val_data = prices[train_size:train_size + val_size]
        test_data = prices[train_size + val_size:]
        
        print(f"✅ Data loaded: {len(prices)} total observations")
        print(f"📈 Training: {len(train_data)} observations")
        print(f"📊 Validation: {len(val_data)} observations")
        print(f"🧪 Test: {len(test_data)} observations")
        print(f"💰 Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        
        self.data = train_data
        return train_data, val_data, test_data
    
    def analyze_seasonality(self, series, period=7):
        """
        Analyze seasonal patterns in Bitcoin prices
        
        Args:
            series: Bitcoin price time series
            period: Seasonal period (7 for weekly patterns)
        """
        print(f"\n📅 Bitcoin Seasonality Analysis (Period: {period} days)")
        print("=" * 55)
        
        try:
            # Seasonal decomposition
            decomposition = seasonal_decompose(series, model='multiplicative', 
                                             period=period, extrapolate_trend='freq')
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            # Original series
            axes[0].plot(decomposition.observed, color='blue', alpha=0.8)
            axes[0].set_title("Original Bitcoin Prices", fontweight='bold')
            axes[0].set_ylabel("Price (USD)")
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')
            
            # Trend component
            axes[1].plot(decomposition.trend, color='red', alpha=0.8)
            axes[1].set_title("Trend Component", fontweight='bold')
            axes[1].set_ylabel("Trend")
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal component
            axes[2].plot(decomposition.seasonal, color='green', alpha=0.8)
            axes[2].set_title(f"Seasonal Component ({period}-day cycle)", fontweight='bold')
            axes[2].set_ylabel("Seasonal")
            axes[2].grid(True, alpha=0.3)
            
            # Residual component
            axes[3].plot(decomposition.resid, color='orange', alpha=0.8)
            axes[3].set_title("Residual Component", fontweight='bold')
            axes[3].set_ylabel("Residuals")
            axes[3].set_xlabel("Date")
            axes[3].grid(True, alpha=0.3)
            
            plt.suptitle("Bitcoin Price Seasonal Decomposition\nExponential Smoothing Component Analysis", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Calculate seasonal strength
            seasonal_strength = 1 - (decomposition.resid.var() / decomposition.observed.var())
            trend_strength = 1 - (decomposition.resid.var() / (decomposition.observed - decomposition.seasonal).var())
            
            print(f"📊 Decomposition Results:")
            print(f"  Seasonal Strength: {seasonal_strength:.3f}")
            print(f"  Trend Strength: {trend_strength:.3f}")
            print(f"  Seasonality: {'✅ Detected' if seasonal_strength > 0.1 else '❌ Weak'}")
            
            return decomposition, seasonal_strength > 0.1
            
        except Exception as e:
            print(f"⚠️ Seasonal decomposition failed: {str(e)}")
            return None, False
    
    def fit_simple_exponential_smoothing(self, series):
        """
        Fit Simple Exponential Smoothing (SES) model
        
        Args:
            series: Bitcoin price time series
        """
        print("\n🔧 Fitting Simple Exponential Smoothing (SES)...")
        
        try:
            model = ExponentialSmoothing(series, trend=None, seasonal=None)
            fitted_model = model.fit(optimized=True)
            
            self.models['SES'] = model
            self.fitted_models['SES'] = fitted_model
            
            print(f"✅ SES Model fitted successfully!")
            print(f"   Alpha (level): {fitted_model.params['smoothing_level']:.4f}")
            print(f"   AIC: {fitted_model.aic:.2f}")
            
            return fitted_model
            
        except Exception as e:
            print(f"❌ SES fitting failed: {str(e)}")
            return None
    
    def fit_double_exponential_smoothing(self, series):
        """
        Fit Holt's Linear Trend (Double Exponential Smoothing) model
        
        Args:
            series: Bitcoin price time series
        """
        print("\n🔧 Fitting Double Exponential Smoothing (Holt's Method)...")
        
        try:
            model = ExponentialSmoothing(series, trend='add', seasonal=None)
            fitted_model = model.fit(optimized=True)
            
            self.models['Holt'] = model
            self.fitted_models['Holt'] = fitted_model
            
            print(f"✅ Holt Model fitted successfully!")
            print(f"   Alpha (level): {fitted_model.params['smoothing_level']:.4f}")
            print(f"   Beta (trend): {fitted_model.params['smoothing_trend']:.4f}")
            print(f"   AIC: {fitted_model.aic:.2f}")
            
            return fitted_model
            
        except Exception as e:
            print(f"❌ Holt fitting failed: {str(e)}")
            return None
    
    def fit_triple_exponential_smoothing(self, series, seasonal_periods=7):
        """
        Fit Holt-Winters (Triple Exponential Smoothing) model
        
        Args:
            series: Bitcoin price time series
            seasonal_periods: Length of seasonal cycle
        """
        print(f"\n🔧 Fitting Triple Exponential Smoothing (Holt-Winters, period={seasonal_periods})...")
        
        if len(series) < 2 * seasonal_periods:
            print(f"⚠️ Insufficient data for seasonality (need >{2*seasonal_periods} observations)")
            return None
        
        try:
            # Try both additive and multiplicative seasonality
            models_to_try = [
                ('add', 'Additive'),
                ('mul', 'Multiplicative')
            ]
            
            best_model = None
            best_aic = np.inf
            best_type = None
            
            for seasonal_type, name in models_to_try:
                try:
                    model = ExponentialSmoothing(series, trend='add', seasonal=seasonal_type, 
                                               seasonal_periods=seasonal_periods)
                    fitted_model = model.fit(optimized=True)
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_model = fitted_model
                        best_type = name
                        
                        self.models[f'HW_{seasonal_type}'] = model
                        self.fitted_models[f'HW_{seasonal_type}'] = fitted_model
                        
                except Exception as e:
                    print(f"  ⚠️ {name} seasonality failed: {str(e)}")
                    continue
            
            if best_model is not None:
                print(f"✅ Holt-Winters ({best_type}) fitted successfully!")
                print(f"   Alpha (level): {best_model.params['smoothing_level']:.4f}")
                print(f"   Beta (trend): {best_model.params['smoothing_trend']:.4f}")
                print(f"   Gamma (seasonal): {best_model.params['smoothing_seasonal']:.4f}")
                print(f"   AIC: {best_model.aic:.2f}")
                
                return best_model
            else:
                print("❌ All Holt-Winters variants failed")
                return None
                
        except Exception as e:
            print(f"❌ Holt-Winters fitting failed: {str(e)}")
            return None
    
    def compare_models(self, series):
        """
        Compare all exponential smoothing models using AIC
        
        Args:
            series: Bitcoin price time series
        """
        print("\n🏆 Model Comparison (Exponential Smoothing Variants)")
        print("=" * 60)
        
        model_results = []
        
        for model_name, fitted_model in self.fitted_models.items():
            if fitted_model is not None:
                aic = fitted_model.aic
                bic = fitted_model.bic if hasattr(fitted_model, 'bic') else np.nan
                
                # Calculate in-sample fit
                fitted_values = fitted_model.fittedvalues
                if len(fitted_values) == len(series):
                    mse = mean_squared_error(series, fitted_values)
                    mae = mean_absolute_error(series, fitted_values)
                else:
                    mse = mae = np.nan
                
                model_results.append({
                    'Model': model_name,
                    'AIC': aic,
                    'BIC': bic,
                    'MSE': mse,
                    'MAE': mae
                })
        
        if len(model_results) == 0:
            print("❌ No models successfully fitted")
            return None
        
        # Sort by AIC
        model_results.sort(key=lambda x: x['AIC'])
        
        print("Rank | Model        | AIC      | BIC      | MSE         | MAE")
        print("-" * 65)
        for i, result in enumerate(model_results, 1):
            marker = "👑" if i == 1 else f"{i}."
            print(f"{marker:4} | {result['Model']:12} | {result['AIC']:8.2f} | "
                  f"{result['BIC']:8.2f} | {result['MSE']:11.2e} | {result['MAE']:8.2f}")
        
        best_model_name = model_results[0]['Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        
        return best_model_name
    
    def predict_bitcoin_price(self, model_name, steps=30):
        """
        Generate Bitcoin price forecasts using specified exponential smoothing model
        
        Args:
            model_name: Name of fitted model to use
            steps: Number of steps ahead to forecast
            
        Returns:
            Predictions with confidence intervals
        """
        if model_name not in self.fitted_models:
            print(f"❌ Model {model_name} not found")
            return None
        
        fitted_model = self.fitted_models[model_name]
        if fitted_model is None:
            print(f"❌ Model {model_name} not fitted properly")
            return None
        
        print(f"\n🔮 Bitcoin Price Forecasting - {model_name} ({steps} days ahead)")
        print("=" * 60)
        
        try:
            # Generate forecasts
            forecast_result = fitted_model.forecast(steps=steps)
            
            # Get prediction intervals (approximate)
            # Note: Exponential smoothing confidence intervals are complex
            # Using simplified approach based on residual variance
            residuals = self.data - fitted_model.fittedvalues
            residual_std = residuals.std()
            
            # Create forecast dates
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                         periods=steps, freq='D')
            
            # Approximate confidence intervals (95%)
            z_score = 1.96
            margin = z_score * residual_std * np.sqrt(np.arange(1, steps + 1))
            
            # Organize results
            predictions = pd.DataFrame({
                'forecast': forecast_result,
                'lower_ci': forecast_result - margin,
                'upper_ci': forecast_result + margin
            }, index=forecast_dates)
            
            print(f"✅ Generated {steps}-day Bitcoin price forecast")
            print(f"📊 Model: {model_name}")
            print(f"💰 Forecast range: ${predictions['forecast'].min():.2f} - ${predictions['forecast'].max():.2f}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Forecasting failed: {str(e)}")
            return None
    
    def evaluate_bitcoin_predictions(self, actual_prices, predictions, model_name):
        """
        Comprehensive evaluation of Bitcoin exponential smoothing predictions
        
        Args:
            actual_prices: True Bitcoin prices
            predictions: Model predictions
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary of evaluation metrics
        """
        if predictions is None or len(actual_prices) == 0:
            print("⚠️ Insufficient data for evaluation")
            return {}
        
        # Align data
        common_index = actual_prices.index.intersection(predictions.index)
        if len(common_index) == 0:
            print("⚠️ No overlapping dates for evaluation")
            return {}
        
        actual = actual_prices.loc[common_index]
        pred = predictions.loc[common_index, 'forecast']
        
        # Forecasting accuracy metrics
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        # Directional accuracy
        if len(actual) > 1:
            actual_direction = (actual.shift(-1) > actual).astype(int)[:-1]
            pred_direction = (pred.shift(-1) > pred).astype(int)[:-1]
            directional_accuracy = (actual_direction == pred_direction).mean() * 100
        else:
            directional_accuracy = 0
        
        # Financial metrics (simplified trading strategy)
        returns_actual = actual.pct_change().dropna()
        returns_pred = pred.pct_change().dropna()
        
        # Simple trend-following strategy
        if len(returns_actual) > 1:
            signals = np.where(pred.shift(1) < pred, 1, -1)  # Buy if price expected to rise
            strategy_returns = signals[1:len(returns_actual)+1] * returns_actual
            
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
            cumulative_returns = strategy_returns.cumsum()
            max_drawdown = (cumulative_returns - cumulative_returns.expanding().max()).min()
            total_return = strategy_returns.sum()
        else:
            sharpe_ratio = max_drawdown = total_return = 0
        
        metrics = {
            'model_name': model_name,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'total_return': total_return * 100
        }
        
        print(f"\n📊 Bitcoin Prediction Evaluation - {model_name}")
        print("=" * 50)
        print(f"💰 Forecasting Accuracy:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
        
        print(f"\n💼 Trading Strategy Performance:")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Total Return: {total_return:.2f}%")
        
        return metrics
    
    def visualize_results(self, actual_data, predictions, model_name, title="Bitcoin Exponential Smoothing"):
        """
        Comprehensive visualization of exponential smoothing results
        
        Args:
            actual_data: Historical Bitcoin prices
            predictions: Model predictions
            model_name: Name of the model
            title: Plot title
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Price forecast plot
        axes[0].plot(actual_data.index, actual_data.values, 
                    label='Historical Bitcoin Price', color='blue', alpha=0.7)
        
        if predictions is not None and not predictions.empty:
            axes[0].plot(predictions.index, predictions['forecast'], 
                        label=f'{model_name} Forecast', color='red', linewidth=2)
            axes[0].fill_between(predictions.index, 
                               predictions['lower_ci'], predictions['upper_ci'],
                               alpha=0.3, color='red', label='95% Confidence Interval')
        
        axes[0].set_title(f"{title} - {model_name}\nClassical Exponential Smoothing Methodology")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Bitcoin Price (USD)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # 2. Model components (if available)
        if model_name in self.fitted_models and self.fitted_models[model_name] is not None:
            fitted_model = self.fitted_models[model_name]
            
            if hasattr(fitted_model, 'level'):
                axes[1].plot(fitted_model.level.index, fitted_model.level, 
                           label='Level Component', color='green', alpha=0.8)
            
            if hasattr(fitted_model, 'trend') and fitted_model.trend is not None:
                axes[1].plot(fitted_model.trend.index, fitted_model.trend, 
                           label='Trend Component', color='orange', alpha=0.8)
            
            axes[1].set_title("Exponential Smoothing Components")
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel("Component Value")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. Innovation comparison
        innovation_text = f"""
        📊 Exponential Smoothing Innovation Impact ({YEAR})
        
        🏭 Historical Foundation:
        • Holt (1957): Linear trend forecasting
        • Winters (1960): Seasonal pattern incorporation
        • Brown (1959): Adaptive forecasting methods
        
        💡 Key Innovations:
        • Weighted averages with exponential decay
        • Automatic adaptation to level, trend, seasonality
        • Simple yet powerful business forecasting
        • Real-time parameter optimization
        
        ⚖️ Bitcoin Application Strengths:
        • Computationally efficient and fast
        • Adapts to changing price levels
        • Handles trend changes dynamically
        • Practical for real-time trading
        
        🚧 Limitations for Crypto Markets:
        • Assumes stable trend/seasonal patterns
        • Cannot handle structural breaks well
        • No multivariate relationships
        • Limited uncertainty quantification
        
        📈 Model: {model_name}
        """
        
        axes[2].text(0.05, 0.95, innovation_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Complete Bitcoin Exponential Smoothing prediction pipeline
    Demonstrates classical smoothing methods on cryptocurrency data
    """
    print("🚀 ERA 1: Exponential Smoothing for Bitcoin Prediction")
    print("=" * 65)
    print(f"📅 Historical Context: {YEAR}")
    print(f"💡 Innovation: {INNOVATION}")
    print(f"🎯 Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"📈 Impact: {IMPACT}")
    print("=" * 65)
    
    # Initialize forecaster
    forecaster = BitcoinExponentialSmoothingForecaster()
    
    # Load Bitcoin data
    train_data, val_data, test_data = forecaster.load_bitcoin_data()
    
    # Step 1: Seasonality analysis
    print("\n" + "="*65)
    print("STEP 1: BITCOIN SEASONALITY ANALYSIS")
    print("="*65)
    
    decomposition, has_seasonality = forecaster.analyze_seasonality(train_data)
    
    # Step 2: Fit all exponential smoothing models
    print("\n" + "="*65)
    print("STEP 2: EXPONENTIAL SMOOTHING MODEL FITTING")
    print("="*65)
    
    # Simple Exponential Smoothing
    ses_model = forecaster.fit_simple_exponential_smoothing(train_data)
    
    # Double Exponential Smoothing (Holt)
    holt_model = forecaster.fit_double_exponential_smoothing(train_data)
    
    # Triple Exponential Smoothing (Holt-Winters)
    if has_seasonality:
        hw_model = forecaster.fit_triple_exponential_smoothing(train_data, seasonal_periods=7)
    else:
        print("\n⚠️ Weak seasonality detected, skipping Holt-Winters model")
    
    # Step 3: Model comparison
    print("\n" + "="*65)
    print("STEP 3: MODEL COMPARISON")
    print("="*65)
    
    best_model_name = forecaster.compare_models(train_data)
    
    if best_model_name is None:
        print("❌ No models successfully fitted!")
        return
    
    # Step 4: Forecasting with best model
    print("\n" + "="*65)
    print("STEP 4: BITCOIN PRICE FORECASTING")
    print("="*65)
    
    # Generate predictions
    predictions = forecaster.predict_bitcoin_price(best_model_name, steps=30)
    
    # Evaluate on validation data
    if len(val_data) > 0 and predictions is not None:
        val_predictions = forecaster.predict_bitcoin_price(best_model_name, steps=len(val_data))
        metrics = forecaster.evaluate_bitcoin_predictions(val_data, val_predictions, best_model_name)
    
    # Step 5: Visualization and analysis
    print("\n" + "="*65)
    print("STEP 5: RESULTS VISUALIZATION")
    print("="*65)
    
    # Combine historical and forecast data for visualization
    full_data = pd.concat([train_data, val_data, test_data])
    forecaster.visualize_results(full_data, predictions, best_model_name,
                               f"Bitcoin {best_model_name} Forecast")
    
    # Compare all models if multiple fitted
    if len(forecaster.fitted_models) > 1:
        print("\n" + "="*65)
        print("STEP 6: COMPARATIVE ANALYSIS")
        print("="*65)
        
        comparison_results = []
        for model_name in forecaster.fitted_models.keys():
            if forecaster.fitted_models[model_name] is not None:
                pred = forecaster.predict_bitcoin_price(model_name, steps=min(30, len(val_data)))
                if pred is not None and len(val_data) > 0:
                    metrics = forecaster.evaluate_bitcoin_predictions(val_data[:min(30, len(val_data))], 
                                                                    pred, model_name)
                    comparison_results.append(metrics)
        
        if comparison_results:
            print("\n📊 Model Performance Comparison:")
            print("Model        | RMSE     | MAPE    | Dir.Acc | Sharpe")
            print("-" * 55)
            for result in comparison_results:
                print(f"{result['model_name']:12} | {result['rmse']:8.2f} | "
                      f"{result['mape']:7.2f}% | {result['directional_accuracy']:7.1f}% | "
                      f"{result['sharpe_ratio']:6.3f}")
    
    # Final summary
    print("\n" + "🏆" + "="*63 + "🏆")
    print("ERA 1 SUMMARY: EXPONENTIAL SMOOTHING FOUNDATIONS")
    print("🏆" + "="*63 + "🏆")
    print(f"""
    📊 Exponential Smoothing Analysis Complete:
    
    🏭 Classical Methods Applied:
    • Best Model: {best_model_name}
    • Models Tested: {len(forecaster.fitted_models)} variants
    • Seasonality: {'✅ Detected' if has_seasonality else '❌ Weak'}
    
    💰 Bitcoin Prediction Performance:
    • 30-day forecast generated
    • Automatic parameter optimization
    • Real-time adaptation capabilities
    
    🎓 Educational Value:
    • Demonstrates adaptive forecasting
    • Shows trend and seasonal modeling
    • Practical business forecasting approach
    
    🚀 Ready for ERA 2: Machine Learning Transition!
    """)


if __name__ == "__main__":
    main()