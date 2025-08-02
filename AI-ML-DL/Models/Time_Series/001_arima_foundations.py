"""
Time Series ERA 1: ARIMA Foundations (1970s-2000s)
==================================================

Historical Context:
Year: 1970-1976 (Box-Jenkins methodology)
Innovation: Systematic approach to ARIMA modeling with diagnostic checking
Previous Limitation: Ad-hoc time series modeling without statistical rigor
Impact: Established the foundation for modern time series analysis and forecasting

This implementation demonstrates the classical Box-Jenkins approach to ARIMA modeling
applied to Bitcoin price prediction, showcasing the power and limitations of traditional
statistical methods when applied to highly volatile cryptocurrency markets.
"""

# Historical Context & Innovation
YEAR = "1970-1976"
INNOVATION = "Box-Jenkins ARIMA methodology with systematic model identification"
PREVIOUS_LIMITATION = "Ad-hoc time series modeling without statistical foundation"
IMPACT = "Established rigorous statistical framework for time series forecasting"

# Imports and Dependencies
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

class BitcoinARIMAForecaster:
    """
    Classical ARIMA implementation for Bitcoin price prediction
    
    Features Box-Jenkins methodology:
    1. Model Identification (ACF/PACF analysis)
    2. Parameter Estimation (Maximum Likelihood)
    3. Diagnostic Checking (Residual analysis)
    4. Forecasting with confidence intervals
    """
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.data = None
        self.differenced_data = None
        self.best_order = None
        
    def load_bitcoin_data(self, start_date='2010-01-01', end_date='2024-01-01'):
        """
        Load Bitcoin OHLCV data from Yahoo Finance
        
        Returns:
            train_data, val_data, test_data: Temporal splits for time series
        """
        print("📊 Loading Bitcoin Data (2010-present)...")
        
        # Download Bitcoin data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_data = btc_data.dropna()
        
        # Use closing prices for ARIMA modeling
        prices = btc_data['Close'].copy()
        
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
    
    def check_stationarity(self, series, name="Series"):
        """
        Comprehensive stationarity testing using ADF and KPSS tests
        
        Args:
            series: Time series data
            name: Name for display purposes
        """
        print(f"\n🔍 Stationarity Analysis for {name}")
        print("=" * 50)
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna(), autolag='AIC')
        print(f"ADF Test Results:")
        print(f"  ADF Statistic: {adf_result[0]:.6f}")
        print(f"  p-value: {adf_result[1]:.6f}")
        print(f"  Critical Values:")
        for key, value in adf_result[4].items():
            print(f"    {key}: {value:.6f}")
        
        adf_stationary = adf_result[1] <= 0.05
        print(f"  ADF Result: {'✅ Stationary' if adf_stationary else '❌ Non-stationary'}")
        
        # KPSS test
        kpss_result = kpss(series.dropna(), regression='c')
        print(f"\nKPSS Test Results:")
        print(f"  KPSS Statistic: {kpss_result[0]:.6f}")
        print(f"  p-value: {kpss_result[1]:.6f}")
        print(f"  Critical Values:")
        for key, value in kpss_result[3].items():
            print(f"    {key}: {value:.6f}")
        
        kpss_stationary = kpss_result[1] >= 0.05
        print(f"  KPSS Result: {'✅ Stationary' if kpss_stationary else '❌ Non-stationary'}")
        
        # Combined interpretation
        if adf_stationary and kpss_stationary:
            conclusion = "✅ STATIONARY (both tests agree)"
        elif not adf_stationary and not kpss_stationary:
            conclusion = "❌ NON-STATIONARY (both tests agree)"
        else:
            conclusion = "⚠️ INCONCLUSIVE (tests disagree)"
        
        print(f"\n📊 Final Assessment: {conclusion}")
        return adf_stationary and kpss_stationary
    
    def make_stationary(self, series, max_diff=2):
        """
        Apply differencing to achieve stationarity
        
        Args:
            series: Time series data
            max_diff: Maximum number of differencing operations
            
        Returns:
            Differenced series and number of differences applied
        """
        print("\n🔄 Making Series Stationary...")
        
        differenced = series.copy()
        diff_order = 0
        
        for d in range(max_diff + 1):
            if self.check_stationarity(differenced, f"Differenced (d={d})"):
                diff_order = d
                break
            if d < max_diff:
                differenced = differenced.diff().dropna()
        
        self.differenced_data = differenced
        print(f"✅ Achieved stationarity with d={diff_order}")
        return differenced, diff_order
    
    def plot_acf_pacf(self, series, lags=40):
        """
        Plot ACF and PACF for model identification
        
        Args:
            series: Stationary time series
            lags: Number of lags to display
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        plot_acf(series.dropna(), lags=lags, ax=ax1, title="Autocorrelation Function (ACF)")
        ax1.set_xlabel("Lags")
        ax1.set_ylabel("ACF")
        ax1.grid(True, alpha=0.3)
        
        # PACF plot
        plot_pacf(series.dropna(), lags=lags, ax=ax2, title="Partial Autocorrelation Function (PACF)")
        ax2.set_xlabel("Lags")
        ax2.set_ylabel("PACF")
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle("Box-Jenkins Model Identification\nACF/PACF Analysis for ARIMA Order Selection", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def identify_arima_order(self, series, max_p=5, max_d=2, max_q=5):
        """
        Automatic ARIMA order selection using AIC criterion
        
        Args:
            series: Time series data
            max_p, max_d, max_q: Maximum orders to test
            
        Returns:
            Best ARIMA order (p, d, q)
        """
        print("\n🔍 ARIMA Order Identification (Box-Jenkins Methodology)")
        print("=" * 60)
        
        best_aic = np.inf
        best_order = None
        aic_results = []
        
        # Grid search over ARIMA orders
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        aic_results.append((p, d, q, aic))
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            
                    except Exception as e:
                        continue
        
        # Display top 5 models
        aic_results.sort(key=lambda x: x[3])
        print("🏆 Top 5 ARIMA Models (by AIC):")
        print("Rank | Order (p,d,q) | AIC Score")
        print("-" * 35)
        for i, (p, d, q, aic) in enumerate(aic_results[:5], 1):
            marker = "👑" if i == 1 else f"{i}."
            print(f"{marker:4} | ({p},{d},{q})       | {aic:.2f}")
        
        self.best_order = best_order
        print(f"\n✅ Selected ARIMA Order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def fit_arima_model(self, series, order):
        """
        Fit ARIMA model with specified order
        
        Args:
            series: Time series data
            order: ARIMA order (p, d, q)
        """
        print(f"\n🔧 Fitting ARIMA{order} Model...")
        
        self.model = ARIMA(series, order=order)
        self.fitted_model = self.model.fit()
        
        print("✅ Model fitted successfully!")
        print("\n📊 Model Summary:")
        print("=" * 40)
        print(f"AIC: {self.fitted_model.aic:.2f}")
        print(f"BIC: {self.fitted_model.bic:.2f}")
        print(f"Log-Likelihood: {self.fitted_model.llf:.2f}")
        
        return self.fitted_model
    
    def diagnostic_checking(self):
        """
        Comprehensive residual analysis for model validation
        """
        if self.fitted_model is None:
            print("❌ No fitted model available for diagnostics")
            return False
        
        print("\n🔍 Diagnostic Checking (Box-Jenkins Step 3)")
        print("=" * 50)
        
        residuals = self.fitted_model.resid
        
        # Ljung-Box test for autocorrelation in residuals
        ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
        autocorr_pvalue = ljung_box['lb_pvalue'].min()
        
        print(f"📊 Ljung-Box Test (Residual Autocorrelation):")
        print(f"  Min p-value: {autocorr_pvalue:.4f}")
        if autocorr_pvalue > 0.05:
            print("  ✅ No significant autocorrelation in residuals")
        else:
            print("  ⚠️ Significant autocorrelation detected")
        
        # Normality test (visual)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals plot
        axes[0, 0].plot(residuals.index, residuals, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title("Residuals vs Time")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot (Normality Check)")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residual ACF
        plot_acf(residuals, lags=20, ax=axes[1, 0], title="Residual ACF")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue')
        axes[1, 1].set_title("Residual Distribution")
        axes[1, 1].set_xlabel("Residuals")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle("ARIMA Model Diagnostic Checking", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return autocorr_pvalue > 0.05
    
    def predict_bitcoin_price(self, steps=30, alpha=0.05):
        """
        Multi-horizon Bitcoin price forecasting with confidence intervals
        
        Args:
            steps: Number of steps ahead to forecast
            alpha: Significance level for confidence intervals
            
        Returns:
            Predictions with confidence intervals
        """
        if self.fitted_model is None:
            print("❌ No fitted model available for prediction")
            return None
        
        print(f"\n🔮 Bitcoin Price Forecasting ({steps} days ahead)")
        print("=" * 50)
        
        # Generate forecasts
        forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
        forecasts = forecast_result
        
        # Get confidence intervals
        conf_int = self.fitted_model.get_forecast(steps=steps, alpha=alpha).conf_int()
        
        # Create forecast dates
        last_date = self.data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=steps, freq='D')
        
        # Organize results
        predictions = pd.DataFrame({
            'forecast': forecasts,
            'lower_ci': conf_int.iloc[:, 0],
            'upper_ci': conf_int.iloc[:, 1]
        }, index=forecast_dates)
        
        print(f"✅ Generated {steps}-day Bitcoin price forecast")
        print(f"📊 Confidence Level: {(1-alpha)*100}%")
        
        return predictions
    
    def evaluate_bitcoin_predictions(self, actual_prices, predictions):
        """
        Comprehensive evaluation of Bitcoin price predictions
        
        Args:
            actual_prices: True Bitcoin prices
            predictions: Model predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(actual_prices) == 0 or len(predictions) == 0:
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
        actual_direction = (actual.shift(-1) > actual).astype(int)[:-1]
        pred_direction = (pred.shift(-1) > pred).astype(int)[:-1]
        directional_accuracy = (actual_direction == pred_direction).mean() * 100
        
        # Financial metrics (simplified)
        returns_actual = actual.pct_change().dropna()
        returns_pred = pred.pct_change().dropna()
        
        # Trading strategy simulation
        signals = np.where(pred.shift(1) < pred, 1, -1)  # Simple momentum
        strategy_returns = signals[1:] * returns_actual[1:]
        
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'total_return': strategy_returns.sum() * 100
        }
        
        print("\n📊 Bitcoin Prediction Evaluation")
        print("=" * 40)
        print(f"💰 Forecasting Accuracy:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
        
        print(f"\n💼 Trading Strategy Performance:")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Total Return: {strategy_returns.sum()*100:.2f}%")
        
        return metrics
    
    def visualize_results(self, actual_data, predictions, title="Bitcoin ARIMA Forecast"):
        """
        Comprehensive visualization of Bitcoin ARIMA results
        
        Args:
            actual_data: Historical Bitcoin prices
            predictions: Model predictions with confidence intervals
            title: Plot title
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Price forecast plot
        axes[0].plot(actual_data.index, actual_data.values, 
                    label='Historical Bitcoin Price', color='blue', alpha=0.7)
        
        if predictions is not None and not predictions.empty:
            axes[0].plot(predictions.index, predictions['forecast'], 
                        label='ARIMA Forecast', color='red', linewidth=2)
            axes[0].fill_between(predictions.index, 
                               predictions['lower_ci'], predictions['upper_ci'],
                               alpha=0.3, color='red', label='95% Confidence Interval')
        
        axes[0].set_title(f"{title}\nClassical Box-Jenkins ARIMA Methodology")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Bitcoin Price (USD)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')  # Log scale for Bitcoin prices
        
        # 2. Model residuals
        if self.fitted_model is not None:
            residuals = self.fitted_model.resid
            axes[1].plot(residuals.index, residuals, alpha=0.7, color='green')
            axes[1].axhline(y=0, color='red', linestyle='--')
            axes[1].set_title("Model Residuals (Should be Random)")
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel("Residuals")
            axes[1].grid(True, alpha=0.3)
        
        # 3. Innovation comparison
        innovation_text = f"""
        📊 ARIMA Innovation Impact ({YEAR})
        
        🏛️ Historical Foundation:
        • Box-Jenkins methodology provided systematic approach
        • Rigorous statistical framework for time series analysis
        • Model identification through ACF/PACF analysis
        
        💡 Key Innovations:
        • Maximum likelihood parameter estimation
        • Diagnostic checking with residual analysis
        • Confidence intervals for forecasts
        • AIC/BIC model selection criteria
        
        ⚖️ Bitcoin Application Strengths:
        • Interpretable statistical model
        • Uncertainty quantification
        • Established theoretical foundation
        • Computationally efficient
        
        🚧 Limitations for Crypto Markets:
        • Linear assumptions inadequate for Bitcoin volatility
        • Cannot capture regime changes
        • No multivariate relationships
        • Struggles with structural breaks
        """
        
        axes[2].text(0.05, 0.95, innovation_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Complete Bitcoin ARIMA prediction pipeline
    Demonstrates classical Box-Jenkins methodology on cryptocurrency data
    """
    print("🚀 ERA 1: ARIMA Foundations for Bitcoin Prediction")
    print("=" * 60)
    print(f"📅 Historical Context: {YEAR}")
    print(f"💡 Innovation: {INNOVATION}")
    print(f"🎯 Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"📈 Impact: {IMPACT}")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = BitcoinARIMAForecaster()
    
    # Load Bitcoin data
    train_data, val_data, test_data = forecaster.load_bitcoin_data()
    
    # Step 1: Stationarity analysis
    print("\n" + "="*60)
    print("STEP 1: STATIONARITY ANALYSIS (Box-Jenkins)")
    print("="*60)
    
    # Check original series stationarity
    forecaster.check_stationarity(train_data, "Original Bitcoin Prices")
    
    # Make stationary if needed
    stationary_data, diff_order = forecaster.make_stationary(train_data)
    
    # Step 2: Model identification
    print("\n" + "="*60)
    print("STEP 2: MODEL IDENTIFICATION (ACF/PACF Analysis)")
    print("="*60)
    
    # Plot ACF/PACF for visual identification
    forecaster.plot_acf_pacf(stationary_data)
    
    # Automatic order selection
    best_order = forecaster.identify_arima_order(train_data)
    
    # Step 3: Parameter estimation
    print("\n" + "="*60)
    print("STEP 3: PARAMETER ESTIMATION")
    print("="*60)
    
    fitted_model = forecaster.fit_arima_model(train_data, best_order)
    
    # Step 4: Diagnostic checking
    print("\n" + "="*60)
    print("STEP 4: DIAGNOSTIC CHECKING")
    print("="*60)
    
    model_adequate = forecaster.diagnostic_checking()
    
    # Step 5: Forecasting
    print("\n" + "="*60)
    print("STEP 5: BITCOIN PRICE FORECASTING")
    print("="*60)
    
    # Generate predictions
    predictions = forecaster.predict_bitcoin_price(steps=30)
    
    # Evaluate on validation data
    if len(val_data) > 0:
        val_predictions = forecaster.predict_bitcoin_price(steps=len(val_data))
        metrics = forecaster.evaluate_bitcoin_predictions(val_data, val_predictions)
    
    # Step 6: Visualization and analysis
    print("\n" + "="*60)
    print("STEP 6: RESULTS VISUALIZATION")
    print("="*60)
    
    # Combine historical and forecast data for visualization
    full_data = pd.concat([train_data, val_data, test_data])
    forecaster.visualize_results(full_data, predictions, 
                               f"Bitcoin ARIMA{best_order} Forecast")
    
    # Final summary
    print("\n" + "🏆" + "="*58 + "🏆")
    print("ERA 1 SUMMARY: ARIMA FOUNDATIONS")
    print("🏆" + "="*58 + "🏆")
    print(f"""
    📊 Classical ARIMA Analysis Complete:
    
    🏛️ Statistical Foundation:
    • Model: ARIMA{best_order}
    • AIC Score: {fitted_model.aic:.2f}
    • Diagnostic Check: {'✅ Passed' if model_adequate else '⚠️ Issues detected'}
    
    💰 Bitcoin Prediction Performance:
    • 30-day forecast generated with confidence intervals
    • Captures linear trends and seasonality
    • Provides uncertainty quantification
    
    🎓 Educational Value:
    • Demonstrates Box-Jenkins methodology
    • Shows systematic approach to time series modeling
    • Establishes baseline for modern methods comparison
    
    🚀 Ready for ERA 2: Machine Learning Transition!
    """)


if __name__ == "__main__":
    main()