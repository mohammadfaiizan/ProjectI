"""
Tools module for Trading Analysis Agent.
Contains market data fetching, technical indicator calculations, sentiment analysis, and risk metrics.
"""

from langchain_core.tools import tool
from typing import Dict, List, Any, Optional
import json
import random
import numpy as np
from datetime import datetime, timedelta
import math


class Market_Data_Generator:
    """Class for generating realistic mock OHLCV market data using random walks."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize market data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def Generate_OHLCV_Data(
        self,
        symbol: str,
        days: int,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.001
    ) -> List[Dict[str, Any]]:
        """
        Generate realistic OHLCV data using random walk with trend and volatility.
        
        Args:
            symbol: Stock symbol
            days: Number of days of data to generate
            initial_price: Starting price
            volatility: Daily volatility factor (e.g., 0.02 for 2%)
            trend: Daily trend factor (e.g., 0.001 for 0.1% daily growth)
            
        Returns:
            List of dictionaries with OHLCV data
        """
        data = []
        current_price = initial_price
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Random walk with trend
            daily_change = np.random.normal(trend, volatility)
            current_price = current_price * (1 + daily_change)
            
            # Generate intraday variation for OHLC
            intraday_volatility = volatility * 0.5
            open_price = current_price * (1 + np.random.normal(0, intraday_volatility * 0.3))
            high_price = max(
                open_price,
                current_price * (1 + abs(np.random.normal(0, intraday_volatility)))
            )
            low_price = min(
                open_price,
                current_price * (1 - abs(np.random.normal(0, intraday_volatility)))
            )
            close_price = current_price
            
            # Generate volume (higher volume on larger price movements)
            price_change_pct = abs((close_price - open_price) / open_price)
            base_volume = 1000000
            volume_multiplier = 1 + (price_change_pct * 10)
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.7, 1.3))
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume
            })
            
            current_price = close_price
        
        return data


class Technical_Calculator:
    """Class for calculating technical indicators."""
    
    @staticmethod
    def Calculate_SMA(prices: List[float], period: int) -> List[Optional[float]]:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of closing prices
            period: Period for SMA calculation
            
        Returns:
            List of SMA values (None for insufficient data)
        """
        sma_values = []
        for i in range(len(prices)):
            if i < period - 1:
                sma_values.append(None)
            else:
                sma = sum(prices[i - period + 1:i + 1]) / period
                sma_values.append(round(sma, 2))
        return sma_values
    
    @staticmethod
    def Calculate_EMA(prices: List[float], period: int) -> List[Optional[float]]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of closing prices
            period: Period for EMA calculation
            
        Returns:
            List of EMA values (None for insufficient data)
        """
        if len(prices) < period:
            return [None] * len(prices)
        
        ema_values = []
        multiplier = 2.0 / (period + 1)
        
        # Initialize with SMA
        initial_sma = sum(prices[:period]) / period
        ema_values.extend([None] * (period - 1))
        ema_values.append(round(initial_sma, 2))
        
        # Calculate EMA for remaining values
        current_ema = initial_sma
        for i in range(period, len(prices)):
            current_ema = (prices[i] - current_ema) * multiplier + current_ema
            ema_values.append(round(current_ema, 2))
        
        return ema_values
    
    @staticmethod
    def Calculate_RSI(prices: List[float], period: int = 14) -> List[Optional[float]]:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: List of closing prices
            period: Period for RSI calculation (default 14)
            
        Returns:
            List of RSI values (None for insufficient data)
        """
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        rsi_values = [None] * period
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(round(rsi, 2))
        
        # Calculate RSI for remaining values
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(round(rsi, 2))
        
        return rsi_values
    
    @staticmethod
    def Calculate_MACD(
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, List[Optional[float]]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: List of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        # Calculate fast and slow EMAs
        fast_ema = Technical_Calculator.Calculate_EMA(prices, fast)
        slow_ema = Technical_Calculator.Calculate_EMA(prices, slow)
        
        # Calculate MACD line
        macd_line = []
        min_period = max(fast, slow)
        for i in range(len(prices)):
            if i < min_period - 1 or fast_ema[i] is None or slow_ema[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(round(fast_ema[i] - slow_ema[i], 2))
        
        # Calculate signal line (EMA of MACD line)
        macd_values = [v for v in macd_line if v is not None]
        if len(macd_values) < signal:
            signal_line = [None] * len(macd_line)
        else:
            signal_line_values = Technical_Calculator.Calculate_EMA(macd_values, signal)
            signal_line = [None] * (min_period - 1 + signal - 1)
            signal_line.extend(signal_line_values[signal - 1:])
        
        # Calculate histogram
        histogram = []
        for i in range(len(macd_line)):
            if macd_line[i] is None or signal_line[i] is None:
                histogram.append(None)
            else:
                histogram.append(round(macd_line[i] - signal_line[i], 2))
        
        return {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram
        }
    
    @staticmethod
    def Calculate_Bollinger_Bands(
        prices: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, List[Optional[float]]]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: List of closing prices
            period: Period for moving average
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper band, middle band (SMA), and lower band
        """
        sma_values = Technical_Calculator.Calculate_SMA(prices, period)
        upper_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if sma_values[i] is None:
                upper_band.append(None)
                lower_band.append(None)
            else:
                # Calculate standard deviation
                start_idx = max(0, i - period + 1)
                period_prices = prices[start_idx:i + 1]
                variance = sum((p - sma_values[i]) ** 2 for p in period_prices) / len(period_prices)
                std = math.sqrt(variance)
                
                upper_band.append(round(sma_values[i] + (std_dev * std), 2))
                lower_band.append(round(sma_values[i] - (std_dev * std), 2))
        
        return {
            "upper_band": upper_band,
            "middle_band": sma_values,
            "lower_band": lower_band
        }


@tool
def Fetch_Market_Data(symbol: str, days: int = 30) -> str:
    """
    Fetch market data (OHLCV) for a given symbol.
    Generates realistic mock data using random walks.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        days: Number of days of historical data to fetch
        
    Returns:
        JSON string containing OHLCV data
    """
    generator = Market_Data_Generator(seed=hash(symbol) % 1000)
    
    # Adjust initial price based on symbol
    symbol_prices = {
        "AAPL": 180.0,
        "GOOGL": 140.0,
        "TSLA": 250.0,
        "MSFT": 420.0,
        "AMZN": 150.0
    }
    initial_price = symbol_prices.get(symbol.upper(), 100.0)
    
    # Adjust volatility based on symbol
    symbol_volatility = {
        "AAPL": 0.018,
        "GOOGL": 0.020,
        "TSLA": 0.035,
        "MSFT": 0.015,
        "AMZN": 0.022
    }
    volatility = symbol_volatility.get(symbol.upper(), 0.02)
    
    data = generator.Generate_OHLCV_Data(
        symbol=symbol,
        days=days,
        initial_price=initial_price,
        volatility=volatility,
        trend=np.random.uniform(-0.001, 0.002)
    )
    
    return json.dumps({
        "symbol": symbol.upper(),
        "data": data,
        "days": days
    })


@tool
def Calculate_Technical_Indicators(data_json: str) -> str:
    """
    Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) from market data.
    
    Args:
        data_json: JSON string containing OHLCV data from Fetch_Market_Data
        
    Returns:
        JSON string containing calculated technical indicators
    """
    data = json.loads(data_json)
    prices = [item["close"] for item in data["data"]]
    volumes = [item["volume"] for item in data["data"]]
    
    calculator = Technical_Calculator()
    
    # Calculate SMAs
    sma_10 = calculator.Calculate_SMA(prices, 10)
    sma_20 = calculator.Calculate_SMA(prices, 20)
    sma_50 = calculator.Calculate_SMA(prices, 50)
    sma_200 = calculator.Calculate_SMA(prices, 200)
    
    # Calculate EMAs
    ema_12 = calculator.Calculate_EMA(prices, 12)
    ema_26 = calculator.Calculate_EMA(prices, 26)
    ema_50 = calculator.Calculate_EMA(prices, 50)
    
    # Calculate RSI
    rsi = calculator.Calculate_RSI(prices, 14)
    
    # Calculate MACD
    macd = calculator.Calculate_MACD(prices, 12, 26, 9)
    
    # Calculate Bollinger Bands
    bollinger = calculator.Calculate_Bollinger_Bands(prices, 20, 2.0)
    
    # Calculate Volume SMA
    volume_sma = calculator.Calculate_SMA(volumes, 20)
    
    # Get latest values
    latest_idx = len(prices) - 1
    
    indicators = {
        "sma": {
            "sma_10": sma_10[latest_idx] if sma_10[latest_idx] is not None else None,
            "sma_20": sma_20[latest_idx] if sma_20[latest_idx] is not None else None,
            "sma_50": sma_50[latest_idx] if sma_50[latest_idx] is not None else None,
            "sma_200": sma_200[latest_idx] if sma_200[latest_idx] is not None else None
        },
        "ema": {
            "ema_12": ema_12[latest_idx] if ema_12[latest_idx] is not None else None,
            "ema_26": ema_26[latest_idx] if ema_26[latest_idx] is not None else None,
            "ema_50": ema_50[latest_idx] if ema_50[latest_idx] is not None else None
        },
        "rsi": rsi[latest_idx] if rsi[latest_idx] is not None else None,
        "macd": {
            "macd_line": macd["macd_line"][latest_idx] if macd["macd_line"][latest_idx] is not None else None,
            "signal_line": macd["signal_line"][latest_idx] if macd["signal_line"][latest_idx] is not None else None,
            "histogram": macd["histogram"][latest_idx] if macd["histogram"][latest_idx] is not None else None
        },
        "bollinger_bands": {
            "upper": bollinger["upper_band"][latest_idx] if bollinger["upper_band"][latest_idx] is not None else None,
            "middle": bollinger["middle_band"][latest_idx] if bollinger["middle_band"][latest_idx] is not None else None,
            "lower": bollinger["lower_band"][latest_idx] if bollinger["lower_band"][latest_idx] is not None else None
        },
        "volume_sma": volume_sma[latest_idx] if volume_sma[latest_idx] is not None else None,
        "current_price": prices[latest_idx]
    }
    
    return json.dumps(indicators)


@tool
def Analyze_Sentiment(symbol: str) -> str:
    """
    Analyze market sentiment for a given symbol.
    Returns mock sentiment analysis with score and headlines.
    
    Args:
        symbol: Stock symbol to analyze sentiment for
        
    Returns:
        JSON string containing sentiment score and sample headlines
    """
    # Mock sentiment data based on symbol
    sentiment_data = {
        "AAPL": {
            "score": 0.65,
            "headlines": [
                "Apple reports strong iPhone sales in Q4",
                "New product launches drive investor optimism",
                "Market analysts upgrade Apple price target"
            ]
        },
        "GOOGL": {
            "score": 0.55,
            "headlines": [
                "Google Cloud revenue growth accelerates",
                "AI competition intensifies in search market",
                "Regulatory concerns weigh on tech sector"
            ]
        },
        "TSLA": {
            "score": 0.45,
            "headlines": [
                "Tesla announces new model launch",
                "Volatile trading continues amid market uncertainty",
                "Production delays concern investors"
            ]
        },
        "MSFT": {
            "score": 0.70,
            "headlines": [
                "Microsoft Azure growth exceeds expectations",
                "Enterprise adoption drives strong performance",
                "Dividend increase signals confidence"
            ]
        },
        "AMZN": {
            "score": 0.60,
            "headlines": [
                "Amazon Web Services maintains market leadership",
                "E-commerce segment shows resilience",
                "Cost optimization efforts pay off"
            ]
        }
    }
    
    # Default sentiment with some randomness
    default_sentiment = {
        "score": round(random.uniform(0.3, 0.7), 2),
        "headlines": [
            f"{symbol} shows mixed signals in recent trading",
            f"Analysts remain cautious on {symbol} outlook",
            f"Market sentiment neutral for {symbol}"
        ]
    }
    
    sentiment = sentiment_data.get(symbol.upper(), default_sentiment)
    
    # Add some randomness to score
    sentiment["score"] = round(
        max(0.0, min(1.0, sentiment["score"] + random.uniform(-0.1, 0.1))),
        2
    )
    
    return json.dumps({
        "symbol": symbol.upper(),
        "sentiment_score": sentiment["score"],
        "headlines": sentiment["headlines"],
        "interpretation": (
            "Bullish" if sentiment["score"] > 0.6
            else "Bearish" if sentiment["score"] < 0.4
            else "Neutral"
        )
    })


@tool
def Calculate_Risk_Metrics(data_json: str, position_size: float = 10000.0) -> str:
    """
    Calculate risk metrics including VaR, max drawdown, and Sharpe ratio.
    
    Args:
        data_json: JSON string containing OHLCV data from Fetch_Market_Data
        position_size: Position size in base currency
        
    Returns:
        JSON string containing risk metrics
    """
    data = json.loads(data_json)
    prices = [item["close"] for item in data["data"]]
    
    if len(prices) < 2:
        return json.dumps({
            "error": "Insufficient data for risk calculations"
        })
    
    # Calculate daily returns
    returns = []
    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i - 1]) / prices[i - 1]
        returns.append(daily_return)
    
    if len(returns) == 0:
        return json.dumps({
            "error": "Cannot calculate returns"
        })
    
    returns_array = np.array(returns)
    
    # Calculate Value at Risk (VaR) at 95% confidence
    var_95 = np.percentile(returns_array, 5)
    var_value = abs(var_95 * position_size)
    
    # Calculate Maximum Drawdown
    cumulative_returns = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(np.min(drawdown))
    
    # Calculate Sharpe Ratio (annualized, assuming 252 trading days)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    if std_return == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
    
    # Calculate volatility (annualized)
    volatility = std_return * np.sqrt(252)
    
    # Calculate current price and price change
    current_price = prices[-1]
    price_change_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
    
    risk_metrics = {
        "position_size": position_size,
        "current_price": round(current_price, 2),
        "price_change_pct": round(price_change_pct, 2),
        "var_95": round(var_value, 2),
        "var_95_pct": round(var_95 * 100, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "volatility": round(volatility * 100, 2),
        "mean_daily_return": round(mean_return * 100, 3)
    }
    
    return json.dumps(risk_metrics)
