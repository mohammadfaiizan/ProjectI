"""
Trading Analysis Agent Implementation

DISCLAIMER: This system is for educational and research purposes only.
It does not constitute financial advice, and trading involves substantial risk of loss.
Past performance does not guarantee future results. Users should conduct their own
research and consult with licensed financial advisors before making trading decisions.

A complete trading analysis agent that analyzes financial instruments using
technical analysis, sentiment analysis, and risk assessment to provide trading insights.
All market data is mocked/simulated for demonstration purposes.
"""

import os
import json
import math
import random
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from openai import OpenAI


@dataclass
class OHLCV_Data:
    """Open, High, Low, Close, Volume data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Technical_Indicators:
    """Technical indicator values."""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    trend: Optional[str] = None  # "up", "down", "sideways"


@dataclass
class Sentiment_Analysis:
    """Sentiment analysis results."""
    overall_sentiment: str  # "bullish", "bearish", "neutral"
    sentiment_score: float  # -1 to 1
    key_events: List[str]
    news_summary: str


@dataclass
class Risk_Assessment:
    """Risk assessment results."""
    position_size: float
    stop_loss: float
    risk_reward_ratio: float
    volatility: float
    max_drawdown: float
    risk_level: str  # "low", "medium", "high"


@dataclass
class Trading_Signal:
    """Trading signal."""
    signal_type: str  # "BUY", "SELL", "HOLD"
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    reasoning: str
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class Position:
    """Trading position."""
    asset: str
    entry_price: float
    quantity: float
    current_price: float
    entry_date: datetime
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None


class Market_Data_Provider:
    """Generates realistic mock OHLCV data."""
    
    def __init__(self):
        self.data_cache: Dict[str, List[OHLCV_Data]] = {}
    
    def generate_ohlcv_data(self, symbol: str, days: int = 100, initial_price: float = 100.0) -> List[OHLCV_Data]:
        """Generate realistic mock OHLCV data."""
        if symbol in self.data_cache:
            return self.data_cache[symbol]
        
        data = []
        current_price = initial_price
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            # Generate realistic price movement
            change_percent = random.gauss(0, 0.02)  # 2% daily volatility
            trend = math.sin(i / 20) * 0.01  # Long-term trend
            
            open_price = current_price
            change = open_price * (change_percent + trend)
            close_price = open_price + change
            
            # Generate high and low
            daily_range = abs(change) * random.uniform(1.5, 3.0)
            high_price = max(open_price, close_price) + daily_range * random.uniform(0.3, 0.7)
            low_price = min(open_price, close_price) - daily_range * random.uniform(0.3, 0.7)
            
            # Generate volume
            base_volume = 1000000
            volume = base_volume * random.uniform(0.5, 2.0)
            
            timestamp = base_date + timedelta(days=i)
            data.append(OHLCV_Data(
                timestamp=timestamp,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=round(volume, 0)
            ))
            
            current_price = close_price
        
        self.data_cache[symbol] = data
        return data
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        data = self.generate_ohlcv_data(symbol)
        return data[-1].close if data else 100.0
    
    def get_historical_data(self, symbol: str, days: int = 100) -> List[OHLCV_Data]:
        """Get historical OHLCV data."""
        return self.generate_ohlcv_data(symbol, days)


class Technical_Analyzer:
    """Calculates technical indicators using raw math/numpy."""
    
    def __init__(self):
        pass
    
    def calculate_sma(self, prices: List[float], period: int) -> List[Optional[float]]:
        """Calculate Simple Moving Average."""
        sma_values = []
        for i in range(len(prices)):
            if i < period - 1:
                sma_values.append(None)
            else:
                sma = sum(prices[i - period + 1:i + 1]) / period
                sma_values.append(sma)
        return sma_values
    
    def calculate_ema(self, prices: List[float], period: int) -> List[Optional[float]]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return [None] * len(prices)
        
        multiplier = 2.0 / (period + 1)
        ema_values = []
        
        # Start with SMA
        sma = sum(prices[:period]) / period
        ema_values.extend([None] * (period - 1))
        ema_values.append(sma)
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
            ema_values.append(ema)
        
        return ema_values
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[Optional[float]]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        rsi_values = [None] * period
        
        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        
        for i in range(period, len(changes)):
            gains = [max(change, 0) for change in changes[i - period + 1:i + 1]]
            losses = [-min(change, 0) for change in changes[i - period + 1:i + 1]]
            
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """Calculate MACD, Signal Line, and Histogram."""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = []
        for i in range(len(prices)):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line.append(ema_fast[i] - ema_slow[i])
            else:
                macd_line.append(None)
        
        # Calculate signal line (EMA of MACD)
        macd_values = [m for m in macd_line if m is not None]
        if len(macd_values) >= signal:
            signal_line_values = self.calculate_ema(macd_values, signal)
            signal_line = []
            none_count = len([m for m in macd_line if m is None])
            signal_line.extend([None] * none_count)
            signal_line.extend(signal_line_values)
        else:
            signal_line = [None] * len(prices)
        
        # Calculate histogram
        histogram = []
        for i in range(len(prices)):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram.append(macd_line[i] - signal_line[i])
            else:
                histogram.append(None)
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """Calculate Bollinger Bands."""
        sma_values = self.calculate_sma(prices, period)
        
        upper_band = []
        middle_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if sma_values[i] is None:
                upper_band.append(None)
                middle_band.append(None)
                lower_band.append(None)
            else:
                # Calculate standard deviation
                period_prices = prices[i - period + 1:i + 1]
                mean = sma_values[i]
                variance = sum((p - mean) ** 2 for p in period_prices) / period
                std = math.sqrt(variance)
                
                middle_band.append(mean)
                upper_band.append(mean + std_dev * std)
                lower_band.append(mean - std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def identify_trend(self, prices: List[float], sma_short: List[Optional[float]], sma_long: List[Optional[float]]) -> Optional[str]:
        """Identify trend direction."""
        if len(sma_short) == 0 or len(sma_long) == 0:
            return None
        
        # Compare recent SMAs
        recent_short = [s for s in sma_short[-10:] if s is not None]
        recent_long = [s for s in sma_long[-10:] if s is not None]
        
        if len(recent_short) < 2 or len(recent_long) < 2:
            return None
        
        short_trend = recent_short[-1] > recent_short[0]
        long_trend = recent_long[-1] > recent_long[0]
        
        if short_trend and long_trend:
            return "up"
        elif not short_trend and not long_trend:
            return "down"
        else:
            return "sideways"
    
    def analyze(self, ohlcv_data: List[OHLCV_Data]) -> Technical_Indicators:
        """Perform complete technical analysis."""
        closes = [d.close for d in ohlcv_data]
        
        sma_20 = self.calculate_sma(closes, 20)
        sma_50 = self.calculate_sma(closes, 50)
        ema_12 = self.calculate_ema(closes, 12)
        ema_26 = self.calculate_ema(closes, 26)
        rsi = self.calculate_rsi(closes, 14)
        macd, macd_signal, macd_hist = self.calculate_macd(closes)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes)
        
        trend = self.identify_trend(closes, sma_20, sma_50)
        
        return Technical_Indicators(
            sma_20=sma_20[-1] if sma_20[-1] is not None else None,
            sma_50=sma_50[-1] if sma_50[-1] is not None else None,
            ema_12=ema_12[-1] if ema_12[-1] is not None else None,
            ema_26=ema_26[-1] if ema_26[-1] is not None else None,
            rsi=rsi[-1] if rsi[-1] is not None else None,
            macd=macd[-1] if macd[-1] is not None else None,
            macd_signal=macd_signal[-1] if macd_signal[-1] is not None else None,
            macd_histogram=macd_hist[-1] if macd_hist[-1] is not None else None,
            bollinger_upper=bb_upper[-1] if bb_upper[-1] is not None else None,
            bollinger_middle=bb_middle[-1] if bb_middle[-1] is not None else None,
            bollinger_lower=bb_lower[-1] if bb_lower[-1] is not None else None,
            trend=trend
        )


class News_Sentiment_Analyzer:
    """Analyzes financial news sentiment using LLM."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate_mock_news(self, symbol: str) -> List[str]:
        """Generate mock financial news articles."""
        news_templates = [
            f"{symbol} reports strong quarterly earnings, beating analyst expectations",
            f"Market analysts upgrade {symbol} price target following positive outlook",
            f"{symbol} announces new product launch, driving investor optimism",
            f"Concerns about {symbol} valuation as stock reaches new highs",
            f"{symbol} faces regulatory challenges in key markets",
            f"Technical analysis suggests {symbol} may be overbought",
            f"{symbol} forms strategic partnership to expand market presence",
            f"Volatility increases for {symbol} amid market uncertainty"
        ]
        return random.sample(news_templates, 3)
    
    def analyze_sentiment(self, news_articles: List[str]) -> Sentiment_Analysis:
        """Analyze sentiment from news articles."""
        news_text = "\n".join([f"- {article}" for article in news_articles])
        
        sentiment_prompt = f"""Analyze the sentiment of the following financial news articles and provide:
1. Overall sentiment (bullish, bearish, or neutral)
2. Sentiment score (-1 to 1, where -1 is very bearish, 1 is very bullish)
3. Key events or factors mentioned
4. Brief summary of the news

News Articles:
{news_text}

Respond with JSON containing:
- "overall_sentiment": one of "bullish", "bearish", "neutral"
- "sentiment_score": number between -1 and 1
- "key_events": list of key events mentioned
- "news_summary": brief summary

JSON Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial news sentiment analyst. Always respond with valid JSON."},
                    {"role": "user", "content": sentiment_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return Sentiment_Analysis(
                overall_sentiment=result.get("overall_sentiment", "neutral"),
                sentiment_score=float(result.get("sentiment_score", 0.0)),
                key_events=result.get("key_events", []),
                news_summary=result.get("news_summary", "")
            )
        except Exception as e:
            return Sentiment_Analysis(
                overall_sentiment="neutral",
                sentiment_score=0.0,
                key_events=[],
                news_summary=f"Error analyzing sentiment: {str(e)}"
            )


class Risk_Assessor:
    """Assesses trading risk and calculates position sizing."""
    
    def __init__(self, account_balance: float = 100000.0, risk_per_trade: float = 0.02):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade  # 2% risk per trade
    
    def calculate_volatility(self, prices: List[float], period: int = 20) -> float:
        """Calculate price volatility."""
        if len(prices) < period:
            return 0.0
        
        recent_prices = prices[-period:]
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                   for i in range(1, len(recent_prices))]
        
        if len(returns) == 0:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        
        # Annualized volatility
        return std_dev * math.sqrt(252)  # 252 trading days
    
    def calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk."""
        risk_amount = self.account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0.0
        
        position_size = risk_amount / price_risk
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, volatility: float, support_level: Optional[float] = None) -> float:
        """Calculate stop-loss level."""
        # Use volatility-based stop loss (2x ATR equivalent)
        volatility_stop = entry_price * (1 - volatility * 2)
        
        if support_level and support_level < entry_price:
            # Use support level if it's tighter
            return max(volatility_stop, support_level * 0.98)
        
        return volatility_stop
    
    def calculate_risk_reward_ratio(self, entry_price: float, target_price: float, stop_loss: float) -> float:
        """Calculate risk-reward ratio."""
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        if risk == 0:
            return 0.0
        
        return reward / risk
    
    def assess_risk(self, current_price: float, prices: List[float], 
                   technical_indicators: Technical_Indicators) -> Risk_Assessment:
        """Perform complete risk assessment."""
        volatility = self.calculate_volatility(prices)
        max_dd = self.calculate_max_drawdown(prices)
        
        # Determine support level from Bollinger Bands
        support_level = technical_indicators.bollinger_lower
        
        stop_loss = self.calculate_stop_loss(current_price, volatility / 100, support_level)
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        # Estimate target price (resistance or 2x risk)
        target_price = current_price + 2 * abs(current_price - stop_loss)
        if technical_indicators.bollinger_upper:
            target_price = min(target_price, technical_indicators.bollinger_upper)
        
        risk_reward = self.calculate_risk_reward_ratio(current_price, target_price, stop_loss)
        
        # Determine risk level
        if volatility < 0.15 and max_dd < 0.1:
            risk_level = "low"
        elif volatility < 0.30 and max_dd < 0.2:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return Risk_Assessment(
            position_size=position_size,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward,
            volatility=volatility,
            max_drawdown=max_dd,
            risk_level=risk_level
        )


class Signal_Generator:
    """Generates trading signals from technical and sentiment analysis."""
    
    def __init__(self):
        self.technical_weight = 0.6
        self.sentiment_weight = 0.4
    
    def generate_signal(self, technical_indicators: Technical_Indicators,
                       sentiment: Sentiment_Analysis,
                       current_price: float) -> Trading_Signal:
        """Generate trading signal."""
        technical_score = 0.0
        reasoning_parts = []
        
        # Technical analysis scoring
        if technical_indicators.trend == "up":
            technical_score += 0.3
            reasoning_parts.append("Uptrend detected")
        elif technical_indicators.trend == "down":
            technical_score -= 0.3
            reasoning_parts.append("Downtrend detected")
        
        if technical_indicators.rsi:
            if technical_indicators.rsi < 30:
                technical_score += 0.2
                reasoning_parts.append("Oversold conditions (RSI < 30)")
            elif technical_indicators.rsi > 70:
                technical_score -= 0.2
                reasoning_parts.append("Overbought conditions (RSI > 70)")
        
        if technical_indicators.macd and technical_indicators.macd_signal:
            if technical_indicators.macd > technical_indicators.macd_signal:
                technical_score += 0.2
                reasoning_parts.append("MACD bullish crossover")
            else:
                technical_score -= 0.2
                reasoning_parts.append("MACD bearish crossover")
        
        if technical_indicators.sma_20 and technical_indicators.sma_50:
            if technical_indicators.sma_20 > technical_indicators.sma_50:
                technical_score += 0.1
                reasoning_parts.append("Price above moving averages")
            else:
                technical_score -= 0.1
                reasoning_parts.append("Price below moving averages")
        
        # Sentiment scoring
        sentiment_score = sentiment.sentiment_score
        
        # Combine scores
        combined_score = (technical_score * self.technical_weight + 
                         sentiment_score * self.sentiment_weight)
        
        # Determine signal
        if combined_score > 0.3:
            signal_type = "BUY"
            strength = min(abs(combined_score), 1.0)
        elif combined_score < -0.3:
            signal_type = "SELL"
            strength = min(abs(combined_score), 1.0)
        else:
            signal_type = "HOLD"
            strength = 1.0 - abs(combined_score)
        
        confidence = abs(combined_score)
        
        reasoning = "; ".join(reasoning_parts)
        if sentiment.overall_sentiment != "neutral":
            reasoning += f"; Sentiment: {sentiment.overall_sentiment}"
        
        return Trading_Signal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            reasoning=reasoning
        )


class Portfolio_Tracker:
    """Tracks trading positions and portfolio performance."""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.account_balance = 100000.0
        self.initial_balance = 100000.0
    
    def add_position(self, asset: str, entry_price: float, quantity: float,
                    stop_loss: Optional[float] = None, target_price: Optional[float] = None):
        """Add a new position."""
        position = Position(
            asset=asset,
            entry_price=entry_price,
            quantity=quantity,
            current_price=entry_price,
            entry_date=datetime.now(),
            stop_loss=stop_loss,
            target_price=target_price
        )
        self.positions[asset] = position
    
    def update_position(self, asset: str, current_price: float):
        """Update position with current price."""
        if asset in self.positions:
            self.positions[asset].current_price = current_price
    
    def calculate_pnl(self, asset: str) -> float:
        """Calculate profit/loss for a position."""
        if asset not in self.positions:
            return 0.0
        
        position = self.positions[asset]
        pnl = (position.current_price - position.entry_price) * position.quantity
        return pnl
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total_value = self.account_balance
        
        for asset, position in self.positions.items():
            if asset in current_prices:
                position.current_price = current_prices[asset]
            total_value += position.current_price * position.quantity
        
        return total_value
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        summary = {
            "total_positions": len(self.positions),
            "positions": [],
            "total_pnl": 0.0
        }
        
        for asset, position in self.positions.items():
            pnl = self.calculate_pnl(asset)
            summary["positions"].append({
                "asset": asset,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "quantity": position.quantity,
                "pnl": pnl,
                "pnl_percent": (position.current_price - position.entry_price) / position.entry_price * 100
            })
            summary["total_pnl"] += pnl
        
        return summary


class Trading_Agent:
    """Main trading analysis agent."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.data_provider = Market_Data_Provider()
        self.technical_analyzer = Technical_Analyzer()
        self.sentiment_analyzer = News_Sentiment_Analyzer(client)
        self.risk_assessor = Risk_Assessor()
        self.signal_generator = Signal_Generator()
        self.portfolio_tracker = Portfolio_Tracker()
    
    def analyze_asset(self, symbol: str) -> Dict[str, Any]:
        """Perform complete analysis for an asset."""
        # Get market data
        ohlcv_data = self.data_provider.get_historical_data(symbol)
        current_price = ohlcv_data[-1].close
        closes = [d.close for d in ohlcv_data]
        
        # Technical analysis
        technical_indicators = self.technical_analyzer.analyze(ohlcv_data)
        
        # Sentiment analysis
        news_articles = self.sentiment_analyzer.generate_mock_news(symbol)
        sentiment = self.sentiment_analyzer.analyze_sentiment(news_articles)
        
        # Risk assessment
        risk_assessment = self.risk_assessor.assess_risk(current_price, closes, technical_indicators)
        
        # Generate signal
        signal = self.signal_generator.generate_signal(technical_indicators, sentiment, current_price)
        signal.entry_price = current_price
        signal.stop_loss = risk_assessment.stop_loss
        signal.target_price = current_price + 2 * abs(current_price - risk_assessment.stop_loss)
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "technical_indicators": technical_indicators,
            "sentiment": sentiment,
            "risk_assessment": risk_assessment,
            "signal": signal,
            "news_articles": news_articles
        }
    
    def get_market_overview(self, symbols: List[str]) -> Dict[str, Any]:
        """Get overview of multiple assets."""
        overview = {
            "analysis_date": datetime.now().isoformat(),
            "assets": []
        }
        
        for symbol in symbols:
            analysis = self.analyze_asset(symbol)
            overview["assets"].append({
                "symbol": symbol,
                "current_price": analysis["current_price"],
                "signal": analysis["signal"].signal_type,
                "signal_strength": analysis["signal"].strength,
                "trend": analysis["technical_indicators"].trend,
                "sentiment": analysis["sentiment"].overall_sentiment
            })
        
        return overview
    
    def generate_report(self, symbol: str) -> str:
        """Generate comprehensive analysis report."""
        analysis = self.analyze_asset(symbol)
        
        report = f"""
TRADING ANALYSIS REPORT
{'=' * 80}
Symbol: {symbol}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Current Price: ${analysis['current_price']:.2f}

TECHNICAL ANALYSIS
{'-' * 80}
Trend: {analysis['technical_indicators'].trend or 'N/A'}
SMA 20: ${analysis['technical_indicators'].sma_20:.2f if analysis['technical_indicators'].sma_20 else 'N/A'}
SMA 50: ${analysis['technical_indicators'].sma_50:.2f if analysis['technical_indicators'].sma_50 else 'N/A'}
RSI: {analysis['technical_indicators'].rsi:.2f if analysis['technical_indicators'].rsi else 'N/A'}
MACD: {analysis['technical_indicators'].macd:.4f if analysis['technical_indicators'].macd else 'N/A'}
Bollinger Upper: ${analysis['technical_indicators'].bollinger_upper:.2f if analysis['technical_indicators'].bollinger_upper else 'N/A'}
Bollinger Lower: ${analysis['technical_indicators'].bollinger_lower:.2f if analysis['technical_indicators'].bollinger_lower else 'N/A'}

SENTIMENT ANALYSIS
{'-' * 80}
Overall Sentiment: {analysis['sentiment'].overall_sentiment}
Sentiment Score: {analysis['sentiment'].sentiment_score:.2f}
Key Events: {', '.join(analysis['sentiment'].key_events[:3])}
News Summary: {analysis['sentiment'].news_summary[:200]}

RISK ASSESSMENT
{'-' * 80}
Risk Level: {analysis['risk_assessment'].risk_level}
Volatility: {analysis['risk_assessment'].volatility:.2%}
Max Drawdown: {analysis['risk_assessment'].max_drawdown:.2%}
Position Size: {analysis['risk_assessment'].position_size:.2f} shares
Stop Loss: ${analysis['risk_assessment'].stop_loss:.2f}
Risk/Reward Ratio: {analysis['risk_assessment'].risk_reward_ratio:.2f}

TRADING SIGNAL
{'-' * 80}
Signal: {analysis['signal'].signal_type}
Strength: {analysis['signal'].strength:.2f}
Confidence: {analysis['signal'].confidence:.2f}
Entry Price: ${analysis['signal'].entry_price:.2f}
Target Price: ${analysis['signal'].target_price:.2f}
Stop Loss: ${analysis['signal'].stop_loss:.2f}
Reasoning: {analysis['signal'].reasoning}

DISCLAIMER
{'-' * 80}
This analysis is for educational purposes only and does not constitute financial advice.
Trading involves substantial risk of loss. Past performance does not guarantee future results.
"""
        
        return report
    
    def backtest_strategy(self, symbol: str, days: int = 100) -> Dict[str, Any]:
        """Simple backtest of trading strategy."""
        ohlcv_data = self.data_provider.get_historical_data(symbol, days)
        
        signals_generated = 0
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        
        # Analyze each day
        for i in range(50, len(ohlcv_data)):  # Start after enough data for indicators
            historical_data = ohlcv_data[:i+1]
            closes = [d.close for d in historical_data]
            
            technical = self.technical_analyzer.analyze(historical_data)
            news = self.sentiment_analyzer.generate_mock_news(symbol)
            sentiment = self.sentiment_analyzer.analyze_sentiment(news)
            
            current_price = historical_data[-1].close
            signal = self.signal_generator.generate_signal(technical, sentiment, current_price)
            
            signals_generated += 1
            if signal.signal_type == "BUY":
                buy_signals += 1
            elif signal.signal_type == "SELL":
                sell_signals += 1
            else:
                hold_signals += 1
        
        return {
            "symbol": symbol,
            "period_days": days,
            "signals_generated": signals_generated,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "buy_percentage": buy_signals / signals_generated * 100 if signals_generated > 0 else 0,
            "sell_percentage": sell_signals / signals_generated * 100 if signals_generated > 0 else 0
        }


def main():
    """Main function demonstrating the trading analysis agent."""
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    client = OpenAI(api_key=api_key)
    agent = Trading_Agent(client)
    
    # Sample assets to analyze
    symbols = ["AAPL", "TSLA", "BTC-USD"]
    
    print("=" * 80)
    print("TRADING ANALYSIS AGENT - Sample Analysis")
    print("=" * 80)
    print("\nDISCLAIMER: This system is for educational purposes only.")
    print("It does not constitute financial advice. Trading involves risk.\n")
    
    # Analyze each asset
    for symbol in symbols:
        print(f"\n{'=' * 80}")
        print(f"Analyzing: {symbol}")
        print('=' * 80)
        
        report = agent.generate_report(symbol)
        print(report)
    
    # Market overview
    print("\n" + "=" * 80)
    print("MARKET OVERVIEW")
    print("=" * 80)
    overview = agent.get_market_overview(symbols)
    print(json.dumps(overview, indent=2))
    
    # Backtest example
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    backtest = agent.backtest_strategy(symbols[0], days=100)
    print(json.dumps(backtest, indent=2))
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
