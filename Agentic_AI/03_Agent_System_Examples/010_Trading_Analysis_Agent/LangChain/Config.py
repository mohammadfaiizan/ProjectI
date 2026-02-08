"""
Configuration module for Trading Analysis Agent.
Contains LLM configuration, market settings, analysis parameters, and risk management settings.
"""

from langchain_openai import ChatOpenAI
from typing import Optional, List, Dict
import os


class LLM_Config:
    """Configuration class for Language Model setup."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM configuration.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature for model generation (0.3 for balanced creativity)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (optional, for custom endpoints)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def Get_LLM(self) -> ChatOpenAI:
        """
        Create and return ChatOpenAI instance.
        
        Returns:
            Configured ChatOpenAI instance
        """
        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "api_key": self.api_key
        }
        
        if self.base_url:
            kwargs["base_url"] = self.base_url
            
        return ChatOpenAI(**kwargs)


class Market_Config:
    """Configuration class for market data and exchange settings."""
    
    def __init__(self):
        """Initialize market configuration with default settings."""
        self.supported_exchanges = [
            "NYSE",
            "NASDAQ",
            "AMEX",
            "LSE",
            "TSE",
            "HKEX"
        ]
        
        self.trading_hours = {
            "NYSE": {"open": "09:30", "close": "16:00", "timezone": "America/New_York"},
            "NASDAQ": {"open": "09:30", "close": "16:00", "timezone": "America/New_York"},
            "AMEX": {"open": "09:30", "close": "16:00", "timezone": "America/New_York"},
            "LSE": {"open": "08:00", "close": "16:30", "timezone": "Europe/London"},
            "TSE": {"open": "09:00", "close": "15:00", "timezone": "Asia/Tokyo"},
            "HKEX": {"open": "09:30", "close": "16:00", "timezone": "Asia/Hong_Kong"}
        }
        
        self.data_intervals = [
            "1min",
            "5min",
            "15min",
            "30min",
            "1hour",
            "4hour",
            "1day",
            "1week"
        ]
        
        self.default_interval = "1day"
        self.default_lookback_days = 30
    
    def Get_Supported_Exchanges(self) -> List[str]:
        """Return list of supported exchanges."""
        return self.supported_exchanges
    
    def Get_Trading_Hours(self, exchange: str) -> Dict[str, str]:
        """Return trading hours for a specific exchange."""
        return self.trading_hours.get(exchange, {})
    
    def Get_Data_Intervals(self) -> List[str]:
        """Return list of supported data intervals."""
        return self.data_intervals
    
    def Get_Default_Interval(self) -> str:
        """Return default data interval."""
        return self.default_interval
    
    def Get_Default_Lookback_Days(self) -> int:
        """Return default lookback period in days."""
        return self.default_lookback_days


class Analysis_Config:
    """Configuration class for technical analysis indicator parameters."""
    
    def __init__(self):
        """Initialize analysis configuration with default indicator periods."""
        self.sma_periods = [10, 20, 50, 200]
        self.ema_periods = [12, 26, 50]
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        
        self.volume_sma_period = 20
        
    def Get_SMA_Periods(self) -> List[int]:
        """Return list of SMA periods."""
        return self.sma_periods
    
    def Get_EMA_Periods(self) -> List[int]:
        """Return list of EMA periods."""
        return self.ema_periods
    
    def Get_RSI_Period(self) -> int:
        """Return RSI calculation period."""
        return self.rsi_period
    
    def Get_RSI_Overbought(self) -> float:
        """Return RSI overbought threshold."""
        return self.rsi_overbought
    
    def Get_RSI_Oversold(self) -> float:
        """Return RSI oversold threshold."""
        return self.rsi_oversold
    
    def Get_MACD_Params(self) -> Dict[str, int]:
        """Return MACD parameters."""
        return {
            "fast": self.macd_fast,
            "slow": self.macd_slow,
            "signal": self.macd_signal
        }
    
    def Get_Bollinger_Params(self) -> Dict[str, float]:
        """Return Bollinger Bands parameters."""
        return {
            "period": self.bollinger_period,
            "std": self.bollinger_std
        }
    
    def Get_Volume_SMA_Period(self) -> int:
        """Return volume SMA period."""
        return self.volume_sma_period


class Risk_Config:
    """Configuration class for risk management parameters."""
    
    def __init__(
        self,
        max_position_size: float = 10000.0,
        stop_loss_percentage: float = 2.0,
        risk_tolerance: str = "moderate"
    ):
        """
        Initialize risk configuration.
        
        Args:
            max_position_size: Maximum position size in base currency
            stop_loss_percentage: Stop loss percentage (e.g., 2.0 for 2%)
            risk_tolerance: Risk tolerance level (conservative, moderate, aggressive)
        """
        self.max_position_size = max_position_size
        self.stop_loss_percentage = stop_loss_percentage
        self.risk_tolerance = risk_tolerance.lower()
        
        self.risk_tolerance_levels = {
            "conservative": {
                "max_position_pct": 5.0,
                "stop_loss_pct": 1.5,
                "max_drawdown_pct": 10.0,
                "sharpe_min": 1.5
            },
            "moderate": {
                "max_position_pct": 10.0,
                "stop_loss_pct": 2.0,
                "max_drawdown_pct": 15.0,
                "sharpe_min": 1.0
            },
            "aggressive": {
                "max_position_pct": 20.0,
                "stop_loss_pct": 3.0,
                "max_drawdown_pct": 25.0,
                "sharpe_min": 0.5
            }
        }
    
    def Get_Max_Position_Size(self) -> float:
        """Return maximum position size."""
        return self.max_position_size
    
    def Get_Stop_Loss_Percentage(self) -> float:
        """Return stop loss percentage."""
        return self.stop_loss_percentage
    
    def Get_Risk_Tolerance(self) -> str:
        """Return risk tolerance level."""
        return self.risk_tolerance
    
    def Get_Risk_Parameters(self) -> Dict[str, float]:
        """Return risk parameters based on risk tolerance level."""
        return self.risk_tolerance_levels.get(
            self.risk_tolerance,
            self.risk_tolerance_levels["moderate"]
        )
    
    def Get_Max_Position_Percentage(self) -> float:
        """Return maximum position percentage of portfolio."""
        return self.Get_Risk_Parameters()["max_position_pct"]
    
    def Get_Max_Drawdown_Percentage(self) -> float:
        """Return maximum acceptable drawdown percentage."""
        return self.Get_Risk_Parameters()["max_drawdown_pct"]
    
    def Get_Minimum_Sharpe_Ratio(self) -> float:
        """Return minimum acceptable Sharpe ratio."""
        return self.Get_Risk_Parameters()["sharpe_min"]
