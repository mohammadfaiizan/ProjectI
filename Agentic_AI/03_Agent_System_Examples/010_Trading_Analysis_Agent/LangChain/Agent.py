"""
Agent module for Trading Analysis Agent.
Contains the LangGraph state machine and trading analysis workflow.
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Dict, Any, Optional, List
import json

from Config import LLM_Config, Market_Config, Analysis_Config, Risk_Config
from Tools import (
    Fetch_Market_Data,
    Calculate_Technical_Indicators,
    Analyze_Sentiment,
    Calculate_Risk_Metrics
)


class Trading_State(TypedDict):
    """State dictionary for trading analysis workflow."""
    symbol: str
    market_data: Optional[str]
    indicators: Optional[str]
    sentiment: Optional[str]
    risk_metrics: Optional[str]
    analysis: Optional[str]
    signal: Optional[str]
    report: Optional[str]
    confidence: Optional[float]


class Trading_Analysis_Graph:
    """LangGraph-based trading analysis agent."""
    
    def __init__(
        self,
        llm_config: Optional[LLM_Config] = None,
        market_config: Optional[Market_Config] = None,
        analysis_config: Optional[Analysis_Config] = None,
        risk_config: Optional[Risk_Config] = None
    ):
        """
        Initialize trading analysis graph.
        
        Args:
            llm_config: LLM configuration instance
            market_config: Market configuration instance
            analysis_config: Analysis configuration instance
            risk_config: Risk configuration instance
        """
        self.llm_config = llm_config or LLM_Config()
        self.market_config = market_config or Market_Config()
        self.analysis_config = analysis_config or Analysis_Config()
        self.risk_config = risk_config or Risk_Config()
        
        self.llm = self.llm_config.Get_LLM()
        self.graph = self.Build_Graph()
    
    def Validate_Symbol(self, symbol: str) -> bool:
        """
        Validate stock symbol format.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        symbol_clean = symbol.strip().upper()
        
        # Basic validation: 1-5 uppercase letters/numbers
        if len(symbol_clean) < 1 or len(symbol_clean) > 5:
            return False
        
        if not symbol_clean.isalnum():
            return False
        
        return True
    
    def Validate_State(self, state: Trading_State, required_fields: List[str]) -> bool:
        """
        Validate that required fields exist in state.
        
        Args:
            state: Trading state to validate
            required_fields: List of required field names
            
        Returns:
            True if all required fields exist, False otherwise
        """
        for field in required_fields:
            if field not in state:
                return False
        return True
    
    def Build_Graph(self) -> StateGraph:
        """
        Build the LangGraph state machine with sequential nodes.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(Trading_State)
        
        # Add nodes
        workflow.add_node("Fetch_Data", self.Fetch_Data_Node)
        workflow.add_node("Technical_Analysis", self.Technical_Analysis_Node)
        workflow.add_node("Sentiment_Analysis", self.Sentiment_Analysis_Node)
        workflow.add_node("Risk_Assessment", self.Risk_Assessment_Node)
        workflow.add_node("Generate_Signal", self.Generate_Signal_Node)
        workflow.add_node("Create_Report", self.Create_Report_Node)
        
        # Set entry point
        workflow.set_entry_point("Fetch_Data")
        
        # Define sequential flow
        workflow.add_edge("Fetch_Data", "Technical_Analysis")
        workflow.add_edge("Technical_Analysis", "Sentiment_Analysis")
        workflow.add_edge("Sentiment_Analysis", "Risk_Assessment")
        workflow.add_edge("Risk_Assessment", "Generate_Signal")
        workflow.add_edge("Generate_Signal", "Create_Report")
        workflow.add_edge("Create_Report", END)
        
        return workflow.compile()
    
    def Fetch_Data_Node(self, state: Trading_State) -> Trading_State:
        """
        Fetch market data for the symbol.
        
        Args:
            state: Current trading state
            
        Returns:
            Updated state with market_data
        """
        symbol = state["symbol"]
        days = self.market_config.Get_Default_Lookback_Days()
        
        print(f"Fetching market data for {symbol}...")
        market_data = Fetch_Market_Data.invoke({"symbol": symbol, "days": days})
        
        state["market_data"] = market_data
        return state
    
    def Technical_Analysis_Node(self, state: Trading_State) -> Trading_State:
        """
        Calculate technical indicators from market data.
        
        Args:
            state: Current trading state
            
        Returns:
            Updated state with indicators
        """
        if not state.get("market_data"):
            state["indicators"] = json.dumps({"error": "No market data available"})
            return state
        
        print("Calculating technical indicators...")
        indicators = Calculate_Technical_Indicators.invoke({
            "data_json": state["market_data"]
        })
        
        state["indicators"] = indicators
        return state
    
    def Sentiment_Analysis_Node(self, state: Trading_State) -> Trading_State:
        """
        Analyze market sentiment for the symbol.
        
        Args:
            state: Current trading state
            
        Returns:
            Updated state with sentiment
        """
        symbol = state["symbol"]
        
        print("Analyzing market sentiment...")
        sentiment = Analyze_Sentiment.invoke({"symbol": symbol})
        
        state["sentiment"] = sentiment
        return state
    
    def Risk_Assessment_Node(self, state: Trading_State) -> Trading_State:
        """
        Calculate risk metrics from market data.
        
        Args:
            state: Current trading state
            
        Returns:
            Updated state with risk_metrics
        """
        if not state.get("market_data"):
            state["risk_metrics"] = json.dumps({"error": "No market data available"})
            return state
        
        position_size = self.risk_config.Get_Max_Position_Size()
        
        print("Assessing risk metrics...")
        risk_metrics = Calculate_Risk_Metrics.invoke({
            "data_json": state["market_data"],
            "position_size": position_size
        })
        
        state["risk_metrics"] = risk_metrics
        return state
    
    def Generate_Signal_Node(self, state: Trading_State) -> Trading_State:
        """
        Generate trading signal (BUY/SELL/HOLD) using LLM synthesis.
        
        Args:
            state: Current trading state
            
        Returns:
            Updated state with signal, analysis, and confidence
        """
        symbol = state["symbol"]
        indicators_json = state.get("indicators", "{}")
        sentiment_json = state.get("sentiment", "{}")
        risk_json = state.get("risk_metrics", "{}")
        
        # Parse JSON data
        try:
            indicators = json.loads(indicators_json)
            sentiment = json.loads(sentiment_json)
            risk = json.loads(risk_json)
        except json.JSONDecodeError:
            state["signal"] = "HOLD"
            state["analysis"] = "Error parsing data"
            state["confidence"] = 0.0
            return state
        
        # Build prompt for LLM
        system_prompt = """You are an expert trading analyst. Analyze the provided technical indicators, 
sentiment, and risk metrics to generate a trading signal. Consider:
1. Technical indicators (RSI, MACD, moving averages, Bollinger Bands)
2. Market sentiment (score and headlines)
3. Risk metrics (VaR, drawdown, Sharpe ratio)

Generate a signal: BUY, SELL, or HOLD, along with:
- Brief analysis (2-3 sentences)
- Confidence level (0.0 to 1.0)

Respond in JSON format:
{
    "signal": "BUY|SELL|HOLD",
    "analysis": "brief analysis text",
    "confidence": 0.75
}"""

        user_prompt = f"""Symbol: {symbol}

Technical Indicators:
{json.dumps(indicators, indent=2)}

Sentiment Analysis:
{json.dumps(sentiment, indent=2)}

Risk Metrics:
{json.dumps(risk, indent=2)}

Generate trading signal and analysis."""

        print("Generating trading signal...")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Try to extract JSON from response
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            signal_data = json.loads(response_text)
            
            state["signal"] = signal_data.get("signal", "HOLD")
            state["analysis"] = signal_data.get("analysis", "No analysis provided")
            state["confidence"] = float(signal_data.get("confidence", 0.5))
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            state["signal"] = "HOLD"
            state["analysis"] = f"Error in signal generation: {str(e)}"
            state["confidence"] = 0.0
        
        return state
    
    def Create_Report_Node(self, state: Trading_State) -> Trading_State:
        """
        Generate comprehensive analysis report.
        
        Args:
            state: Current trading state
            
        Returns:
            Updated state with report
        """
        symbol = state["symbol"]
        signal = state.get("signal", "HOLD")
        analysis = state.get("analysis", "")
        confidence = state.get("confidence", 0.0)
        
        # Parse all data for report
        try:
            market_data = json.loads(state.get("market_data", "{}"))
            indicators = json.loads(state.get("indicators", "{}"))
            sentiment = json.loads(state.get("sentiment", "{}"))
            risk = json.loads(state.get("risk_metrics", "{}"))
        except json.JSONDecodeError:
            state["report"] = "Error generating report: Invalid data format"
            return state
        
        print("Creating analysis report...")
        
        # Build comprehensive report
        report_parts = []
        report_parts.append("=" * 80)
        report_parts.append(f"TRADING ANALYSIS REPORT: {symbol}")
        report_parts.append("=" * 80)
        report_parts.append("")
        
        # Market Data Summary
        if "data" in market_data and len(market_data["data"]) > 0:
            latest = market_data["data"][-1]
            report_parts.append("MARKET DATA SUMMARY:")
            report_parts.append(f"  Current Price: ${latest.get('close', 'N/A')}")
            report_parts.append(f"  Date Range: {market_data['data'][0].get('date', 'N/A')} to {latest.get('date', 'N/A')}")
            report_parts.append(f"  Days Analyzed: {market_data.get('days', 'N/A')}")
            report_parts.append("")
        
        # Technical Indicators
        report_parts.append("TECHNICAL INDICATORS:")
        if "sma" in indicators:
            sma = indicators["sma"]
            report_parts.append(f"  SMA 10: ${sma.get('sma_10', 'N/A')}")
            report_parts.append(f"  SMA 20: ${sma.get('sma_20', 'N/A')}")
            report_parts.append(f"  SMA 50: ${sma.get('sma_50', 'N/A')}")
            report_parts.append(f"  SMA 200: ${sma.get('sma_200', 'N/A')}")
        
        if "rsi" in indicators and indicators["rsi"] is not None:
            rsi = indicators["rsi"]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            report_parts.append(f"  RSI: {rsi} ({rsi_status})")
        
        if "macd" in indicators:
            macd = indicators["macd"]
            report_parts.append(f"  MACD Line: {macd.get('macd_line', 'N/A')}")
            report_parts.append(f"  Signal Line: {macd.get('signal_line', 'N/A')}")
            report_parts.append(f"  Histogram: {macd.get('histogram', 'N/A')}")
        
        if "bollinger_bands" in indicators:
            bb = indicators["bollinger_bands"]
            report_parts.append(f"  Bollinger Upper: ${bb.get('upper', 'N/A')}")
            report_parts.append(f"  Bollinger Middle: ${bb.get('middle', 'N/A')}")
            report_parts.append(f"  Bollinger Lower: ${bb.get('lower', 'N/A')}")
        
        report_parts.append("")
        
        # Sentiment Analysis
        report_parts.append("SENTIMENT ANALYSIS:")
        if "sentiment_score" in sentiment:
            score = sentiment["sentiment_score"]
            interpretation = sentiment.get("interpretation", "Neutral")
            report_parts.append(f"  Sentiment Score: {score} ({interpretation})")
        
        if "headlines" in sentiment:
            report_parts.append("  Key Headlines:")
            for headline in sentiment["headlines"][:3]:
                report_parts.append(f"    - {headline}")
        
        report_parts.append("")
        
        # Risk Metrics
        report_parts.append("RISK METRICS:")
        if "var_95" in risk:
            report_parts.append(f"  Value at Risk (95%): ${risk.get('var_95', 'N/A')} ({risk.get('var_95_pct', 'N/A')}%)")
        
        if "max_drawdown" in risk:
            report_parts.append(f"  Maximum Drawdown: {risk.get('max_drawdown', 'N/A')}%")
        
        if "sharpe_ratio" in risk:
            report_parts.append(f"  Sharpe Ratio: {risk.get('sharpe_ratio', 'N/A')}")
        
        if "volatility" in risk:
            report_parts.append(f"  Annualized Volatility: {risk.get('volatility', 'N/A')}%")
        
        report_parts.append("")
        
        # Trading Signal
        report_parts.append("TRADING SIGNAL:")
        report_parts.append(f"  Signal: {signal}")
        report_parts.append(f"  Confidence: {confidence:.1%}")
        report_parts.append("")
        report_parts.append("ANALYSIS:")
        report_parts.append(f"  {analysis}")
        report_parts.append("")
        report_parts.append("=" * 80)
        
        report = "\n".join(report_parts)
        state["report"] = report
        
        return state
    
    def Analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Run complete trading analysis for a symbol.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Dictionary containing final state with all analysis results
            
        Raises:
            ValueError: If symbol is invalid
        """
        # Validate symbol
        if not self.Validate_Symbol(symbol):
            raise ValueError(f"Invalid stock symbol: {symbol}")
        
        symbol_upper = symbol.upper()
        
        initial_state: Trading_State = {
            "symbol": symbol_upper,
            "market_data": None,
            "indicators": None,
            "sentiment": None,
            "risk_metrics": None,
            "analysis": None,
            "signal": None,
            "report": None,
            "confidence": None
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Validate final state has required fields
            if not self.Validate_State(final_state, ["symbol", "signal"]):
                print("Warning: Final state missing required fields")
            
            return final_state
        
        except Exception as e:
            print(f"Error during analysis: {e}")
            # Return partial state with error information
            initial_state["error"] = str(e)
            initial_state["signal"] = "ERROR"
            return initial_state
