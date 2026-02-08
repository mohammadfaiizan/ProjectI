"""
Sample Input module for Trading Analysis Agent.
Contains predefined stock symbols with context and analysis scenarios for testing.
"""

from Main import Setup_Trading_System, Analyze_Stock, Compare_Stocks
from typing import List, Dict, Any
import json


# Predefined stock symbols with mock context
STOCK_SYMBOLS = [
    {
        "symbol": "AAPL",
        "context": "Post-earnings, strong iPhone sales"
    },
    {
        "symbol": "GOOGL",
        "context": "AI competition heating up, cloud revenue growing"
    },
    {
        "symbol": "TSLA",
        "context": "New model launch, volatile trading"
    }
]


# Analysis scenarios for different market conditions
ANALYSIS_SCENARIOS = [
    {
        "name": "Bullish Market",
        "description": "Analyzing stocks in a bullish market environment",
        "risk_tolerance": "moderate"
    },
    {
        "name": "Bearish Market",
        "description": "Analyzing stocks in a bearish market environment",
        "risk_tolerance": "conservative"
    },
    {
        "name": "Sideways Market",
        "description": "Analyzing stocks in a sideways/range-bound market",
        "risk_tolerance": "moderate"
    }
]


def Run_Samples():
    """
    Run sample analysis on predefined stocks.
    Analyzes all stocks, compares signals, and prints comprehensive reports.
    """
    print("=" * 80)
    print("TRADING ANALYSIS AGENT - Sample Analysis")
    print("=" * 80)
    print()
    print("This sample analyzes predefined stocks with different market contexts.")
    print()
    
    # Extract symbols from STOCK_SYMBOLS
    symbols = [stock["symbol"] for stock in STOCK_SYMBOLS]
    
    # Display stock contexts
    print("STOCK CONTEXTS:")
    print("-" * 80)
    for stock in STOCK_SYMBOLS:
        print(f"  {stock['symbol']}: {stock['context']}")
    print()
    
    # Setup system with moderate risk tolerance
    graph = Setup_Trading_System(risk_tolerance="moderate")
    
    # Analyze all stocks
    print("=" * 80)
    print("RUNNING ANALYSIS ON ALL STOCKS")
    print("=" * 80)
    print()
    
    all_results = {}
    
    for stock in STOCK_SYMBOLS:
        symbol = stock["symbol"]
        context = stock["context"]
        
        print(f"\n{'=' * 80}")
        print(f"ANALYZING: {symbol} - {context}")
        print('=' * 80)
        
        try:
            results = graph.Analyze(symbol)
            all_results[symbol] = {
                "results": results,
                "context": context
            }
            
            if results.get("report"):
                print(results["report"])
            else:
                print(f"Analysis completed for {symbol} but report not generated.")
        
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            all_results[symbol] = {
                "results": None,
                "context": context,
                "error": str(e)
            }
        
        print()
    
    # Detailed comparison
    print("=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    print()
    
    # Signal comparison table
    print("SIGNAL COMPARISON:")
    print("-" * 80)
    print(f"{'Symbol':<10} {'Context':<40} {'Signal':<10} {'Confidence':<15}")
    print("-" * 80)
    
    for stock in STOCK_SYMBOLS:
        symbol = stock["symbol"]
        context = stock["context"]
        
        if symbol in all_results and all_results[symbol]["results"]:
            results = all_results[symbol]["results"]
            signal = results.get("signal", "N/A")
            confidence = results.get("confidence", 0.0)
            confidence_str = f"{confidence:.1%}" if confidence else "N/A"
            
            print(f"{symbol:<10} {context:<40} {signal:<10} {confidence_str:<15}")
        else:
            print(f"{symbol:<10} {context:<40} {'ERROR':<10} {'N/A':<15}")
    
    print()
    
    # Technical indicators comparison
    print("TECHNICAL INDICATORS COMPARISON:")
    print("-" * 80)
    
    for stock in STOCK_SYMBOLS:
        symbol = stock["symbol"]
        
        if symbol in all_results and all_results[symbol]["results"]:
            results = all_results[symbol]["results"]
            
            try:
                indicators = json.loads(results.get("indicators", "{}"))
                current_price = indicators.get("current_price", "N/A")
                rsi = indicators.get("rsi", "N/A")
                
                print(f"\n{symbol}:")
                print(f"  Current Price: ${current_price}")
                if rsi != "N/A" and rsi is not None:
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    print(f"  RSI: {rsi} ({rsi_status})")
                else:
                    print(f"  RSI: {rsi}")
                
                if "macd" in indicators:
                    macd = indicators["macd"]
                    macd_line = macd.get("macd_line", "N/A")
                    signal_line = macd.get("signal_line", "N/A")
                    histogram = macd.get("histogram", "N/A")
                    
                    print(f"  MACD Line: {macd_line}")
                    print(f"  Signal Line: {signal_line}")
                    if histogram != "N/A" and histogram is not None:
                        macd_signal = "Bullish" if histogram > 0 else "Bearish"
                        print(f"  Histogram: {histogram} ({macd_signal})")
            
            except Exception as e:
                print(f"\n{symbol}: Error parsing indicators - {e}")
    
    print()
    
    # Risk metrics comparison
    print("RISK METRICS COMPARISON:")
    print("-" * 80)
    
    for stock in STOCK_SYMBOLS:
        symbol = stock["symbol"]
        
        if symbol in all_results and all_results[symbol]["results"]:
            results = all_results[symbol]["results"]
            
            try:
                risk = json.loads(results.get("risk_metrics", "{}"))
                
                print(f"\n{symbol}:")
                if "var_95_pct" in risk:
                    print(f"  VaR (95%): {risk.get('var_95_pct', 'N/A')}%")
                if "max_drawdown" in risk:
                    print(f"  Max Drawdown: {risk.get('max_drawdown', 'N/A')}%")
                if "sharpe_ratio" in risk:
                    sharpe = risk.get("sharpe_ratio", "N/A")
                    print(f"  Sharpe Ratio: {sharpe}")
                if "volatility" in risk:
                    print(f"  Volatility: {risk.get('volatility', 'N/A')}%")
            
            except Exception as e:
                print(f"\n{symbol}: Error parsing risk metrics - {e}")
    
    print()
    
    # Sentiment comparison
    print("SENTIMENT COMPARISON:")
    print("-" * 80)
    
    for stock in STOCK_SYMBOLS:
        symbol = stock["symbol"]
        
        if symbol in all_results and all_results[symbol]["results"]:
            results = all_results[symbol]["results"]
            
            try:
                sentiment = json.loads(results.get("sentiment", "{}"))
                
                print(f"\n{symbol}:")
                if "sentiment_score" in sentiment:
                    score = sentiment["sentiment_score"]
                    interpretation = sentiment.get("interpretation", "Neutral")
                    print(f"  Sentiment Score: {score} ({interpretation})")
                
                if "headlines" in sentiment:
                    print(f"  Key Headlines:")
                    for headline in sentiment["headlines"][:2]:
                        print(f"    - {headline}")
            
            except Exception as e:
                print(f"\n{symbol}: Error parsing sentiment - {e}")
    
    print()
    
    # Final recommendations
    print("=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    buy_signals = []
    sell_signals = []
    hold_signals = []
    
    for stock in STOCK_SYMBOLS:
        symbol = stock["symbol"]
        
        if symbol in all_results and all_results[symbol]["results"]:
            results = all_results[symbol]["results"]
            signal = results.get("signal", "HOLD")
            confidence = results.get("confidence", 0.0)
            
            if signal == "BUY":
                buy_signals.append((symbol, confidence))
            elif signal == "SELL":
                sell_signals.append((symbol, confidence))
            else:
                hold_signals.append((symbol, confidence))
    
    if buy_signals:
        print("BUY SIGNALS:")
        for symbol, confidence in sorted(buy_signals, key=lambda x: x[1], reverse=True):
            print(f"  {symbol}: {confidence:.1%} confidence")
        print()
    
    if sell_signals:
        print("SELL SIGNALS:")
        for symbol, confidence in sorted(sell_signals, key=lambda x: x[1], reverse=True):
            print(f"  {symbol}: {confidence:.1%} confidence")
        print()
    
    if hold_signals:
        print("HOLD SIGNALS:")
        for symbol, confidence in sorted(hold_signals, key=lambda x: x[1], reverse=True):
            print(f"  {symbol}: {confidence:.1%} confidence")
        print()
    
    print("=" * 80)
    print("Sample analysis completed!")
    print("=" * 80)
    
    return all_results


def Run_Scenario_Analysis(scenario_name: str):
    """
    Run analysis for a specific market scenario.
    
    Args:
        scenario_name: Name of the scenario to run
    """
    scenario = next(
        (s for s in ANALYSIS_SCENARIOS if s["name"] == scenario_name),
        None
    )
    
    if not scenario:
        print(f"Scenario '{scenario_name}' not found.")
        return
    
    print("=" * 80)
    print(f"SCENARIO ANALYSIS: {scenario['name']}")
    print("=" * 80)
    print(f"Description: {scenario['description']}")
    print(f"Risk Tolerance: {scenario['risk_tolerance']}")
    print()
    
    graph = Setup_Trading_System(risk_tolerance=scenario["risk_tolerance"])
    symbols = [stock["symbol"] for stock in STOCK_SYMBOLS]
    
    Compare_Stocks(symbols, graph)


if __name__ == "__main__":
    print("Running sample analysis with predefined stocks...")
    print()
    
    Run_Samples()
    
    print()
    print("=" * 80)
    print("You can also run scenario-specific analysis:")
    print("  - Bullish Market")
    print("  - Bearish Market")
    print("  - Sideways Market")
    print("=" * 80)
