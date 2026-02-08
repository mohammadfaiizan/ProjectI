"""
Main module for Trading Analysis Agent.
Provides high-level interface for running trading analysis.
"""

from Agent import Trading_Analysis_Graph
from Config import LLM_Config, Market_Config, Analysis_Config, Risk_Config
from typing import List, Dict, Any, Optional
import sys


def Setup_Trading_System(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.3,
    risk_tolerance: str = "moderate"
) -> Trading_Analysis_Graph:
    """
    Setup and initialize the trading analysis system.
    
    Args:
        model_name: LLM model name to use
        temperature: Temperature for LLM generation
        risk_tolerance: Risk tolerance level (conservative, moderate, aggressive)
        
    Returns:
        Initialized Trading_Analysis_Graph instance
    """
    print("Initializing Trading Analysis System...")
    print(f"  Model: {model_name}")
    print(f"  Temperature: {temperature}")
    print(f"  Risk Tolerance: {risk_tolerance}")
    print()
    
    llm_config = LLM_Config(model_name=model_name, temperature=temperature)
    market_config = Market_Config()
    analysis_config = Analysis_Config()
    risk_config = Risk_Config(risk_tolerance=risk_tolerance)
    
    graph = Trading_Analysis_Graph(
        llm_config=llm_config,
        market_config=market_config,
        analysis_config=analysis_config,
        risk_config=risk_config
    )
    
    print("System initialized successfully!")
    print()
    return graph


def Analyze_Stock(symbol: str, graph: Optional[Trading_Analysis_Graph] = None) -> Dict[str, Any]:
    """
    Run full trading analysis for a single stock.
    
    Args:
        symbol: Stock symbol to analyze
        graph: Optional pre-initialized graph instance
        
    Returns:
        Dictionary containing analysis results
    """
    if graph is None:
        graph = Setup_Trading_System()
    
    print(f"Analyzing {symbol}...")
    print("-" * 80)
    
    results = graph.Analyze(symbol)
    
    # Print report
    if results.get("report"):
        print(results["report"])
    else:
        print("Analysis completed but report not generated.")
    
    return results


def Compare_Stocks(
    symbols: List[str],
    graph: Optional[Trading_Analysis_Graph] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple stocks and compare their signals.
    
    Args:
        symbols: List of stock symbols to analyze
        graph: Optional pre-initialized graph instance
        
    Returns:
        Dictionary mapping symbols to their analysis results
    """
    if graph is None:
        graph = Setup_Trading_System()
    
    print(f"Comparing {len(symbols)} stocks: {', '.join(symbols)}")
    print("=" * 80)
    print()
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING: {symbol}")
        print('=' * 80)
        results = graph.Analyze(symbol)
        all_results[symbol] = results
        
        if results.get("report"):
            print(results["report"])
        print()
    
    # Comparison summary
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Symbol':<10} {'Signal':<10} {'Confidence':<15} {'Current Price':<15}")
    print("-" * 80)
    
    for symbol, results in all_results.items():
        signal = results.get("signal", "N/A")
        confidence = results.get("confidence", 0.0)
        
        # Extract current price from indicators
        current_price = "N/A"
        try:
            import json
            indicators = json.loads(results.get("indicators", "{}"))
            current_price = f"${indicators.get('current_price', 'N/A')}"
        except:
            pass
        
        confidence_str = f"{confidence:.1%}" if confidence else "N/A"
        print(f"{symbol:<10} {signal:<10} {confidence_str:<15} {current_price:<15}")
    
    print()
    print("=" * 80)
    
    return all_results


def Run_Demo():
    """Run interactive demo where user enters stock symbol."""
    print("=" * 80)
    print("TRADING ANALYSIS AGENT - Interactive Demo")
    print("=" * 80)
    print()
    print("This demo analyzes stocks using technical indicators, sentiment,")
    print("and risk metrics to generate trading signals.")
    print()
    
    graph = Setup_Trading_System()
    
    while True:
        print("-" * 80)
        symbol = input("Enter stock symbol to analyze (or 'quit' to exit): ").strip().upper()
        
        if symbol.lower() in ['quit', 'exit', 'q']:
            print("Exiting demo. Goodbye!")
            break
        
        if not symbol:
            print("Please enter a valid stock symbol.")
            continue
        
        try:
            Analyze_Stock(symbol, graph)
        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted by user.")
            break
        except Exception as e:
            print(f"\nError analyzing {symbol}: {e}")
            print("Please try again with a different symbol.")
        
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode: analyze provided symbols
        symbols = [s.upper() for s in sys.argv[1:]]
        
        if len(symbols) == 1:
            Analyze_Stock(symbols[0])
        else:
            Compare_Stocks(symbols)
    else:
        # Interactive demo mode
        Run_Demo()
