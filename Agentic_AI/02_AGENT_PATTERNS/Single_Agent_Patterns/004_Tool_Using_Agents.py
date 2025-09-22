#!/usr/bin/env python3
"""
Tool Using Agents: Extending AI with External Capabilities
========================================================

WHAT IS THE PROBLEM?
==================
Imagine you have a brilliant assistant, but they're stuck in a room with no internet, no calculator, no phone, and no books. They can think really well, but they can't:

❌ Look up current information (like today's weather)
❌ Do complex calculations (like calculating compound interest)  
❌ Send emails or messages
❌ Access databases or files
❌ Use specialized software

This is like most AI systems - they're smart but limited!

REAL WORLD EXAMPLE:
=================
User: "What's the weather in Tokyo right now, and if I invest $10,000 at 5% compound interest for 10 years, how much will I have?"

❌ LIMITED AI: 
"I can't check current weather or do complex calculations"

✅ TOOL-USING AI:
"Let me help you with both questions!

Step 1: Checking Tokyo weather...
[Uses Weather API Tool]
🌤️ Current weather in Tokyo: 22°C, partly cloudy, humidity 65%

Step 2: Calculating compound interest...  
[Uses Financial Calculator Tool]
🧮 $10,000 at 5% for 10 years = $16,288.95

Complete Answer: Tokyo is currently 22°C and partly cloudy. Your $10,000 investment would grow to $16,288.95 in 10 years."

THE ALGORITHM:
=============
1. ANALYZE the user's request
2. IDENTIFY what tools are needed
3. SELECT the appropriate tools  
4. EXECUTE tools in the right order
5. COMBINE results into coherent answer

PSEUDO CODE:
===========
def solve_with_tools(user_request):
    # Analyze what's needed
    required_capabilities = analyze_request(user_request)
    
    # Find matching tools
    available_tools = find_tools(required_capabilities)
    
    # Execute tools
    results = []
    for tool in available_tools:
        if tool.can_handle(user_request):
            result = tool.execute(extract_parameters(user_request, tool))
            results.append(result)
    
    # Combine and present results
    final_answer = synthesize_results(results, user_request)
    return final_answer

WHY IS THIS REVOLUTIONARY?
========================
- AI can access real-time information (weather, news, stock prices)
- AI can perform specialized calculations (financial, scientific, engineering)
- AI can interact with systems (send emails, update databases, control devices)
- AI can use domain-specific tools (image editors, code compilers, data analyzers)
- AI becomes truly useful for real-world tasks
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

class ToolCategory(Enum):
    INFORMATION = "information"      # Web search, weather, news
    CALCULATION = "calculation"     # Math, financial, scientific
    COMMUNICATION = "communication" # Email, messaging, notifications
    DATA_PROCESSING = "data_processing" # File operations, data analysis
    CREATIVE = "creative"           # Image generation, text writing
    AUTOMATION = "automation"       # System control, task automation

@dataclass
class ToolResult:
    """Result from executing a tool"""
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None

class BaseTool(ABC):
    """Base class for all tools that agents can use"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get tool name"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get tool description"""
        pass
    
    @abstractmethod
    def get_category(self) -> ToolCategory:
        """Get tool category"""
        pass
    
    @abstractmethod
    async def execute(self, **parameters) -> Any:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def can_handle(self, request: str) -> bool:
        """Check if this tool can handle the request"""
        pass

class WeatherTool(BaseTool):
    """Tool for getting weather information"""
    
    def get_name(self) -> str:
        return "Weather Lookup"
    
    def get_description(self) -> str:
        return "Get current weather conditions for any city worldwide"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.INFORMATION
    
    async def execute(self, city: str) -> Dict[str, Any]:
        """Get weather for a city (simulated)"""
        # Simulate API call delay
        await asyncio.sleep(0.2)
        
        # Simulated weather data
        weather_db = {
            "tokyo": {"temp": 22, "condition": "partly cloudy", "humidity": 65, "wind": "10 km/h"},
            "new york": {"temp": 18, "condition": "sunny", "humidity": 45, "wind": "15 km/h"},
            "london": {"temp": 15, "condition": "rainy", "humidity": 80, "wind": "20 km/h"},
            "paris": {"temp": 20, "condition": "overcast", "humidity": 70, "wind": "12 km/h"},
            "sydney": {"temp": 25, "condition": "sunny", "humidity": 55, "wind": "8 km/h"}
        }
        
        city_key = city.lower()
        if city_key in weather_db:
            data = weather_db[city_key]
            return {
                "city": city.title(),
                "temperature": f"{data['temp']}°C",
                "condition": data["condition"],
                "humidity": f"{data['humidity']}%",
                "wind": data["wind"],
                "status": "success"
            }
        else:
            return {
                "city": city.title(),
                "status": "not_found",
                "message": f"Weather data not available for {city}"
            }
    
    def can_handle(self, request: str) -> bool:
        weather_keywords = ["weather", "temperature", "rain", "sunny", "cloudy", "forecast"]
        return any(keyword in request.lower() for keyword in weather_keywords)

class CalculatorTool(BaseTool):
    """Tool for mathematical calculations"""
    
    def get_name(self) -> str:
        return "Advanced Calculator"
    
    def get_description(self) -> str:
        return "Perform mathematical calculations including compound interest, percentages, and complex equations"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.CALCULATION
    
    async def execute(self, operation: str, **parameters) -> Dict[str, Any]:
        """Perform calculation based on operation type"""
        await asyncio.sleep(0.1)  # Simulate calculation time
        
        try:
            if operation == "compound_interest":
                principal = parameters.get("principal", 0)
                rate = parameters.get("rate", 0) / 100  # Convert percentage
                time = parameters.get("time", 0)
                compounds_per_year = parameters.get("compounds_per_year", 1)
                
                # A = P(1 + r/n)^(nt)
                amount = principal * (1 + rate/compounds_per_year) ** (compounds_per_year * time)
                interest_earned = amount - principal
                
                return {
                    "operation": "Compound Interest",
                    "principal": f"${principal:,.2f}",
                    "rate": f"{parameters.get('rate')}% annually",
                    "time": f"{time} years",
                    "final_amount": f"${amount:,.2f}",
                    "interest_earned": f"${interest_earned:,.2f}",
                    "status": "success"
                }
            
            elif operation == "percentage":
                percentage = parameters.get("percentage", 0)
                of_value = parameters.get("of_value", 0)
                result = (percentage / 100) * of_value
                
                return {
                    "operation": "Percentage Calculation", 
                    "calculation": f"{percentage}% of {of_value}",
                    "result": result,
                    "status": "success"
                }
            
            elif operation == "basic_math":
                expression = parameters.get("expression", "")
                # Safe evaluation for basic math
                allowed_chars = set('0123456789+-*/.() ')
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                    return {
                        "operation": "Basic Math",
                        "expression": expression,
                        "result": result,
                        "status": "success"
                    }
                else:
                    return {"status": "error", "message": "Invalid mathematical expression"}
            
            else:
                return {"status": "error", "message": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Calculation error: {str(e)}"}
    
    def can_handle(self, request: str) -> bool:
        calc_keywords = ["calculate", "math", "interest", "percentage", "+", "-", "*", "/", "="]
        return any(keyword in request.lower() for keyword in calc_keywords)

class EmailTool(BaseTool):
    """Tool for sending emails"""
    
    def get_name(self) -> str:
        return "Email Sender"
    
    def get_description(self) -> str:
        return "Send emails to specified recipients with custom subject and content"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.COMMUNICATION
    
    async def execute(self, to: str, subject: str, body: str, cc: List[str] = None) -> Dict[str, Any]:
        """Send email (simulated)"""
        await asyncio.sleep(0.3)  # Simulate email sending
        
        # Basic email validation
        if "@" not in to:
            return {"status": "error", "message": "Invalid email address"}
        
        # Simulate successful email sending
        return {
            "status": "success",
            "message": f"Email sent successfully to {to}",
            "to": to,
            "subject": subject,
            "cc": cc or [],
            "sent_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def can_handle(self, request: str) -> bool:
        email_keywords = ["email", "send", "message", "notify", "mail"]
        return any(keyword in request.lower() for keyword in email_keywords)

class WebSearchTool(BaseTool):
    """Tool for web search"""
    
    def get_name(self) -> str:
        return "Web Search"
    
    def get_description(self) -> str:
        return "Search the internet for current information on any topic"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.INFORMATION
    
    async def execute(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Perform web search (simulated)"""
        await asyncio.sleep(0.5)  # Simulate search time
        
        # Simulated search results
        knowledge_base = {
            "python programming": [
                {"title": "Python.org - Official Website", "url": "python.org", "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively."},
                {"title": "Python Tutorial for Beginners", "url": "tutorial.python.org", "snippet": "Learn Python programming from scratch with examples and exercises."}
            ],
            "artificial intelligence": [
                {"title": "What is Artificial Intelligence?", "url": "ai-guide.com", "snippet": "AI refers to the simulation of human intelligence in machines programmed to think and learn."},
                {"title": "AI Applications in 2024", "url": "ai-news.org", "snippet": "Current applications of AI include healthcare, finance, transportation, and education."}
            ],
            "machine learning": [
                {"title": "Machine Learning Explained", "url": "ml-basics.edu", "snippet": "Machine learning is a subset of AI that enables computers to learn without explicit programming."},
                {"title": "ML Algorithms Guide", "url": "algorithms.ml", "snippet": "Comprehensive guide to machine learning algorithms and their applications."}
            ]
        }
        
        # Find relevant results
        query_lower = query.lower()
        results = []
        
        for topic, articles in knowledge_base.items():
            if any(word in query_lower for word in topic.split()):
                results.extend(articles[:max_results])
        
        if not results:
            results = [{"title": "General Search Result", "url": "search-engine.com", "snippet": f"Search results for: {query}"}]
        
        return {
            "query": query,
            "results": results[:max_results],
            "total_found": len(results),
            "status": "success"
        }
    
    def can_handle(self, request: str) -> bool:
        search_keywords = ["search", "find", "look up", "information about", "what is", "tell me about"]
        return any(keyword in request.lower() for keyword in search_keywords)

class ToolUsingAgent:
    """
    An AI agent that can use external tools to solve complex problems
    
    EXAMPLE USAGE:
    =============
    agent = ToolUsingAgent()
    agent.add_tool(WeatherTool())
    agent.add_tool(CalculatorTool())
    
    result = await agent.process_request("What's the weather in Tokyo and calculate 15% of 1000?")
    
    This will:
    1. Identify that it needs weather and calculation tools
    2. Use WeatherTool to get Tokyo weather
    3. Use CalculatorTool to calculate 15% of 1000  
    4. Combine results into a coherent answer
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.usage_stats: Dict[str, int] = {}
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent's toolkit"""
        self.tools[tool.get_name()] = tool
        self.usage_stats[tool.get_name()] = 0
        print(f"🔧 Added tool: {tool.get_name()} ({tool.get_category().value})")
    
    def list_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools"""
        return [
            {
                "name": tool.get_name(),
                "description": tool.get_description(),
                "category": tool.get_category().value
            }
            for tool in self.tools.values()
        ]
    
    async def process_request(self, request: str) -> Dict[str, Any]:
        """
        Process a user request using available tools
        
        This is the main method that:
        1. Analyzes the request
        2. Identifies needed tools
        3. Executes tools
        4. Combines results
        """
        print(f"\n🎯 REQUEST: {request}")
        print("=" * 60)
        
        # Step 1: Analyze request and identify needed tools
        needed_tools = self.identify_needed_tools(request)
        print(f"🔍 IDENTIFIED TOOLS NEEDED: {[tool.get_name() for tool in needed_tools]}")
        
        if not needed_tools:
            return {
                "request": request,
                "response": "I don't have the right tools to handle this request.",
                "tools_used": [],
                "success": False
            }
        
        # Step 2: Execute tools
        tool_results = []
        for tool in needed_tools:
            print(f"\n🔧 USING TOOL: {tool.get_name()}")
            
            try:
                # Extract parameters for this tool
                parameters = self.extract_tool_parameters(tool, request)
                print(f"   Parameters: {parameters}")
                
                # Execute tool
                start_time = time.time()
                result = await tool.execute(**parameters)
                execution_time = time.time() - start_time
                
                tool_result = ToolResult(
                    tool_name=tool.get_name(),
                    success=True,
                    result=result,
                    execution_time=execution_time
                )
                
                tool_results.append(tool_result)
                self.usage_stats[tool.get_name()] += 1
                
                print(f"   ✅ Success: {result}")
                
            except Exception as e:
                tool_result = ToolResult(
                    tool_name=tool.get_name(),
                    success=False,
                    result=None,
                    execution_time=0,
                    error_message=str(e)
                )
                tool_results.append(tool_result)
                print(f"   ❌ Error: {str(e)}")
        
        # Step 3: Synthesize final response
        final_response = self.synthesize_response(request, tool_results)
        
        return {
            "request": request,
            "response": final_response,
            "tools_used": [r.tool_name for r in tool_results],
            "tool_results": tool_results,
            "success": any(r.success for r in tool_results)
        }
    
    def identify_needed_tools(self, request: str) -> List[BaseTool]:
        """Identify which tools can handle this request"""
        needed_tools = []
        
        for tool in self.tools.values():
            if tool.can_handle(request):
                needed_tools.append(tool)
        
        return needed_tools
    
    def extract_tool_parameters(self, tool: BaseTool, request: str) -> Dict[str, Any]:
        """Extract parameters needed for a specific tool"""
        import re
        
        if isinstance(tool, WeatherTool):
            # Extract city name
            cities = ["tokyo", "new york", "london", "paris", "sydney", "chicago", "berlin", "mumbai"]
            for city in cities:
                if city in request.lower():
                    return {"city": city}
            
            # If no specific city found, try to extract from "weather in X"
            match = re.search(r'weather in (\w+)', request.lower())
            if match:
                return {"city": match.group(1)}
            
            return {"city": "tokyo"}  # Default
        
        elif isinstance(tool, CalculatorTool):
            # Check for compound interest calculation
            if "compound interest" in request.lower() or "invest" in request.lower():
                # Extract numbers for compound interest
                numbers = re.findall(r'\$?(\d+(?:,\d+)*(?:\.\d+)?)', request)
                percentages = re.findall(r'(\d+(?:\.\d+)?)%', request)
                years = re.findall(r'(\d+)\s*years?', request)
                
                if numbers and percentages and years:
                    principal = float(numbers[0].replace(',', ''))
                    rate = float(percentages[0])
                    time = int(years[0])
                    
                    return {
                        "operation": "compound_interest",
                        "principal": principal,
                        "rate": rate,
                        "time": time,
                        "compounds_per_year": 1
                    }
            
            # Check for percentage calculation
            elif "%" in request:
                percentages = re.findall(r'(\d+(?:\.\d+)?)%', request)
                numbers = re.findall(r'(\d+(?:\.\d+)?)', request)
                
                if percentages and numbers:
                    return {
                        "operation": "percentage",
                        "percentage": float(percentages[0]),
                        "of_value": float(numbers[-1])  # Last number is usually the base
                    }
            
            # Basic math expression
            else:
                # Extract mathematical expression
                math_expr = re.search(r'[\d\+\-\*/\(\)\s]+', request)
                if math_expr:
                    return {
                        "operation": "basic_math",
                        "expression": math_expr.group().strip()
                    }
            
            return {"operation": "basic_math", "expression": "1+1"}
        
        elif isinstance(tool, EmailTool):
            # Extract email parameters
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', request)
            
            return {
                "to": emails[0] if emails else "example@example.com",
                "subject": f"Regarding: {request[:50]}...",
                "body": f"This email is about: {request}"
            }
        
        elif isinstance(tool, WebSearchTool):
            # Extract search query
            # Remove command words and focus on the topic
            stop_words = {"search", "find", "look", "up", "information", "about", "what", "is", "tell", "me"}
            words = request.lower().split()
            search_words = [w for w in words if w not in stop_words and len(w) > 2]
            
            query = " ".join(search_words[:4])  # Top 4 relevant words
            return {"query": query if query else request}
        
        return {}
    
    def synthesize_response(self, request: str, tool_results: List[ToolResult]) -> str:
        """Combine tool results into a coherent response"""
        if not tool_results:
            return "I couldn't process your request with the available tools."
        
        successful_results = [r for r in tool_results if r.success]
        
        if not successful_results:
            return "I encountered errors while trying to help you with this request."
        
        response_parts = []
        
        for result in successful_results:
            if result.tool_name == "Weather Lookup":
                weather_data = result.result
                if weather_data.get("status") == "success":
                    response_parts.append(
                        f"🌤️ Weather in {weather_data['city']}: {weather_data['temperature']}, "
                        f"{weather_data['condition']}, humidity {weather_data['humidity']}"
                    )
            
            elif result.tool_name == "Advanced Calculator":
                calc_data = result.result
                if calc_data.get("status") == "success":
                    if "compound_interest" in calc_data.get("operation", "").lower():
                        response_parts.append(
                            f"💰 Investment calculation: {calc_data['principal']} at {calc_data['rate']} "
                            f"for {calc_data['time']} grows to {calc_data['final_amount']} "
                            f"(earning {calc_data['interest_earned']} in interest)"
                        )
                    elif "percentage" in calc_data.get("operation", "").lower():
                        response_parts.append(f"🧮 Calculation: {calc_data['calculation']} = {calc_data['result']}")
                    else:
                        response_parts.append(f"🧮 {calc_data['expression']} = {calc_data['result']}")
            
            elif result.tool_name == "Email Sender":
                email_data = result.result
                if email_data.get("status") == "success":
                    response_parts.append(f"📧 {email_data['message']}")
            
            elif result.tool_name == "Web Search":
                search_data = result.result
                if search_data.get("status") == "success":
                    results_summary = f"🔍 Found {search_data['total_found']} results for '{search_data['query']}'"
                    top_result = search_data['results'][0] if search_data['results'] else None
                    if top_result:
                        results_summary += f": {top_result['snippet']}"
                    response_parts.append(results_summary)
        
        if response_parts:
            return "\n\n".join(response_parts)
        else:
            return "I processed your request but couldn't generate a meaningful response."
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get statistics about tool usage"""
        total_uses = sum(self.usage_stats.values())
        
        return {
            "total_tool_uses": total_uses,
            "tools_available": len(self.tools),
            "usage_by_tool": self.usage_stats.copy(),
            "most_used_tool": max(self.usage_stats.items(), key=lambda x: x[1])[0] if self.usage_stats else None
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_weather_and_calculation():
    """Demo: Using multiple tools for a complex request"""
    print("\n" + "="*70)
    print("DEMO 1: WEATHER + CALCULATION")
    print("="*70)
    
    agent = ToolUsingAgent()
    agent.add_tool(WeatherTool())
    agent.add_tool(CalculatorTool())
    
    await agent.process_request("What's the weather in Tokyo and calculate 20% of 500?")

async def demo_investment_calculation():
    """Demo: Complex financial calculation"""
    print("\n" + "="*70)
    print("DEMO 2: INVESTMENT CALCULATION")
    print("="*70)
    
    agent = ToolUsingAgent()
    agent.add_tool(CalculatorTool())
    
    await agent.process_request("If I invest $15,000 at 6% compound interest for 8 years, how much will I have?")

async def demo_search_and_email():
    """Demo: Information gathering and communication"""
    print("\n" + "="*70)
    print("DEMO 3: SEARCH + EMAIL")
    print("="*70)
    
    agent = ToolUsingAgent()
    agent.add_tool(WebSearchTool())
    agent.add_tool(EmailTool())
    
    await agent.process_request("Search for information about machine learning and send an email to john@example.com")

async def main():
    """
    Demonstrate Tool Using Agents with practical examples
    
    WHAT YOU'LL LEARN:
    ================
    1. How AI agents can use external tools to extend capabilities
    2. How to identify which tools are needed for different requests
    3. How to extract parameters and execute tools properly
    4. How to combine tool results into coherent responses
    5. Why tool-using agents are crucial for real-world AI applications
    
    REAL WORLD APPLICATIONS:
    =======================
    - Personal assistants that can check weather, send emails, make calculations
    - Research agents that can search web, analyze data, generate reports
    - Customer service bots that can look up information, process orders
    - Educational AI that can access real-time information and examples
    - Business automation that integrates multiple systems and APIs
    """
    
    print("🔧 Tool Using Agents Demonstration")
    print("This shows how AI can use tools to become truly useful!")
    
    await demo_weather_and_calculation()
    await demo_investment_calculation()
    await demo_search_and_email()
    
    print("\n" + "="*70)
    print("🎓 WHAT WE LEARNED:")
    print("="*70)
    print("✅ Tools extend AI capabilities beyond just thinking")
    print("✅ Agents can identify and use multiple tools per request")
    print("✅ Parameter extraction enables proper tool usage")
    print("✅ Results synthesis creates coherent final responses")
    print("✅ Tool-using agents solve real-world problems effectively")
    print("\n🔧 TRY IT YOURSELF:")
    print("- Add more tools (image generation, file operations, etc.)")
    print("- Implement tool chaining (output of one tool feeds another)")
    print("- Add real API integrations instead of simulated ones")
    print("- Create specialized tool sets for different domains")
    print("- Add error handling and retry logic for tool failures")

if __name__ == "__main__":
    asyncio.run(main())
