"""
OpenAI Function Calling Examples
================================

This module demonstrates various function calling patterns with OpenAI's
Chat Completions API. Each section is standalone and can be run independently.

Sections:
1. Basic Function Calling
2. Multiple Functions
3. Parallel Function Calling
4. Structured Output
5. Pydantic Integration
6. Conversation with Function Calling
7. Error Handling
"""

import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, Field


# Initialize OpenAI client
client = OpenAI(api_key="your-api-key-here")  # Replace with your API key


# ============================================================================
# SECTION 1: Basic Function Calling
# ============================================================================

def get_weather_basic(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Mock weather function for demonstration.
    In production, this would call a real weather API.
    """
    # Simulate API call
    time.sleep(0.1)
    
    mock_data = {
        "New York": {"temp": 22, "condition": "sunny", "humidity": 65},
        "London": {"temp": 15, "condition": "cloudy", "humidity": 80},
        "Tokyo": {"temp": 28, "condition": "rainy", "humidity": 75},
    }
    
    default = {"temp": 20, "condition": "unknown", "humidity": 70}
    weather = mock_data.get(location, default)
    
    if unit == "fahrenheit":
        weather["temp"] = (weather["temp"] * 9/5) + 32
    
    return {
        "location": location,
        "temperature": weather["temp"],
        "unit": unit,
        "condition": weather["condition"],
        "humidity": weather["humidity"]
    }


def basic_function_calling():
    """
    Demonstrates basic function calling with a single function.
    """
    print("=" * 70)
    print("SECTION 1: Basic Function Calling")
    print("=" * 70)
    
    # Define the function schema
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather_basic",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit for temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # User message
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in New York?"
        }
    ]
    
    # First API call - model decides to call function
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        tools=functions,
        tool_choice="auto"
    )
    
    # Check if function was called
    message = response.choices[0].message
    
    if message.tool_calls:
        # Extract function call details
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"\nFunction called: {function_name}")
        print(f"Arguments: {function_args}")
        
        # Execute the function
        function_result = get_weather_basic(**function_args)
        
        print(f"Function result: {function_result}")
        
        # Add assistant's message with tool call
        messages.append(message)
        
        # Add function result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(function_result)
        })
        
        # Second API call - model uses function result
        second_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )
        
        final_message = second_response.choices[0].message.content
        print(f"\nFinal response: {final_message}")
    else:
        print(f"Response: {message.content}")


# ============================================================================
# SECTION 2: Multiple Functions
# ============================================================================

def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """Get weather information for a location."""
    time.sleep(0.1)
    mock_data = {
        "New York": {"temp": 22, "condition": "sunny"},
        "London": {"temp": 15, "condition": "cloudy"},
    }
    weather = mock_data.get(location, {"temp": 20, "condition": "unknown"})
    if unit == "fahrenheit":
        weather["temp"] = (weather["temp"] * 9/5) + 32
    return {"location": location, "temperature": weather["temp"], "condition": weather["condition"]}


def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a mathematical expression safely."""
    try:
        # Simple safe evaluation (in production, use a proper math parser)
        result = eval(expression.replace("^", "**"))
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def search_knowledge_base(query: str) -> Dict[str, Any]:
    """Search a knowledge base for information."""
    time.sleep(0.1)
    mock_kb = {
        "python": "Python is a high-level programming language.",
        "openai": "OpenAI is an AI research company.",
        "function calling": "Function calling allows models to call external functions."
    }
    
    results = []
    query_lower = query.lower()
    for key, value in mock_kb.items():
        if key in query_lower:
            results.append({"topic": key, "info": value})
    
    return {"query": query, "results": results}


def multiple_functions():
    """
    Demonstrates an assistant with multiple available functions.
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Multiple Functions")
    print("=" * 70)
    
    # Define multiple functions
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate, e.g., '2 + 2' or '10 * 5'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base for information about a topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Multi-part user query
    messages = [
        {
            "role": "user",
            "content": "What's the weather in London? Also calculate 15 * 23, and tell me about Python."
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        tools=functions,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    # Handle multiple tool calls
    if message.tool_calls:
        print(f"\nNumber of tool calls: {len(message.tool_calls)}")
        
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\nCalling: {function_name} with args: {function_args}")
            
            # Execute appropriate function
            if function_name == "get_weather":
                result = get_weather(**function_args)
            elif function_name == "calculate":
                result = calculate(**function_args)
            elif function_name == "search_knowledge_base":
                result = search_knowledge_base(**function_args)
            else:
                result = {"error": "Unknown function"}
            
            print(f"Result: {result}")
            
            # Add tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        # Get final response
        final_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )
        
        print(f"\nFinal response: {final_response.choices[0].message.content}")


# ============================================================================
# SECTION 3: Parallel Function Calling
# ============================================================================

def parallel_function_calling():
    """
    Demonstrates handling multiple simultaneous function calls.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Parallel Function Calling")
    print("=" * 70)
    
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "Get weather for New York and London, calculate 25 * 17, and search for OpenAI."
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        tools=functions,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    if message.tool_calls:
        print(f"\nProcessing {len(message.tool_calls)} parallel function calls...")
        
        # Execute all functions (can be parallelized in production)
        tool_outputs = []
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"Executing: {function_name}({function_args})")
            
            # Execute function
            if function_name == "get_weather":
                result = get_weather(**function_args)
            elif function_name == "calculate":
                result = calculate(**function_args)
            elif function_name == "search_knowledge_base":
                result = search_knowledge_base(**function_args)
            else:
                result = {"error": "Unknown function"}
            
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": json.dumps(result)
            })
        
        # Add all tool outputs at once
        messages.extend(tool_outputs)
        
        # Get final response
        final_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )
        
        print(f"\nFinal response: {final_response.choices[0].message.content}")


# ============================================================================
# SECTION 4: Structured Output
# ============================================================================

def structured_output():
    """
    Demonstrates forcing structured JSON output using response_format.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: Structured Output")
    print("=" * 70)
    
    # Request structured output
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": "Extract information about three cities: New York, Paris, Tokyo. "
                          "For each city, provide: name, country, population (in millions), "
                          "and a famous landmark."
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    content = response.choices[0].message.content
    print(f"\nStructured JSON response:")
    print(json.dumps(json.loads(content), indent=2))
    
    # Parse and use the structured data
    data = json.loads(content)
    if "cities" in data:
        print("\nExtracted cities:")
        for city in data["cities"]:
            print(f"  - {city.get('name', 'N/A')}: {city.get('landmark', 'N/A')}")


# ============================================================================
# SECTION 5: Pydantic Integration
# ============================================================================

class WeatherRequest(BaseModel):
    """Request schema for weather function."""
    location: str = Field(description="The city name")
    unit: str = Field(default="celsius", enum=["celsius", "fahrenheit"], description="Temperature unit")


class WeatherResponse(BaseModel):
    """Response schema for weather function."""
    location: str
    temperature: float
    unit: str
    condition: str


class CalculationRequest(BaseModel):
    """Request schema for calculation function."""
    expression: str = Field(description="Mathematical expression to evaluate")


class CalculationResponse(BaseModel):
    """Response schema for calculation function."""
    expression: str
    result: float
    error: Optional[str] = None


def pydantic_to_json_schema(pydantic_model: type[BaseModel]) -> Dict[str, Any]:
    """Convert Pydantic model to JSON Schema."""
    return pydantic_model.model_json_schema()


def pydantic_integration():
    """
    Demonstrates using Pydantic models for function schemas and validation.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: Pydantic Integration")
    print("=" * 70)
    
    # Convert Pydantic models to JSON schemas
    weather_schema = pydantic_to_json_schema(WeatherRequest)
    calculation_schema = pydantic_to_json_schema(CalculationRequest)
    
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather_validated",
                "description": "Get weather with validated input",
                "parameters": weather_schema
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_validated",
                "description": "Calculate with validated input",
                "parameters": calculation_schema
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "Get weather for Tokyo in fahrenheit and calculate 42 * 8"
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        tools=functions,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args_json = tool_call.function.arguments
            
            print(f"\nFunction: {function_name}")
            print(f"Raw arguments: {function_args_json}")
            
            # Validate with Pydantic
            if function_name == "get_weather_validated":
                try:
                    validated_args = WeatherRequest.model_validate_json(function_args_json)
                    print(f"Validated args: {validated_args}")
                    
                    # Execute function
                    result = get_weather(validated_args.location, validated_args.unit)
                    
                    # Validate response
                    validated_result = WeatherResponse(**result)
                    print(f"Validated result: {validated_result}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": validated_result.model_dump_json()
                    })
                except Exception as e:
                    print(f"Validation error: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": str(e)})
                    })
            
            elif function_name == "calculate_validated":
                try:
                    validated_args = CalculationRequest.model_validate_json(function_args_json)
                    result = calculate(validated_args.expression)
                    validated_result = CalculationResponse(**result)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": validated_result.model_dump_json()
                    })
                except Exception as e:
                    print(f"Validation error: {e}")
        
        final_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )
        
        print(f"\nFinal response: {final_response.choices[0].message.content}")


# ============================================================================
# SECTION 6: Conversation with Function Calling
# ============================================================================

def conversation_with_functions():
    """
    Demonstrates a multi-turn conversation with function calling.
    """
    print("\n" + "=" * 70)
    print("SECTION 6: Conversation with Function Calling")
    print("=" * 70)
    
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can get weather and perform calculations."
        }
    ]
    
    # Turn 1
    print("\n--- Turn 1 ---")
    user_message_1 = "What's the weather in New York?"
    print(f"User: {user_message_1}")
    messages.append({"role": "user", "content": user_message_1})
    
    response_1 = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        tools=functions
    )
    
    message_1 = response_1.choices[0].message
    messages.append(message_1)
    
    if message_1.tool_calls:
        for tool_call in message_1.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = get_weather(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        response_1_final = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )
        print(f"Assistant: {response_1_final.choices[0].message.content}")
        messages.append(response_1_final.choices[0].message)
    
    # Turn 2
    print("\n--- Turn 2 ---")
    user_message_2 = "Now calculate 15 * 23"
    print(f"User: {user_message_2}")
    messages.append({"role": "user", "content": user_message_2})
    
    response_2 = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        tools=functions
    )
    
    message_2 = response_2.choices[0].message
    messages.append(message_2)
    
    if message_2.tool_calls:
        for tool_call in message_2.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = calculate(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        response_2_final = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )
        print(f"Assistant: {response_2_final.choices[0].message.content}")
    
    # Turn 3
    print("\n--- Turn 3 ---")
    user_message_3 = "Compare that to the temperature in New York"
    print(f"User: {user_message_3}")
    messages.append({"role": "user", "content": user_message_3})
    
    response_3 = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        tools=functions
    )
    
    print(f"Assistant: {response_3.choices[0].message.content}")


# ============================================================================
# SECTION 7: Error Handling
# ============================================================================

def function_with_error():
    """Function that may raise errors."""
    raise ValueError("Simulated error")


def robust_function_call(func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute function with error handling."""
    try:
        if func_name == "get_weather":
            return get_weather(**args)
        elif func_name == "calculate":
            return calculate(**args)
        elif func_name == "function_with_error":
            return function_with_error()
        else:
            return {"error": f"Unknown function: {func_name}"}
    except Exception as e:
        return {"error": f"Function execution failed: {str(e)}"}


def error_handling():
    """
    Demonstrates error handling in function calling.
    """
    print("\n" + "=" * 70)
    print("SECTION 7: Error Handling")
    print("=" * 70)
    
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "function_with_error",
                "description": "A function that may error",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "Get weather for London and also call the error function."
        }
    ]
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=functions,
                tool_choice="auto",
                timeout=30
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            if message.tool_calls:
                print(f"\nProcessing {len(message.tool_calls)} tool calls...")
                
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        function_args = {}
                    
                    print(f"Calling: {function_name}")
                    
                    # Execute with error handling
                    result = robust_function_call(function_name, function_args)
                    
                    print(f"Result: {result}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                # Get final response
                final_response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    timeout=30
                )
                
                print(f"\nFinal response: {final_response.choices[0].message.content}")
                break
                
            else:
                print(f"Response: {message.content}")
                break
                
        except Exception as e:
            retry_count += 1
            print(f"\nError occurred (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Giving up.")
                raise


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run all examples.
    Uncomment the section you want to run.
    """
    print("\n" + "=" * 70)
    print("OpenAI Function Calling Examples")
    print("=" * 70)
    print("\nNote: Make sure to set your OpenAI API key in the client initialization.")
    print("Each section is standalone and can be run independently.\n")
    
    # Uncomment sections to run:
    
    # basic_function_calling()
    # multiple_functions()
    # parallel_function_calling()
    # structured_output()
    # pydantic_integration()
    # conversation_with_functions()
    # error_handling()
    
    print("\n" + "=" * 70)
    print("Examples completed. Uncomment sections in main() to run them.")
    print("=" * 70)


if __name__ == "__main__":
    main()
