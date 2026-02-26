"""
Tools: Optional parameters with default values.

Syntax: def tool(param: str, optional: Optional[str] = "default") -> str:

invoke can omit optional: {"param": "value"}
"""

from langchain_core.tools import tool
from typing import Optional


@tool
def get_weather(city: str, unit: Optional[str] = "celsius") -> str:
    """Get weather. unit: celsius or fahrenheit."""
    return f"Weather in {city}: 20{unit[0]}"


# get_weather.invoke({"city": "NYC"}) -> uses unit="celsius"
# get_weather.invoke({"city": "NYC", "unit": "fahrenheit"})
