"""
mcp_server.py — MCP server exposing weather and soil tools for AI Crop Doctor.
"""
import os
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

load_dotenv()
weather_api_key = os.getenv("OPENWEATHER_API_KEY")

mcp = FastMCP("crop-doctor-tools")


SOIL_DATABASE = {
    "lucknow": "Alluvial soil, rich in potash but poor in phosphorus",
    "punjab": "Loamy to sandy loam, good for wheat and rice",
    "maharashtra": "Black cotton soil, retains moisture well",
    "kerala": "Laterite soil, acidic in nature",
    "gujarat": "Sandy loam to clay loam, moderate fertility",
    "rajasthan": "Sandy and arid soil, low organic matter",
    "karnataka": "Red laterite soil, slightly acidic",
    "andhra pradesh": "Black and red loamy soil",
    "tamil nadu": "Red loam and alluvial, varies by region",
    "west bengal": "Alluvial soil, good for paddy cultivation",
}


OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


@mcp.tool()
def get_weather(location: str) -> str:
    """Fetches current weather condition for a given location.
    Args:
        location: City name
    Returns:
        A short human-readable weather summary.
    Raises:
        An error if no OpenWeatherMap API key is configured, the location is not
        found, or the request fails. The caller treats this as "weather
        unavailable" instead of receiving made-up conditions.
    """
    if not weather_api_key:
        raise ToolError("OPENWEATHER_API_KEY is not configured.")

    # params= URL-encodes the location (e.g. '&', '#', spaces), and HTTPS keeps
    # the API key off the wire in plaintext.
    response = requests.get(
        OPENWEATHER_URL,
        params={"q": location, "appid": weather_api_key, "units": "metric"},
        timeout=10,
    )
    if response.status_code == 404:
        raise ToolError(f"Location not found by OpenWeatherMap: {location!r}")
    response.raise_for_status()

    res = response.json()
    try:
        temp = res["main"]["temp"]
        humidity = res["main"]["humidity"]
        desc = res["weather"][0]["description"]
    except (KeyError, IndexError, TypeError) as e:
        raise ToolError(f"Unexpected OpenWeatherMap response: {e}") from e
    return f"Temperature: {temp}°C, Humidity: {humidity}%, Condition: {desc.capitalize()}"


@mcp.tool()
def get_soil_condition(location: str) -> str:
    """Look up the soil typefor a given Indian state or city.
    Args:
        location: City or state name
    Returns:
        A short description of the typical local soil type.
        Falls back to a generic default if the location isn't in the lookup table.
    """
    city = location.lower().split(",")[0].strip()
    return SOIL_DATABASE.get(city, "Standard Loamy Soil")


if __name__ == "__main__":
    mcp.run(transport="stdio")