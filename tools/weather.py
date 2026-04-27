"""Weather tool — wraps OpenWeatherMapAPIWrapper as a LangChain tool."""

from langchain.tools import tool
from langchain_community.utilities import OpenWeatherMapAPIWrapper

_weather = OpenWeatherMapAPIWrapper()


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location.

    Queries OpenWeatherMap and returns temperature, humidity, wind speed,
    and a short description for the requested city or region.

    Args:
        location: City name or coordinate string (e.g. "Rome", "Rome,IT").
    """
    return _weather.run(location)
