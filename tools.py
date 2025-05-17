import requests
import wikipedia
from langchain_core.tools  import tool
from pydantic import BaseModel, Field
from typing import Optional

class GetWeatherInput(BaseModel):
    latitude: float = Field(description="latitude of the location")
    longitude: float = Field(description="longitude of the location")

class SearchWikipediaInput(BaseModel):
    query: str = Field(description="search query for wikipedia")
    lang: Optional[str] = Field(description="language of the search query in ISO 639-1 format")

class PythonCodeInput(BaseModel):
    code: str = Field(description="Python code to run")

@tool("weather-tool", args_schema=GetWeatherInput)
def get_weather(latitude: float, longitude: float) -> str:
    """Get the weather for a given location."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    response = requests.get(
        BASE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,weather_code",
            "forecast_days": 1,
        },
    )

    if response.status_code == 200:
        data = response.json()
        
        temperature = data["hourly"]["temperature_2m"][0]
        humidity = data["hourly"]["relative_humidity_2m"][0]
        precipitation = data["hourly"]["precipitation"][0]
        wind_speed = data["hourly"]["wind_speed_10m"][0]
        wind_direction = data["hourly"]["wind_direction_10m"][0]
        weather_code = data["hourly"]["weather_code"][0]
        
        weather_descriptions = {
            0: "Clear sky",
            1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            66: "Light freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        
        weather_description = weather_descriptions.get(weather_code, "Unknown")
        
        cardinal_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        cardinal_index = round(wind_direction / 45) % 8
        cardinal_direction = cardinal_directions[cardinal_index]
        
        return f"""Weather Report:
• Condition: {weather_description}
• Temperature: {temperature}°C
• Humidity: {humidity}%
• Wind: {wind_speed} km/h from {cardinal_direction} ({wind_direction}°)
• Precipitation: {precipitation} mm
"""

    return "Unable to fetch weather data."

@tool("wikipedia-tool", args_schema=SearchWikipediaInput)
def get_wikipedia(query: str, lang: str = "en") -> str:
    """Get the first paragraph of a Wikipedia page."""
    wikipedia.set_lang(lang)
    return wikipedia.summary(query, sentences=5)

@tool("python-code-runner-tool", args_schema=PythonCodeInput)
def run_python_code(code: str) -> str:
    """Run a Python code snippet and return the output. Can be used for mathematical calculations and general Python code execution."""
    try:
        output = eval(code)
        return str(output)
    except Exception as e:
        return str(e)
