"""Weather data via Open-Meteo API.
Free, no API key required.
https://open-meteo.com/
"""

import httpx
from datetime import datetime
from zoneinfo import ZoneInfo

import os

# Default coordinates — configure via DORIS_HOME_LAT / DORIS_HOME_LON env vars
# Falls back to NYC city center (40.7128, -74.0060) if not set
DEFAULT_LAT = float(os.getenv("DORIS_HOME_LAT", "40.7128"))
DEFAULT_LON = float(os.getenv("DORIS_HOME_LON", "-74.0060"))

# Known locations — customize for your area
# Add your own aliases to this dict or configure via environment variables
# e.g., "home": (lat, lon), "office": (lat, lon)
_secondary_lat = float(os.getenv("DORIS_SECONDARY_LAT", "0"))
_secondary_lon = float(os.getenv("DORIS_SECONDARY_LON", "0"))

KNOWN_LOCATIONS = {
    # Primary location aliases
    "nyc": (40.7128, -74.0060),
    "new york": (40.7128, -74.0060),
    "new york city": (40.7128, -74.0060),
    "manhattan": (40.7831, -73.9712),
    "home": (DEFAULT_LAT, DEFAULT_LON),
    # Secondary location (if configured)
    **({"secondary": (_secondary_lat, _secondary_lon)} if _secondary_lat else {}),
}

# Weather code descriptions (WMO codes)
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy", 
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "light snow showers",
    86: "snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def get_temp_descriptor(temp: int) -> str:
    """Get a natural language descriptor for a temperature."""
    if temp <= 10:
        return "brutal"
    elif temp <= 20:
        return "frigid"
    elif temp <= 32:
        return "freezing"
    elif temp <= 45:
        return "chilly"
    elif temp <= 55:
        return "cool"
    elif temp <= 68:
        return "mild"
    elif temp <= 78:
        return "nice"
    elif temp <= 85:
        return "warm"
    elif temp <= 95:
        return "hot"
    else:
        return "scorching"


def get_time_of_day() -> str:
    """Get current time of day for contextual phrasing."""
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    elif hour < 21:
        return "evening"
    else:
        return "night"


def resolve_location(location: str | None) -> tuple[float, float, str]:
    """Resolve a location string to coordinates.

    Returns (lat, lon, location_name).
    """
    if not location:
        return DEFAULT_LAT, DEFAULT_LON, "NYC"

    # Check known locations (case-insensitive)
    location_lower = location.lower().strip()
    if location_lower in KNOWN_LOCATIONS:
        lat, lon = KNOWN_LOCATIONS[location_lower]
        return lat, lon, location.title()

    # Try geocoding via Open-Meteo's geocoding API
    try:
        response = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=3.0
        )
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            result = data["results"][0]
            return result["latitude"], result["longitude"], result.get("name", location)
    except Exception as e:
        print(f"Geocoding error for '{location}': {e}")

    # Fall back to NYC
    return DEFAULT_LAT, DEFAULT_LON, "NYC"


def get_current_weather(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, location: str = None) -> dict | None:
    """Get current weather conditions.

    Args:
        lat: Latitude (ignored if location is provided)
        lon: Longitude (ignored if location is provided)
        location: Location string (e.g., "NYC", "Upstate NY", "Paris")
    """
    try:
        # Resolve location string to coordinates
        if location:
            lat, lon, resolved_name = resolve_location(location)
        else:
            # Validate coordinates and fall back to defaults if invalid
            try:
                if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                    lat, lon = DEFAULT_LAT, DEFAULT_LON
            except (TypeError, ValueError):
                lat, lon = DEFAULT_LAT, DEFAULT_LON
            resolved_name = "NYC"
            
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "America/New_York",
            "forecast_days": 1
        }
        
        response = httpx.get(url, params=params, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        daily = data.get("daily", {})
        
        temperature = current.get("temperature_2m")
        weather_code = current.get("weather_code", 0)
        
        return {
            "location": resolved_name,
            "temperature": round(temperature) if temperature is not None else None,
            "feels_like": round(current.get("apparent_temperature", temperature or 0)),
            "humidity": current.get("relative_humidity_2m", 0),
            "wind_speed": round(current.get("wind_speed_10m", 0)),
            "conditions": WEATHER_CODES.get(weather_code, "unknown"),
            "high": round(daily.get("temperature_2m_max", [temperature or 0])[0]),
            "low": round(daily.get("temperature_2m_min", [temperature or 0])[0]),
            "precipitation_chance": daily.get("precipitation_probability_max", [0])[0],
            "sunrise": daily.get("sunrise", [""])[0],
            "sunset": daily.get("sunset", [""])[0],
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return None


def get_forecast(location: str = None, days: int = 3) -> list[dict] | None:
    """Get multi-day weather forecast.

    Args:
        location: Location string (e.g., "NYC", "Upstate NY", "Paris")
        days: Number of forecast days (1-7)

    Returns:
        List of daily forecasts, each with date, high, low, conditions, precipitation_chance
    """
    try:
        lat, lon, resolved_name = resolve_location(location)
        days = max(1, min(7, days))  # Clamp to 1-7

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "forecast_days": days
        }

        response = httpx.get(url, params=params, timeout=5.0)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        codes = daily.get("weather_code", [])
        precips = daily.get("precipitation_probability_max", [])

        forecast = []
        for i, date in enumerate(dates):
            forecast.append({
                "date": date,
                "location": resolved_name,
                "high": round(highs[i]) if i < len(highs) else None,
                "low": round(lows[i]) if i < len(lows) else None,
                "conditions": WEATHER_CODES.get(codes[i] if i < len(codes) else 0, "unknown"),
                "precipitation_chance": precips[i] if i < len(precips) else 0,
            })

        return forecast
    except Exception as e:
        print(f"Forecast API error: {e}")
        return None


def format_forecast(forecast: list[dict]) -> str:
    """Format multi-day forecast as natural language."""
    if not forecast:
        return "Forecast data unavailable."

    from datetime import datetime

    parts = []
    for i, day in enumerate(forecast):
        date_obj = datetime.strptime(day["date"], "%Y-%m-%d")
        if i == 0:
            day_name = "Today"
        elif i == 1:
            day_name = "Tomorrow"
        else:
            day_name = date_obj.strftime("%A")

        conditions = day["conditions"]
        high = day["high"]
        low = day["low"]
        precip = day.get("precipitation_chance", 0)

        line = f"{day_name}: {conditions}, high of {high}°, low of {low}°"
        if precip >= 40:
            line += f" ({precip}% chance of rain)"
        parts.append(line)

    return ". ".join(parts) + "."


def get_weather(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> str:
    """Get weather in the expected format."""
    weather_data = get_current_weather(lat, lon)
    if not weather_data:
        return "Current conditions: unknown, Unknown°F"
    
    conditions = weather_data.get("conditions", "unknown")
    temperature = weather_data.get("temperature")
    
    temp_display = f"{temperature}" if temperature is not None else "Unknown"
    return f"Current conditions: {conditions}, {temp_display}°F"


def format_weather(weather: dict) -> str:
    """Format weather data as natural, conversational text."""
    if not weather:
        return "Weather data unavailable."

    temp = weather["temperature"]
    feels = weather["feels_like"]
    conditions = weather["conditions"]
    high = weather["high"]
    low = weather["low"]
    precip = weather["precipitation_chance"]
    wind = weather.get("wind_speed", 0)
    humidity = weather.get("humidity", 0)

    time_of_day = get_time_of_day()
    temp_desc = get_temp_descriptor(temp)
    feels_diff = temp - feels

    # Determine if snow-related
    is_snow = "snow" in conditions.lower()
    is_rain = any(w in conditions.lower() for w in ["rain", "drizzle", "shower"])
    is_storm = "thunder" in conditions.lower()
    is_nice = conditions in ("clear sky", "mainly clear", "sunny") and 60 <= temp <= 80

    parts = []

    # Opening line - vary the structure based on conditions
    if is_nice:
        if time_of_day == "morning":
            parts.append(f"Beautiful day ahead — {temp_desc} in the low {_round_to_tens(temp)}s and {conditions}.")
        else:
            parts.append(f"Gorgeous out there — {temp_desc}, {conditions}, around {temp} degrees.")
    elif is_storm:
        parts.append(f"Heads up — {conditions} today. Currently {temp} degrees.")
    elif temp_desc in ("brutal", "frigid", "freezing"):
        parts.append(f"Bundle up — it's {temp_desc} out there, only {temp} degrees.")
    elif temp_desc in ("hot", "scorching"):
        parts.append(f"It's gonna be a scorcher — {_temp_phrase(temp)} and {conditions}.")
    else:
        # Standard varied openings
        openers = [
            f"It's {temp_desc} out — {_temp_phrase(temp)} and {conditions}.",
            f"Currently {_temp_phrase(temp)} with {conditions}.",
            f"Looking at {_temp_phrase(temp)} and {conditions} right now.",
        ]
        # Pick based on temp to add variety
        parts.append(openers[temp % 3])

    # Feels-like: only if significant AND explain why
    if abs(feels_diff) >= 8:
        if feels_diff > 0 and wind >= 10:
            parts.append(f"Wind's making it feel closer to {feels} though.")
        elif feels_diff > 0 and temp <= 45:
            parts.append(f"The wind chill brings it down to {feels}.")
        elif feels_diff < 0 and humidity >= 60 and temp >= 75:
            parts.append(f"Humidity's pushing it to feel more like {feels}.")
        elif feels_diff < 0 and temp >= 80:
            parts.append(f"With the humidity, it'll feel more like {feels}.")
        else:
            parts.append(f"Feels more like {feels} out there.")

    # Practical suggestions based on conditions
    if is_snow:
        if "heavy" in conditions.lower():
            parts.append("Expect accumulation — plan for slick roads.")
        elif "light" in conditions.lower():
            parts.append("Light snow expected, shouldn't be too bad.")
    elif precip >= 70:
        if is_rain:
            parts.append("Definitely grab an umbrella.")
        else:
            parts.append("Rain's looking likely — bring an umbrella.")
    elif precip >= 40:
        if time_of_day == "morning":
            parts.append("Might want an umbrella later today.")
        else:
            parts.append("Could see some rain, maybe grab an umbrella.")

    # Cold + wind layering advice
    if temp <= 40 and wind >= 12 and not any("umbrella" in p for p in parts):
        parts.append("Layer up — that wind's no joke.")

    # Hot + humid advice (if not already covered)
    if temp >= 85 and humidity >= 65 and abs(feels_diff) < 8:
        parts.append("Stay hydrated if you're heading out.")

    # Keep it short - max 3 sentences
    result = " ".join(parts)
    sentences = result.split(". ")
    if len(sentences) > 3:
        result = ". ".join(sentences[:3])
        if not result.endswith("."):
            result += "."

    return result


def _temp_phrase(temp: int) -> str:
    """Convert temp to natural phrasing like 'mid-30s' or 'low 70s'."""
    tens = (temp // 10) * 10
    remainder = temp % 10

    if remainder <= 3:
        prefix = "low"
    elif remainder <= 6:
        prefix = "mid"
    else:
        prefix = "upper"

    return f"{prefix} {tens}s"


def _round_to_tens(temp: int) -> int:
    """Round temperature to nearest tens for casual speech."""
    return (temp // 10) * 10


def get_weather_summary() -> str:
    """Get a quick weather summary for context injection."""
    weather = get_current_weather()
    if not weather:
        return "Weather: unavailable"
    
    precip_note = ""
    if weather["precipitation_chance"] >= 40:
        precip_note = f", {weather['precipitation_chance']}% chance of rain"
    
    return (
        f"Weather: {weather['temperature']}°F ({weather['conditions']}), "
        f"high {weather['high']}°, low {weather['low']}°{precip_note}"
    )


if __name__ == "__main__":
    # Quick test
    weather = get_current_weather()
    if weather:
        print("Raw data:", weather)
        print("\nFormatted:", format_weather(weather))
        print("\nContext summary:", get_weather_summary())
    else:
        print("Failed to get weather")