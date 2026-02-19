"""
Weather Scout

Monitors weather conditions and alerts on significant changes.
Runs every hour via the daemon scheduler.

Uses existing tools/weather.py for weather data.
"""

from datetime import datetime
from typing import Optional
import sys
import logging
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scouts.base import Scout, Observation, Relevance
from tools.weather import get_current_weather, get_forecast, WEATHER_CODES
from security.injection_scanner import scan_for_injection

logger = logging.getLogger(__name__)


# Weather conditions that warrant alerts
SEVERE_CONDITIONS = {
    66: "freezing rain",
    67: "freezing rain",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}

NOTABLE_CONDITIONS = {
    65: "heavy rain",
    75: "heavy snow",
    82: "heavy showers",
    86: "snow showers",
}


class WeatherScout(Scout):
    """
    Scout that monitors weather conditions.

    Checks:
    - Significant temperature changes
    - Severe weather alerts
    - High precipitation chances
    - Conditions relevant to outdoor plans

    Runs every hour by default.

    Note: This scout is rule-based (no Haiku) since weather
    alerting logic is straightforward.
    """

    name = "weather-scout"

    def __init__(self):
        super().__init__()
        self._last_weather: Optional[dict] = None
        self._last_alert_conditions: set[int] = set()

    async def observe(self) -> list[Observation]:
        """
        Check weather for significant changes.

        Returns observations for:
        - Severe weather conditions
        - Large temperature changes
        - High precipitation chances
        """
        observations = []
        now = datetime.now()

        try:
            weather = get_current_weather()
            if not weather:
                return observations

            # Scan weather conditions for injection patterns (external API data)
            conditions_text = weather.get('conditions', '')
            scan = scan_for_injection(conditions_text, source="weather-scout")
            if scan.is_suspicious:
                logger.warning(
                    f"[{self.name}] Suspicious weather conditions text "
                    f"(risk={scan.risk_level}): {conditions_text[:100]!r}"
                )

            # Check for severe weather
            # We need to get the raw weather code - not available in current API
            # So we'll check conditions string instead
            conditions = weather.get('conditions', '').lower()

            # Severe weather alert
            if any(severe in conditions for severe in ['thunderstorm', 'freezing rain', 'hail']):
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation=f"Severe weather alert: {weather['conditions']}",
                    relevance=Relevance.HIGH,
                    escalate=True,
                    context_tags=["weather", "severe", "alert"],
                    raw_data=weather
                ))

            # Heavy precipitation
            elif any(heavy in conditions for heavy in ['heavy rain', 'heavy snow', 'heavy showers']):
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation=f"Heavy precipitation: {weather['conditions']}",
                    relevance=Relevance.MEDIUM,
                    escalate=False,
                    context_tags=["weather", "precipitation"],
                    raw_data=weather
                ))

            # High precipitation probability
            precip_chance = weather.get('precipitation_chance', 0)
            if precip_chance >= 80:
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation=f"High chance of rain/snow today ({precip_chance}%)",
                    relevance=Relevance.MEDIUM,
                    escalate=False,
                    context_tags=["weather", "precipitation", "forecast"],
                    raw_data=weather
                ))

            # Significant temperature changes from last check
            if self._last_weather:
                temp_change = abs(
                    weather.get('temperature', 0) -
                    self._last_weather.get('temperature', 0)
                )
                if temp_change >= 15:
                    observations.append(Observation(
                        scout=self.name,
                        timestamp=now,
                        observation=f"Temperature changed significantly: {temp_change}°F in the last hour",
                        relevance=Relevance.MEDIUM,
                        escalate=False,
                        context_tags=["weather", "temperature", "change"],
                        raw_data=weather
                    ))

            # Extreme temperatures - escalate these, they affect everyone
            temp = weather.get('temperature', 50)
            if temp <= 20:
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation=f"Extreme cold: {temp}°F - bundle up!",
                    relevance=Relevance.HIGH,
                    escalate=True,
                    context_tags=["weather", "cold", "extreme"],
                    raw_data=weather
                ))
            elif temp >= 95:
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation=f"Extreme heat: {temp}°F - stay cool and hydrated",
                    relevance=Relevance.HIGH,
                    escalate=True,
                    context_tags=["weather", "heat", "extreme"],
                    raw_data=weather
                ))

            # Check tomorrow's forecast for significant weather
            forecast = get_forecast(days=2)
            if forecast and len(forecast) > 1:
                tomorrow = forecast[1]
                # Scan forecast conditions text
                tomorrow_raw = tomorrow.get('conditions', '')
                forecast_scan = scan_for_injection(tomorrow_raw, source="weather-scout-forecast")
                if forecast_scan.is_suspicious:
                    logger.warning(
                        f"[{self.name}] Suspicious forecast text "
                        f"(risk={forecast_scan.risk_level}): {tomorrow_raw[:100]!r}"
                    )
                tomorrow_conditions = tomorrow_raw.lower()
                tomorrow_precip = tomorrow.get('precipitation_chance', 0)
                tomorrow_low = tomorrow.get('low', 50)

                # Significant weather coming tomorrow
                significant_tomorrow = []

                # Snow or storms tomorrow
                if any(w in tomorrow_conditions for w in ['snow', 'thunderstorm', 'freezing rain', 'hail']):
                    significant_tomorrow.append(f"{tomorrow['conditions']}")

                # Very high precipitation chance
                if tomorrow_precip >= 70:
                    significant_tomorrow.append(f"{tomorrow_precip}% chance of precipitation")

                # Extreme cold tomorrow
                if tomorrow_low <= 15:
                    significant_tomorrow.append(f"low of {tomorrow_low}°F")

                if significant_tomorrow:
                    observations.append(Observation(
                        scout=self.name,
                        timestamp=now,
                        observation=f"Tomorrow's forecast: {', '.join(significant_tomorrow)}",
                        relevance=Relevance.HIGH,
                        escalate=True,
                        context_tags=["weather", "forecast", "tomorrow"],
                        raw_data=tomorrow
                    ))

            # Wind advisory (high winds)
            wind = weather.get('wind_speed', 0)
            if wind >= 30:
                observations.append(Observation(
                    scout=self.name,
                    timestamp=now,
                    observation=f"High winds: {wind} mph",
                    relevance=Relevance.MEDIUM,
                    escalate=False,
                    context_tags=["weather", "wind"],
                    raw_data=weather
                ))

            self._last_weather = weather

        except Exception as e:
            print(f"[{self.name}] Error checking weather: {e}")
            observations.append(Observation(
                scout=self.name,
                timestamp=now,
                observation=f"Error checking weather: {str(e)[:100]}",
                relevance=Relevance.LOW,
                escalate=False,
                context_tags=["error", "weather"],
            ))

        return observations


# For testing
if __name__ == "__main__":
    import asyncio

    async def test():
        scout = WeatherScout()
        print(f"Running {scout.name}...")

        observations = await scout.run()

        if not observations:
            print("No weather alerts")
        else:
            print(f"Found {len(observations)} observations:\n")
            for obs in observations:
                print(f"[{obs.relevance.value.upper()}] {obs.observation}")
                print(f"  Escalate: {obs.escalate}")
                print(f"  Tags: {obs.context_tags}")
                print()

        # Show current conditions
        weather = get_current_weather()
        if weather:
            print("\nCurrent conditions:")
            print(f"  Temperature: {weather['temperature']}°F (feels like {weather['feels_like']}°F)")
            print(f"  Conditions: {weather['conditions']}")
            print(f"  Precipitation chance: {weather['precipitation_chance']}%")
            print(f"  Wind: {weather.get('wind_speed', 0)} mph")

    asyncio.run(test())
