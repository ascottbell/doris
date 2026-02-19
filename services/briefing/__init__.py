"""
Morning briefing components for Doris.

Components:
- weather_calendar: Weather correlated with today's events
- news_digest: 4 curated news stories
- deliveries: Expected packages from email
- kids_corner: Fun content for kids
- orchestrator: Combines all components
"""

from .weather_calendar import get_weather_calendar_fusion
from .news_digest import get_news_digest
from .deliveries import get_expected_deliveries
from .kids_corner import get_kids_content
from .orchestrator import get_morning_briefing
