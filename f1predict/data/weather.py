"""Open-Meteo weather client.

Weather is the one materially predictive fact about a *future* race that can be
obtained in advance, which is why it earns a place in the pre-race feature set.

Two endpoints, both free and keyless:

* ``archive-api.open-meteo.com`` -- ERA5 reanalysis, back to 1940, but lagging
  roughly five days behind the present.
* ``api.open-meteo.com/v1/forecast`` -- covers the recent past through +16 days,
  which is what fills the gap the archive leaves and what supplies the forecast
  for an upcoming race.

Resolution matters here. A grand prix is a ~2 hour window, so daily aggregates
blur a wet start into a dry afternoon. This client pulls *hourly* values and
aggregates over the race window, falling back to daily only when the session
start time is unknown. The original pipeline scraped a free-text label off
Wikipedia ("sunny", "wet"); this is a strict improvement on that.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_CACHE = Path("data/cache/weather")

HOURLY_VARS = ["temperature_2m", "precipitation", "windspeed_10m", "relativehumidity_2m"]
DAILY_VARS = ["temperature_2m_max", "precipitation_sum", "windspeed_10m_max"]

#: The archive lags reality; inside this many days, ask the forecast endpoint.
ARCHIVE_LAG_DAYS = 10

#: Total precipitation over the race window (mm) above which we call it wet.
WET_THRESHOLD_MM = 0.5

#: Nominal race length used to build the aggregation window.
RACE_WINDOW_HOURS = 2


class WeatherError(RuntimeError):
    pass


class WeatherClient:
    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE,
        session: requests.Session | None = None,
        offline: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = session or requests.Session()
        self.offline = offline
        self.requests_made = 0

    # -- transport ----------------------------------------------------------

    def _cache_path(self, lat: float, lon: float, day: date, kind: str) -> Path:
        return self.cache_dir / f"{kind}_{lat:.4f}_{lon:.4f}_{day.isoformat()}.json"

    def _fetch(
        self, url: str, params: dict[str, Any], cache_path: Path | None, retries: int = 3
    ) -> dict:
        # Historical weather is immutable, so a cache hit is always valid.
        # Forecasts are never cached -- a stale forecast is worse than none.
        if cache_path is not None and cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log.warning("Discarding corrupt weather cache %s", cache_path)

        if self.offline:
            raise WeatherError(f"offline=True and no cached weather at {cache_path}")

        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self._session.get(url, params=params, timeout=45)
                self.requests_made += 1
                response.raise_for_status()
                payload = response.json()
                if cache_path is not None:
                    cache_path.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        raise WeatherError(f"failed to fetch weather from {url}: {last_error}")

    # -- public API ---------------------------------------------------------

    def race_weather(
        self,
        latitude: float,
        longitude: float,
        race_date: date,
        race_time_utc: str | None = None,
    ) -> dict[str, float | None]:
        """Weather over the race window at a circuit.

        Args:
            race_date: date of the race.
            race_time_utc: session start as ``"14:00:00Z"`` if known. When
                omitted, the whole day is used instead of the race window.

        Returns a dict of the ``weather_*`` features declared in
        :mod:`f1predict.data.contracts`.
        """
        today = datetime.now(timezone.utc).date()
        use_archive = race_date < today - timedelta(days=ARCHIVE_LAG_DAYS)
        url = ARCHIVE_URL if use_archive else FORECAST_URL

        # Only immutable past weather is cached.
        cache_path = (
            self._cache_path(latitude, longitude, race_date, "archive")
            if use_archive
            else None
        )

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": race_date.isoformat(),
            "end_date": race_date.isoformat(),
            "hourly": ",".join(HOURLY_VARS),
            "daily": ",".join(DAILY_VARS),
            "timezone": "UTC",
        }

        try:
            payload = self._fetch(url, params, cache_path)
        except WeatherError:
            log.warning(
                "No weather for (%.3f, %.3f) on %s; features will be null",
                latitude,
                longitude,
                race_date,
            )
            return _null_weather()

        return _summarise(payload, race_date, race_time_utc)


def _null_weather() -> dict[str, float | None]:
    return {
        "weather_temp_max": None,
        "weather_precipitation": None,
        "weather_windspeed_max": None,
        "weather_is_wet": None,
    }


def _window_indices(
    times: list[str], race_date: date, race_time_utc: str | None
) -> list[int]:
    """Indices of the hourly samples covering the race, or the full day."""
    if not race_time_utc:
        return list(range(len(times)))
    try:
        hour = int(str(race_time_utc)[:2])
    except (ValueError, TypeError):
        return list(range(len(times)))

    start = datetime.combine(race_date, datetime.min.time()).replace(hour=hour)
    end = start + timedelta(hours=RACE_WINDOW_HOURS)

    indices = []
    for i, stamp in enumerate(times):
        try:
            parsed = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        if start <= parsed <= end:
            indices.append(i)
    # A race can start late in the day and run past the last sample.
    return indices or list(range(len(times)))


def _summarise(
    payload: dict, race_date: date, race_time_utc: str | None
) -> dict[str, float | None]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []

    if times:
        idx = _window_indices(times, race_date, race_time_utc)

        def pick(name: str) -> list[float]:
            series = hourly.get(name) or []
            return [series[i] for i in idx if i < len(series) and series[i] is not None]

        temps = pick("temperature_2m")
        precip = pick("precipitation")
        wind = pick("windspeed_10m")

        if temps or precip or wind:
            total_precip = sum(precip) if precip else None
            return {
                "weather_temp_max": max(temps) if temps else None,
                "weather_precipitation": total_precip,
                "weather_windspeed_max": max(wind) if wind else None,
                "weather_is_wet": (
                    float(total_precip > WET_THRESHOLD_MM)
                    if total_precip is not None
                    else None
                ),
            }

    # Fall back to daily aggregates.
    daily = payload.get("daily") or {}

    def first(name: str) -> float | None:
        series = daily.get(name) or []
        return series[0] if series and series[0] is not None else None

    precipitation = first("precipitation_sum")
    return {
        "weather_temp_max": first("temperature_2m_max"),
        "weather_precipitation": precipitation,
        "weather_windspeed_max": first("windspeed_10m_max"),
        "weather_is_wet": (
            float(precipitation > WET_THRESHOLD_MM) if precipitation is not None else None
        ),
    }
