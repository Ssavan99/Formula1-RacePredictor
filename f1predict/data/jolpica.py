"""Jolpica-F1 API client.

Jolpica is the community successor to the Ergast API, which was frozen after the
2024 season and shut down in early 2025. It serves the same JSON schema at
``https://api.jolpi.ca/ergast/f1/``, so the original project's mental model of
the data still applies.

Two constraints shape this client:

* **200 requests/hour, unauthenticated.** Everything is cached to disk, and the
  rate limiter is a sliding window rather than a fixed sleep, so a warm cache
  costs nothing and a cold backfill self-throttles instead of getting banned.
* **Server-side page cap of 100 rows**, regardless of the ``limit`` asked for.
  Budget accordingly: a season of race results is ~5 requests.

Cache policy: a completed season is immutable and cached forever; the current
season is cached with a short TTL because results land during the year.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100  # server-side maximum; larger values are silently capped
DEFAULT_CACHE = Path("data/cache/jolpica")

#: Stay under the documented 200/hour with headroom for retries.
DEFAULT_RATE_LIMIT = 180
RATE_WINDOW_SECONDS = 3600

#: How long current-season responses stay fresh.
CURRENT_SEASON_TTL = 6 * 3600


class JolpicaError(RuntimeError):
    """Raised when the API cannot be reached or returns an unusable payload."""


class _SlidingWindowLimiter:
    """Allow at most ``limit`` requests per ``window`` seconds."""

    def __init__(self, limit: int = DEFAULT_RATE_LIMIT, window: int = RATE_WINDOW_SECONDS):
        self.limit = limit
        self.window = window
        self._times: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._times and now - self._times[0] > self.window:
            self._times.popleft()
        if len(self._times) >= self.limit:
            sleep_for = self.window - (now - self._times[0]) + 1
            log.warning(
                "Jolpica rate limit reached (%d/%ds); sleeping %.0fs",
                self.limit,
                self.window,
                sleep_for,
            )
            time.sleep(sleep_for)
            return self.acquire()
        self._times.append(now)


class JolpicaClient:
    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        session: requests.Session | None = None,
        offline: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = _SlidingWindowLimiter(rate_limit)
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "Formula1-RacePredictor (github.com/Ssavan99)"}
        )
        self.offline = offline
        self.requests_made = 0

    # -- caching ------------------------------------------------------------

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        slug = path.strip("/").replace("/", "_") or "root"
        if params:
            slug += "__" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        return self.cache_dir / f"{slug}.json"

    @staticmethod
    def _ttl_for(season: int | str | None) -> float | None:
        """Completed seasons never change; the current one does."""
        if season is None:
            return CURRENT_SEASON_TTL
        try:
            season_int = int(season)
        except (TypeError, ValueError):
            return CURRENT_SEASON_TTL
        return None if season_int < date.today().year else CURRENT_SEASON_TTL

    def _read_cache(self, path: Path, ttl: float | None) -> dict | None:
        if not path.exists():
            return None
        if ttl is not None and time.time() - path.stat().st_mtime > ttl:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("Discarding corrupt cache entry %s", path)
            return None

    # -- fetching -----------------------------------------------------------

    def get(
        self,
        path: str,
        season: int | str | None = None,
        retries: int = 3,
        **params: Any,
    ) -> dict:
        """GET ``path`` under the API root, with caching and rate limiting."""
        params.setdefault("format", "json")
        cache_path = self._cache_path(path, params)
        ttl = self._ttl_for(season)

        cached = self._read_cache(cache_path, ttl)
        if cached is not None:
            return cached

        if self.offline:
            raise JolpicaError(
                f"offline=True and no cache entry for {path} ({cache_path}). "
                "Run the backfill first."
            )

        url = f"{BASE_URL}/{path.strip('/')}/"
        last_error: Exception | None = None
        for attempt in range(retries):
            self._limiter.acquire()
            try:
                response = self._session.get(url, params=params, timeout=45)
                self.requests_made += 1
                if response.status_code == 429:
                    wait = int(response.headers.get("Retry-After", 60))
                    log.warning("HTTP 429 from Jolpica; waiting %ds", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                payload = response.json()
                if "MRData" not in payload:
                    raise JolpicaError(f"unexpected payload shape from {url}")
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                backoff = 2 ** attempt
                log.warning("Jolpica request failed (%s); retrying in %ds", exc, backoff)
                time.sleep(backoff)

        # A stale cache entry beats no data at all, but say so loudly.
        stale = self._read_cache(cache_path, ttl=None)
        if stale is not None:
            log.error("Serving STALE cache for %s after %d failures", path, retries)
            return stale
        raise JolpicaError(f"failed to fetch {url}: {last_error}") from last_error

    def paginate(self, path: str, season: int | str | None = None) -> Iterator[dict]:
        """Yield every page of a paginated endpoint."""
        offset = 0
        while True:
            payload = self.get(path, season=season, limit=PAGE_SIZE, offset=offset)
            data = payload["MRData"]
            yield payload
            total = int(data.get("total", 0))
            offset += PAGE_SIZE
            if offset >= total:
                return

    # -- typed-ish helpers --------------------------------------------------

    def calendar(self, season: int | str) -> list[dict]:
        """Race calendar for a season, including circuit coordinates.

        Refetched on every scheduled run rather than once in January: the
        calendar genuinely changes mid-season (2026 round 16 is listed as
        "Bahrain Grand Prix in Malaysia"), and this is one cheap request.
        """
        races: list[dict] = []
        for page in self.paginate(f"{season}/races", season=season):
            races.extend(page["MRData"]["RaceTable"]["Races"])
        return races

    def results(self, season: int | str) -> list[dict]:
        """All race results for a season, as race records with nested Results."""
        return self._collect_races(f"{season}/results", season)

    def qualifying(self, season: int | str) -> list[dict]:
        return self._collect_races(f"{season}/qualifying", season)

    def sprint(self, season: int | str) -> list[dict]:
        """Sprint results. Empty before 2021; sprint points count for standings."""
        return self._collect_races(f"{season}/sprint", season)

    def _collect_races(self, path: str, season: int | str) -> list[dict]:
        """Merge paginated race records, concatenating their nested result lists."""
        merged: dict[str, dict] = {}
        nested_keys = ("Results", "QualifyingResults", "SprintResults")
        for page in self.paginate(path, season=season):
            for race in page["MRData"]["RaceTable"]["Races"]:
                key = f"{race['season']}_{race['round']}"
                if key not in merged:
                    merged[key] = race
                    continue
                for nested in nested_keys:
                    if nested in race:
                        merged[key].setdefault(nested, []).extend(race[nested])
        return [merged[k] for k in sorted(merged, key=_season_round_key)]

    def next_race(self, today: date | None = None) -> dict | None:
        """The next race not yet run, or ``None`` if the season is over.

        Reads the live calendar, so a cancelled or relocated round is picked up
        automatically rather than being assumed from a stale schedule.
        """
        today = today or datetime.now(timezone.utc).date()
        for season in (today.year, today.year + 1):
            try:
                races = self.calendar(season)
            except JolpicaError:
                continue
            upcoming = [
                r
                for r in races
                if date.fromisoformat(r["date"]) >= today
            ]
            if upcoming:
                return min(upcoming, key=lambda r: r["date"])
        return None

    def last_completed_race(self, today: date | None = None) -> dict | None:
        today = today or datetime.now(timezone.utc).date()
        for season in (today.year, today.year - 1):
            try:
                races = self.calendar(season)
            except JolpicaError:
                continue
            past = [r for r in races if date.fromisoformat(r["date"]) < today]
            if past:
                return max(past, key=lambda r: r["date"])
        return None


def _season_round_key(key: str) -> tuple[int, int]:
    season, rnd = key.split("_")
    return int(season), int(rnd)
