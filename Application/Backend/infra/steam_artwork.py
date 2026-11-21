"""Best-effort Steam header artwork, independent of recommendation data."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from time import monotonic
from typing import Any, Callable
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)
HTTP_TIMEOUT_SECONDS = 3
SUCCESS_TTL_SECONDS = 24 * 60 * 60
FAILURE_TTL_SECONDS = 10 * 60
MAX_CACHE_ENTRIES = 4096
_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="steam-artwork")


def _fetch_store_image(appid: int) -> str | None:
    query = urlencode({"appids": appid, "filters": "basic", "l": "english"})
    request = Request(
        f"https://store.steampowered.com/api/appdetails?{query}",
        headers={"User-Agent": "GamesRecommender/1.0", "Accept": "application/json"},
    )
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read(2 * 1024 * 1024))
    entry = payload.get(str(appid)) if isinstance(payload, dict) else None
    if not isinstance(entry, dict) or entry.get("success") is not True:
        return None
    data = entry.get("data")
    image = data.get("header_image") if isinstance(data, dict) else None
    if not isinstance(image, str):
        return None
    parsed = urlparse(image)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not (
        host == "steamstatic.com" or host.endswith(".steamstatic.com")
        or host == "steamcdn-a.akamaihd.net"
    ):
        return None
    return image


def _fetch_cdn_image(appid: int) -> str | None:
    """Verify Steam's legacy header asset when Store metadata is unavailable."""
    image = f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"
    request = Request(image, method="HEAD", headers={"User-Agent": "GamesRecommender/1.0"})
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        content_type = response.headers.get("Content-Type", "")
        if response.status == 200 and isinstance(content_type, str) and content_type.lower().startswith("image/"):
            return image
    return None


def _fetch_image(appid: int) -> str | None:
    try:
        image = _fetch_store_image(appid)
        if image:
            return image
    except (OSError, ValueError) as exc:
        logger.info("Steam Store artwork lookup failed for AppID %s: %s", appid, exc)
    try:
        return _fetch_cdn_image(appid)
    except (OSError, ValueError) as exc:
        logger.info("Steam CDN artwork unavailable for AppID %s: %s", appid, exc)
        return None


class SteamArtwork:
    def __init__(
        self,
        fetcher: Callable[[int], str | None] = _fetch_image,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._fetcher = fetcher
        self._clock = clock
        self._cache: OrderedDict[int, tuple[float, str | None]] = OrderedDict()
        self._lock = RLock()

    def lookup(self, appid: int) -> str | None:
        if type(appid) is not int or appid <= 0:
            return None
        with self._lock:
            cached = self._cache.get(appid)
            if cached and cached[0] > self._clock():
                self._cache.move_to_end(appid)
                return cached[1]
        try:
            image = self._fetcher(appid)
        except Exception as exc:
            # Artwork must never turn a valid recommendation into an API error.
            logger.info("Steam artwork unavailable for AppID %s: %s", appid, exc)
            image = None
        ttl = SUCCESS_TTL_SECONDS if image else FAILURE_TTL_SECONDS
        with self._lock:
            self._cache[appid] = (self._clock() + ttl, image)
            self._cache.move_to_end(appid)
            while len(self._cache) > MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)
        return image

    def enrich(self, games: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique_ids = dict.fromkeys(game["appid"] for game in games)
        futures = {appid: _executor.submit(self.lookup, appid) for appid in unique_ids}
        images = {appid: future.result() for appid, future in futures.items()}
        return [{**game, "image_url": images[game["appid"]]} for game in games]


steam_artwork = SteamArtwork()
