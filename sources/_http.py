"""Shared async HTTP client with TTL caching and retry-on-transient-failure.

All outbound calls from `sources/*` go through `cached_get`. Failures on
client errors (4xx) are not retried; transient failures (timeouts, 5xx) are
retried with exponential backoff.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
from cachetools import TTLCache
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

DEFAULT_TIMEOUT = 15.0
_CACHE_MAXSIZE = 2048
_CACHE_TTL_SECONDS = 900  # 15 minutes

_cache: TTLCache[str, Any] = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL_SECONDS)


class UpstreamError(RuntimeError):
    """Raised when an upstream API returns an unrecoverable error."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return False


def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    payload = json.dumps({"u": url, "p": params or {}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
async def _fetch(url: str, params: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


async def cached_get(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """GET `url` with `params`, caching successful JSON responses for 15 minutes."""
    key = _cache_key(url, params)
    if key in _cache:
        return _cache[key]
    data = await _fetch(url, params, timeout)
    _cache[key] = data
    return data
