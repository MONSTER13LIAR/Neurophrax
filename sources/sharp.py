"""SHARP-on-MCP context propagation.

Implements the headers-based healthcare context model expected by MCP hosts
that follow the SHARP-on-MCP specification (https://sharponmcp.com/). Every
inbound MCP request may carry:

- ``x-fhir-server-url``    — base URL of the FHIR R4 server to use
- ``x-fhir-access-token``  — bearer JWT (its ``patient`` claim, when present,
                             implicitly identifies the active patient)
- ``x-patient-id``         — explicit patient ID fallback when no token

Tools that need patient context call :func:`get_fhir_context` and
:func:`resolve_patient_id` to extract these from the active request, then
build a :class:`FhirClient` to query the server. Tools that do not need
patient context ignore this module entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from mcp.server.fastmcp import Context

FHIR_SERVER_URL_HEADER = "x-fhir-server-url"
FHIR_ACCESS_TOKEN_HEADER = "x-fhir-access-token"
PATIENT_ID_HEADER = "x-patient-id"


@dataclass(slots=True)
class FhirContext:
    """Active FHIR connection details, parsed from request headers."""

    url: str
    token: str | None = None


def get_fhir_context(ctx: Context | None) -> FhirContext | None:
    """Return the FHIR context for the active request, or None if absent."""
    if ctx is None:
        return None
    req = getattr(ctx.request_context, "request", None)
    if req is None:
        return None
    url = req.headers.get(FHIR_SERVER_URL_HEADER)
    if not url:
        return None
    token = req.headers.get(FHIR_ACCESS_TOKEN_HEADER)
    return FhirContext(url=url, token=token)


def resolve_patient_id(ctx: Context | None, override: str | None = None) -> str | None:
    """Resolve the active patient ID.

    Order: explicit ``override`` arg → ``patient`` claim in JWT → header.
    """
    if override:
        return override
    if ctx is None:
        return None
    req = getattr(ctx.request_context, "request", None)
    if req is None:
        return None
    token = req.headers.get(FHIR_ACCESS_TOKEN_HEADER)
    if token:
        try:
            claims = jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError:
            claims = {}
        patient = claims.get("patient")
        if patient:
            return str(patient)
    return req.headers.get(PATIENT_ID_HEADER)


class FhirClient:
    """Minimal async FHIR R4 read/search client."""

    def __init__(self, fhir_context: FhirContext, *, timeout: float = 15.0) -> None:
        self._base = fhir_context.url.rstrip("/")
        self._token = fhir_context.token
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/fhir+json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        path = path.lstrip("/")
        url = f"{self._base}/{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(url, headers=self._headers(), params=params)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()

    async def read(self, path: str) -> dict[str, Any] | None:
        """Read a single FHIR resource by path (e.g. ``Patient/123``)."""
        return await self._get(path)

    async def search(
        self, resource_type: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Search a FHIR resource type, returning the raw Bundle."""
        return await self._get(resource_type, params=params)


def bundle_entries(bundle: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract resource entries from a FHIR Bundle (handles None safely)."""
    if not bundle:
        return []
    return [e["resource"] for e in (bundle.get("entry") or []) if e.get("resource")]
