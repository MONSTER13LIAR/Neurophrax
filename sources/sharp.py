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


# ── Chart-level fetch (re-used by every patient-aware tool) ────────────────


@dataclass(slots=True)
class ChartFacts:
    """Patient-chart projection used by every SHARP-aware tool."""

    patient_id: str
    name: str | None = None
    age: int = 0
    birth_date: str | None = None
    medications: list[str] = None  # type: ignore[assignment]
    conditions: list[str] = None  # type: ignore[assignment]
    allergies: list[str] = None  # type: ignore[assignment]
    pregnant: bool = False
    medications_source: str = "none"  # "MedicationStatement" | "MedicationRequest" | "none"
    unparseable_medication_ids: list[str] = None  # type: ignore[assignment]
    unparseable_condition_ids: list[str] = None  # type: ignore[assignment]
    unparseable_allergy_ids: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.medications is None:
            self.medications = []
        if self.conditions is None:
            self.conditions = []
        if self.allergies is None:
            self.allergies = []
        if self.unparseable_medication_ids is None:
            self.unparseable_medication_ids = []
        if self.unparseable_condition_ids is None:
            self.unparseable_condition_ids = []
        if self.unparseable_allergy_ids is None:
            self.unparseable_allergy_ids = []


def _name_from_codeable(cc: dict[str, Any] | None) -> str | None:
    if not cc:
        return None
    text = (cc.get("text") or "").strip()
    if text:
        return text
    for c in cc.get("coding") or []:
        disp = (c.get("display") or "").strip()
        if disp:
            return disp
    return None


def _humanize_name(patient: dict[str, Any]) -> str | None:
    names = patient.get("name") or []
    if not names:
        return None
    first = names[0] or {}
    text = (first.get("text") or "").strip()
    if text:
        return text
    given = " ".join(first.get("given") or []).strip()
    family = (first.get("family") or "").strip()
    full = f"{given} {family}".strip()
    return full or None


def _years_since(date_iso: str) -> int | None:
    try:
        from datetime import date as _date

        d = _date.fromisoformat(date_iso[:10])
    except (ValueError, TypeError):
        return None
    from datetime import datetime as _dt

    today = _dt.now().date()
    return today.year - d.year - ((today.month, today.day) < (d.month, d.day))


async def chart_from_fhir(fhir: "FhirClient", patient_id: str) -> ChartFacts | None:
    """Pull Patient + active MedicationStatement / Condition / AllergyIntolerance.

    Returns None when the patient cannot be read. Failed sub-fetches degrade
    gracefully (the corresponding list is left empty).
    """
    import asyncio as _asyncio

    patient_task = fhir.read(f"Patient/{patient_id}")
    meds_task = fhir.search(
        "MedicationStatement", {"patient": patient_id, "status": "active"}
    )
    conds_task = fhir.search(
        "Condition", {"patient": patient_id, "clinical-status": "active"}
    )
    allergies_task = fhir.search("AllergyIntolerance", {"patient": patient_id})

    # Use return_exceptions=True so a single resource type that the FHIR
    # server rejects (e.g., some sandboxes don't allow AllergyIntolerance
    # search by patient) doesn't drop the whole chart on the floor.
    results = await _asyncio.gather(
        patient_task, meds_task, conds_task, allergies_task, return_exceptions=True
    )
    patient = results[0] if not isinstance(results[0], BaseException) else None
    meds_bundle = results[1] if not isinstance(results[1], BaseException) else None
    conds_bundle = results[2] if not isinstance(results[2], BaseException) else None
    allergies_bundle = results[3] if not isinstance(results[3], BaseException) else None
    if not patient:
        return None

    # Some FHIR sandboxes (and some configurations of HAPI) don't index the
    # MedicationStatement.status or Condition.clinicalStatus token params, so
    # the filtered searches return zero entries even when the resources exist.
    # When the filtered query came back empty, retry without the status filter
    # and rely on each resource's own status field for the active/inactive cut.
    if not bundle_entries(meds_bundle):
        try:
            meds_bundle = await fhir.search("MedicationStatement", {"patient": patient_id})
        except Exception:
            pass
    if not bundle_entries(conds_bundle):
        try:
            conds_bundle = await fhir.search("Condition", {"patient": patient_id})
        except Exception:
            pass

    birth_date = patient.get("birthDate") or ""
    age = _years_since(birth_date) or 0

    medications: list[str] = []
    unparseable_med_ids: list[str] = []
    medications_source = "none"
    for ms in bundle_entries(meds_bundle):
        # Respect status on the resource itself — needed when the server
        # didn't honor `status=active` and returned everything.
        st = (ms.get("status") or "").lower()
        if st and st not in ("active", "intended"):
            continue
        nm = _name_from_codeable(ms.get("medicationCodeableConcept"))
        if nm:
            medications.append(nm)
        else:
            unparseable_med_ids.append(str(ms.get("id") or "<no-id>"))
    if medications:
        medications_source = "MedicationStatement"

    # Many EHR FHIR servers populate MedicationRequest (provider intent) but
    # not MedicationStatement (patient attestation). Fall back when the
    # primary resource yielded zero parseable meds, so safety review isn't
    # silently performed against an empty list.
    if not medications:
        try:
            mreq_bundle = await fhir.search(
                "MedicationRequest", {"patient": patient_id, "status": "active"}
            )
        except Exception:
            mreq_bundle = None
        if not bundle_entries(mreq_bundle):
            try:
                mreq_bundle = await fhir.search("MedicationRequest", {"patient": patient_id})
            except Exception:
                pass
        for mr in bundle_entries(mreq_bundle):
            st = (mr.get("status") or "").lower()
            if st and st not in ("active", "draft"):
                continue
            nm = _name_from_codeable(mr.get("medicationCodeableConcept"))
            if nm:
                medications.append(nm)
            else:
                unparseable_med_ids.append(str(mr.get("id") or "<no-id>"))
        if medications:
            medications_source = "MedicationRequest"

    conditions: list[str] = []
    unparseable_cond_ids: list[str] = []
    pregnant = False
    for c in bundle_entries(conds_bundle):
        # Filter on clinicalStatus when present — needed for fallback queries
        # that returned both active and resolved/inactive conditions.
        cs_code = None
        for coding in (c.get("clinicalStatus") or {}).get("coding") or []:
            code = (coding.get("code") or "").lower()
            if code:
                cs_code = code
                break
        if cs_code and cs_code not in ("active", "recurrence", "relapse"):
            continue
        nm = _name_from_codeable(c.get("code"))
        if not nm:
            unparseable_cond_ids.append(str(c.get("id") or "<no-id>"))
            continue
        conditions.append(nm)
        if "pregnan" in nm.lower():
            pregnant = True

    allergies: list[str] = []
    unparseable_allergy_ids: list[str] = []
    for a in bundle_entries(allergies_bundle):
        nm = _name_from_codeable(a.get("code"))
        if nm:
            allergies.append(nm)
        else:
            unparseable_allergy_ids.append(str(a.get("id") or "<no-id>"))

    return ChartFacts(
        patient_id=patient_id,
        name=_humanize_name(patient),
        age=age,
        birth_date=birth_date or None,
        medications=medications,
        conditions=conditions,
        allergies=allergies,
        pregnant=pregnant,
        medications_source=medications_source,
        unparseable_medication_ids=unparseable_med_ids,
        unparseable_condition_ids=unparseable_cond_ids,
        unparseable_allergy_ids=unparseable_allergy_ids,
    )
