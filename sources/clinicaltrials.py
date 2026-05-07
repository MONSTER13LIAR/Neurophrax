"""ClinicalTrials.gov v2 API (NIH/NLM, public domain).

Free, no-auth JSON API for matching patients to actively recruiting clinical
trials. We focus on a narrow projection of the protocol so the tool output
stays small enough for an agent to reason over.

API docs: https://clinicaltrials.gov/data-api/api
"""
from __future__ import annotations

from typing import Any

from ._http import cached_get

BASE = "https://clinicaltrials.gov/api/v2"
WEB_BASE = "https://clinicaltrials.gov/study"

# Active recruitment statuses an agent normally cares about.
RECRUITING_STATUSES = ("RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION")


def _study_url(nct_id: str) -> str:
    return f"{WEB_BASE}/{nct_id}" if nct_id else ""


def _project_study(study: dict[str, Any]) -> dict[str, Any]:
    proto = study.get("protocolSection") or {}
    ident = proto.get("identificationModule") or {}
    status_mod = proto.get("statusModule") or {}
    design = proto.get("designModule") or {}
    eligibility = proto.get("eligibilityModule") or {}
    cond_mod = proto.get("conditionsModule") or {}
    arms = proto.get("armsInterventionsModule") or {}
    contacts = proto.get("contactsLocationsModule") or {}

    locations: list[dict[str, str]] = []
    for loc in (contacts.get("locations") or [])[:5]:
        locations.append({
            "facility": loc.get("facility") or "",
            "city": loc.get("city") or "",
            "state": loc.get("state") or "",
            "country": loc.get("country") or "",
            "status": loc.get("status") or "",
        })

    interventions: list[str] = []
    for iv in arms.get("interventions") or []:
        name = iv.get("name") or ""
        if name:
            interventions.append(name)

    nct_id = ident.get("nctId") or ""
    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle") or "",
        "status": status_mod.get("overallStatus") or "",
        "phase": (design.get("phases") or [None])[0],
        "study_type": design.get("studyType") or "",
        "conditions": cond_mod.get("conditions") or [],
        "interventions": interventions,
        "minimum_age": eligibility.get("minimumAge") or "",
        "maximum_age": eligibility.get("maximumAge") or "",
        "sex": eligibility.get("sex") or "",
        "eligibility_summary": (eligibility.get("eligibilityCriteria") or "")[:600],
        "locations": locations,
        "url": _study_url(nct_id),
    }


async def search_trials(
    *,
    condition: str,
    location: str | None = None,
    extra_term: str | None = None,
    statuses: tuple[str, ...] = RECRUITING_STATUSES,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search ClinicalTrials.gov for trials matching ``condition``.

    Returns a projection containing the fields a clinician or agent typically
    needs: title, status, phase, eligibility summary, interventions, top
    locations, and the canonical study URL.
    """
    params: dict[str, Any] = {
        "query.cond": condition,
        "filter.overallStatus": "|".join(statuses),
        "pageSize": min(max(limit, 1), 50),
        "format": "json",
    }
    if location:
        params["query.locn"] = location
    if extra_term:
        params["query.term"] = extra_term

    try:
        data = await cached_get(f"{BASE}/studies", params)
    except Exception:
        return []
    studies = data.get("studies") or []
    return [_project_study(s) for s in studies[:limit]]
