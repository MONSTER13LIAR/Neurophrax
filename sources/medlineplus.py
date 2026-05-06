"""MedlinePlus Connect — patient-friendly health topic lookup (NLM).

Given a clinical code (RxNorm RxCUI, ICD-10-CM, etc.) returns plain-language
health topics in English or Spanish, sourced from medlineplus.gov.

API docs: https://medlineplus.gov/connect/technical.html
"""
from __future__ import annotations

from typing import Any

from ._http import cached_get

BASE = "https://connect.medlineplus.gov/service"

# OID code-system identifiers used by MedlinePlus Connect
CS_RXNORM = "2.16.840.1.113883.6.88"
CS_ICD10CM = "2.16.840.1.113883.6.90"
CS_SNOMED = "2.16.840.1.113883.6.96"
CS_LOINC = "2.16.840.1.113883.6.1"


def _coerce_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


async def info_for(
    code_system: str, code: str, *, language: str = "en"
) -> list[dict[str, Any]]:
    """Look up patient-friendly health topics for a clinical code.

    Returns a list of {title, url, summary} entries. Empty list if not found.
    """
    if not code:
        return []
    try:
        data = await cached_get(
            BASE,
            {
                "mainSearchCriteria.v.cs": code_system,
                "mainSearchCriteria.v.c": code,
                "knowledgeResponseType": "application/json",
                "informationRecipient.languageCode.c": language,
            },
        )
    except Exception:
        return []

    feed = data.get("feed") or {}
    entries = _coerce_list(feed.get("entry"))
    out: list[dict[str, Any]] = []
    for entry in entries:
        title = ((entry.get("title") or {}).get("_value") or "").strip()
        links = _coerce_list(entry.get("link"))
        url = ""
        for link in links:
            href = (link.get("href") or "").strip()
            if href:
                url = href.split("?", 1)[0]  # strip tracking params
                break
        summary = ((entry.get("summary") or {}).get("_value") or "").strip()
        if title or url:
            out.append({
                "title": title,
                "url": url or None,
                "summary": (summary[:500] + "…") if len(summary) > 500 else (summary or None),
            })
    return out
