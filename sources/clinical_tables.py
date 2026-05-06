"""NIH Clinical Tables Search Service — fast, free, no-key clinical lookups.

Endpoints used:
- conditions  → FHIR-curated condition names + MedlinePlus consumer links
- icd10cm     → official ICD-10-CM billable diagnosis codes
- loinc_items → LOINC laboratory test/observation codes

API docs: https://clinicaltables.nlm.nih.gov/apidoc/

Response shape is a 4-element list: [total, [keys], extras, [[col, ...], ...]].
We extract the row data column-by-column based on the `df` we requested.
"""
from __future__ import annotations

from typing import Any

from ._http import cached_get

BASE = "https://clinicaltables.nlm.nih.gov/api"


def _rows(data: Any) -> list[list[Any]]:
    if not isinstance(data, list) or len(data) < 4:
        return []
    return data[3] or []


async def search_conditions(term: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search the FHIR-curated Conditions Value Set.

    Each result has a curated primary name, an internal key, a consumer-
    friendly synonym, and a MedlinePlus reference URL where available.
    """
    df = "primary_name,key_id,consumer_name,info_link_data"
    try:
        data = await cached_get(
            f"{BASE}/conditions/v3/search",
            {"terms": term, "maxList": limit, "df": df},
        )
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for row in _rows(data):
        padded = (list(row) + ["", "", "", ""])[:4]
        primary, key_id, consumer, info_link = padded
        url = topic = ""
        if info_link:
            parts = info_link.split(",", 1)
            url = parts[0].strip()
            topic = parts[1].strip() if len(parts) > 1 else ""
        out.append({
            "name": primary,
            "consumer_name": consumer or primary,
            "key_id": key_id,
            "medlineplus_url": url or None,
            "medlineplus_topic": topic or None,
        })
    return out


async def search_icd10(term: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search ICD-10-CM diagnosis codes by term."""
    try:
        data = await cached_get(
            f"{BASE}/icd10cm/v3/search",
            {"terms": term, "maxList": limit, "sf": "name", "df": "code,name"},
        )
    except Exception:
        return []
    return [
        {"code": r[0], "name": r[1]}
        for r in _rows(data)
        if isinstance(r, list) and len(r) >= 2
    ]


async def search_loinc(term: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search LOINC for laboratory tests/observations."""
    try:
        data = await cached_get(
            f"{BASE}/loinc_items/v3/search",
            {"terms": term, "maxList": limit, "df": "LOINC_NUM,LONG_COMMON_NAME"},
        )
    except Exception:
        return []
    return [
        {"loinc_num": r[0], "name": r[1]}
        for r in _rows(data)
        if isinstance(r, list) and len(r) >= 2
    ]
