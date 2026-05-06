"""RxClass — drug classification cross-walks (NLM, public domain).

Maps an RxNorm RxCUI to its therapeutic classifications:
- ATC (WHO Anatomical Therapeutic Chemical, international)
- EPC (FDA Established Pharmacologic Class, US labeling)
"""
from __future__ import annotations

from typing import Any

from ._http import cached_get

BASE = "https://rxnav.nlm.nih.gov/REST/rxclass"


async def _fetch_classes(
    params: dict[str, Any], own_rxcui: str | None = None
) -> list[dict[str, Any]]:
    """Run a byRxcui query and return de-duplicated class entries.

    When `own_rxcui` is given, prefers rows whose minConcept.rxcui matches —
    this filters out combination-product class memberships (e.g., simvastatin
    only, not sitagliptin/simvastatin combos). Falls back to all rows if the
    strict filter would yield nothing.
    """
    try:
        data = await cached_get(f"{BASE}/class/byRxcui.json", params)
    except Exception:
        return []
    rows = (data.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo") or []
    if not rows:
        return []

    def collect(filter_strict: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            min_concept = row.get("minConcept") or {}
            if filter_strict and own_rxcui and min_concept.get("rxcui") != own_rxcui:
                continue
            item = row.get("rxclassMinConceptItem") or {}
            class_id = item.get("classId")
            if not class_id or class_id in seen:
                continue
            seen.add(class_id)
            out.append({
                "class_id": class_id,
                "class_name": item.get("className"),
                "class_type": item.get("classType"),
            })
        return out

    return collect(filter_strict=True) or collect(filter_strict=False)


async def classes_for_rxcui(rxcui: str) -> dict[str, list[dict[str, Any]]]:
    """Return ATC and EPC class memberships for an RxCUI."""
    if not rxcui:
        return {"atc": [], "epc": []}
    atc = await _fetch_classes(
        {"rxcui": rxcui, "relaSource": "ATC"}, own_rxcui=rxcui
    )
    epc = await _fetch_classes(
        {"rxcui": rxcui, "relaSource": "DAILYMED", "relas": "has_epc"},
        own_rxcui=rxcui,
    )
    return {"atc": atc, "epc": epc}
