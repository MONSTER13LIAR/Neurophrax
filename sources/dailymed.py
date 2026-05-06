"""DailyMed — NLM's official SPL (Structured Product Labeling) repository.

Useful for getting links to manufacturer-specific FDA labels with publication
dates. Complements OpenFDA which provides parsed label sections.

API docs: https://dailymed.nlm.nih.gov/dailymed/app-support-mapi.cfm
"""
from __future__ import annotations

from typing import Any

from ._http import cached_get

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
WEB_BASE = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"


async def search_spls(drug_name: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search Structured Product Labels by drug name.

    Returns label metadata with set_ids and consumer-facing DailyMed URLs.
    """
    try:
        data = await cached_get(
            f"{BASE}/spls.json",
            {"drug_name": drug_name, "pagesize": limit},
        )
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for item in (data.get("data") or [])[:limit]:
        set_id = item.get("setid", "")
        out.append({
            "set_id": set_id,
            "title": item.get("title", ""),
            "published_date": item.get("published_date", ""),
            "spl_version": item.get("spl_version"),
            "url": f"{WEB_BASE}?setid={set_id}" if set_id else None,
        })
    return out
