"""RxNorm — drug name normalization (NIH/NLM, public domain)."""
from __future__ import annotations

from ._http import cached_get

BASE = "https://rxnav.nlm.nih.gov/REST"


async def name_to_rxcui(drug: str) -> str | None:
    """Resolve a drug name to its RxNorm Concept Unique Identifier (RxCUI)."""
    try:
        data = await cached_get(f"{BASE}/rxcui.json", {"name": drug, "search": 1})
    except Exception:
        return None
    ids = data.get("idGroup", {}).get("rxnormId") or []
    return ids[0] if ids else None
