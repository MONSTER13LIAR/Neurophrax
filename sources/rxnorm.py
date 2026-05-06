"""RxNorm — drug name normalization (NIH/NLM, public domain)."""
from __future__ import annotations

from ._http import cached_get

BASE = "https://rxnav.nlm.nih.gov/REST"


async def name_to_rxcui(drug: str) -> str | None:
    """Resolve a drug name to its RxNorm Concept Unique Identifier (RxCUI).

    Uses normalized search; returns whatever match RxNorm prioritizes — may
    be a salt form, brand, or precise ingredient depending on the input.
    """
    try:
        data = await cached_get(f"{BASE}/rxcui.json", {"name": drug, "search": 1})
    except Exception:
        return None
    ids = data.get("idGroup", {}).get("rxnormId") or []
    return ids[0] if ids else None


async def name_to_ingredient_rxcui(drug: str) -> str | None:
    """Resolve a drug name to its Ingredient (IN-tty) RxCUI when possible.

    Strategy: get any RxCUI for the name, then traverse `/related.json?tty=IN`
    to find the canonical ingredient. The IN RxCUI is what MedlinePlus
    Connect and RxClass treat as authoritative — salt-form or brand RxCUIs
    sometimes map to wrong patient content downstream (e.g., RxCUI 235743
    "metformin hydrochloride" returns Canagliflozin info on MedlinePlus).
    """
    base_cui = await name_to_rxcui(drug)
    if not base_cui:
        return None
    try:
        data = await cached_get(
            f"{BASE}/rxcui/{base_cui}/related.json", {"tty": "IN"}
        )
    except Exception:
        return base_cui
    groups = (data.get("relatedGroup") or {}).get("conceptGroup") or []
    for g in groups:
        if g.get("tty") != "IN":
            continue
        for concept in g.get("conceptProperties") or []:
            cui = concept.get("rxcui")
            if cui:
                return cui
    return base_cui
