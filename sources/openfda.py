"""OpenFDA — FDA drug labels (public domain)."""
from __future__ import annotations

from typing import Any

import httpx

from ._http import cached_get

BASE = "https://api.fda.gov/drug"


async def label_for(
    medication: str, *, prefer_monotherapy: bool = True
) -> dict[str, Any] | None:
    """Fetch an FDA label for a drug, trying generic name then brand.

    When `prefer_monotherapy` is set, scans the top results and prefers a
    label whose `substance_name` is a single ingredient matching the query —
    avoids returning combination-drug labels (e.g. for "metformin", skips
    "SITAGLIPTIN AND METFORMIN HYDROCHLORIDE" in favor of plain metformin).
    """
    needle = medication.strip().upper()
    for field in ("openfda.generic_name", "openfda.brand_name"):
        try:
            data = await cached_get(
                f"{BASE}/label.json",
                {
                    "search": f'{field}:"{medication}"',
                    "limit": 5 if prefer_monotherapy else 1,
                },
            )
        except httpx.HTTPStatusError:
            continue
        except Exception:
            continue
        results = data.get("results") or []
        if not results:
            continue
        if prefer_monotherapy:
            for r in results:
                subs = (r.get("openfda", {}) or {}).get("substance_name") or []
                if len(subs) == 1 and needle in subs[0].upper():
                    return r
        return results[0]
    return None
