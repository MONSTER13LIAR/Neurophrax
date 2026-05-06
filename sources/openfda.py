"""OpenFDA — FDA drug labels (public domain)."""
from __future__ import annotations

from typing import Any

import httpx

from ._http import cached_get

BASE = "https://api.fda.gov/drug"


async def label_for(medication: str) -> dict[str, Any] | None:
    """Fetch the latest FDA label for a drug, trying generic name then brand."""
    for field in ("openfda.generic_name", "openfda.brand_name"):
        try:
            data = await cached_get(
                f"{BASE}/label.json",
                {"search": f'{field}:"{medication}"', "limit": 1},
            )
        except httpx.HTTPStatusError:
            continue
        except Exception:
            continue
        results = data.get("results", [])
        if results:
            return results[0]
    return None
