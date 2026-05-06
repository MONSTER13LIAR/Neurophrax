"""Drug-drug interaction lookups.

Primary: bundled DDInter 2.0 dataset in `data/neurophrax.sqlite`.
Fallback: live parsing of the OpenFDA drug label `drug_interactions` field
when a pair is missing from DDInter.

The lookup is RxCUI-aware: each DDInter drug carries its RxNorm RxCUI, so a
user input of "aspirin" (RxCUI 1191) will resolve to DDInter's canonical
"Acetylsalicylic acid" entry even though the strings differ.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from . import openfda

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "neurophrax.sqlite"

DDINTER_HOME = "http://ddinter.scbdd.com/"


def _canon(a: str, b: str) -> tuple[str, str]:
    a_n = a.strip().lower()
    b_n = b.strip().lower()
    return (a_n, b_n) if a_n <= b_n else (b_n, a_n)


def _connect() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH)


def dataset_version() -> str | None:
    """Return the DDInter version recorded at ingest time, or None if not built."""
    conn = _connect()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT version FROM dataset_versions WHERE name = 'ddinter'"
        ).fetchone()
        return row[0] if row else None
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()


def _resolve_canonical_name(
    conn: sqlite3.Connection, drug: str, rxcui: str | None
) -> str | None:
    """Map user input to DDInter's canonical normalized name.

    Tries direct name match first, then RxCUI cross-walk via ddinter_drugs.
    Returns the `name_norm` of the matching DDInter entry, or None.
    """
    direct = drug.strip().lower()
    row = conn.execute(
        "SELECT name_norm FROM ddinter_drugs WHERE name_norm = ?", (direct,)
    ).fetchone()
    if row:
        return row[0]
    if rxcui:
        row = conn.execute(
            "SELECT name_norm FROM ddinter_drugs WHERE rxcui = ? LIMIT 1", (rxcui,)
        ).fetchone()
        if row:
            return row[0]
    return None


def _lookup_pair_canonical(
    conn: sqlite3.Connection, a_norm: str, b_norm: str
) -> dict[str, Any] | None:
    """Look up a DDInter interaction by already-canonicalized name_norms."""
    if a_norm > b_norm:
        a_norm, b_norm = b_norm, a_norm
    row = conn.execute(
        "SELECT drug_a_name, drug_b_name, severity, drug_a_id, drug_b_id "
        "FROM ddinter_interactions "
        "WHERE drug_a_norm = ? AND drug_b_norm = ?",
        (a_norm, b_norm),
    ).fetchone()
    if not row:
        return None
    return {
        "drug_a": row[0],
        "drug_b": row[1],
        "severity": row[2],
        "description": None,
        "source": "DDInter 2.0",
        "evidence": {"drug_a_id": row[3], "drug_b_id": row[4]},
    }


_SENTENCE_RE = re.compile(r"[^.]*\.")


def _extract_snippet(text: str, needle: str) -> str:
    needle_re = re.compile(rf"\b{re.escape(needle)}\b", re.IGNORECASE)
    for match in _SENTENCE_RE.finditer(text):
        sentence = match.group(0).strip()
        if needle_re.search(sentence):
            return sentence[:400]
    return ""


async def lookup_pair_label(drug_a: str, drug_b: str) -> dict[str, Any] | None:
    """OpenFDA fallback: scan either drug's label for mention of the other."""
    for primary, other in ((drug_a, drug_b), (drug_b, drug_a)):
        label = await openfda.label_for(primary)
        if not label:
            continue
        raw = label.get("drug_interactions") or []
        if isinstance(raw, str):
            raw = [raw]
        text = " ".join(raw)
        if not text:
            continue
        if not re.search(rf"\b{re.escape(other)}\b", text, re.IGNORECASE):
            continue
        snippet = _extract_snippet(text, other)
        return {
            "drug_a": primary.title(),
            "drug_b": other.title(),
            "severity": "Unknown",
            "description": snippet,
            "source": "OpenFDA Drug Label",
            "evidence": {
                "label_set_id": label.get("set_id") or label.get("id"),
            },
        }
    return None


async def check_pairs(
    drugs: list[str], rxcuis: list[str | None]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Check every unique pair in `drugs` for interactions.

    `rxcuis[i]` is the pre-resolved RxCUI for `drugs[i]` (or None). DDInter is
    queried first by canonical name (direct or RxCUI-mapped); pairs missing
    there fall back to OpenFDA label parsing.
    """
    interactions: list[dict[str, Any]] = []
    counts = {"ddinter": 0, "openfda_label": 0}
    seen: set[tuple[str, str]] = set()

    conn = _connect()
    canonical: list[str | None] = []
    if conn is not None:
        try:
            for d, c in zip(drugs, rxcuis):
                canonical.append(_resolve_canonical_name(conn, d, c))
        finally:
            pass  # keep open until after pair lookups
    else:
        canonical = [None] * len(drugs)

    try:
        for i, a in enumerate(drugs):
            for j in range(i + 1, len(drugs)):
                b = drugs[j]
                key = _canon(a, b)
                if key in seen or key[0] == key[1]:
                    continue
                seen.add(key)

                local: dict[str, Any] | None = None
                if conn is not None and canonical[i] and canonical[j]:
                    local = _lookup_pair_canonical(conn, canonical[i], canonical[j])

                if local:
                    interactions.append(local)
                    counts["ddinter"] += 1
                    continue

                via_label = await lookup_pair_label(a, b)
                if via_label:
                    interactions.append(via_label)
                    counts["openfda_label"] += 1
    finally:
        if conn is not None:
            conn.close()

    return interactions, counts
