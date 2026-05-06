"""Rebuild data/neurophrax.sqlite from upstream reference datasets.

Usage:
    python scripts/build_db.py

Downloads source files into `scripts/raw/` (gitignored) and writes a fresh
SQLite database at `data/neurophrax.sqlite` (committed to the repo).

Currently ingests:
- DDInter 2.0 (drug-drug interactions, ~160k pairs across 8 ATC groups).
- A drugs table with RxNorm RxCUI cross-walks for each DDInter drug — this
  is what allows lookup of common synonyms (e.g., user says "aspirin",
  DDInter calls it "Acetylsalicylic acid").

Future ingests (planned): Beers Criteria 2023, ATC classification, ICD-10-CM.
"""
from __future__ import annotations

import asyncio
import csv
import re
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = Path(__file__).resolve().parent / "raw"
DB_PATH = DATA_DIR / "neurophrax.sqlite"

DDINTER_GROUPS = ["A", "B", "D", "H", "L", "P", "R", "V"]
DDINTER_URL = "http://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_{}.csv"
DDINTER_HOME = "http://ddinter.scbdd.com/"
DDINTER_LICENSE = "Academic / non-commercial use, with attribution"

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
RXNORM_CONCURRENCY = 12

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_versions (
    name        TEXT PRIMARY KEY,
    version     TEXT NOT NULL,
    source_url  TEXT,
    license     TEXT,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ddinter_drugs (
    name_norm     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    ddinter_id    TEXT NOT NULL,
    rxcui         TEXT
);

CREATE INDEX IF NOT EXISTS idx_ddinter_drugs_rxcui
    ON ddinter_drugs(rxcui)
    WHERE rxcui IS NOT NULL;

CREATE TABLE IF NOT EXISTS ddinter_interactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_a_norm     TEXT NOT NULL,
    drug_b_norm     TEXT NOT NULL,
    drug_a_name     TEXT NOT NULL,
    drug_b_name     TEXT NOT NULL,
    drug_a_id       TEXT NOT NULL,
    drug_b_id       TEXT NOT NULL,
    severity        TEXT NOT NULL,
    atc_group       TEXT,
    UNIQUE(drug_a_norm, drug_b_norm)
);

CREATE INDEX IF NOT EXISTS idx_ddinter_a ON ddinter_interactions(drug_a_norm);
CREATE INDEX IF NOT EXISTS idx_ddinter_b ON ddinter_interactions(drug_b_norm);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _download(url: str, target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    req = urllib.request.Request(url, headers={"User-Agent": "Neurophrax/0.2 build_db"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        target.write_bytes(resp.read())
    return target


def _ingest_ddinter_csvs(conn: sqlite3.Connection) -> tuple[int, dict[str, str]]:
    """Ingest DDInter CSVs. Returns (interaction_rows_inserted, drugs_dict)
    where drugs_dict maps name_norm → ddinter_id.
    """
    print("Ingesting DDInter 2.0 CSVs ...")
    total_inserted = 0
    drugs: dict[str, tuple[str, str]] = {}  # name_norm -> (display_name, ddinter_id)

    for code in DDINTER_GROUPS:
        url = DDINTER_URL.format(code)
        target = RAW_DIR / f"ddinter_{code}.csv"
        try:
            _download(url, target)
        except Exception as e:
            print(f"  [{code}] download failed: {e}")
            continue

        rows: list[tuple] = []
        with target.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                a_id = r["DDInterID_A"].strip()
                b_id = r["DDInterID_B"].strip()
                a_name = r["Drug_A"].strip()
                b_name = r["Drug_B"].strip()
                level = r["Level"].strip()
                a_norm = a_name.lower()
                b_norm = b_name.lower()
                drugs.setdefault(a_norm, (a_name, a_id))
                drugs.setdefault(b_norm, (b_name, b_id))
                if a_norm > b_norm:
                    a_id, b_id = b_id, a_id
                    a_name, b_name = b_name, a_name
                    a_norm, b_norm = b_norm, a_norm
                rows.append((a_norm, b_norm, a_name, b_name, a_id, b_id, level, code))

        cur = conn.executemany(
            "INSERT OR IGNORE INTO ddinter_interactions "
            "(drug_a_norm, drug_b_norm, drug_a_name, drug_b_name, "
            " drug_a_id, drug_b_id, severity, atc_group) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        print(f"  [{code}] {len(rows):>6} rows read, {cur.rowcount:>6} inserted")
        total_inserted += cur.rowcount

    conn.executemany(
        "INSERT OR REPLACE INTO ddinter_drugs (name_norm, name, ddinter_id, rxcui) "
        "VALUES (?, ?, ?, NULL)",
        [(norm, name, did) for norm, (name, did) in drugs.items()],
    )
    print(f"  drugs table populated: {len(drugs)} unique drugs")
    return total_inserted, {n: d for n, (_, d) in drugs.items()}


_PAREN = re.compile(r"\s*\([^)]*\)\s*")


async def _resolve_rxcui(client: httpx.AsyncClient, name: str) -> str | None:
    """Try the name as-is, then with parentheticals stripped."""
    candidates = [name]
    stripped = _PAREN.sub(" ", name).strip()
    if stripped and stripped != name:
        candidates.append(stripped)
    for cand in candidates:
        try:
            r = await client.get(
                f"{RXNORM_BASE}/rxcui.json",
                params={"name": cand, "search": 1},
            )
            r.raise_for_status()
            ids = r.json().get("idGroup", {}).get("rxnormId") or []
            if ids:
                return ids[0]
        except Exception:
            continue
    return None


async def _resolve_all_rxcuis(names: list[str]) -> dict[str, str | None]:
    sem = asyncio.Semaphore(RXNORM_CONCURRENCY)
    out: dict[str, str | None] = {}
    progress = {"done": 0, "found": 0}
    total = len(names)

    async with httpx.AsyncClient(timeout=15.0) as client:
        async def one(name: str) -> None:
            async with sem:
                cui = await _resolve_rxcui(client, name)
                out[name] = cui
                progress["done"] += 1
                if cui:
                    progress["found"] += 1
                if progress["done"] % 100 == 0 or progress["done"] == total:
                    print(
                        f"    RxCUI resolution: {progress['done']}/{total}"
                        f" ({progress['found']} resolved)"
                    )

        await asyncio.gather(*(one(n) for n in names))
    return out


def _resolve_and_store_rxcuis(conn: sqlite3.Connection) -> int:
    print("Resolving DDInter drug names to RxNorm RxCUIs ...")
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM ddinter_drugs WHERE rxcui IS NULL ORDER BY name"
        )
    ]
    if not names:
        print("  (no unresolved drugs)")
        return 0
    print(f"  {len(names)} drugs to resolve")
    resolved = asyncio.run(_resolve_all_rxcuis(names))
    updates = [(cui, name.lower()) for name, cui in resolved.items() if cui]
    conn.executemany(
        "UPDATE ddinter_drugs SET rxcui = ? WHERE name_norm = ?", updates
    )
    print(f"  {len(updates)} RxCUIs stored")
    return len(updates)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        _ingest_ddinter_csvs(conn)
        conn.commit()
        _resolve_and_store_rxcuis(conn)
        conn.execute(
            "INSERT OR REPLACE INTO dataset_versions "
            "(name, version, source_url, license, ingested_at) VALUES (?,?,?,?,?)",
            ("ddinter", "2.0", DDINTER_HOME, DDINTER_LICENSE, _now()),
        )
        conn.commit()

        n_pairs = conn.execute("SELECT COUNT(*) FROM ddinter_interactions").fetchone()[0]
        n_drugs = conn.execute("SELECT COUNT(*) FROM ddinter_drugs").fetchone()[0]
        n_with_cui = conn.execute(
            "SELECT COUNT(*) FROM ddinter_drugs WHERE rxcui IS NOT NULL"
        ).fetchone()[0]
        print(
            f"\nFinal: {n_pairs} pairs, {n_drugs} drugs ({n_with_cui} with RxCUI)"
        )
        print(f"DB: {DB_PATH} ({DB_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
