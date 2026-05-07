"""End-to-end demo of the Neurophrax tool surface.

Runs every tool against a single coherent scenario — an 82-year-old on a
medication list designed to exercise interactions, Beers Criteria, boxed
warnings, and (post hoc) clinical-trial / evidence / vaccine matching.

Usage:
    .venv/bin/python scripts/demo.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources import safety_review, clinicaltrials, pubmed, vaccines


SCENARIO = {
    "name": "Margaret O.",
    "age": 82,
    "medications": [
        "warfarin",
        "amiodarone",
        "amoxicillin",
        "lorazepam",
        "diphenhydramine",
    ],
    "conditions": ["atrial fibrillation", "insomnia", "copd"],
    "pregnant": False,
    "patient_id": "demo-margaret-o",
}


def _hr() -> None:
    print("─" * 78)


async def run_safety_review() -> None:
    _hr()
    print("1. medication_safety_review — composite risk across 4 signal types")
    _hr()
    p = safety_review.PatientFacts(
        age=SCENARIO["age"],
        medications=SCENARIO["medications"],
        conditions=SCENARIO["conditions"],
        pregnant=SCENARIO["pregnant"],
        patient_id=SCENARIO["patient_id"],
        source="direct",
    )
    r = await safety_review.run(p)
    print(f"  patient    : {SCENARIO['name']} ({SCENARIO['age']} yo)")
    print(f"  meds       : {', '.join(SCENARIO['medications'])}")
    print(f"  conditions : {', '.join(SCENARIO['conditions'])}")
    print(f"  findings   : {len(r.findings)}")
    for f in r.findings:
        print(f"    [{f.severity:>17}] {f.category:<16}  {f.outcome[:80]}")
    print(f"\n  FHIR RiskAssessment predictions: {len(r.fhir_resource['prediction'])}")


async def run_trials() -> None:
    _hr()
    print("2. find_clinical_trials — actively recruiting AF studies")
    _hr()
    trials = await clinicaltrials.search_trials(
        condition="atrial fibrillation", limit=5
    )
    for t in trials:
        print(f"  {t['nct_id']}  [{t['status']}]  {t['title'][:80]}")
        print(f"      {t['url']}")


async def run_evidence() -> None:
    _hr()
    print("3. search_evidence — meta-analyses & RCTs on Beers-flagged combo")
    _hr()
    out = await pubmed.search(
        question="elderly polypharmacy warfarin amiodarone bleeding risk",
        study_types=["meta-analysis", "rct", "systematic-review"],
        max_age_years=10,
        limit=5,
    )
    print(f"  composed query: {out['query']}")
    print(f"  total hits    : {out['total']}    returned: {len(out['results'])}")
    for r in out["results"]:
        print(f"  PMID {r['pmid']}  {r['pub_date']}  {r['title'][:75]}")


def run_vaccines() -> None:
    _hr()
    print("4. recommend_vaccines — ACIP recommendations for the patient context")
    _hr()
    matches = vaccines.recommendations_for(
        age=SCENARIO["age"],
        conditions=SCENARIO["conditions"],
        pregnant=SCENARIO["pregnant"],
    )
    for m in matches:
        print(f"  [{m.indication:>15}]  {m.name}")
        print(f"      schedule: {m.schedule}")


async def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                    NEUROPHRAX — END-TO-END DEMO                          ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print(f"  scenario: {SCENARIO['name']}, {SCENARIO['age']} yo on {len(SCENARIO['medications'])} medications")
    print()

    await run_safety_review()
    print()
    await run_trials()
    print()
    await run_evidence()
    print()
    run_vaccines()
    print()
    _hr()
    print("Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
