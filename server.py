"""Neurophrax — Healthcare data layer for AI agents.

Tools synthesize live data from NIH RxNorm, FDA OpenFDA, and SNOMED CT
(Snowstorm) into structured responses suitable for agent reasoning. Each
tool returns both a human-readable `summary` and machine-readable `data`
plus explicit source attribution. Educational use only.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from sources import clinical_tables, interactions, openfda, rxnorm

mcp = FastMCP(
    "Neurophrax",
    instructions=(
        "Healthcare data layer backed by NIH RxNorm, FDA OpenFDA, and SNOMED CT. "
        "Tools return both a human-readable summary and structured data with "
        "explicit source attribution. Educational use only — not for clinical decisions."
    ),
)

DISCLAIMER = "For informational use only. Not for clinical decisions."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _response(
    *,
    summary: str,
    data: dict[str, Any],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "summary": summary,
        "data": data,
        "sources": sources,
        "disclaimer": DISCLAIMER,
    }


# ── Tool 1: Drug Interactions ───────────────────────────────────────────────

_SEVERITY_EMOJI = {
    "MAJOR": "🔴",
    "MODERATE": "🟡",
    "MINOR": "🟢",
    "UNKNOWN": "⚪",
}


@mcp.tool()
async def check_drug_interactions(
    drugs: Annotated[
        list[str],
        Field(min_length=2, description="Two or more drug names to check for interactions"),
    ],
) -> dict[str, Any]:
    """Check for drug-drug interactions across every unique pair.

    Primary source: bundled DDInter 2.0 dataset (severity-rated, ~160k pairs).
    Fallback: parses the OpenFDA drug label `drug_interactions` section for
    pairs not in DDInter. RxNorm RxCUIs are resolved in parallel for downstream
    use by agents.
    """
    rxcui_results = await asyncio.gather(*[rxnorm.name_to_rxcui(d) for d in drugs])
    found, counts = await interactions.check_pairs(drugs, rxcui_results)

    rxcuis = dict(zip(drugs, rxcui_results))
    resolved = {d: c for d, c in rxcuis.items() if c}
    unresolved = [d for d, c in rxcuis.items() if not c]
    ddinter_version = interactions.dataset_version()

    sources_meta: list[dict[str, Any]] = [
        {"name": "RxNorm", "url": rxnorm.BASE, "accessed_at": _now()},
        {
            "name": "DDInter 2.0",
            "url": interactions.DDINTER_HOME,
            "version": ddinter_version or "not ingested",
            "accessed_at": _now(),
        },
    ]
    if counts.get("openfda_label", 0):
        sources_meta.append(
            {"name": "OpenFDA Drug Label", "url": openfda.BASE, "accessed_at": _now()}
        )

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║        NEUROPHRAX — DRUG INTERACTION CHECK       ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Drugs    : {', '.join(d.title() for d in drugs)}",
        f"  Sources  : DDInter {ddinter_version or '(not ingested)'} + OpenFDA labels (fallback)",
        "",
    ]
    if unresolved:
        lines += [f"  ⚠️  Could not resolve to RxCUI: {', '.join(unresolved)}", ""]

    if found:
        lines.append(f"  🚨 {len(found)} INTERACTION(S) DETECTED:")
        lines.append(
            f"     ({counts['ddinter']} from DDInter, "
            f"{counts['openfda_label']} from OpenFDA labels)"
        )
        lines.append("")
        for inter in found:
            severity = (inter["severity"] or "Unknown").upper()
            emoji = _SEVERITY_EMOJI.get(severity, "⚪")
            names = f"{inter['drug_a']} + {inter['drug_b']}"
            lines += [
                f"  {emoji} {names}",
                f"     Severity : {severity.title()}",
                f"     Source   : {inter['source']}",
            ]
            if inter.get("description"):
                lines.append(f"     Detail   : {inter['description'][:400]}")
            lines.append("")
    else:
        lines += [
            "  ✅ No interactions found in DDInter or FDA label text.",
            "     Always verify with a licensed pharmacist or prescriber.",
            "",
        ]

    lines.append(f"  ⚕️  DISCLAIMER: {DISCLAIMER}")

    return _response(
        summary="\n".join(lines),
        data={
            "queried_drugs": list(drugs),
            "resolved_rxcuis": resolved,
            "unresolved": unresolved,
            "interactions": found,
            "source_counts": counts,
            "ddinter_version": ddinter_version,
        },
        sources=sources_meta,
    )


# ── Tool 2: Medication Advice ───────────────────────────────────────────────

@mcp.tool()
async def get_medication_advice(
    medication: Annotated[str, Field(description="Drug name to look up")],
) -> dict[str, Any]:
    """Fetch dosage, warnings, adverse reactions, and drug interaction info.

    Sources the FDA OpenFDA drug label database. Tries generic name first,
    then brand name.
    """
    label = await openfda.label_for(medication)
    sources = [{"name": "OpenFDA Drug Label", "url": openfda.BASE, "accessed_at": _now()}]

    if not label:
        return _response(
            summary="\n".join([
                f"  ❌ '{medication}' not found in OpenFDA drug label database.",
                "",
                f"  ⚕️  DISCLAIMER: {DISCLAIMER}",
            ]),
            data={"medication": medication, "found": False},
            sources=sources,
        )

    openfda_meta = label.get("openfda", {})

    def _first(key: str, default: str = "Not available") -> str:
        val = label.get(key, [])
        if not val:
            return default
        text = val[0].strip()
        return text[:600] + "…" if len(text) > 600 else text

    generic = openfda_meta.get("generic_name", [medication.title()])
    brand = openfda_meta.get("brand_name", ["—"])
    drug_class = openfda_meta.get("pharm_class_epc", ["—"])

    summary = "\n".join([
        "╔══════════════════════════════════════════════════╗",
        "║         NEUROPHRAX — MEDICATION ADVICE           ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Generic      : {', '.join(generic)}",
        f"  Brand        : {', '.join(brand)}",
        f"  Class        : {', '.join(drug_class)}",
        f"  Source       : OpenFDA Drug Label",
        "",
        f"  DOSAGE       : {_first('dosage_and_administration')}",
        "",
        f"  WARNINGS     : {_first('warnings')}",
        "",
        f"  ADVERSE RX   : {_first('adverse_reactions')}",
        "",
        f"  INTERACTIONS : {_first('drug_interactions')}",
        "",
        f"  ⚕️  DISCLAIMER: {DISCLAIMER}",
    ])

    return _response(
        summary=summary,
        data={
            "medication": medication,
            "found": True,
            "generic_name": generic,
            "brand_name": brand,
            "pharm_class": drug_class,
            "dosage_and_administration": _first("dosage_and_administration"),
            "warnings": _first("warnings"),
            "adverse_reactions": _first("adverse_reactions"),
            "drug_interactions": _first("drug_interactions"),
        },
        sources=sources,
    )


# ── Tool 3: Symptom → Conditions + ICD-10-CM ────────────────────────────────

@mcp.tool()
async def map_symptoms_to_conditions(
    symptoms: Annotated[
        list[str],
        Field(min_length=1, description="Symptoms or condition terms to map"),
    ],
) -> dict[str, Any]:
    """Map symptoms to clinical conditions and ICD-10-CM diagnosis codes.

    Returns two parallel views per symptom: a curated FHIR Conditions Value
    Set match (with MedlinePlus consumer URL where available) and the
    matching ICD-10-CM billable codes. Both come from the NIH Clinical
    Tables Search Service.
    """
    tasks: list[Any] = []
    for s in symptoms:
        tasks.append(clinical_tables.search_conditions(s, limit=5))
        tasks.append(clinical_tables.search_icd10(s, limit=5))
    results = await asyncio.gather(*tasks)

    mapped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║      NEUROPHRAX — SYMPTOM-TO-CONDITION MAP       ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Symptoms : {', '.join(symptoms)}",
        f"  Sources  : NIH Clinical Tables (Conditions VS + ICD-10-CM)",
        "",
    ]

    for i, symptom in enumerate(symptoms):
        conds: list[dict[str, Any]] = results[i * 2] or []
        icd10: list[dict[str, Any]] = results[i * 2 + 1] or []
        mapped[symptom] = {"conditions": conds, "icd10": icd10}

        lines.append(f"  • {symptom.title()}")
        if conds:
            lines.append("      Conditions:")
            for c in conds:
                lines.append(f"        → {c['name']}  (key {c['key_id']})")
                if c.get("medlineplus_url"):
                    lines.append(f"            MedlinePlus: {c['medlineplus_url']}")
        else:
            lines.append("      Conditions: (no curated match)")
        if icd10:
            lines.append("      ICD-10-CM:")
            for code in icd10:
                lines.append(f"        → {code['code']}  {code['name']}")
        else:
            lines.append("      ICD-10-CM: (no match)")
        lines.append("")

    lines.append(f"  ⚕️  DISCLAIMER: {DISCLAIMER}")

    return _response(
        summary="\n".join(lines),
        data={"mappings": mapped},
        sources=[
            {
                "name": "NIH Clinical Tables — Conditions",
                "url": f"{clinical_tables.BASE}/conditions/v3",
                "accessed_at": _now(),
            },
            {
                "name": "NIH Clinical Tables — ICD-10-CM",
                "url": f"{clinical_tables.BASE}/icd10cm/v3",
                "accessed_at": _now(),
            },
        ],
    )


# ── Tool 4: Patient Summary ─────────────────────────────────────────────────

@mcp.tool()
async def generate_patient_summary(
    name: Annotated[str, Field(description="Patient's full name")],
    age: Annotated[int, Field(ge=0, le=130, description="Patient age in years")],
    conditions: Annotated[list[str], Field(description="Active diagnosed medical conditions")],
    medications: Annotated[list[str], Field(description="Current medications")],
    allergies: Annotated[list[str], Field(description="Known drug/food allergies")],
    last_visit: Annotated[
        str, Field(description="Date of last clinical visit (YYYY-MM-DD)")
    ] = "Not recorded",
) -> dict[str, Any]:
    """Generate a structured patient summary.

    Validates conditions against SNOMED CT, resolves medications via RxNorm,
    and computes a heuristic risk level.
    """
    risk = "LOW"
    risk_reasons: list[str] = []
    if age >= 65:
        risk = "MODERATE"
        risk_reasons.append("age ≥ 65")
    if len(medications) >= 5:
        risk = "HIGH"
        risk_reasons.append("polypharmacy (≥5 medications)")
    if len(conditions) >= 4:
        risk = "HIGH"
        risk_reasons.append("multiple comorbidities")
    critical_keywords = {"cancer", "heart failure", "sepsis", "stroke", "ards"}
    for c in conditions:
        if any(kw in c.lower() for kw in critical_keywords):
            risk = "CRITICAL"
            risk_reasons.append(f"critical condition: {c}")
            break

    condition_results, rxnorm_results = await asyncio.gather(
        asyncio.gather(
            *[clinical_tables.search_conditions(c, limit=1) for c in conditions[:5]]
        ),
        asyncio.gather(*[rxnorm.name_to_rxcui(m) for m in medications[:10]]),
    )

    condition_lookups: list[dict[str, Any]] = []
    for cond, conds in zip(conditions[:5], condition_results):
        if not conds:
            continue
        condition_lookups.append({
            "condition": cond,
            "matched_name": conds[0]["name"],
            "medlineplus_url": conds[0].get("medlineplus_url"),
        })

    rxnorm_map = {m: c for m, c in zip(medications[:10], rxnorm_results) if c}

    def _fmt_list(items: list[str], empty: str) -> str:
        return ("\n" + " " * 20).join(f"• {x}" for x in items) if items else empty

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║          NEUROPHRAX — PATIENT SUMMARY            ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Name             : {name}",
        f"  Age              : {age} years old",
        f"  Last Visit       : {last_visit}",
        f"  Risk Level       : {risk}"
        + (f"  ({'; '.join(risk_reasons)})" if risk_reasons else ""),
        "",
        f"  CONDITIONS [{len(conditions)}]   : {_fmt_list(conditions, 'None recorded')}",
        "",
        f"  MEDICATIONS [{len(medications)}]  : {_fmt_list(medications, 'None recorded')}",
        "",
        f"  ALLERGIES        : {_fmt_list(allergies, 'NKDA — No Known Drug Allergies')}",
    ]
    if condition_lookups:
        lines += ["", "  CONDITION VALIDATION:", ""]
        for entry in condition_lookups:
            lines.append(f"    {entry['condition']} → {entry['matched_name']}")
            if entry.get("medlineplus_url"):
                lines.append(f"      MedlinePlus: {entry['medlineplus_url']}")
    if rxnorm_map:
        lines += ["", "  RXNORM MEDICATION IDs:", ""]
        for med, cid in rxnorm_map.items():
            lines.append(f"    {med.title()} → RxCUI {cid}")
    lines += ["", f"  ⚕️  DISCLAIMER: {DISCLAIMER}"]

    return _response(
        summary="\n".join(lines),
        data={
            "patient": {"name": name, "age": age, "last_visit": last_visit},
            "conditions": conditions,
            "medications": medications,
            "allergies": allergies,
            "risk": {"level": risk, "reasons": risk_reasons},
            "condition_lookups": condition_lookups,
            "rxnorm_map": rxnorm_map,
        },
        sources=[
            {
                "name": "NIH Clinical Tables — Conditions + ICD-10-CM",
                "url": clinical_tables.BASE,
                "accessed_at": _now(),
            },
            {"name": "RxNorm", "url": rxnorm.BASE, "accessed_at": _now()},
        ],
    )


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
