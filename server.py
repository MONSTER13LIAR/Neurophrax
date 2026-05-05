"""
Neurophrax MCP Server
Live medical data from OpenFDA, RxNorm, and SNOMED CT APIs.
"""

import asyncio
import httpx
from typing import Annotated
from pydantic import Field
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Neurophrax",
    instructions=(
        "Healthcare assistant backed by live OpenFDA, RxNorm, and SNOMED CT data. "
        "For educational and informational use only — not for clinical decisions."
    ),
)

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug"
SNOMED_BASE = "https://browser.ihtsdotools.org/snowstorm/snomed-ct/MAIN"
TIMEOUT = 15.0


async def _get(url: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


async def _rxcui(drug: str) -> str | None:
    """Resolve a drug name to its RxNorm CUI."""
    try:
        data = await _get(f"{RXNORM_BASE}/rxcui.json", {"name": drug, "search": 1})
        ids = data.get("idGroup", {}).get("rxnormId") or []
        return ids[0] if ids else None
    except Exception:
        return None


# ── Tool 1: Drug Interactions ──────────────────────────────────────────────────

@mcp.tool()
async def check_drug_interactions(
    drugs: Annotated[
        list[str],
        Field(min_length=2, description="Two or more drug names to check for interactions"),
    ]
) -> str:
    """
    Check for drug interactions using the RxNorm Drug Interaction API.
    Resolves each name to a standard RxCUI, then queries the live interaction database.
    """
    rxcuis = dict(zip(drugs, await asyncio.gather(*[_rxcui(d) for d in drugs])))
    resolved = {d: cid for d, cid in rxcuis.items() if cid}
    unresolved = [d for d, cid in rxcuis.items() if not cid]

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║        NEUROPHRAX — DRUG INTERACTION CHECK       ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Drugs    : {', '.join(d.title() for d in drugs)}",
        f"  Source   : RxNorm Drug Interaction API",
        "",
    ]

    if unresolved:
        lines.append(f"  ⚠️  Could not resolve to RxCUI: {', '.join(unresolved)}")
        lines.append("")

    if len(resolved) < 2:
        lines.append("  ❌ Need at least 2 resolved drugs to check interactions.")
        return "\n".join(lines)

    try:
        data = await _get(
            f"{RXNORM_BASE}/interaction/list.json",
            {"rxcuis": " ".join(resolved.values())},
        )
    except Exception as e:
        lines.append(f"  ❌ RxNorm API error: {e}")
        return "\n".join(lines)

    interactions = []
    for group in data.get("fullInteractionTypeGroup", []):
        source = group.get("sourceName", "Unknown")
        for itype in group.get("fullInteractionType", []):
            for pair in itype.get("interactionPair", []):
                severity = pair.get("severity", "N/A").upper()
                description = pair.get("description", "")
                drug_names = " + ".join(
                    ic.get("minConceptItem", {}).get("name", "")
                    for ic in pair.get("interactionConcept", [])
                )
                interactions.append((severity, drug_names or "—", description, source))

    if interactions:
        lines.append(f"  🚨 {len(interactions)} INTERACTION(S) DETECTED:")
        lines.append("")
        for severity, names, desc, source in interactions:
            emoji = "🔴" if severity in ("HIGH", "N/A") else "🟡"
            lines += [
                f"  {emoji} {names}",
                f"     Severity  : {severity}",
                f"     Detail    : {desc[:400]}",
                f"     Source    : {source}",
                "",
            ]
    else:
        lines += [
            "  ✅ No known interactions found in the RxNorm database.",
            "     Always verify with a licensed pharmacist or prescriber.",
            "",
        ]

    lines.append("  ⚕️  DISCLAIMER: For informational use only. Not for clinical decisions.")
    return "\n".join(lines)


# ── Tool 2: Medication Advice ──────────────────────────────────────────────────

@mcp.tool()
async def get_medication_advice(
    medication: Annotated[str, Field(description="Drug name to look up")],
) -> str:
    """
    Fetch dosage, warnings, adverse reactions, and drug interaction info
    from the OpenFDA drug label database. Tries generic name first, then brand name.
    """
    label = None
    for field in ("openfda.generic_name", "openfda.brand_name"):
        try:
            data = await _get(
                f"{OPENFDA_BASE}/label.json",
                {"search": f'{field}:"{medication}"', "limit": 1},
            )
            results = data.get("results", [])
            if results:
                label = results[0]
                break
        except httpx.HTTPStatusError:
            continue

    if not label:
        return "\n".join([
            f"  ❌ '{medication}' not found in OpenFDA drug label database.",
            "",
            "  ⚕️  DISCLAIMER: For informational use only. Not for clinical decisions.",
        ])

    openfda = label.get("openfda", {})

    def first(key: str, default: str = "Not available") -> str:
        val = label.get(key, [])
        if not val:
            return default
        text = val[0].strip()
        return text[:600] + "…" if len(text) > 600 else text

    generic = ", ".join(openfda.get("generic_name", [medication.title()]))
    brand = ", ".join(openfda.get("brand_name", ["—"]))
    drug_class = ", ".join(openfda.get("pharm_class_epc", ["—"]))

    return "\n".join([
        "╔══════════════════════════════════════════════════╗",
        "║         NEUROPHRAX — MEDICATION ADVICE           ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Generic      : {generic}",
        f"  Brand        : {brand}",
        f"  Class        : {drug_class}",
        f"  Source       : OpenFDA Drug Label",
        "",
        f"  DOSAGE       : {first('dosage_and_administration')}",
        "",
        f"  WARNINGS     : {first('warnings')}",
        "",
        f"  ADVERSE RX   : {first('adverse_reactions')}",
        "",
        f"  INTERACTIONS : {first('drug_interactions')}",
        "",
        "  ⚕️  DISCLAIMER: For informational use only. Not for clinical decisions.",
    ])


# ── Tool 3: Symptom → SNOMED CT Concepts ──────────────────────────────────────

@mcp.tool()
async def map_symptoms_to_conditions(
    symptoms: Annotated[
        list[str],
        Field(min_length=1, description="Symptoms to map to SNOMED CT clinical findings"),
    ],
) -> str:
    """
    Map symptoms to standardized SNOMED CT clinical concepts.
    Returns the preferred term, concept ID, and semantic type for each match.
    Searches within the Clinical Finding hierarchy (SCTID 404684003).
    """

    async def lookup(symptom: str) -> tuple[str, list[dict]]:
        try:
            data = await _get(
                f"{SNOMED_BASE}/concepts",
                {
                    "term": symptom,
                    "activeFilter": "true",
                    "limit": 5,
                    "ecl": "<<404684003",  # Clinical finding + all descendants
                },
            )
            return symptom, data.get("items", [])
        except Exception:
            return symptom, []

    results = await asyncio.gather(*[lookup(s) for s in symptoms])

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║      NEUROPHRAX — SYMPTOM-TO-CONDITION MAP       ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Symptoms : {', '.join(symptoms)}",
        f"  Source   : SNOMED CT International (Snowstorm API)",
        "",
    ]

    for symptom, items in results:
        lines.append(f"  • {symptom.title()}")
        if items:
            for item in items:
                pt = (item.get("pt") or {}).get("term") or (item.get("fsn") or {}).get("term", "—")
                sctid = item.get("conceptId", "—")
                fsn_term = (item.get("fsn") or {}).get("term", "")
                semantic = fsn_term[fsn_term.rfind("(") + 1: fsn_term.rfind(")")] if "(" in fsn_term else ""
                tag = f"  [{semantic}]" if semantic else ""
                lines.append(f"      → {pt}{tag}  [SCTID: {sctid}]")
        else:
            lines.append("      No SNOMED CT concepts found.")
        lines.append("")

    lines.append("  ⚕️  DISCLAIMER: For informational use only. Not for clinical decisions.")
    return "\n".join(lines)


# ── Tool 4: Patient Summary ────────────────────────────────────────────────────

@mcp.tool()
async def generate_patient_summary(
    name: Annotated[str, Field(description="Patient's full name")],
    age: Annotated[int, Field(ge=0, le=130, description="Patient age in years")],
    conditions: Annotated[list[str], Field(description="Active diagnosed medical conditions")],
    medications: Annotated[list[str], Field(description="Current medications")],
    allergies: Annotated[list[str], Field(description="Known drug/food allergies")],
    last_visit: Annotated[str, Field(description="Date of last clinical visit (YYYY-MM-DD)")] = "Not recorded",
) -> str:
    """
    Generate a structured patient summary with SNOMED CT condition validation
    and RxNorm-based medication resolution. Risk level is auto-assessed.
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

    def fmt_list(items: list[str], empty: str) -> str:
        return ("\n" + " " * 20).join(f"• {x}" for x in items) if items else empty

    async def snomed_lookup(term: str) -> str | None:
        try:
            data = await _get(
                f"{SNOMED_BASE}/concepts",
                {"term": term, "activeFilter": "true", "limit": 1, "ecl": "<<404684003"},
            )
            items = data.get("items", [])
            if items:
                pt = (items[0].get("pt") or {}).get("term", term)
                sctid = items[0].get("conceptId", "")
                return f"    {term} → {pt}  [SCTID: {sctid}]"
        except Exception:
            pass
        return None

    # Run SNOMED condition validation and RxNorm medication resolution concurrently
    snomed_results, rxnorm_results = await asyncio.gather(
        asyncio.gather(*[snomed_lookup(c) for c in conditions[:5]]),
        asyncio.gather(*[_rxcui(m) for m in medications[:10]]),
    )

    snomed_notes = [r for r in snomed_results if r]
    rxnorm_map = {m: cid for m, cid in zip(medications[:10], rxnorm_results) if cid}

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║          NEUROPHRAX — PATIENT SUMMARY            ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Name             : {name}",
        f"  Age              : {age} years old",
        f"  Last Visit       : {last_visit}",
        f"  Risk Level       : {risk}" + (f"  ({'; '.join(risk_reasons)})" if risk_reasons else ""),
        "",
        f"  CONDITIONS [{len(conditions)}]   : {fmt_list(conditions, 'None recorded')}",
        "",
        f"  MEDICATIONS [{len(medications)}]  : {fmt_list(medications, 'None recorded')}",
        "",
        f"  ALLERGIES        : {fmt_list(allergies, 'NKDA — No Known Drug Allergies')}",
    ]

    if snomed_notes:
        lines += ["", "  SNOMED CT CONDITION LOOKUP:", ""]
        lines += snomed_notes

    if rxnorm_map:
        lines += ["", "  RXNORM MEDICATION IDs:", ""]
        for med, cid in rxnorm_map.items():
            lines.append(f"    {med.title()} → RxCUI {cid}")

    lines += [
        "",
        "  ⚕️  DISCLAIMER: For informational use only. Not for clinical decisions.",
    ]
    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
