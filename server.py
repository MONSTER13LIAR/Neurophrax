"""Neurophrax — Healthcare data layer for AI agents.

Tools synthesize live data from NIH RxNorm, FDA OpenFDA, and SNOMED CT
(Snowstorm) into structured responses suitable for agent reasoning. Each
tool returns both a human-readable `summary` and machine-readable `data`
plus explicit source attribution. Educational use only.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from mcp.server.fastmcp import Context

from sources import (
    clinical_tables,
    clinicaltrials,
    dailymed,
    interactions,
    medlineplus,
    openfda,
    pubmed,
    rxclass,
    rxnorm,
    safety_brief,
    safety_review,
    vaccines,
)
from sources.beers import ATTRIBUTION as BEERS_ATTRIBUTION, SOURCE_URL as BEERS_URL
from sources.sharp import (
    FhirClient,
    bundle_entries,
    chart_from_fhir,
    get_fhir_context,
    resolve_patient_id,
)
from sources.vaccines import (
    ATTRIBUTION as VACCINES_ATTRIBUTION,
    SOURCE_URL as VACCINES_URL,
)

mcp = FastMCP(
    "Neurophrax",
    instructions=(
        "Healthcare data layer backed by NIH RxNorm, FDA OpenFDA, and SNOMED CT. "
        "Tools return both a human-readable summary and structured data with "
        "explicit source attribution. Educational use only — not for clinical decisions."
    ),
    stateless_http=True,
    host="0.0.0.0",
)

# Advertise SHARP-on-MCP FHIR context capability — tells the host which FHIR
# scopes this server can consume when the platform propagates patient context
# via x-fhir-server-url / x-fhir-access-token headers.
_original_get_capabilities = mcp._mcp_server.get_capabilities


def _patched_get_capabilities(notification_options, experimental_capabilities):
    caps = _original_get_capabilities(notification_options, experimental_capabilities)
    caps.model_extra["extensions"] = {
        "ai.promptopinion/fhir-context": {
            "scopes": [
                {"name": "patient/Patient.rs", "required": True},
                {"name": "patient/Condition.rs"},
                {"name": "patient/MedicationStatement.rs"},
                {"name": "patient/AllergyIntolerance.rs"},
                {"name": "patient/Observation.rs"},
            ]
        }
    }
    return caps


mcp._mcp_server.get_capabilities = _patched_get_capabilities

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
async def drug_interaction_check(
    drugs: Annotated[
        list[str] | None,
        Field(description="Two or more drug names to check. Omit if SHARP/FHIR patient context is set."),
    ] = None,
    patientId: Annotated[  # noqa: N803
        str | None,
        Field(description="Explicit FHIR Patient ID. Auto-resolved from SHARP context if omitted."),
    ] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Use this ONLY when the user asks specifically about pairwise drug-drug
    interactions and does not need Beers Criteria, boxed warnings, or
    duplicate-class checks. For a full medication safety review that
    cross-references all four signals, prefer ``medication_safety_review``.
    For a one-call patient brief (safety + evidence + vaccines + trials),
    prefer ``prescription_safety_brief``.

    Primary source: bundled DDInter 2.0 dataset (severity-rated, ~160k pairs).
    Fallback: parses the OpenFDA drug label `drug_interactions` section for
    pairs not in DDInter. RxNorm RxCUIs are resolved in parallel for downstream
    use by agents.

    When the request carries SHARP-on-MCP headers and ``drugs`` is omitted,
    the tool fetches active ``MedicationStatement`` (with ``MedicationRequest``
    fallback) resources from the FHIR server and runs the all-pair check on
    those. Returns ``data.chart_status`` (``"ok"`` | ``"empty_chart"`` |
    ``"single_medication"``) — agents MUST check this before reporting "no
    interactions found." Empty/single-med charts return a structured response,
    not an HTTP error.
    """
    fhir_ctx = get_fhir_context(ctx)
    resolved_pid = resolve_patient_id(ctx, patientId)
    fhir_error: str | None = None
    source = "direct"
    chart: Any = None
    used_fhir = False

    if not drugs and fhir_ctx and resolved_pid:
        used_fhir = True
        try:
            chart = await chart_from_fhir(FhirClient(fhir_ctx), resolved_pid)
            if chart and chart.medications:
                drugs = chart.medications
                source = "fhir"
        except Exception as exc:
            fhir_error = f"FHIR fetch failed: {exc}"

    # Genuine missing-input — agent gave nothing to work with.
    if not drugs and not used_fhir:
        raise ValueError(
            "Provide at least two drug names, or call with SHARP/FHIR patient context."
        )

    # FHIR was attempted but returned 0 or 1 medications — return a structured
    # empty/degraded response so the foreign agent can distinguish this from
    # "checked, no interactions found."
    if not drugs or len(drugs) < 2:
        n = len(drugs or [])
        chart_status = "empty_chart" if n == 0 else "single_medication"
        meds_src = chart.medications_source if chart else "none"
        warnings = [
            (
                "FHIR chart contained 0 active medications "
                "(checked MedicationStatement and MedicationRequest with status=active). "
                "Cannot run pairwise interaction check."
                if n == 0 else
                f"FHIR chart contained only 1 active medication ({(drugs or ['?'])[0]}). "
                "Pairwise interaction check needs ≥2 drugs."
            )
        ]
        if chart and chart.unparseable_medication_ids:
            warnings.append(
                f"{len(chart.unparseable_medication_ids)} medication entry/entries had "
                f"missing code/display and were dropped: {chart.unparseable_medication_ids}."
            )
        return _response(
            summary="\n".join([
                "╔══════════════════════════════════════════════════╗",
                "║        NEUROPHRAX — DRUG INTERACTION CHECK       ║",
                "╚══════════════════════════════════════════════════╝",
                f"  Source       : FHIR R4 (SHARP context, meds via {meds_src})",
                f"  Patient ID   : {resolved_pid or '—'}",
                f"  Medications  : {n}",
                f"  Chart Status : {chart_status}",
                "",
                *(f"  ⚠️  {w}" for w in warnings),
                "",
                "  ⛔ NO CHECK PERFORMED — insufficient medications. This is NOT",
                "     equivalent to 'no interactions found'.",
                "",
                f"  ⚕️  DISCLAIMER: {DISCLAIMER}",
            ]),
            data={
                "queried_drugs": list(drugs or []),
                "resolved_rxcuis": {},
                "unresolved": [],
                "interactions": [],
                "source_counts": {"ddinter": 0, "openfda_label": 0},
                "ddinter_version": interactions.dataset_version(),
                "patient_id": resolved_pid,
                "source": "fhir",
                "chart_status": chart_status,
                "warnings": warnings,
                "medications_source": meds_src,
                "fhir_error": fhir_error,
            },
            sources=[
                {"name": "FHIR R4 Server", "url": fhir_ctx.url if fhir_ctx else None, "accessed_at": _now()},
            ],
        )

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
        f"  Source   : {'FHIR R4 (SHARP context)' if source == 'fhir' else 'Direct input'}",
        f"  Drugs    : {', '.join(d.title() for d in drugs)}",
        f"  Sources  : DDInter {ddinter_version or '(not ingested)'} + OpenFDA labels (fallback)",
        "",
    ]
    if fhir_error:
        lines += [f"  ⚠️  {fhir_error} — used direct args.", ""]
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

    warnings: list[str] = []
    if unresolved:
        warnings.append(
            f"{len(unresolved)} of {len(drugs)} drug(s) could not be normalized via RxNorm: "
            f"{unresolved}. Interaction lookup against DDInter relies on RxCUIs, so "
            "interactions involving these drugs may not have been detected."
        )
    if chart and chart.unparseable_medication_ids:
        warnings.append(
            f"{len(chart.unparseable_medication_ids)} medication entry/entries had missing "
            f"code/display and were dropped: {chart.unparseable_medication_ids}."
        )

    return _response(
        summary="\n".join(lines),
        data={
            "queried_drugs": list(drugs),
            "resolved_rxcuis": resolved,
            "unresolved": unresolved,
            "interactions": found,
            "source_counts": counts,
            "ddinter_version": ddinter_version,
            "patient_id": resolved_pid,
            "source": source,
            "chart_status": "ok",
            "warnings": warnings,
            "medications_source": chart.medications_source if chart else ("fhir" if source == "fhir" else "direct"),
            "fhir_error": fhir_error,
        },
        sources=sources_meta,
    )


# ── Tool 2: Medication Advice ───────────────────────────────────────────────

async def _empty_classes() -> dict[str, list[dict[str, Any]]]:
    return {"atc": [], "epc": []}


async def _empty_list() -> list[dict[str, Any]]:
    return []


@mcp.tool()
async def medication_profile(
    medication: Annotated[str, Field(description="Drug name to look up")],
) -> dict[str, Any]:
    """Use this when the user asks about a SINGLE medication — dosage,
    warnings, side effects, mechanism, or patient-friendly explanation.
    Not for multi-drug safety analysis (use ``medication_safety_review``)
    nor pairwise interactions (use ``drug_interaction_check``).

    Synthesizes from four NLM/FDA sources:
    - OpenFDA: parsed FDA label sections (dosage, warnings, adverse reactions,
      interactions, boxed warning).
    - DailyMed: links to manufacturer-specific SPLs with publication dates.
    - RxClass: ATC and EPC therapeutic class memberships.
    - MedlinePlus Connect: patient-friendly explanation in plain language.
    """
    rxcui = await rxnorm.name_to_ingredient_rxcui(medication)
    label, dailymed_results, classes, mp_entries = await asyncio.gather(
        openfda.label_for(medication),
        dailymed.search_spls(medication, limit=3),
        rxclass.classes_for_rxcui(rxcui) if rxcui else _empty_classes(),
        medlineplus.info_for(medlineplus.CS_RXNORM, rxcui) if rxcui else _empty_list(),
    )

    sources = [
        {"name": "RxNorm", "url": rxnorm.BASE, "accessed_at": _now()},
        {"name": "OpenFDA Drug Label", "url": openfda.BASE, "accessed_at": _now()},
        {"name": "DailyMed", "url": dailymed.BASE, "accessed_at": _now()},
        {"name": "RxClass", "url": rxclass.BASE, "accessed_at": _now()},
        {"name": "MedlinePlus Connect", "url": medlineplus.BASE, "accessed_at": _now()},
    ]

    if not label and not dailymed_results and not mp_entries:
        return _response(
            summary="\n".join([
                f"  ❌ '{medication}' not found across OpenFDA / DailyMed / MedlinePlus.",
                "",
                f"  ⚕️  DISCLAIMER: {DISCLAIMER}",
            ]),
            data={"medication": medication, "found": False, "rxcui": rxcui},
            sources=sources,
        )

    openfda_meta = (label or {}).get("openfda") or {}

    def _first(key: str, default: str = "Not available") -> str:
        val = (label or {}).get(key) or []
        if not val:
            return default
        text = (val[0] or "").strip()
        return text[:600] + "…" if len(text) > 600 else text

    generic = openfda_meta.get("generic_name") or [medication.title()]
    brand = openfda_meta.get("brand_name") or ["—"]
    pharm_class_epc = openfda_meta.get("pharm_class_epc") or []
    boxed_warning = ((label or {}).get("boxed_warning") or [None])[0]
    if boxed_warning:
        boxed_warning = boxed_warning.strip()
        if len(boxed_warning) > 600:
            boxed_warning = boxed_warning[:600] + "…"

    atc_classes = classes.get("atc", [])
    epc_classes = classes.get("epc", [])
    patient_info = mp_entries[0] if mp_entries else None

    # RxClass returns combination-product memberships alongside the drug's own
    # therapeutic class — prefer non-combo classes for the displayed summary.
    def _non_combo(cs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [c for c in cs if "combination" not in (c.get("class_name") or "").lower()]

    atc_display = _non_combo(atc_classes) or atc_classes
    epc_display = _non_combo(epc_classes) or epc_classes

    atc_str = (
        ", ".join(f"{c['class_id']} — {c['class_name']}" for c in atc_display[:3])
        if atc_display
        else "—"
    )
    epc_label = (
        ", ".join(c["class_name"] for c in epc_display[:3])
        if epc_display
        else (", ".join(pharm_class_epc) if pharm_class_epc else "—")
    )

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║         NEUROPHRAX — MEDICATION ADVICE           ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Medication   : {medication.title()}",
        f"  RxCUI        : {rxcui or '—'}",
        f"  Generic      : {', '.join(generic)}",
        f"  Brand        : {', '.join(brand)}",
        f"  Class (ATC)  : {atc_str}",
        f"  Class (EPC)  : {epc_label}",
        f"  Sources      : OpenFDA + DailyMed + RxClass + MedlinePlus",
        "",
    ]

    if patient_info:
        lines += [
            "  📋 PATIENT EXPLANATION (MedlinePlus):",
            f"     {patient_info['title']}",
        ]
        if patient_info.get("url"):
            lines.append(f"     {patient_info['url']}")
        if patient_info.get("summary"):
            lines.append(f"     {patient_info['summary']}")
        lines.append("")

    if boxed_warning:
        lines += ["  ⚠️  BOXED WARNING:", f"     {boxed_warning}", ""]

    lines += [
        f"  DOSAGE       : {_first('dosage_and_administration')}",
        "",
        f"  WARNINGS     : {_first('warnings')}",
        "",
        f"  ADVERSE RX   : {_first('adverse_reactions')}",
        "",
        f"  INTERACTIONS : {_first('drug_interactions')}",
        "",
    ]

    if dailymed_results:
        lines.append("  📚 DAILYMED LABELS:")
        for entry in dailymed_results:
            title = (entry.get("title") or "")[:80]
            lines.append(f"     • {title}")
            if entry.get("published_date"):
                lines.append(f"       Published: {entry['published_date']}")
            if entry.get("url"):
                lines.append(f"       {entry['url']}")
        lines.append("")

    lines.append(f"  ⚕️  DISCLAIMER: {DISCLAIMER}")

    return _response(
        summary="\n".join(lines),
        data={
            "medication": medication,
            "found": True,
            "rxcui": rxcui,
            "generic_name": generic,
            "brand_name": brand,
            "pharm_class_epc": pharm_class_epc,
            "atc_classes": atc_classes,
            "epc_classes": epc_classes,
            "patient_info": patient_info,
            "boxed_warning": boxed_warning,
            "dosage_and_administration": _first("dosage_and_administration"),
            "warnings": _first("warnings"),
            "adverse_reactions": _first("adverse_reactions"),
            "drug_interactions": _first("drug_interactions"),
            "dailymed_labels": dailymed_results,
        },
        sources=sources,
    )


# ── Tool 3: Symptom → Conditions + ICD-10-CM ────────────────────────────────

@mcp.tool()
async def symptom_assessment(
    symptoms: Annotated[
        list[str],
        Field(min_length=1, description="Symptoms or condition terms to map"),
    ],
) -> dict[str, Any]:
    """Use this when the user describes symptoms or needs ICD-10-CM
    diagnosis codes / SNOMED-style condition mappings. This is a CODING
    LOOKUP — it does not diagnose, triage, or rank likelihood. Pair with
    ``clinical_evidence_search`` if the user wants supporting evidence on
    the mapped conditions.

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
async def patient_summary(
    name: Annotated[
        str | None, Field(description="Patient's full name. Omit if SHARP/FHIR context is set.")
    ] = None,
    age: Annotated[
        int | None,
        Field(ge=0, le=130, description="Patient age. Omit if SHARP/FHIR context is set."),
    ] = None,
    conditions: Annotated[
        list[str] | None,
        Field(description="Active diagnosed conditions. Pulled from FHIR if context is set."),
    ] = None,
    medications: Annotated[
        list[str] | None,
        Field(description="Current medications. Pulled from FHIR if context is set."),
    ] = None,
    allergies: Annotated[
        list[str] | None,
        Field(description="Known allergies. Pulled from FHIR if context is set."),
    ] = None,
    last_visit: Annotated[
        str, Field(description="Date of last clinical visit (YYYY-MM-DD).")
    ] = "Not recorded",
    patientId: Annotated[  # noqa: N803
        str | None,
        Field(description="Explicit FHIR Patient ID. Auto-resolved from SHARP context if omitted."),
    ] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Use this when the user wants a quick patient overview / risk
    snapshot (name, age, conditions, meds, allergies, heuristic risk level).
    This is a READ-ONLY snapshot — it does NOT analyze drug safety; for
    that use ``medication_safety_review`` or ``prescription_safety_brief``.

    Validates conditions against the FHIR Conditions Value Set, resolves
    medications via RxNorm, and computes a heuristic risk level.

    When the request carries SHARP-on-MCP headers, any omitted argument is
    pulled from the FHIR server (Patient → name + age, MedicationStatement
    with MedicationRequest fallback, Condition, AllergyIntolerance).
    Explicit args always win over chart data. Returns ``data.chart_status``
    and ``data.warnings`` so agents can detect a sparsely-populated chart.
    """
    fhir_ctx = get_fhir_context(ctx)
    resolved_pid = resolve_patient_id(ctx, patientId)
    fhir_error: str | None = None
    chart_source = "direct"
    chart: Any = None
    medications_source = "direct"

    if fhir_ctx and resolved_pid and (
        name is None or age is None or conditions is None or medications is None or allergies is None
    ):
        try:
            chart = await chart_from_fhir(FhirClient(fhir_ctx), resolved_pid)
            if chart is not None:
                chart_source = "fhir"
                medications_source = chart.medications_source
                if name is None:
                    name = chart.name
                if age is None:
                    age = chart.age
                if conditions is None:
                    conditions = list(chart.conditions)
                if medications is None:
                    medications = list(chart.medications)
                if allergies is None:
                    allergies = list(chart.allergies)
        except Exception as exc:
            fhir_error = f"FHIR fetch failed: {exc}"

    if name is None or age is None:
        raise ValueError(
            "name and age are required when SHARP/FHIR patient context is not provided"
        )
    conditions = list(conditions or [])
    medications = list(medications or [])
    allergies = list(allergies or [])

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
        f"  Source           : {'FHIR R4 (SHARP context)' if chart_source == 'fhir' else 'Direct input'}",
        f"  Patient ID       : {resolved_pid or '—'}",
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
    if fhir_error:
        lines += ["", f"  ⚠️  {fhir_error} — used direct args."]
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

    warnings: list[str] = []
    if chart and chart.unparseable_medication_ids:
        warnings.append(
            f"{len(chart.unparseable_medication_ids)} medication FHIR entry/entries had "
            f"missing code/display: {chart.unparseable_medication_ids}."
        )
    if chart and chart.unparseable_condition_ids:
        warnings.append(
            f"{len(chart.unparseable_condition_ids)} condition FHIR entry/entries had "
            f"missing code/display: {chart.unparseable_condition_ids}."
        )
    if chart and chart.unparseable_allergy_ids:
        warnings.append(
            f"{len(chart.unparseable_allergy_ids)} allergy FHIR entry/entries had "
            f"missing code/display: {chart.unparseable_allergy_ids}."
        )
    chart_status = "ok"
    if chart_source == "fhir" and not medications and not conditions:
        chart_status = "empty_chart"
        warnings.insert(
            0,
            "FHIR chart returned no active medications and no active conditions. "
            "Patient summary may reflect a sparsely-populated record rather than a healthy patient.",
        )
    if warnings:
        lines += ["", "  ⚠️  WARNINGS:", *(f"     • {w}" for w in warnings)]
    lines += ["", f"  ⚕️  DISCLAIMER: {DISCLAIMER}"]

    return _response(
        summary="\n".join(lines),
        data={
            "patient": {
                "name": name,
                "age": age,
                "last_visit": last_visit,
                "patient_id": resolved_pid,
                "source": chart_source,
                "medications_source": medications_source,
            },
            "conditions": conditions,
            "medications": medications,
            "allergies": allergies,
            "risk": {"level": risk, "reasons": risk_reasons},
            "condition_lookups": condition_lookups,
            "rxnorm_map": rxnorm_map,
            "chart_status": chart_status,
            "warnings": warnings,
            "unparseable_medication_ids": list(chart.unparseable_medication_ids) if chart else [],
            "unparseable_condition_ids": list(chart.unparseable_condition_ids) if chart else [],
            "unparseable_allergy_ids": list(chart.unparseable_allergy_ids) if chart else [],
            "fhir_error": fhir_error,
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


# ── Tool 5: Medication Safety Review (composer) ─────────────────────────────

_SEVERITY_BANNER = {
    "MAJOR": "🔴",
    "AVOID": "🔴",
    "AVOID_CHRONIC_USE": "🟠",
    "USE_WITH_CAUTION": "🟡",
    "MODERATE": "🟡",
    "DOSE_ADJUST": "🟢",
    "MINOR": "🟢",
    "UNKNOWN": "⚪",
}


@mcp.tool()
async def medication_safety_review(
    medications: Annotated[
        list[str] | None,
        Field(description="Medications to review. Omit if SHARP/FHIR patient context is set."),
    ] = None,
    age: Annotated[
        int | None,
        Field(ge=0, le=130, description="Patient age. Omit if SHARP/FHIR patient context is set."),
    ] = None,
    conditions: Annotated[
        list[str] | None,
        Field(description="Active conditions (optional, used for pregnancy detection)."),
    ] = None,
    pregnant: Annotated[
        bool, Field(description="Whether the patient is pregnant or planning to be.")
    ] = False,
    patientId: Annotated[  # noqa: N803
        str | None,
        Field(description="Explicit FHIR Patient ID. Auto-resolved from SHARP context if omitted."),
    ] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Use this when the user asks "is this medication list safe?" or any
    patient-level medication-safety question. PREFERRED tool for any
    multi-drug + patient-context safety analysis. For an even broader
    one-call brief that ALSO returns supporting evidence, vaccines, and
    matching trials, use ``prescription_safety_brief`` instead.

    Composite medication safety review.

    Cross-references four independent risk signals against patient context:

    1. **Pairwise drug-drug interactions** (DDInter 2.0 + OpenFDA fallback).
    2. **AGS Beers Criteria 2023** flags for adults ≥ 65 (and per-rule thresholds).
    3. **Duplicate therapeutic class** detection via RxClass ATC.
    4. **FDA boxed warnings** parsed from current drug labels (and pregnancy
       contraindications when ``pregnant=True``).

    When the request carries SHARP-on-MCP headers (``x-fhir-server-url`` /
    ``x-fhir-access-token`` / ``x-patient-id``), the tool fetches the patient,
    active medications, and active conditions from the FHIR server directly —
    no need to pass them in. Otherwise it falls back to the explicit args.

    Returns a single prioritized risk list, with each finding tagged by
    severity, rationale, and recommended mitigation. Output includes a FHIR
    ``RiskAssessment`` resource ready to persist back to the patient's record.

    The response payload also exposes ``data.chart_status`` (``"ok"`` or
    ``"empty_chart"``), ``data.warnings`` (human-readable caveats),
    ``data.unresolved_medications`` (drug names RxNorm could not normalize —
    interaction lookup may be incomplete for these), and
    ``data.unparseable_medication_ids`` / ``data.unparseable_condition_ids``
    (FHIR entries with missing code/display, dropped from the chart). Agents
    consuming this tool MUST check these fields before reporting safety:
    ``chart_status="empty_chart"`` means the chart was fetched but contained
    zero active medications, so no review was performed — this is NOT
    equivalent to "no safety signals."
    """
    fhir_ctx = get_fhir_context(ctx)
    resolved_pid = resolve_patient_id(ctx, patientId)
    patient_facts: safety_review.PatientFacts | None = None
    fhir_error: str | None = None

    if fhir_ctx and resolved_pid:
        try:
            client = FhirClient(fhir_ctx)
            patient_facts = await safety_review.patient_facts_from_fhir(client, resolved_pid)
        except Exception as exc:  # graceful fallback to direct args
            fhir_error = f"FHIR fetch failed: {exc}"
            patient_facts = None

    if patient_facts is None:
        if not medications or age is None:
            raise ValueError(
                "medications and age are required when SHARP/FHIR patient context is not provided"
            )
        patient_facts = safety_review.PatientFacts(
            age=age,
            medications=list(medications),
            conditions=list(conditions or []),
            pregnant=pregnant,
            patient_id=resolved_pid,
            source="direct",
            medications_source="direct",
        )

    # Distinguish "FHIR chart fetched, but contained zero active meds" from
    # "checked, found nothing" — the latter is vacuously true for an empty
    # input set and will mislead any downstream agent into reporting safety.
    chart_status = "ok"
    if patient_facts.source == "fhir" and not patient_facts.medications:
        chart_status = "empty_chart"

    review = await safety_review.run(patient_facts)
    findings = review.findings
    unresolved_medications = review.unresolved_medications

    warnings: list[str] = []
    if chart_status == "empty_chart":
        warnings.append(
            "FHIR chart was fetched but contained 0 active medications "
            "(checked MedicationStatement and MedicationRequest with status=active). "
            "Safety review was not performed — there is no data to assess."
        )
    if patient_facts.unparseable_medication_ids:
        warnings.append(
            f"{len(patient_facts.unparseable_medication_ids)} medication entry/entries could not "
            f"be parsed (missing code/display): {patient_facts.unparseable_medication_ids}. "
            "Chart context may be incomplete."
        )
    if patient_facts.unparseable_condition_ids:
        warnings.append(
            f"{len(patient_facts.unparseable_condition_ids)} condition entry/entries could not "
            f"be parsed: {patient_facts.unparseable_condition_ids}."
        )
    if unresolved_medications:
        warnings.append(
            f"{len(unresolved_medications)} of {len(patient_facts.medications)} medication(s) "
            f"could not be normalized via RxNorm: {unresolved_medications}. "
            "Interaction lookup against DDInter relies on RxCUIs, so interactions involving "
            "these drugs may not have been detected."
        )

    sources_meta: list[dict[str, Any]] = [
        {"name": "RxNorm", "url": rxnorm.BASE, "accessed_at": _now()},
        {"name": "RxClass (ATC/EPC)", "url": rxclass.BASE, "accessed_at": _now()},
        {"name": "OpenFDA Drug Label", "url": openfda.BASE, "accessed_at": _now()},
        {
            "name": "DDInter 2.0",
            "url": interactions.DDINTER_HOME,
            "version": interactions.dataset_version() or "not ingested",
            "accessed_at": _now(),
        },
        {
            "name": "AGS Beers Criteria 2023",
            "url": BEERS_URL,
            "attribution": BEERS_ATTRIBUTION,
            "accessed_at": _now(),
        },
    ]
    if patient_facts.source == "fhir":
        sources_meta.insert(
            0,
            {
                "name": "FHIR R4 Server",
                "url": fhir_ctx.url if fhir_ctx else None,
                "accessed_at": _now(),
            },
        )

    # ── Human-readable summary ──
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.category] = counts.get(f.category, 0) + 1

    if patient_facts.source == "fhir":
        meds_src = patient_facts.medications_source or "none"
        source_label = f"FHIR R4 (SHARP context, meds via {meds_src})"
    else:
        source_label = "Direct input"

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║      NEUROPHRAX — MEDICATION SAFETY REVIEW       ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Source       : {source_label}",
        f"  Patient ID   : {patient_facts.patient_id or '—'}",
        f"  Age          : {patient_facts.age} years",
        f"  Medications  : {len(patient_facts.medications)}  ({', '.join(m.title() for m in patient_facts.medications) or '—'})",
        f"  Conditions   : {len(patient_facts.conditions)}",
        f"  Pregnant     : {'Yes' if patient_facts.pregnant else 'No'}",
        f"  Chart Status : {chart_status}",
        "",
        f"  ◇ {len(findings)} finding(s)"
        + (f"   [interactions={counts.get('interaction', 0)}, beers={counts.get('beers', 0)}, "
           f"duplicate_class={counts.get('duplicate_class', 0)}, boxed={counts.get('boxed_warning', 0)}, "
           f"pregnancy={counts.get('pregnancy', 0)}]" if findings else ""),
        "",
    ]
    if fhir_error:
        lines += [f"  ⚠️  {fhir_error} — used direct args.", ""]
    for w in warnings:
        lines += [f"  ⚠️  {w}", ""]

    if chart_status == "empty_chart":
        lines += [
            "  ⛔ NO REVIEW PERFORMED — empty medication list from FHIR chart.",
            "     This is NOT a clean bill of health. The agent should ask the user",
            "     to verify the FHIR server populates MedicationStatement or",
            "     MedicationRequest, or pass `medications` and `age` explicitly.",
        ]
    elif not findings:
        lines += [
            "  ✅ No safety signals matched across all four signal types",
            f"     (checked {len(patient_facts.medications)} medication(s)).",
            "     Continue routine pharmacist review and monitoring.",
        ]
    else:
        for f in findings:
            emoji = _SEVERITY_BANNER.get(f.severity.upper(), "⚪")
            lines += [
                f"  {emoji} [{f.severity}] {f.outcome}",
                f"     Why : {f.rationale[:300]}{'…' if len(f.rationale) > 300 else ''}",
            ]
            if f.mitigation:
                lines.append(f"     Do  : {f.mitigation}")
            lines.append("")

    lines.append(f"  ⚕️  DISCLAIMER: {DISCLAIMER}")

    return _response(
        summary="\n".join(lines),
        data={
            "patient": {
                "age": patient_facts.age,
                "patient_id": patient_facts.patient_id,
                "medications": patient_facts.medications,
                "conditions": patient_facts.conditions,
                "pregnant": patient_facts.pregnant,
                "source": patient_facts.source,
                "medications_source": patient_facts.medications_source,
            },
            "chart_status": chart_status,
            "warnings": warnings,
            "unresolved_medications": unresolved_medications,
            "unparseable_medication_ids": patient_facts.unparseable_medication_ids,
            "unparseable_condition_ids": patient_facts.unparseable_condition_ids,
            "findings": [f.to_dict() for f in findings],
            "fhir": {"RiskAssessment": review.fhir_resource},
            "fhir_error": fhir_error,
        },
        sources=sources_meta,
    )


# ── Tool 6: Find Clinical Trials ────────────────────────────────────────────


def _name_from_codeable(cc: dict[str, Any] | None) -> str | None:
    if not cc:
        return None
    text = (cc.get("text") or "").strip()
    if text:
        return text
    for c in cc.get("coding") or []:
        disp = (c.get("display") or "").strip()
        if disp:
            return disp
    return None


@mcp.tool()
async def clinical_trial_search(
    condition: Annotated[
        str | None,
        Field(description="Condition to search. Omit if SHARP/FHIR patient context is set; conditions will be pulled from the chart."),
    ] = None,
    location: Annotated[
        str | None, Field(description="City, state, or country to bias results by proximity.")
    ] = None,
    extra_term: Annotated[
        str | None, Field(description="Additional free-text filter (e.g. specific intervention).")
    ] = None,
    include_not_yet_recruiting: Annotated[
        bool, Field(description="Include studies not yet open to enrollment.")
    ] = True,
    limit: Annotated[
        int, Field(ge=1, le=25, description="Maximum trials to return per condition.")
    ] = 10,
    patientId: Annotated[  # noqa: N803
        str | None,
        Field(description="Explicit FHIR Patient ID. Auto-resolved from SHARP context if omitted."),
    ] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Use this when the user asks specifically about clinical trials,
    recruitment, or research opportunities for a condition or patient.
    For a one-call patient brief that bundles trials with safety + evidence
    + vaccines, use ``prescription_safety_brief`` instead.

    Uses ClinicalTrials.gov v2. When the request carries SHARP-on-MCP headers
    and ``condition`` is omitted, the tool fetches the patient's active
    ``Condition`` resources from the FHIR server and runs one search per
    distinct condition. Results are projected to the protocol fields a
    clinician typically scans: status, phase, eligibility summary,
    interventions, and the top study locations.
    """
    fhir_ctx = get_fhir_context(ctx)
    resolved_pid = resolve_patient_id(ctx, patientId)
    conditions: list[str] = []
    fhir_error: str | None = None

    if condition:
        conditions = [condition]
    elif fhir_ctx and resolved_pid:
        try:
            client = FhirClient(fhir_ctx)
            bundle = await client.search(
                "Condition", {"patient": resolved_pid, "clinical-status": "active"}
            )
            for c in bundle_entries(bundle):
                name = _name_from_codeable(c.get("code"))
                if name:
                    conditions.append(name)
        except Exception as exc:
            fhir_error = f"FHIR fetch failed: {exc}"
    if not conditions:
        raise ValueError(
            "Provide a `condition` or call with SHARP/FHIR patient context that has active Condition resources."
        )

    statuses = clinicaltrials.RECRUITING_STATUSES
    if not include_not_yet_recruiting:
        statuses = ("RECRUITING",)

    per_condition: list[dict[str, Any]] = []
    total_studies = 0
    for cond in conditions:
        trials = await clinicaltrials.search_trials(
            condition=cond,
            location=location,
            extra_term=extra_term,
            statuses=statuses,
            limit=limit,
        )
        total_studies += len(trials)
        per_condition.append({"condition": cond, "trials": trials, "count": len(trials)})

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║       NEUROPHRAX — CLINICAL TRIALS MATCH         ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Source       : {'FHIR R4 (SHARP context)' if (fhir_ctx and resolved_pid and not condition) else 'Direct input'}",
        f"  Conditions   : {', '.join(conditions)}",
        f"  Location     : {location or '—'}",
        f"  Total trials : {total_studies}",
        "",
    ]
    if fhir_error:
        lines += [f"  ⚠️  {fhir_error}", ""]
    if total_studies == 0:
        lines += ["  ✅ No matching trials in the requested status set.", ""]
    for block in per_condition:
        lines.append(f"  • {block['condition']}  ({block['count']} trial(s))")
        for t in block["trials"]:
            phase = t.get("phase") or t.get("study_type") or "—"
            locs = t.get("locations") or []
            loc_str = ", ".join(filter(None, [locs[0].get("city", ""), locs[0].get("country", "")])) if locs else "—"
            lines += [
                f"      → {t['nct_id']} — {t['title'][:80]}",
                f"        Status: {t['status']}    Phase/Type: {phase}",
                f"        First location: {loc_str}",
                f"        URL: {t['url']}",
            ]
        lines.append("")

    lines.append(f"  ⚕️  DISCLAIMER: {DISCLAIMER}")

    return _response(
        summary="\n".join(lines),
        data={
            "conditions": conditions,
            "location": location,
            "results": per_condition,
            "total": total_studies,
            "fhir_error": fhir_error,
        },
        sources=[
            {"name": "ClinicalTrials.gov v2", "url": clinicaltrials.BASE, "accessed_at": _now()},
        ],
    )


# ── Tool 7: Evidence Search (PubMed) ────────────────────────────────────────

_VALID_STUDY_TYPES = tuple(pubmed.STUDY_TYPE_FILTERS.keys())


@mcp.tool()
async def clinical_evidence_search(
    question: Annotated[
        str, Field(description="Clinical question or topic (e.g. 'metformin in elderly CKD').")
    ],
    study_types: Annotated[
        list[str] | None,
        Field(
            description=(
                "Restrict to publication types: 'meta-analysis', 'systematic-review', "
                "'rct', 'review', 'guideline', 'practice-guideline'. "
                "Omit for unrestricted search."
            )
        ),
    ] = None,
    max_age_years: Annotated[
        int, Field(ge=0, le=50, description="Restrict to articles published in the last N years (0 = no limit).")
    ] = 5,
    limit: Annotated[
        int, Field(ge=1, le=30, description="Maximum citations to return.")
    ] = 10,
) -> dict[str, Any]:
    """Use this when the user asks for PUBLISHED EVIDENCE on a clinical
    question — RCTs, meta-analyses, systematic reviews, or guidelines. This
    tool is NOT patient-aware on its own; pair it with ``patient_summary``
    or ``medication_safety_review`` if the question is patient-specific. For
    a one-call patient brief that auto-attaches evidence to safety findings,
    use ``prescription_safety_brief``.

    Composes the search with Humans + English filters and an optional study-
    type restriction (meta-analyses, RCTs, systematic reviews, guidelines)
    so callers consistently receive evidence-grade hits. Results are sorted
    by publication date and projected to title, journal, year, authors, and
    publication types — small enough for an agent to reason over.
    """
    valid_types = [s for s in (study_types or []) if s in pubmed.STUDY_TYPE_FILTERS]
    invalid_types = sorted(set(study_types or []) - set(valid_types))
    result = await pubmed.search(
        question=question,
        max_age_years=max_age_years if max_age_years > 0 else None,
        study_types=valid_types or None,
        limit=limit,
    )
    items = result["results"]

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║         NEUROPHRAX — EVIDENCE SEARCH             ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Question     : {question}",
        f"  Filters      : {', '.join(valid_types) or 'none'}"
        + (f"  (window: last {max_age_years}y)" if max_age_years else "  (no date window)"),
        f"  Total hits   : {result['total']}    Returned: {len(items)}",
        "",
    ]
    if invalid_types:
        lines += [
            f"  ⚠️  Unknown study_types ignored: {', '.join(invalid_types)}",
            f"      Valid options: {', '.join(_VALID_STUDY_TYPES)}",
            "",
        ]
    if not items:
        lines += [
            "  ✅ No matching articles for this composed query.",
            "     Consider broadening study_types or extending max_age_years.",
        ]
    for r in items:
        ptypes = ", ".join(r.get("publication_types") or []) or "—"
        first_authors = ", ".join((r.get("authors") or [])[:3])
        lines += [
            f"  • PMID {r['pmid']} — {r['title'][:90]}",
            f"      {r.get('journal') or '—'}    {r.get('pub_date') or '—'}",
            f"      Authors: {first_authors or '—'}",
            f"      Types  : {ptypes}",
            f"      URL    : {r['url']}",
            "",
        ]

    lines.append(f"  ⚕️  DISCLAIMER: {DISCLAIMER}")

    return _response(
        summary="\n".join(lines),
        data={
            "question": question,
            "composed_query": result["query"],
            "study_types": valid_types,
            "max_age_years": max_age_years,
            "total": result["total"],
            "results": items,
        },
        sources=[
            {"name": "NCBI PubMed (E-Utilities)", "url": pubmed.BASE, "accessed_at": _now()},
        ],
    )


# ── Tool 8: Vaccine Recommendations (ACIP) ──────────────────────────────────


_VACCINE_INDICATION_BANNER = {
    vaccines.ROUTINE: "✅",
    vaccines.RISK_BASED: "🟡",
    vaccines.SHARED_DECISION: "🟦",
}


def _build_immunization_recommendation(
    rules: list[Any], patient_id: str | None
) -> dict[str, Any]:
    """Emit a FHIR R4 ``ImmunizationRecommendation`` from matching ACIP rules."""
    recs: list[dict[str, Any]] = []
    for rule in rules:
        recs.append(
            {
                "vaccineCode": [{"text": rule.name}],
                "forecastStatus": {"text": rule.indication},
                "dateCriterion": [],
                "description": rule.rationale,
                "supportingPatientInformation": [],
            }
        )
    resource: dict[str, Any] = {
        "resourceType": "ImmunizationRecommendation",
        "date": _now(),
        "recommendation": recs,
    }
    if patient_id:
        resource["patient"] = {"reference": f"Patient/{patient_id}"}
    return resource


@mcp.tool()
async def vaccine_recommendations(
    age: Annotated[
        int | None,
        Field(ge=0, le=130, description="Patient age. Omit if SHARP/FHIR patient context is set."),
    ] = None,
    conditions: Annotated[
        list[str] | None,
        Field(description="Active conditions used to enable risk-based vaccinations."),
    ] = None,
    pregnant: Annotated[
        bool, Field(description="Whether the patient is currently pregnant.")
    ] = False,
    patientId: Annotated[  # noqa: N803
        str | None,
        Field(description="Explicit FHIR Patient ID. Auto-resolved from SHARP context if omitted."),
    ] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Use this when the user asks specifically about vaccines, immunizations,
    or what shots a patient is due for. For a comprehensive patient brief
    that bundles vaccine recommendations with medication safety, evidence,
    and matching trials, use ``prescription_safety_brief`` instead.

    ACIP-curated adult vaccine recommendations based on age and chart conditions.
    Returns the routine, risk-based, and shared-decision-making vaccines
    that apply to the patient context. When SHARP/FHIR headers are present,
    the tool pulls the patient's birthDate and active conditions directly
    so the call can be made with no arguments at all. Output includes a FHIR
    ``ImmunizationRecommendation`` resource.
    """
    fhir_ctx = get_fhir_context(ctx)
    resolved_pid = resolve_patient_id(ctx, patientId)
    fhir_error: str | None = None
    source = "direct"
    facts: safety_review.PatientFacts | None = None

    if (age is None) and fhir_ctx and resolved_pid:
        try:
            client = FhirClient(fhir_ctx)
            facts = await safety_review.patient_facts_from_fhir(client, resolved_pid)
            if facts is not None:
                age = facts.age
                conditions = list(facts.conditions)
                pregnant = facts.pregnant
                source = "fhir"
        except Exception as exc:
            fhir_error = f"FHIR fetch failed: {exc}"

    if age is None:
        raise ValueError(
            "age is required when SHARP/FHIR patient context is not provided"
        )

    matches = vaccines.recommendations_for(
        age=age, conditions=conditions or [], pregnant=pregnant
    )
    fhir_resource = _build_immunization_recommendation(matches, resolved_pid)

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║         NEUROPHRAX — VACCINE RECOMMENDATIONS      ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Source       : {'FHIR R4 (SHARP context)' if source == 'fhir' else 'Direct input'}",
        f"  Patient ID   : {resolved_pid or '—'}",
        f"  Age          : {age} years",
        f"  Conditions   : {', '.join(conditions or []) or '—'}",
        f"  Pregnant     : {'Yes' if pregnant else 'No'}",
        f"  Sources      : CDC ACIP Adult Immunization Schedule",
        "",
    ]
    if fhir_error:
        lines += [f"  ⚠️  {fhir_error} — used direct args.", ""]
    if not matches:
        lines += [
            "  ✅ No additional vaccines matched the supplied context.",
            "     Always confirm against the live CDC ACIP schedule.",
        ]
    for rule in matches:
        emoji = _VACCINE_INDICATION_BANNER.get(rule.indication, "▫")
        lines += [
            f"  {emoji} [{rule.indication}] {rule.name}",
            f"     Why     : {rule.rationale}",
            f"     Schedule: {rule.schedule}",
            "",
        ]

    lines.append(f"  ⚕️  DISCLAIMER: {DISCLAIMER}")

    warnings: list[str] = []
    if facts and facts.unparseable_condition_ids:
        warnings.append(
            f"{len(facts.unparseable_condition_ids)} condition FHIR entry/entries had missing "
            f"code/display: {facts.unparseable_condition_ids}. Risk-based vaccine matching may "
            "be incomplete."
        )

    return _response(
        summary="\n".join(lines),
        data={
            "patient": {
                "age": age,
                "patient_id": resolved_pid,
                "conditions": list(conditions or []),
                "pregnant": pregnant,
                "source": source,
            },
            "recommendations": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "indication": r.indication,
                    "rationale": r.rationale,
                    "schedule": r.schedule,
                }
                for r in matches
            ],
            "chart_status": "ok",
            "warnings": warnings,
            "unparseable_condition_ids": list(facts.unparseable_condition_ids) if facts else [],
            "fhir": {"ImmunizationRecommendation": fhir_resource},
            "fhir_error": fhir_error,
        },
        sources=[
            {
                "name": "CDC ACIP Adult Immunization Schedule",
                "url": VACCINES_URL,
                "attribution": VACCINES_ATTRIBUTION,
                "accessed_at": _now(),
            },
        ],
    )


# ── Tool 9: Prescription Safety Brief (mega-composer) ──────────────────────


@mcp.tool()
async def prescription_safety_brief(
    medications: Annotated[
        list[str] | None,
        Field(description="Medications to review. Omit if SHARP/FHIR patient context is set."),
    ] = None,
    age: Annotated[
        int | None,
        Field(ge=0, le=130, description="Patient age. Omit if SHARP/FHIR context is set."),
    ] = None,
    conditions: Annotated[
        list[str] | None,
        Field(description="Active conditions. Pulled from FHIR if context is set."),
    ] = None,
    pregnant: Annotated[
        bool, Field(description="Whether the patient is pregnant or planning to be.")
    ] = False,
    location: Annotated[
        str | None,
        Field(description="City/state/country to bias clinical-trial proximity matching."),
    ] = None,
    evidence_per_finding: Annotated[
        int,
        Field(ge=0, le=5, description="Citations to attach per safety finding (0 = skip evidence)."),
    ] = 2,
    trials_per_condition: Annotated[
        int,
        Field(ge=0, le=10, description="Trial matches per active condition (0 = skip trials)."),
    ] = 3,
    patientId: Annotated[  # noqa: N803
        str | None,
        Field(description="Explicit FHIR Patient ID. Auto-resolved from SHARP context if omitted."),
    ] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """PREFERRED default for any patient-aware clinical question. Use this
    when the user wants the full picture on a patient — safety analysis,
    supporting evidence, vaccines due, and clinical trials, all in ONE call.
    Skips the need to orchestrate ``medication_safety_review`` +
    ``clinical_evidence_search`` + ``vaccine_recommendations`` +
    ``clinical_trial_search`` separately.

    Single-call clinical brief that synthesizes safety, evidence, prevention, and trials.

    Internally chains four primitives the agent would otherwise have to
    orchestrate by hand:

    1. ``medication_safety_review`` — pairwise interactions (DDInter +
       OpenFDA), Beers Criteria flags, duplicate therapeutic classes, and
       boxed-warning / pregnancy contraindications.
    2. ``clinical_evidence_search`` — for every flagged finding, a
       publication-type filtered PubMed query for guidelines, systematic
       reviews, and meta-analyses supporting the warning.
    3. ``vaccine_recommendations`` — ACIP adult immunization rules matched
       against the patient's age, conditions, and pregnancy status.
    4. ``clinical_trial_search`` — actively recruiting trials on
       ClinicalTrials.gov v2 keyed to the patient's distinct active
       conditions, optionally biased by ``location``.

    When the request carries SHARP-on-MCP headers (``x-fhir-server-url`` /
    ``x-fhir-access-token`` / ``x-patient-id``), patient context is fetched
    from the FHIR server with a single round trip — no arguments required.

    Returns the standard envelope plus a FHIR transaction ``Bundle``
    containing ``RiskAssessment`` + ``ImmunizationRecommendation`` +
    ``DocumentReference`` (the brief narrative, base64-encoded), ready to
    POST back to the EHR's FHIR base URL.
    """
    fhir_ctx = get_fhir_context(ctx)
    resolved_pid = resolve_patient_id(ctx, patientId)
    patient_facts: safety_review.PatientFacts | None = None
    fhir_error: str | None = None

    if fhir_ctx and resolved_pid:
        try:
            client = FhirClient(fhir_ctx)
            patient_facts = await safety_review.patient_facts_from_fhir(client, resolved_pid)
        except Exception as exc:
            fhir_error = f"FHIR fetch failed: {exc}"
            patient_facts = None

    if patient_facts is None:
        if not medications or age is None:
            raise ValueError(
                "medications and age are required when SHARP/FHIR patient context is not provided"
            )
        patient_facts = safety_review.PatientFacts(
            age=age,
            medications=list(medications),
            conditions=list(conditions or []),
            pregnant=pregnant,
            patient_id=resolved_pid,
            source="direct",
            medications_source="direct",
        )

    chart_status = "empty_chart" if (
        patient_facts.source == "fhir" and not patient_facts.medications
    ) else "ok"

    brief = await safety_brief.compose(
        patient_facts,
        location=location,
        evidence_per_finding=evidence_per_finding,
        trials_per_condition=trials_per_condition,
        include_trials=trials_per_condition > 0,
    )

    sources_meta: list[dict[str, Any]] = []
    if patient_facts.source == "fhir":
        sources_meta.append(
            {
                "name": "FHIR R4 Server",
                "url": fhir_ctx.url if fhir_ctx else None,
                "accessed_at": _now(),
            }
        )
    sources_meta += [
        {"name": "RxNorm", "url": rxnorm.BASE, "accessed_at": _now()},
        {"name": "RxClass (ATC/EPC)", "url": rxclass.BASE, "accessed_at": _now()},
        {"name": "OpenFDA Drug Label", "url": openfda.BASE, "accessed_at": _now()},
        {
            "name": "DDInter 2.0",
            "url": interactions.DDINTER_HOME,
            "version": interactions.dataset_version() or "not ingested",
            "accessed_at": _now(),
        },
        {
            "name": "AGS Beers Criteria 2023",
            "url": BEERS_URL,
            "attribution": BEERS_ATTRIBUTION,
            "accessed_at": _now(),
        },
        {"name": "NCBI PubMed (E-Utilities)", "url": pubmed.BASE, "accessed_at": _now()},
        {
            "name": "CDC ACIP Adult Immunization Schedule",
            "url": VACCINES_URL,
            "attribution": VACCINES_ATTRIBUTION,
            "accessed_at": _now(),
        },
        {"name": "ClinicalTrials.gov v2", "url": clinicaltrials.BASE, "accessed_at": _now()},
    ]

    findings_payload = [fe.to_dict() for fe in brief.findings_with_evidence]
    vaccine_payload = [
        {
            "rule_id": r.rule_id,
            "name": r.name,
            "indication": r.indication,
            "rationale": r.rationale,
            "schedule": r.schedule,
        }
        for r in brief.vaccine_rules
    ]

    warnings: list[str] = []
    if chart_status == "empty_chart":
        warnings.append(
            "FHIR chart was fetched but contained 0 active medications "
            "(checked MedicationStatement and MedicationRequest with status=active). "
            "Safety brief was generated against an empty medication list — findings, "
            "vaccine recommendations, and trial matches reflect chart conditions only, "
            "not pharmacotherapy."
        )
    if patient_facts.unparseable_medication_ids:
        warnings.append(
            f"{len(patient_facts.unparseable_medication_ids)} medication entry/entries had "
            f"missing code/display: {patient_facts.unparseable_medication_ids}."
        )
    if patient_facts.unparseable_condition_ids:
        warnings.append(
            f"{len(patient_facts.unparseable_condition_ids)} condition entry/entries had "
            f"missing code/display: {patient_facts.unparseable_condition_ids}."
        )

    summary = render_brief_summary(brief, fhir_error=fhir_error)
    if warnings:
        summary = (
            summary
            + "\n\n  ⚠️  WARNINGS:\n"
            + "\n".join(f"     • {w}" for w in warnings)
        )

    return _response(
        summary=summary,
        data={
            "patient": {
                "age": patient_facts.age,
                "patient_id": patient_facts.patient_id,
                "medications": patient_facts.medications,
                "conditions": patient_facts.conditions,
                "pregnant": patient_facts.pregnant,
                "source": patient_facts.source,
                "medications_source": patient_facts.medications_source,
            },
            "chart_status": chart_status,
            "warnings": warnings,
            "unparseable_medication_ids": list(patient_facts.unparseable_medication_ids),
            "unparseable_condition_ids": list(patient_facts.unparseable_condition_ids),
            "findings": findings_payload,
            "vaccine_recommendations": vaccine_payload,
            "trials": brief.trial_blocks,
            "fhir": {
                "Bundle": brief.fhir_bundle,
                "RiskAssessment": brief.fhir_resources["RiskAssessment"],
                "ImmunizationRecommendation": brief.fhir_resources["ImmunizationRecommendation"],
                "DocumentReference": brief.fhir_resources["DocumentReference"],
            },
            "fhir_error": fhir_error,
        },
        sources=sources_meta,
    )


def render_brief_summary(brief: safety_brief.SafetyBrief, *, fhir_error: str | None) -> str:
    """Wrap the rendered narrative with a leading FHIR error notice if any."""
    body = safety_brief.render_brief(
        patient=brief.patient,
        findings_with_evidence=brief.findings_with_evidence,
        vaccine_rules=brief.vaccine_rules,
        trial_blocks=brief.trial_blocks,
    )
    if fhir_error:
        return f"  ⚠️  {fhir_error} — used direct args.\n\n{body}"
    return body


# ── HTTP transport (FastAPI + streamable HTTP at /mcp) ──────────────────────


@asynccontextmanager
async def _lifespan(_: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(lifespan=_lifespan, title="Neurophrax MCP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/", mcp.streamable_http_app())


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
