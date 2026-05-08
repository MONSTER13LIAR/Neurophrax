"""Prescription safety brief — synthesis composer.

Single entry point that chains the medication safety review, per-finding
literature evidence, ACIP vaccine matching, and clinical-trial matching into
one structured clinical brief, then bundles the FHIR-shaped outputs
(``RiskAssessment`` + ``ImmunizationRecommendation`` + ``DocumentReference``)
into a transaction ``Bundle`` ready to POST back to an EHR.

The value of this module is the cross-referencing — none of the upstream
sources alone produces this output. Given a patient context it answers:
"what is risky, why, what does the literature say, what should we add for
prevention, and where could this patient enroll?"
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from . import fhir_emit, pubmed, vaccines
from . import clinicaltrials as ctgov
from .safety_review import Finding, PatientFacts, ReviewResult
from .safety_review import run as run_safety_review


# ── Per-finding evidence query construction ─────────────────────────────────


def _quote(term: str) -> str:
    """Quote a multi-word term for PubMed; pass single tokens through."""
    term = (term or "").strip()
    if not term:
        return ""
    return f'"{term}"' if " " in term else term


def _evidence_query(finding: Finding) -> tuple[str, list[str]] | None:
    """Build (question, study_types) for a finding, or None to skip.

    Selects publication-type filters per category so the returned citations
    bias toward guidance-grade evidence rather than primary case reports.
    """
    drugs = [_quote(d) for d in finding.drugs if d]
    cat = finding.category

    if cat == "interaction" and len(drugs) >= 2:
        q = f"{drugs[0]} AND {drugs[1]} AND interaction"
        return q, ["meta-analysis", "systematic-review", "guideline"]
    if cat == "duplicate_class" and len(drugs) >= 2:
        q = f"{drugs[0]} AND {drugs[1]} AND coadministration"
        return q, ["systematic-review", "guideline", "review"]
    if cat == "beers" and drugs:
        q = f'{drugs[0]} AND ("older adults" OR elderly) AND (Beers OR "potentially inappropriate")'
        return q, ["guideline", "systematic-review", "practice-guideline"]
    if cat == "boxed_warning" and drugs:
        q = f"{drugs[0]} AND (adverse OR safety)"
        return q, ["systematic-review", "meta-analysis"]
    if cat == "pregnancy" and drugs:
        q = f"{drugs[0]} AND pregnancy"
        return q, ["guideline", "systematic-review", "practice-guideline"]
    return None


async def _evidence_for_finding(
    finding: Finding, *, per_finding: int, max_age_years: int
) -> list[dict[str, Any]]:
    """Run a single targeted PubMed search for one finding.

    Falls back to an unrestricted study-type search if the filtered query
    returns nothing — guidance-grade hits are preferred but absence of one
    shouldn't leave the finding evidence-less.
    """
    spec = _evidence_query(finding)
    if spec is None:
        return []
    question, study_types = spec
    result = await pubmed.search(
        question=question,
        study_types=study_types,
        max_age_years=max_age_years,
        limit=per_finding,
    )
    items = result.get("results") or []
    if items:
        return items
    fallback = await pubmed.search(
        question=question,
        study_types=None,
        max_age_years=max_age_years,
        limit=per_finding,
    )
    return fallback.get("results") or []


# ── Trial fan-out across active conditions ─────────────────────────────────


def _dedupe_conditions(conditions: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in conditions:
        key = (c or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c.strip())
    return out


async def _trials_for_conditions(
    conditions: list[str],
    *,
    location: str | None,
    per_condition: int,
) -> list[dict[str, Any]]:
    if not conditions:
        return []
    tasks = [
        ctgov.search_trials(
            condition=cond,
            location=location,
            statuses=ctgov.RECRUITING_STATUSES,
            limit=per_condition,
        )
        for cond in conditions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    blocks: list[dict[str, Any]] = []
    for cond, res in zip(conditions, results):
        trials = res if isinstance(res, list) else []
        blocks.append({"condition": cond, "trials": trials, "count": len(trials)})
    return blocks


# ── Vaccine recommendation → FHIR ImmunizationRecommendation ───────────────


def _build_immunization_recommendation(
    rules: list[Any], patient_id: str | None, *, now: str
) -> dict[str, Any]:
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
        "date": now,
        "recommendation": recs,
    }
    if patient_id:
        resource["patient"] = {"reference": f"Patient/{patient_id}"}
    return resource


# ── Top-level brief ────────────────────────────────────────────────────────


@dataclass(slots=True)
class FindingWithEvidence:
    finding: Finding
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.finding.to_dict()
        d["evidence"] = self.evidence
        return d


@dataclass(slots=True)
class SafetyBrief:
    patient: PatientFacts
    review: ReviewResult
    findings_with_evidence: list[FindingWithEvidence]
    vaccine_rules: list[Any]
    trial_blocks: list[dict[str, Any]]
    fhir_bundle: dict[str, Any]
    fhir_resources: dict[str, dict[str, Any]]


async def compose(
    patient: PatientFacts,
    *,
    location: str | None = None,
    evidence_per_finding: int = 2,
    evidence_max_age_years: int = 10,
    trials_per_condition: int = 3,
    include_trials: bool = True,
) -> SafetyBrief:
    """Run the full prescription safety brief pipeline.

    Stages:

    1. Run :func:`sources.safety_review.run` to get prioritized findings + a
       ``RiskAssessment`` resource for the patient's medications.
    2. For each finding, fan out a targeted PubMed search (publication-type
       filtered toward guidelines and systematic reviews) for evidence.
    3. Match the patient against the curated ACIP rules and emit a FHIR
       ``ImmunizationRecommendation``.
    4. Search ClinicalTrials.gov v2 for actively recruiting trials against
       each distinct active condition.
    5. Render a clinical-brief narrative and wrap it in a FHIR
       ``DocumentReference``; bundle all three resources as a transaction
       ``Bundle`` ready to POST back to an EHR.
    """
    now = fhir_emit._now()  # noqa: SLF001 — internal helper, single source of "now"

    review = await run_safety_review(patient)

    evidence_lists, vaccine_rules, trial_blocks = await asyncio.gather(
        asyncio.gather(
            *[
                _evidence_for_finding(
                    f,
                    per_finding=evidence_per_finding,
                    max_age_years=evidence_max_age_years,
                )
                for f in review.findings
            ]
        )
        if review.findings
        else asyncio.sleep(0, result=[]),
        asyncio.sleep(
            0,
            result=vaccines.recommendations_for(
                age=patient.age,
                conditions=list(patient.conditions),
                pregnant=patient.pregnant,
            ),
        ),
        _trials_for_conditions(
            _dedupe_conditions(patient.conditions),
            location=location,
            per_condition=trials_per_condition,
        )
        if include_trials
        else asyncio.sleep(0, result=[]),
    )

    findings_with_evidence = [
        FindingWithEvidence(finding=f, evidence=list(ev))
        for f, ev in zip(review.findings, evidence_lists)
    ]

    immunization_rec = _build_immunization_recommendation(
        vaccine_rules, patient.patient_id, now=now
    )

    narrative = render_brief(
        patient=patient,
        findings_with_evidence=findings_with_evidence,
        vaccine_rules=vaccine_rules,
        trial_blocks=trial_blocks,
    )
    doc_ref = fhir_emit.document_reference(
        content_text=narrative,
        title="Neurophrax Prescription Safety Brief",
        description=(
            f"Synthesized brief covering {len(findings_with_evidence)} safety finding(s), "
            f"{len(vaccine_rules)} vaccine recommendation(s), and "
            f"{sum(b['count'] for b in trial_blocks)} matched trial(s)."
        ),
        patient_id=patient.patient_id,
        type_text="Medication safety + preventive care brief",
    )

    bundle_resources = [review.fhir_resource, immunization_rec, doc_ref]
    fhir_bundle = fhir_emit.transaction_bundle(bundle_resources)

    return SafetyBrief(
        patient=patient,
        review=review,
        findings_with_evidence=findings_with_evidence,
        vaccine_rules=vaccine_rules,
        trial_blocks=trial_blocks,
        fhir_bundle=fhir_bundle,
        fhir_resources={
            "RiskAssessment": review.fhir_resource,
            "ImmunizationRecommendation": immunization_rec,
            "DocumentReference": doc_ref,
        },
    )


# ── Narrative rendering ─────────────────────────────────────────────────────


_SEVERITY_BANNER: dict[str, str] = {
    "MAJOR": "🔴",
    "AVOID": "🔴",
    "AVOID_CHRONIC_USE": "🟠",
    "USE_WITH_CAUTION": "🟡",
    "MODERATE": "🟡",
    "DOSE_ADJUST": "🟢",
    "MINOR": "🟢",
    "UNKNOWN": "⚪",
}

_VACCINE_BANNER: dict[str, str] = {
    vaccines.ROUTINE: "✅",
    vaccines.RISK_BASED: "🟡",
    vaccines.SHARED_DECISION: "🟦",
}


def render_brief(
    *,
    patient: PatientFacts,
    findings_with_evidence: list[FindingWithEvidence],
    vaccine_rules: list[Any],
    trial_blocks: list[dict[str, Any]],
) -> str:
    """Render the human-readable clinical brief used as the tool ``summary``.

    Also stored verbatim inside the FHIR ``DocumentReference`` attachment so
    a downstream EHR persists the same narrative an agent reads.
    """
    lines: list[str] = [
        "╔══════════════════════════════════════════════════╗",
        "║   NEUROPHRAX — PRESCRIPTION SAFETY BRIEF         ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Source       : {'FHIR R4 (SHARP context)' if patient.source == 'fhir' else 'Direct input'}",
        f"  Patient ID   : {patient.patient_id or '—'}",
        f"  Age          : {patient.age} years",
        f"  Medications  : {len(patient.medications)} ({', '.join(m.title() for m in patient.medications) or '—'})",
        f"  Conditions   : {len(patient.conditions)} ({', '.join(patient.conditions) or '—'})",
        f"  Pregnant     : {'Yes' if patient.pregnant else 'No'}",
        "",
        "  ── SAFETY FINDINGS ──",
    ]

    if not findings_with_evidence:
        lines += [
            "  ✅ No safety signals matched.",
            "     Continue routine pharmacist review and monitoring.",
            "",
        ]
    else:
        for fe in findings_with_evidence:
            f = fe.finding
            emoji = _SEVERITY_BANNER.get(f.severity.upper(), "⚪")
            rationale = f.rationale[:280] + ("…" if len(f.rationale) > 280 else "")
            lines += [
                f"  {emoji} [{f.severity}] {f.outcome}",
                f"     Why : {rationale}",
            ]
            if f.mitigation:
                lines.append(f"     Do  : {f.mitigation}")
            if fe.evidence:
                lines.append("     Evidence:")
                for cit in fe.evidence:
                    title = (cit.get("title") or "")[:90]
                    journal = cit.get("journal") or "—"
                    pub = cit.get("pub_date") or "—"
                    lines.append(f"       • PMID {cit.get('pmid')} — {title}")
                    lines.append(f"         {journal}, {pub}")
                    if cit.get("url"):
                        lines.append(f"         {cit['url']}")
            else:
                lines.append("     Evidence: (no guidance-grade citations matched)")
            lines.append("")

    lines.append("  ── PREVENTIVE OPPORTUNITIES (ACIP) ──")
    if not vaccine_rules:
        lines += [
            "  ✅ No additional vaccines matched the supplied context.",
            "",
        ]
    else:
        for rule in vaccine_rules:
            emoji = _VACCINE_BANNER.get(rule.indication, "▫")
            lines += [
                f"  {emoji} [{rule.indication}] {rule.name}",
                f"     {rule.rationale}",
                f"     Schedule: {rule.schedule}",
                "",
            ]

    lines.append("  ── ACTIVELY RECRUITING TRIALS ──")
    total_trials = sum(b["count"] for b in trial_blocks)
    if total_trials == 0:
        lines += [
            "  (no actively recruiting trials matched this patient's conditions)",
            "",
        ]
    else:
        for block in trial_blocks:
            if not block["count"]:
                continue
            lines.append(f"  • {block['condition']} ({block['count']} trial(s))")
            for t in block["trials"]:
                phase = t.get("phase") or t.get("study_type") or "—"
                locs = t.get("locations") or []
                loc_str = (
                    ", ".join(filter(None, [locs[0].get("city", ""), locs[0].get("country", "")]))
                    if locs
                    else "—"
                )
                lines += [
                    f"      → {t['nct_id']} — {t['title'][:80]}",
                    f"        Status: {t['status']}    Phase/Type: {phase}",
                    f"        First location: {loc_str}",
                    f"        URL: {t['url']}",
                ]
            lines.append("")

    lines.append("  ── BUNDLE READY FOR EHR POST ──")
    lines.append(
        "  FHIR transaction Bundle: RiskAssessment + ImmunizationRecommendation + DocumentReference."
    )
    lines.append("")
    lines.append("  ⚕️  For informational use only. Not for clinical decisions.")
    return "\n".join(lines)
