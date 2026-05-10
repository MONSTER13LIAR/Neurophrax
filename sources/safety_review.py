"""Composite medication safety review.

Combines four independent signals — pairwise drug-drug interactions, Beers
Criteria flags for older adults, duplicate therapeutic-class detection, and
pregnancy/boxed-warning detection from FDA labels — into a single prioritized
risk list with both a human-readable summary and a FHIR ``RiskAssessment``
output.

This is the AI-factor centerpiece: no single source produces this output;
the value comes from cross-referencing them against patient context.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from . import beers, fhir_emit, interactions, openfda, rxclass, rxnorm
from .sharp import FhirClient, FhirContext, chart_from_fhir  # noqa: F401


# ── Severity ranking for sorting (higher = more urgent) ─────────────────────
_SEVERITY_RANK: dict[str, int] = {
    "MAJOR": 100,
    "AVOID": 95,
    "AVOID_CHRONIC_USE": 80,
    "USE_WITH_CAUTION": 70,
    "MODERATE": 65,
    "DOSE_ADJUST": 50,
    "MINOR": 30,
    "UNKNOWN": 10,
}


@dataclass(slots=True)
class Finding:
    """A single risk finding in the prioritized output."""

    category: str  # "interaction" | "beers" | "duplicate_class" | "boxed_warning" | "pregnancy"
    outcome: str
    severity: str
    rationale: str
    mitigation: str = ""
    drugs: tuple[str, ...] = field(default_factory=tuple)
    rule_id: str = ""

    def sort_key(self) -> tuple[int, str]:
        return (-_SEVERITY_RANK.get(self.severity.upper(), 0), self.outcome)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "outcome": self.outcome,
            "severity": self.severity,
            "rationale": self.rationale,
            "mitigation": self.mitigation,
            "drugs": list(self.drugs),
            "rule_id": self.rule_id,
        }


@dataclass(slots=True)
class MedFacts:
    """Per-medication resolved facts gathered in parallel."""

    name: str
    rxcui: str | None = None
    ingredient_name: str | None = None
    atc_classes: list[dict[str, Any]] = field(default_factory=list)
    epc_classes: list[dict[str, Any]] = field(default_factory=list)
    label: dict[str, Any] | None = None


@dataclass(slots=True)
class PatientFacts:
    """Inputs to the review, resolved either from FHIR or direct args."""

    age: int
    medications: list[str]
    conditions: list[str] = field(default_factory=list)
    pregnant: bool = False
    patient_id: str | None = None
    source: str = "direct"  # "fhir" | "direct"
    medications_source: str = "none"  # "MedicationStatement" | "MedicationRequest" | "none" | "direct"
    unparseable_medication_ids: list[str] = field(default_factory=list)
    unparseable_condition_ids: list[str] = field(default_factory=list)


# ── FHIR ingestion (SHARP path) ─────────────────────────────────────────────


async def patient_facts_from_fhir(
    fhir: FhirClient, patient_id: str
) -> PatientFacts | None:
    """Pull (age, medications, conditions, pregnant?) from a FHIR R4 server.

    Thin projection over :func:`sources.sharp.chart_from_fhir` so the safety
    review and the patient-summary tool share the same FHIR fetch path.
    """
    chart = await chart_from_fhir(fhir, patient_id)
    if chart is None:
        return None
    return PatientFacts(
        age=chart.age,
        medications=list(chart.medications),
        conditions=list(chart.conditions),
        pregnant=chart.pregnant,
        patient_id=chart.patient_id,
        source="fhir",
        medications_source=chart.medications_source,
        unparseable_medication_ids=list(chart.unparseable_medication_ids),
        unparseable_condition_ids=list(chart.unparseable_condition_ids),
    )


# ── Drug fact gathering ─────────────────────────────────────────────────────


async def _gather_med_facts(name: str) -> MedFacts:
    rxcui = await rxnorm.name_to_ingredient_rxcui(name)
    classes_task = rxclass.classes_for_rxcui(rxcui) if rxcui else asyncio.sleep(0, result={"atc": [], "epc": []})
    label_task = openfda.label_for(name)
    classes, label = await asyncio.gather(classes_task, label_task)

    ingredient_name: str | None = None
    if label:
        subs = (label.get("openfda", {}) or {}).get("substance_name") or []
        if len(subs) == 1:
            ingredient_name = subs[0].lower()
    if not ingredient_name:
        # Fall back to user-provided name (lowercased) when label resolution failed.
        ingredient_name = name.lower()

    return MedFacts(
        name=name,
        rxcui=rxcui,
        ingredient_name=ingredient_name,
        atc_classes=classes.get("atc") or [],
        epc_classes=classes.get("epc") or [],
        label=label,
    )


# ── Composers ──────────────────────────────────────────────────────────────


_PREGNANCY_RX = re.compile(
    r"\b(pregnan|fetal|teratogen|category\s*[dx]|contraindicat\w*\s+in\s+pregnan)",
    re.IGNORECASE,
)


def _label_text(label: dict[str, Any] | None, key: str) -> str:
    if not label:
        return ""
    val = label.get(key)
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val or "")


def _atc_l3_prefix(class_id: str) -> str:
    """Return the ATC level-3 'pharmacological/therapeutic subgroup' prefix.

    ATC code structure: L1=1 char, L2=3 chars, L3=4 chars, L4=5 chars, L5=7
    chars. Level 3 is the standard granularity for "duplicate therapeutic
    class" safety checks: two anticoagulants from different L4 chemical
    subgroups (e.g., warfarin B01AA03 and apixaban B01AF02) still share the
    L3 antithrombotic class B01A and concurrent use is clinically dangerous.
    """
    return class_id[:4] if class_id else ""


def _interaction_findings(
    inter_results: list[dict[str, Any]],
) -> list[Finding]:
    out: list[Finding] = []
    for inter in inter_results:
        sev = (inter.get("severity") or "Unknown").upper()
        out.append(
            Finding(
                category="interaction",
                outcome=f"Interaction: {inter['drug_a']} + {inter['drug_b']}",
                severity=sev,
                rationale=(inter.get("description") or f"Reported by {inter.get('source', 'source')}.").strip(),
                mitigation="Consider therapeutic alternative or close monitoring.",
                drugs=(inter["drug_a"], inter["drug_b"]),
            )
        )
    return out


def _beers_findings(facts: list[MedFacts], age: int) -> list[Finding]:
    if age <= 0:
        return []
    out: list[Finding] = []
    for fact in facts:
        atc_ids = tuple(c["class_id"] for c in fact.atc_classes if c.get("class_id"))
        rules = beers.flags_for(
            age=age,
            ingredient_name=fact.ingredient_name,
            atc_class_ids=atc_ids,
        )
        for rule in rules:
            out.append(
                Finding(
                    category="beers",
                    outcome=f"Beers: {rule.title} — {fact.name}",
                    severity=rule.severity,
                    rationale=rule.rationale,
                    mitigation=rule.recommendation,
                    drugs=(fact.name,),
                    rule_id=rule.rule_id,
                )
            )
    return out


def _duplicate_class_findings(facts: list[MedFacts]) -> list[Finding]:
    """Flag two or more medications sharing an ATC level-3 class.

    Skips combination-product class memberships (those whose class name
    contains the word "combination") so a fixed-dose combo isn't reported as
    a duplicate of one of its ingredients.
    """
    by_prefix: dict[str, list[tuple[str, str]]] = {}
    for fact in facts:
        for cls in fact.atc_classes:
            cid = cls.get("class_id") or ""
            cname = (cls.get("class_name") or "").lower()
            if not cid or "combination" in cname:
                continue
            prefix = _atc_l3_prefix(cid)
            if not prefix:
                continue
            by_prefix.setdefault(prefix, []).append((fact.name, cls.get("class_name") or cid))

    out: list[Finding] = []
    for prefix, members in by_prefix.items():
        unique_meds = {name for name, _ in members}
        if len(unique_meds) < 2:
            continue
        class_label = members[0][1]
        meds_sorted = sorted(unique_meds)
        out.append(
            Finding(
                category="duplicate_class",
                outcome=f"Duplicate therapeutic class ({class_label})",
                severity="USE_WITH_CAUTION",
                rationale=(
                    f"{', '.join(meds_sorted)} share ATC class {prefix}. "
                    "Concurrent use within the same pharmacological subgroup may compound effects."
                ),
                mitigation="Confirm both are clinically indicated; consider consolidation.",
                drugs=tuple(meds_sorted),
            )
        )
    return out


def _boxed_warning_findings(facts: list[MedFacts]) -> list[Finding]:
    out: list[Finding] = []
    for fact in facts:
        boxed = _label_text(fact.label, "boxed_warning").strip()
        if not boxed:
            continue
        snippet = boxed[:300] + ("…" if len(boxed) > 300 else "")
        out.append(
            Finding(
                category="boxed_warning",
                outcome=f"FDA boxed warning: {fact.name}",
                severity="MAJOR",
                rationale=snippet,
                mitigation="Review boxed-warning indication, monitoring, and contraindications before continuing.",
                drugs=(fact.name,),
            )
        )
    return out


def _pregnancy_findings(facts: list[MedFacts]) -> list[Finding]:
    out: list[Finding] = []
    for fact in facts:
        text = " ".join(
            _label_text(fact.label, k)
            for k in ("pregnancy", "use_in_specific_populations", "contraindications", "boxed_warning")
        )
        if not text:
            continue
        m = _PREGNANCY_RX.search(text)
        if not m:
            continue
        # Pull a sentence around the match for a useful rationale.
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 200)
        snippet = text[start:end].strip()
        out.append(
            Finding(
                category="pregnancy",
                outcome=f"Pregnancy concern: {fact.name}",
                severity="MAJOR",
                rationale=(snippet[:400] + "…") if len(snippet) > 400 else snippet,
                mitigation="Verify pregnancy status and review safer alternatives before continuing.",
                drugs=(fact.name,),
            )
        )
    return out


# ── Top-level review ────────────────────────────────────────────────────────


@dataclass(slots=True)
class ReviewResult:
    findings: list[Finding]
    facts: list[MedFacts]
    patient: PatientFacts
    fhir_resource: dict[str, Any]
    unresolved_medications: list[str] = field(default_factory=list)


async def run(patient: PatientFacts) -> ReviewResult:
    """Run the full safety review and return prioritized findings + FHIR resource."""
    if not patient.medications:
        empty = ReviewResult(
            findings=[],
            facts=[],
            patient=patient,
            fhir_resource=fhir_emit.risk_assessment(
                findings=[],
                patient_id=patient.patient_id,
                method_text="Neurophrax composite medication safety review",
                note="No active medications were available for review.",
            ),
            unresolved_medications=[],
        )
        return empty

    # Fan out per-med fact gathering and pairwise interaction check in parallel.
    facts_task = asyncio.gather(*[_gather_med_facts(m) for m in patient.medications])
    rxcuis_task = asyncio.gather(
        *[rxnorm.name_to_rxcui(m) for m in patient.medications]
    )
    facts, rxcuis = await asyncio.gather(facts_task, rxcuis_task)
    unresolved_medications = [m for m, c in zip(patient.medications, rxcuis) if not c]
    inter_results, _counts = await interactions.check_pairs(patient.medications, rxcuis)

    findings: list[Finding] = []
    findings += _interaction_findings(inter_results)
    findings += _beers_findings(facts, patient.age)
    findings += _duplicate_class_findings(facts)
    findings += _boxed_warning_findings(facts)
    if patient.pregnant:
        findings += _pregnancy_findings(facts)

    findings.sort(key=lambda f: f.sort_key())

    fhir_resource = fhir_emit.risk_assessment(
        findings=[f.to_dict() for f in findings],
        patient_id=patient.patient_id,
        method_text="Neurophrax composite medication safety review",
        note=(
            f"Inputs: age {patient.age}, {len(patient.medications)} medication(s), "
            f"{len(patient.conditions)} active condition(s); source={patient.source}."
        ),
    )

    return ReviewResult(
        findings=findings,
        facts=facts,
        patient=patient,
        fhir_resource=fhir_resource,
        unresolved_medications=unresolved_medications,
    )
