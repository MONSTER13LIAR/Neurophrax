"""FHIR R4 resource emitters.

Helpers that build minimal but conformant FHIR resources from Neurophrax tool
output, so callers (or the host platform) can persist or pipe results back
into a FHIR server. We deliberately keep these small and dependency-free
rather than pulling a full FHIR library.
"""
from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone
from typing import Any

# Map Neurophrax severity vocabularies to FHIR risk-probability codes.
RISK_PROBABILITY_SYSTEM = "http://terminology.hl7.org/CodeSystem/risk-probability"

_SEVERITY_TO_RISK_PROBABILITY: dict[str, str] = {
    # Drug-interaction severities (DDInter / OpenFDA)
    "MAJOR": "high",
    "MODERATE": "moderate",
    "MINOR": "low",
    "UNKNOWN": "negligible",
    # Beers Criteria severities
    "AVOID": "high",
    "AVOID_CHRONIC_USE": "moderate",
    "USE_WITH_CAUTION": "moderate",
    "DOSE_ADJUST": "low",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _patient_ref(patient_id: str | None) -> dict[str, Any] | None:
    if not patient_id:
        return None
    return {"reference": f"Patient/{patient_id}"}


def risk_probability(severity: str) -> dict[str, Any]:
    """Build a FHIR ``CodeableConcept`` for a risk-probability code."""
    code = _SEVERITY_TO_RISK_PROBABILITY.get(severity.upper(), "negligible")
    return {
        "coding": [{"system": RISK_PROBABILITY_SYSTEM, "code": code}],
        "text": severity,
    }


def risk_assessment(
    *,
    findings: list[dict[str, Any]],
    patient_id: str | None,
    method_text: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Build a FHIR ``RiskAssessment`` resource.

    ``findings`` is a list of dicts with at minimum ``outcome`` (str),
    ``severity`` (str), and ``rationale`` (str). Optional ``mitigation`` (str)
    is appended to the rationale.
    """
    predictions: list[dict[str, Any]] = []
    for f in findings:
        rationale = f.get("rationale") or ""
        if f.get("mitigation"):
            rationale = f"{rationale} Recommendation: {f['mitigation']}".strip()
        predictions.append(
            {
                "outcome": {"text": f["outcome"]},
                "qualitativeRisk": risk_probability(f["severity"]),
                "rationale": rationale,
            }
        )

    resource: dict[str, Any] = {
        "resourceType": "RiskAssessment",
        "status": "final",
        "occurrenceDateTime": _now(),
        "method": {"text": method_text},
        "prediction": predictions,
    }
    subject = _patient_ref(patient_id)
    if subject:
        resource["subject"] = subject
    if note:
        resource["note"] = [{"text": note}]
    return resource


def document_reference(
    *,
    content_text: str,
    title: str,
    description: str | None = None,
    patient_id: str | None = None,
    type_text: str = "Clinical synthesis brief",
    category_text: str = "Clinical Note",
) -> dict[str, Any]:
    """Build a FHIR ``DocumentReference`` wrapping a text narrative.

    The narrative is base64-encoded into a single ``content[].attachment``.
    Suitable for persisting an AI-generated clinical brief back to an EHR
    alongside the structured resources it summarizes.
    """
    encoded = base64.b64encode(content_text.encode("utf-8")).decode("ascii")
    resource: dict[str, Any] = {
        "resourceType": "DocumentReference",
        "status": "current",
        "docStatus": "final",
        "type": {"text": type_text},
        "category": [{"text": category_text}],
        "date": _now(),
        "content": [
            {
                "attachment": {
                    "contentType": "text/plain; charset=utf-8",
                    "data": encoded,
                    "title": title,
                    "creation": _now(),
                }
            }
        ],
    }
    if description:
        resource["description"] = description
    subject = _patient_ref(patient_id)
    if subject:
        resource["subject"] = subject
    return resource


def transaction_bundle(resources: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a list of FHIR resources into a transaction ``Bundle``.

    Each resource gets a ``urn:uuid:`` ``fullUrl`` and a POST request entry
    keyed to the resource's ``resourceType`` — the standard shape an EHR's
    FHIR endpoint accepts at ``POST /[base]`` to commit them atomically.
    """
    entries: list[dict[str, Any]] = []
    for res in resources:
        rtype = res.get("resourceType")
        if not rtype:
            continue
        entries.append(
            {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": res,
                "request": {"method": "POST", "url": rtype},
            }
        )
    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "timestamp": _now(),
        "entry": entries,
    }
