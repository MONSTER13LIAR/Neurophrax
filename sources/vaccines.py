"""Adult vaccine recommendations — curated from CDC ACIP schedules.

A compact, attribution-only subset of the **CDC Advisory Committee on
Immunization Practices (ACIP) Adult Immunization Schedule**. Bundled offline
because there is no stable, no-auth ACIP JSON endpoint; the schedule is
revised yearly and any clinical use must consult the latest CDC publication.

Each :class:`VaccineRule` declares the population indications under which it
applies (age ranges, condition-keyword triggers, pregnancy gating). The tool
returns rules whose indications match the patient context.

Attribution (do not remove):
    Centers for Disease Control and Prevention. Recommended Adult
    Immunization Schedule by Age and Medical Condition (ACIP). The list
    below paraphrases the public CDC schedule current as of 2025; consult
    https://www.cdc.gov/vaccines/schedules/ for the authoritative version.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SOURCE_URL = "https://www.cdc.gov/vaccines/schedules/hcp/imz/adult.html"
ATTRIBUTION = (
    "Adapted from the CDC ACIP Recommended Adult Immunization Schedule. "
    "Consult the live CDC schedule for clinical use; the bundled subset is "
    "intended for software flagging only."
)

ROUTINE = "ROUTINE"
RISK_BASED = "RISK_BASED"
SHARED_DECISION = "SHARED_DECISION"


@dataclass(frozen=True, slots=True)
class VaccineRule:
    rule_id: str
    name: str
    indication: str
    rationale: str
    schedule: str
    age_min: int = 0
    age_max: int = 130
    condition_keywords: tuple[str, ...] = field(default_factory=tuple)
    pregnancy: str = "any"  # "any" | "preferred" | "contraindicated"


RULES: tuple[VaccineRule, ...] = (
    VaccineRule(
        rule_id="acip-influenza",
        name="Seasonal influenza",
        indication=ROUTINE,
        rationale="ACIP recommends annual influenza vaccination for all persons aged ≥ 6 months without contraindications.",
        schedule="1 dose annually each influenza season.",
        age_min=6,  # months — kept for completeness even though we accept years.
    ),
    VaccineRule(
        rule_id="acip-tdap-td",
        name="Tdap / Td (tetanus, diphtheria, ± pertussis)",
        indication=ROUTINE,
        rationale="One Tdap dose for all adults who have not received it; Td or Tdap booster every 10 years.",
        schedule="1 Tdap once, then Td or Tdap every 10 years; Tdap during each pregnancy (27–36 weeks).",
        age_min=19,
        pregnancy="preferred",
    ),
    VaccineRule(
        rule_id="acip-zoster",
        name="Recombinant zoster vaccine (RZV / Shingrix)",
        indication=ROUTINE,
        rationale="ACIP recommends RZV for adults ≥ 50 to prevent shingles and post-herpetic neuralgia.",
        schedule="2 doses, 2–6 months apart (≥ 50). Also recommended ≥ 19 if immunocompromised.",
        age_min=50,
    ),
    VaccineRule(
        rule_id="acip-pneumococcal-65",
        name="Pneumococcal (PCV20 or PCV15 + PPSV23)",
        indication=ROUTINE,
        rationale="Routine pneumococcal vaccination for adults ≥ 65 to reduce invasive pneumococcal disease.",
        schedule="PCV20 alone, OR PCV15 followed ≥ 1 year later by PPSV23.",
        age_min=65,
    ),
    VaccineRule(
        rule_id="acip-pneumococcal-risk",
        name="Pneumococcal (risk-based, age 19–64)",
        indication=RISK_BASED,
        rationale="Adults 19–64 with chronic conditions (heart, lung, liver, diabetes, immunocompromise) qualify for pneumococcal vaccination.",
        schedule="PCV20 alone, OR PCV15 followed by PPSV23.",
        age_min=19,
        age_max=64,
        condition_keywords=(
            "diabetes", "heart failure", "copd", "asthma", "cirrhosis", "ckd",
            "kidney", "liver", "hiv", "asplenia", "sickle", "cochlear",
            "transplant", "immunocompromised", "leukemia", "lymphoma", "multiple myeloma",
        ),
    ),
    VaccineRule(
        rule_id="acip-rsv-75",
        name="RSV vaccine (≥ 75)",
        indication=ROUTINE,
        rationale="ACIP recommends RSV vaccination for all adults ≥ 75 (single dose).",
        schedule="1 dose; not currently recommended as annual.",
        age_min=75,
    ),
    VaccineRule(
        rule_id="acip-rsv-60-74-risk",
        name="RSV vaccine (60–74 with risk factors)",
        indication=SHARED_DECISION,
        rationale="Shared clinical decision-making for adults 60–74 with chronic heart/lung disease, diabetes, severe obesity, immunocompromise, or residence in long-term care.",
        schedule="1 dose if indicated.",
        age_min=60,
        age_max=74,
        condition_keywords=(
            "heart failure", "copd", "asthma", "diabetes", "obesity",
            "ckd", "kidney", "immunocompromised", "transplant", "hiv",
        ),
    ),
    VaccineRule(
        rule_id="acip-rsv-pregnancy",
        name="Maternal RSV vaccine (RSVpreF)",
        indication=ROUTINE,
        rationale="One dose between 32 0/7 and 36 6/7 weeks gestation during RSV season to protect the newborn.",
        schedule="1 dose during third trimester (Sept–Jan in most regions).",
        age_min=15,
        age_max=49,
        pregnancy="preferred",
    ),
    VaccineRule(
        rule_id="acip-covid",
        name="COVID-19 (updated formula)",
        indication=ROUTINE,
        rationale="ACIP recommends one dose of the current season's updated COVID-19 vaccine for all adults.",
        schedule="≥ 1 dose of the most recent formulation per season; additional doses for immunocompromised.",
        age_min=18,
    ),
    VaccineRule(
        rule_id="acip-hpv",
        name="HPV vaccine",
        indication=ROUTINE,
        rationale="Routine vaccination through age 26; shared clinical decision-making for some adults 27–45.",
        schedule="2 or 3 doses depending on age at initiation and immune status.",
        age_min=9,
        age_max=26,
    ),
    VaccineRule(
        rule_id="acip-hpv-sdm",
        name="HPV vaccine (shared decision-making, 27–45)",
        indication=SHARED_DECISION,
        rationale="Adults 27–45 may benefit from HPV vaccination based on individual risk discussion.",
        schedule="3 doses if initiating in this age range.",
        age_min=27,
        age_max=45,
    ),
    VaccineRule(
        rule_id="acip-hepb-19-59",
        name="Hepatitis B (universal, 19–59)",
        indication=ROUTINE,
        rationale="Universal Hep B vaccination for all adults 19–59; risk-based for ≥ 60.",
        schedule="2- or 3-dose series depending on product.",
        age_min=19,
        age_max=59,
    ),
    VaccineRule(
        rule_id="acip-hepb-60-risk",
        name="Hepatitis B (risk-based, ≥ 60)",
        indication=RISK_BASED,
        rationale="Adults ≥ 60 with chronic liver disease, diabetes, HIV, hemodialysis, or other risk factors.",
        schedule="2- or 3-dose series depending on product.",
        age_min=60,
        condition_keywords=(
            "diabetes", "cirrhosis", "hcv", "hepatitis c", "liver", "hiv",
            "dialysis", "ckd stage 5",
        ),
    ),
    VaccineRule(
        rule_id="acip-hepa-risk",
        name="Hepatitis A (risk-based)",
        indication=RISK_BASED,
        rationale="Recommended for adults with chronic liver disease, HIV, MSM, IDU, travel to endemic regions, homelessness.",
        schedule="2-dose series, 6 months apart.",
        age_min=19,
        condition_keywords=("cirrhosis", "hcv", "hepatitis c", "liver", "hiv"),
    ),
    VaccineRule(
        rule_id="acip-mmr-catchup",
        name="MMR (measles, mumps, rubella) catch-up",
        indication=RISK_BASED,
        rationale="Adults born in 1957 or later without immunity evidence should have ≥ 1 MMR dose; healthcare workers and students need 2.",
        schedule="1–2 doses depending on risk group.",
        age_min=19,
        pregnancy="contraindicated",
    ),
    VaccineRule(
        rule_id="acip-varicella-catchup",
        name="Varicella catch-up",
        indication=RISK_BASED,
        rationale="Adults without immunity evidence (especially < 50) should be vaccinated.",
        schedule="2 doses, 4–8 weeks apart.",
        age_min=19,
        age_max=49,
        pregnancy="contraindicated",
    ),
    VaccineRule(
        rule_id="acip-meningococcal-risk",
        name="Meningococcal (MenACWY ± MenB)",
        indication=RISK_BASED,
        rationale="Recommended for adults with asplenia, complement deficiency, HIV, microbiologists, military, hajj travelers, college freshmen in dorms.",
        schedule="MenACWY series; MenB shared decision-making for healthy adolescents.",
        age_min=16,
        condition_keywords=("asplenia", "sickle", "complement", "hiv", "transplant"),
    ),
)


def _matches_age(rule: VaccineRule, age: int) -> bool:
    """Compare against rule.age_min in years (treat months-style values < 18 as years too)."""
    return rule.age_min <= age <= rule.age_max


def _matches_conditions(rule: VaccineRule, conditions_blob: str) -> bool:
    if not rule.condition_keywords:
        return True
    if not conditions_blob:
        return False
    return any(kw in conditions_blob for kw in rule.condition_keywords)


def recommendations_for(
    *,
    age: int,
    conditions: list[str] | None = None,
    pregnant: bool = False,
) -> list[VaccineRule]:
    """Return vaccine rules whose indications match the patient context.

    A risk-based rule with ``condition_keywords`` is only returned when at
    least one keyword appears in the supplied conditions. Pregnancy gates
    are honored (contraindicated rules drop out; ``preferred`` rules require
    ``pregnant=True``).
    """
    blob = " | ".join((c or "").lower() for c in (conditions or []))
    out: list[VaccineRule] = []
    for rule in RULES:
        if not _matches_age(rule, age):
            continue
        if rule.pregnancy == "contraindicated" and pregnant:
            continue
        if rule.pregnancy == "preferred" and not pregnant:
            continue
        if rule.indication in (RISK_BASED, SHARED_DECISION) and rule.condition_keywords:
            if not _matches_conditions(rule, blob):
                continue
        out.append(rule)
    return out
