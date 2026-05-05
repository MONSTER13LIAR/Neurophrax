"""
Neurophrax MCP Server
Synthetic healthcare data only — not for clinical use.
"""

from typing import Annotated
from pydantic import Field
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Neurophrax",
    instructions="Synthetic healthcare assistant. All data is fictional and for educational/demo purposes only.",
)

# ── Synthetic Data ─────────────────────────────────────────────────────────────

DRUG_INTERACTIONS: dict[frozenset, dict] = {
    frozenset(["warfarin", "aspirin"]): {
        "severity": "DANGEROUS",
        "effect": "Concurrent use greatly increases hemorrhage risk — aspirin inhibits platelet aggregation",
        "recommendation": "Avoid combination. Use acetaminophen for pain if anticoagulation is needed.",
    },
    frozenset(["warfarin", "ibuprofen"]): {
        "severity": "DANGEROUS",
        "effect": "NSAIDs displace warfarin from plasma proteins, raising free-drug levels and bleeding risk",
        "recommendation": "Use acetaminophen instead. Monitor INR closely if NSAID unavoidable.",
    },
    frozenset(["metformin", "alcohol"]): {
        "severity": "DANGEROUS",
        "effect": "Alcohol inhibits gluconeogenesis and raises lactic acid — lactic acidosis risk is potentially fatal",
        "recommendation": "Avoid all alcohol while on metformin.",
    },
    frozenset(["lisinopril", "potassium"]): {
        "severity": "HIGH",
        "effect": "ACE inhibitors reduce potassium excretion; supplements can cause life-threatening hyperkalemia",
        "recommendation": "Avoid potassium supplements unless prescribed. Monitor serum potassium regularly.",
    },
    frozenset(["sertraline", "tramadol"]): {
        "severity": "DANGEROUS",
        "effect": "Both increase serotonergic activity — serotonin syndrome risk (hyperthermia, seizures, death)",
        "recommendation": "Do not combine. Use a non-serotonergic analgesic.",
    },
    frozenset(["simvastatin", "amiodarone"]): {
        "severity": "HIGH",
        "effect": "Amiodarone inhibits CYP3A4, raising simvastatin plasma levels — myopathy and rhabdomyolysis risk",
        "recommendation": "Lower simvastatin dose (max 20 mg/day) or switch to a non-CYP3A4-metabolised statin.",
    },
    frozenset(["ciprofloxacin", "antacids"]): {
        "severity": "MODERATE",
        "effect": "Divalent cations in antacids chelate ciprofloxacin, reducing absorption by up to 90%",
        "recommendation": "Take ciprofloxacin 2 h before or 6 h after antacids.",
    },
    frozenset(["atorvastatin", "clarithromycin"]): {
        "severity": "HIGH",
        "effect": "CYP3A4 inhibition by clarithromycin raises atorvastatin AUC — increased myopathy risk",
        "recommendation": "Withhold atorvastatin for the duration of antibiotic course.",
    },
    frozenset(["clopidogrel", "omeprazole"]): {
        "severity": "MODERATE",
        "effect": "Omeprazole inhibits CYP2C19, reducing clopidogrel conversion to active metabolite",
        "recommendation": "Use pantoprazole as a safer gastroprotective alternative.",
    },
    frozenset(["lithium", "ibuprofen"]): {
        "severity": "HIGH",
        "effect": "NSAIDs reduce renal lithium clearance, causing toxic lithium accumulation",
        "recommendation": "Use acetaminophen. Monitor lithium levels if NSAID is unavoidable.",
    },
    frozenset(["sildenafil", "nitrates"]): {
        "severity": "DANGEROUS",
        "effect": "Synergistic vasodilation causes severe, potentially fatal hypotension",
        "recommendation": "Absolute contraindication — never combine.",
    },
}

SYMPTOMS_TO_CONDITIONS: dict[str, list[str]] = {
    "fever": ["Influenza", "COVID-19", "Bacterial Infection", "Sepsis (if high-grade ⚠️)"],
    "cough": ["Common Cold", "Influenza", "COVID-19", "Bronchitis", "Pneumonia"],
    "fatigue": ["Anemia", "Hypothyroidism", "Type 2 Diabetes", "Depression", "Chronic Fatigue Syndrome"],
    "chest pain": ["Angina Pectoris", "Myocardial Infarction ⚠️", "Costochondritis", "GERD", "Panic Disorder"],
    "shortness of breath": ["Asthma", "COPD", "Heart Failure", "Pulmonary Embolism ⚠️", "Anemia"],
    "headache": ["Tension Headache", "Migraine", "Hypertension", "Dehydration", "Meningitis (if severe ⚠️)"],
    "nausea": ["Gastroenteritis", "Food Poisoning", "Pregnancy", "Medication Side Effect", "Appendicitis ⚠️"],
    "dizziness": ["Benign Positional Vertigo", "Orthostatic Hypotension", "Anemia", "Inner Ear Infection"],
    "joint pain": ["Osteoarthritis", "Rheumatoid Arthritis", "Gout", "Lupus", "Lyme Disease"],
    "rash": ["Allergic Reaction", "Eczema", "Psoriasis", "Contact Dermatitis", "Drug Reaction"],
    "abdominal pain": ["IBS", "Appendicitis ⚠️", "Peptic Ulcer", "Gallstones", "Crohn's Disease"],
    "frequent urination": ["Type 2 Diabetes", "UTI", "Overactive Bladder", "Prostate Hyperplasia"],
    "weight loss": ["Hyperthyroidism", "Type 1 Diabetes", "Malignancy ⚠️", "Depression", "Malabsorption"],
    "palpitations": ["Atrial Fibrillation ⚠️", "Anxiety", "Hyperthyroidism", "Anemia", "Excessive Caffeine"],
    "blurred vision": ["Diabetes (retinopathy)", "Hypertension", "Glaucoma", "Cataracts", "Stroke ⚠️"],
    "swelling": ["Heart Failure", "DVT ⚠️", "Kidney Disease", "Venous Insufficiency", "Medication Side Effect"],
}

MEDICATIONS: dict[str, dict] = {
    "metformin": {
        "generic": "Metformin HCl",
        "drug_class": "Biguanide — Antidiabetic",
        "dosage": "500–1000 mg twice daily (max 2550 mg/day)",
        "timing": "Take with meals to minimise GI side effects",
        "reminders": ["After breakfast (08:00)", "After dinner (18:00)"],
        "side_effects": ["Nausea", "Diarrhoea", "Stomach upset", "Lactic acidosis (rare)"],
        "monitoring": "Fasting glucose & HbA1c every 3 months; renal function annually",
        "storage": "Room temperature, away from moisture",
    },
    "lisinopril": {
        "generic": "Lisinopril",
        "drug_class": "ACE Inhibitor — Antihypertensive",
        "dosage": "10–40 mg once daily",
        "timing": "Same time each day; morning preferred",
        "reminders": ["Morning with or without food (08:00)"],
        "side_effects": ["Dry cough", "Dizziness on standing", "Hyperkalemia"],
        "monitoring": "Blood pressure daily; renal function & serum potassium at 1–2 weeks and periodically",
        "storage": "Room temperature",
    },
    "atorvastatin": {
        "generic": "Atorvastatin calcium",
        "drug_class": "HMG-CoA Reductase Inhibitor — Statin",
        "dosage": "10–80 mg once daily",
        "timing": "Any time of day; evening slightly preferred",
        "reminders": ["Evening (21:00)"],
        "side_effects": ["Myalgia", "Elevated liver enzymes", "GI upset"],
        "monitoring": "Fasting lipid panel at 6 weeks, then every 3–6 months; CK if muscle symptoms arise",
        "storage": "Room temperature",
    },
    "warfarin": {
        "generic": "Warfarin sodium",
        "drug_class": "Vitamin K Antagonist — Anticoagulant",
        "dosage": "Individualised by INR target (typical 2–10 mg/day)",
        "timing": "Same time every evening",
        "reminders": ["Evening (18:00) — CRITICAL: never skip or double dose"],
        "side_effects": ["Bruising", "Prolonged bleeding", "Hair thinning"],
        "monitoring": "INR every 1–4 weeks; maintain consistent Vitamin K dietary intake",
        "storage": "Room temperature, protect from light",
    },
    "sertraline": {
        "generic": "Sertraline HCl",
        "drug_class": "SSRI — Antidepressant / Anxiolytic",
        "dosage": "50–200 mg once daily",
        "timing": "Morning or evening; consistency matters more than time of day",
        "reminders": ["Morning with breakfast (08:00)"],
        "side_effects": ["Nausea (first 1–2 weeks)", "Insomnia", "Reduced libido"],
        "monitoring": "Assess mood and suicidality at 2 and 4 weeks; therapeutic effect in 4–6 weeks",
        "storage": "Room temperature",
    },
    "amoxicillin": {
        "generic": "Amoxicillin trihydrate",
        "drug_class": "Penicillin Antibiotic",
        "dosage": "250–500 mg every 8 h  OR  500–875 mg every 12 h",
        "timing": "With or without food; complete the full prescribed course",
        "reminders": ["08:00", "16:00", "00:00  (if 8-hourly dosing)"],
        "side_effects": ["Diarrhoea", "Rash", "Nausea"],
        "monitoring": "Report urticaria or throat swelling immediately — may indicate allergy",
        "storage": "Room temperature (tablets/capsules); refrigerate oral suspension",
    },
    "omeprazole": {
        "generic": "Omeprazole",
        "drug_class": "Proton Pump Inhibitor — Antacid",
        "dosage": "20–40 mg once daily",
        "timing": "30–60 minutes before breakfast for maximum effect",
        "reminders": ["Before breakfast (07:30)"],
        "side_effects": ["Headache", "Diarrhoea", "Vitamin B12 deficiency (long-term)"],
        "monitoring": "Long-term use: check magnesium, B12, and bone density annually",
        "storage": "Room temperature",
    },
    "ibuprofen": {
        "generic": "Ibuprofen",
        "drug_class": "NSAID — Analgesic / Anti-inflammatory",
        "dosage": "200–800 mg every 6–8 h (max 3200 mg/day)",
        "timing": "Always take with food or milk to reduce gastric irritation",
        "reminders": ["With meals as needed — do not exceed recommended daily dose"],
        "side_effects": ["GI irritation / ulceration", "Fluid retention", "Raised blood pressure"],
        "monitoring": "Renal function with prolonged use; avoid in heart failure or CKD",
        "storage": "Room temperature",
    },
    "amlodipine": {
        "generic": "Amlodipine besylate",
        "drug_class": "Calcium Channel Blocker — Antihypertensive",
        "dosage": "5–10 mg once daily",
        "timing": "Same time each day; morning or evening",
        "reminders": ["Morning (08:00)"],
        "side_effects": ["Peripheral oedema", "Flushing", "Headache"],
        "monitoring": "Blood pressure and heart rate at each visit",
        "storage": "Room temperature",
    },
    "levothyroxine": {
        "generic": "Levothyroxine sodium",
        "drug_class": "Thyroid Hormone Replacement",
        "dosage": "25–200 mcg once daily (dose titrated by TSH)",
        "timing": "30–60 minutes before breakfast on an empty stomach — calcium/iron supplements reduce absorption",
        "reminders": ["Before breakfast on an empty stomach (07:00)"],
        "side_effects": ["Palpitations (if over-replaced)", "Insomnia", "Weight loss"],
        "monitoring": "TSH every 6–8 weeks until stable; then annually",
        "storage": "Room temperature, protect from light and moisture",
    },
}

# ── Tool 1: Drug Interactions ──────────────────────────────────────────────────

@mcp.tool()
def check_drug_interactions(
    drugs: Annotated[
        list[str],
        Field(min_length=2, description="Two or more drug names to check for interactions"),
    ]
) -> str:
    """
    Check for dangerous or clinically significant interactions between two or more medications.
    Uses synthetic reference data only — not for real clinical decisions.
    """
    normalised = [d.lower().strip() for d in drugs]
    interactions_found: list[str] = []

    for i in range(len(normalised)):
        for j in range(i + 1, len(normalised)):
            pair = frozenset([normalised[i], normalised[j]])
            if pair in DRUG_INTERACTIONS:
                info = DRUG_INTERACTIONS[pair]
                interactions_found.append(
                    f"  ⚠️  {drugs[i].title()} + {drugs[j].title()}\n"
                    f"     Severity     : {info['severity']}\n"
                    f"     Effect       : {info['effect']}\n"
                    f"     Recommend    : {info['recommendation']}"
                )

    pair_count = len(normalised) * (len(normalised) - 1) // 2
    header = [
        "╔══════════════════════════════════════════════════╗",
        "║        NEUROPHRAX — DRUG INTERACTION CHECK       ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Drugs    : {', '.join(d.title() for d in drugs)}",
        f"  Pairs    : {pair_count} checked",
        "",
    ]

    if interactions_found:
        body = [f"  🚨 {len(interactions_found)} INTERACTION(S) DETECTED:", ""] + interactions_found
    else:
        body = [
            "  ✅ No known interactions found in the synthetic database.",
            "     Always verify with a licensed pharmacist or prescriber.",
        ]

    footer = [
        "",
        "  ⚕️  DISCLAIMER: Synthetic/educational data only. Not for clinical use.",
    ]
    return "\n".join(header + body + footer)


# ── Tool 2: Patient Summary ────────────────────────────────────────────────────

@mcp.tool()
def generate_patient_summary(
    name: Annotated[str, Field(description="Patient's full name")],
    age: Annotated[int, Field(ge=0, le=130, description="Patient age in years")],
    conditions: Annotated[list[str], Field(description="Active diagnosed medical conditions")],
    medications: Annotated[list[str], Field(description="Current medications")],
    allergies: Annotated[list[str], Field(description="Known drug/food allergies")],
    last_visit: Annotated[str, Field(description="Date of last clinical visit (YYYY-MM-DD)")] = "Not recorded",
) -> str:
    """
    Generate a structured patient summary from provided synthetic patient data.
    Risk level is automatically assessed based on age, conditions, and medication count.
    """
    # Simple synthetic risk scoring
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

    return "\n".join([
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
        "",
        "  ⚕️  DISCLAIMER: Synthetic/educational data only. Not for clinical use.",
    ])


# ── Tool 3: Symptoms → Conditions ─────────────────────────────────────────────

@mcp.tool()
def map_symptoms_to_conditions(
    symptoms: Annotated[
        list[str],
        Field(min_length=1, description="Symptoms the patient is experiencing"),
    ]
) -> str:
    """
    Map reported symptoms to possible medical conditions using a synthetic reference database.
    Conditions are ranked by how many symptoms point to them.
    """
    matched: dict[str, list[str]] = {}
    condition_votes: dict[str, int] = {}

    for symptom in symptoms:
        s = symptom.lower().strip()
        for db_key, conditions in SYMPTOMS_TO_CONDITIONS.items():
            if s == db_key or db_key in s or s in db_key:
                matched[symptom] = conditions
                for c in conditions:
                    condition_votes[c] = condition_votes.get(c, 0) + 1
                break

    unrecognised = [s for s in symptoms if s not in matched]
    ranked = sorted(condition_votes.items(), key=lambda x: x[1], reverse=True)

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║      NEUROPHRAX — SYMPTOM-TO-CONDITION MAP       ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Symptoms : {', '.join(symptoms)}",
        "",
    ]

    if ranked:
        lines += ["  POSSIBLE CONDITIONS (ranked by symptom overlap):", ""]
        for condition, votes in ranked[:10]:
            tag = "⚠️ " if "⚠️" in condition else "   "
            lines.append(f"  {tag}{condition:<45}  [{votes} symptom(s)]")
        lines.append("")
        lines += ["  SYMPTOM BREAKDOWN:", ""]
        for orig_symptom, conds in matched.items():
            lines.append(f"    • {orig_symptom.title()}")
            for c in conds[:4]:
                lines.append(f"        → {c}")
    else:
        lines.append("  No recognised symptoms found in the synthetic database.")

    if unrecognised:
        lines += ["", f"  Unrecognised: {', '.join(unrecognised)}"]

    lines += [
        "",
        "  ⚠️  Items marked ⚠️ may indicate emergencies — seek immediate care.",
        "  ⚕️  DISCLAIMER: Synthetic/educational data only. Not for clinical use.",
    ]
    return "\n".join(lines)


# ── Tool 4: Medication Advice ──────────────────────────────────────────────────

@mcp.tool()
def get_medication_advice(
    medication: Annotated[str, Field(description="Name of the medication to look up")]
) -> str:
    """
    Return dosage, timing, reminder schedule, side effects, and monitoring info
    for a given medication using synthetic reference data.
    """
    key = medication.lower().strip()
    info = MEDICATIONS.get(key)

    if not info:
        available = ", ".join(m.title() for m in sorted(MEDICATIONS))
        return "\n".join([
            f"  ❌ '{medication}' not found in the synthetic database.",
            "",
            f"  Available medications: {available}",
            "",
            "  ⚕️  DISCLAIMER: Synthetic/educational data only. Not for clinical use.",
        ])

    reminders_fmt = "\n     ".join(f"🔔 {r}" for r in info["reminders"])
    side_effects_fmt = " | ".join(info["side_effects"])

    return "\n".join([
        "╔══════════════════════════════════════════════════╗",
        "║         NEUROPHRAX — MEDICATION ADVICE           ║",
        "╚══════════════════════════════════════════════════╝",
        f"  Medication   : {medication.title()}",
        f"  Generic      : {info['generic']}",
        f"  Class        : {info['drug_class']}",
        "",
        f"  DOSAGE       : {info['dosage']}",
        f"  TIMING       : {info['timing']}",
        "",
        "  REMINDERS    :",
        f"     {reminders_fmt}",
        "",
        f"  SIDE EFFECTS : {side_effects_fmt}",
        f"  MONITORING   : {info['monitoring']}",
        f"  STORAGE      : {info['storage']}",
        "",
        "  ⚕️  DISCLAIMER: Synthetic/educational data only. Not for clinical use.",
    ])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
