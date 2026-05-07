"""Beers Criteria — potentially inappropriate medications in older adults.

A curated, attribution-only subset of the **2023 AGS Beers Criteria®** — the
American Geriatrics Society's expert-panel guidance on medications that are
generally best avoided in adults aged ≥ 65. This module bundles a clinically
representative selection of *categories* (not the full table) so the safety-
review tool can flag risk patterns offline and instantly. The full criteria
remain the authoritative reference — see attribution below.

Matching strategy:
- ``ingredient`` names are matched against the lower-cased ingredient name
  RxNorm returns for each medication (ingredient TTY = "IN").
- ``atc_prefixes`` match against the RxClass ATC ``classId`` (any prefix).
  This is how class-wide rules (e.g. "first-generation antihistamines") cover
  every member without enumerating brands.

Attribution (do not remove):
    The 2023 American Geriatrics Society Beers Criteria® for Potentially
    Inappropriate Medication Use in Older Adults. J Am Geriatr Soc. 2023.
    The selection below paraphrases categories from the published criteria
    and is intended for software flagging only; it is not a substitute for
    the full criteria, which clinicians should consult directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Severity vocabulary used in returned flags.
AVOID = "AVOID"
CAUTION = "USE_WITH_CAUTION"
AVOID_CHRONIC = "AVOID_CHRONIC_USE"
DOSE_ADJUST = "DOSE_ADJUST"


@dataclass(frozen=True, slots=True)
class BeersRule:
    rule_id: str
    title: str
    severity: str
    rationale: str
    recommendation: str
    ingredients: tuple[str, ...] = field(default_factory=tuple)
    atc_prefixes: tuple[str, ...] = field(default_factory=tuple)
    age_threshold: int = 65


# Curated 2023 Beers categories. Names are RxNorm ingredient names (lowercase).
# ATC prefixes use WHO ATC classification codes.
RULES: tuple[BeersRule, ...] = (
    BeersRule(
        rule_id="beers-antihist-1g",
        title="First-generation antihistamines",
        severity=AVOID,
        rationale=(
            "Strong anticholinergic burden; associated with confusion, dry mouth, "
            "constipation, urinary retention, and falls in older adults."
        ),
        recommendation=(
            "Prefer second-generation antihistamines (loratadine, cetirizine, "
            "fexofenadine) for allergy indications; non-pharmacologic measures for sleep."
        ),
        ingredients=(
            "diphenhydramine", "hydroxyzine", "chlorpheniramine", "brompheniramine",
            "doxylamine", "promethazine", "cyproheptadine", "meclizine", "dimenhydrinate",
        ),
        atc_prefixes=("R06AA", "R06AB", "R06AD", "R06AX"),
    ),
    BeersRule(
        rule_id="beers-benzo",
        title="Benzodiazepines",
        severity=AVOID,
        rationale=(
            "Older adults have increased sensitivity and decreased metabolism. "
            "Strongly associated with cognitive impairment, delirium, falls, fractures, and motor-vehicle crashes."
        ),
        recommendation=(
            "Avoid for insomnia, agitation, and delirium. Reserve for severe "
            "generalized anxiety, alcohol withdrawal, or end-of-life care."
        ),
        ingredients=(
            "alprazolam", "lorazepam", "diazepam", "clonazepam", "temazepam",
            "oxazepam", "triazolam", "chlordiazepoxide", "clorazepate", "estazolam",
        ),
        atc_prefixes=("N05BA", "N05CD"),
    ),
    BeersRule(
        rule_id="beers-zdrug",
        title="Non-benzodiazepine hypnotics (Z-drugs)",
        severity=AVOID_CHRONIC,
        rationale=(
            "Similar adverse effects to benzodiazepines: delirium, falls, fractures, "
            "motor-vehicle crashes; minimal sleep-quality benefit beyond a few weeks."
        ),
        recommendation="Avoid chronic use (>90 days). Prefer cognitive-behavioral therapy for insomnia.",
        ingredients=("zolpidem", "zaleplon", "eszopiclone"),
    ),
    BeersRule(
        rule_id="beers-tca",
        title="Tricyclic antidepressants (highly anticholinergic)",
        severity=AVOID,
        rationale=(
            "Strong anticholinergic and sedative effects; orthostatic hypotension; "
            "associated with delirium, falls, and cardiac conduction abnormalities."
        ),
        recommendation=(
            "Prefer SSRIs/SNRIs for depression. For neuropathic pain, consider "
            "duloxetine, gabapentin, or pregabalin (with renal dose adjustment)."
        ),
        ingredients=("amitriptyline", "imipramine", "clomipramine", "doxepin", "trimipramine"),
        atc_prefixes=("N06AA",),
    ),
    BeersRule(
        rule_id="beers-skeletal-muscle-relax",
        title="Skeletal muscle relaxants",
        severity=AVOID,
        rationale=(
            "Anticholinergic, sedating, and associated with fractures; effectiveness at "
            "tolerated doses in older adults is questionable."
        ),
        recommendation="Prefer physical therapy, topical analgesics, or scheduled acetaminophen.",
        ingredients=(
            "cyclobenzaprine", "carisoprodol", "chlorzoxazone", "metaxalone",
            "methocarbamol", "orphenadrine",
        ),
    ),
    BeersRule(
        rule_id="beers-sulfonylurea-long",
        title="Long-acting sulfonylureas",
        severity=AVOID,
        rationale="Prolonged hypoglycemia in older adults; increased risk of severe events.",
        recommendation="Prefer short-acting sulfonylureas (glipizide) or non-sulfonylurea agents.",
        ingredients=("glyburide", "glibenclamide", "chlorpropamide", "glimepiride"),
    ),
    BeersRule(
        rule_id="beers-nsaid-chronic",
        title="Non-COX-selective NSAIDs (chronic use)",
        severity=AVOID_CHRONIC,
        rationale=(
            "Increased risk of GI bleeding and peptic ulcer disease in adults ≥ 75 or on "
            "anticoagulants/antiplatelets/corticosteroids; can also worsen heart failure and CKD."
        ),
        recommendation=(
            "Avoid chronic use unless other alternatives are not effective and patient can take a "
            "gastroprotective agent (PPI/misoprostol). Topical NSAIDs may be acceptable."
        ),
        ingredients=(
            "ibuprofen", "naproxen", "diclofenac", "indomethacin", "ketorolac",
            "meloxicam", "piroxicam", "nabumetone", "etodolac", "sulindac",
        ),
        atc_prefixes=("M01A",),
    ),
    BeersRule(
        rule_id="beers-aspirin-primary",
        title="Aspirin for primary cardiovascular prevention",
        severity=AVOID,
        rationale="Net benefit is unclear and bleeding risk increases substantially with age.",
        recommendation=(
            "Do not initiate for primary prevention in adults ≥ 70. Continue only after shared "
            "decision-making in patients already tolerating it for established CVD."
        ),
        ingredients=("aspirin", "acetylsalicylic acid"),
        age_threshold=70,
    ),
    BeersRule(
        rule_id="beers-nitrofurantoin-renal",
        title="Nitrofurantoin (with reduced kidney function)",
        severity=DOSE_ADJUST,
        rationale="Avoid in CrCl < 30 mL/min — pulmonary toxicity and lack of efficacy at low concentrations.",
        recommendation="Confirm CrCl before prescribing; pick an alternative agent if CrCl < 30.",
        ingredients=("nitrofurantoin",),
    ),
    BeersRule(
        rule_id="beers-digoxin",
        title="Digoxin as first-line for atrial fibrillation or heart failure",
        severity=CAUTION,
        rationale=(
            "Higher serum levels (especially > 0.125 mg/day) associated with toxicity; "
            "decreased renal clearance with age."
        ),
        recommendation=(
            "Avoid as first-line for AF or HFrEF. If used, keep daily dose ≤ 0.125 mg and "
            "monitor levels and renal function."
        ),
        ingredients=("digoxin",),
    ),
    BeersRule(
        rule_id="beers-alpha1-blocker-htn",
        title="Peripheral α1-blockers for hypertension",
        severity=AVOID,
        rationale="High risk of orthostatic hypotension; associated with falls.",
        recommendation=(
            "Do not use as antihypertensive monotherapy. Acceptable for benign prostatic hyperplasia "
            "with BP monitoring."
        ),
        ingredients=("doxazosin", "prazosin", "terazosin"),
    ),
    BeersRule(
        rule_id="beers-central-alpha2",
        title="Central α-agonists as first-line antihypertensives",
        severity=AVOID,
        rationale="High risk of CNS adverse effects, bradycardia, and orthostatic hypotension.",
        recommendation="Avoid clonidine as first-line; avoid methyldopa, guanfacine, and guanabenz.",
        ingredients=("clonidine", "methyldopa", "guanfacine", "guanabenz"),
    ),
    BeersRule(
        rule_id="beers-antiarrhythmic-af",
        title="Class Ia/Ic/III antiarrhythmics for AF (first-line)",
        severity=CAUTION,
        rationale=(
            "Rate control generally preferred over rhythm control; agents have significant "
            "toxicity (pulmonary, hepatic, thyroid, QT prolongation)."
        ),
        recommendation="Reserve amiodarone, dronedarone, flecainide, propafenone for specialist-directed care.",
        ingredients=("amiodarone", "dronedarone", "flecainide", "propafenone", "disopyramide", "quinidine", "sotalol"),
    ),
    BeersRule(
        rule_id="beers-ppi-chronic",
        title="Proton-pump inhibitors (>8 weeks without indication)",
        severity=AVOID_CHRONIC,
        rationale=(
            "Long-term PPI use associated with C. difficile infection, pneumonia, bone loss/fractures, "
            "and B12 deficiency."
        ),
        recommendation=(
            "Limit to 8 weeks unless ongoing indication (Barrett's, chronic NSAID, severe esophagitis, "
            "pathologic hypersecretion). Re-evaluate need at each visit."
        ),
        ingredients=("omeprazole", "esomeprazole", "lansoprazole", "pantoprazole", "rabeprazole", "dexlansoprazole"),
        atc_prefixes=("A02BC",),
    ),
    BeersRule(
        rule_id="beers-antipsychotic-dementia",
        title="Antipsychotics for behavioral symptoms of dementia",
        severity=CAUTION,
        rationale=(
            "FDA boxed warning: increased risk of stroke and mortality in older adults with dementia. "
            "Modest efficacy for behavioral symptoms."
        ),
        recommendation=(
            "Prefer non-pharmacologic interventions. Reserve antipsychotics for severe symptoms with "
            "significant harm risk; reassess and taper periodically."
        ),
        ingredients=(
            "haloperidol", "risperidone", "olanzapine", "quetiapine", "aripiprazole",
            "ziprasidone", "chlorpromazine", "fluphenazine", "perphenazine", "thioridazine",
        ),
        atc_prefixes=("N05A",),
    ),
    BeersRule(
        rule_id="beers-estrogen",
        title="Estrogens (oral or transdermal) without progestin",
        severity=AVOID,
        rationale="Carcinogenic potential (breast, endometrial); lack of cardioprotective effect.",
        recommendation="Vaginal estrogen at low doses for genitourinary symptoms is acceptable.",
        ingredients=("estradiol", "estrone", "conjugated estrogens", "esterified estrogens", "ethinyl estradiol"),
    ),
    BeersRule(
        rule_id="beers-megestrol",
        title="Megestrol for appetite stimulation",
        severity=AVOID,
        rationale="Minimal effect on weight; increased risk of thrombosis and possibly mortality.",
        recommendation="Address reversible causes of poor intake; avoid megestrol.",
        ingredients=("megestrol",),
    ),
    BeersRule(
        rule_id="beers-meperidine",
        title="Meperidine",
        severity=AVOID,
        rationale="Not effective at common oral doses; risk of neurotoxicity (delirium) in older adults.",
        recommendation="Use alternative opioid analgesics (avoid in renal impairment).",
        ingredients=("meperidine", "pethidine"),
    ),
    BeersRule(
        rule_id="beers-metoclopramide",
        title="Metoclopramide (chronic)",
        severity=AVOID_CHRONIC,
        rationale="Extrapyramidal effects, including tardive dyskinesia; risk increases with duration and age.",
        recommendation="Limit to <12 weeks unless gastroparesis warrants ongoing use with informed consent.",
        ingredients=("metoclopramide",),
    ),
    BeersRule(
        rule_id="beers-desmopressin-nocturia",
        title="Desmopressin for nocturia or nocturnal polyuria",
        severity=AVOID,
        rationale="High risk of hyponatremia; safer alternatives available.",
        recommendation="Use behavioral/lifestyle strategies and treat underlying causes.",
        ingredients=("desmopressin",),
    ),
    BeersRule(
        rule_id="beers-sliding-scale-insulin",
        title="Sliding-scale insulin without basal coverage",
        severity=CAUTION,
        rationale="Higher risk of hypoglycemia without improving hyperglycemia control.",
        recommendation="Use basal or basal-bolus regimens. Sliding-scale alone should not be the regimen.",
        ingredients=(),  # captured at workflow level; included for documentation
        atc_prefixes=("A10A",),
    ),
)


_SALT_SUFFIXES = (
    "hydrochloride", "hcl", "sulfate", "sulphate", "sodium", "potassium",
    "citrate", "tartrate", "succinate", "fumarate", "maleate", "malate",
    "phosphate", "acetate", "besylate", "mesylate", "tosylate", "bromide",
    "chloride", "calcium", "magnesium", "monohydrate", "dihydrate", "anhydrous",
)


def _normalize_ingredient(name: str) -> set[str]:
    """Return candidate ingredient tokens from a label string.

    Handles common label forms like ``"diphenhydramine hcl"`` →
    ``{"diphenhydramine", "diphenhydramine hcl"}`` so a Beers rule keyed on
    the base name still matches.
    """
    base = name.strip().lower()
    if not base:
        return set()
    tokens: set[str] = {base}
    parts = base.split()
    if len(parts) > 1 and parts[-1] in _SALT_SUFFIXES:
        tokens.add(" ".join(parts[:-1]))
    if parts:
        tokens.add(parts[0])
    return tokens


def flags_for(
    *,
    age: int,
    ingredient_name: str | None = None,
    atc_class_ids: tuple[str, ...] = (),
) -> list[BeersRule]:
    """Return Beers rules that match the given (age, ingredient, ATC classes).

    Returns an empty list when ``age`` is below every rule's threshold. Matching
    is the union of an ingredient-name hit OR any ATC-prefix hit. Ingredient
    matching tolerates common salt forms (e.g. ``"diphenhydramine hcl"``).
    """
    if age <= 0:
        return []
    candidates = _normalize_ingredient(ingredient_name or "")
    matches: list[BeersRule] = []
    for rule in RULES:
        if age < rule.age_threshold:
            continue
        if candidates and any(c in rule.ingredients for c in candidates):
            matches.append(rule)
            continue
        if rule.atc_prefixes and any(
            cid.startswith(prefix) for cid in atc_class_ids for prefix in rule.atc_prefixes
        ):
            matches.append(rule)
    return matches


ATTRIBUTION = (
    "Categories adapted from the American Geriatrics Society 2023 Beers Criteria® "
    "for Potentially Inappropriate Medication Use in Older Adults (J Am Geriatr Soc. 2023). "
    "Consult the full criteria for clinical decisions."
)

SOURCE_URL = "https://geriatricscareonline.org/ProductAbstract/2023-american-geriatrics-society-beers-criteria/CL001"
