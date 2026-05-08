# Neurophrax

A reliable medical-data layer that any healthcare AI agent can plug into. Eight
MCP tools that compose authoritative public sources (NIH RxNorm, FDA OpenFDA,
NLM Clinical Tables, RxClass, MedlinePlus, DailyMed, ClinicalTrials.gov,
PubMed) with curated clinical rule packs (DDInter 2.0, AGS Beers Criteria
2023, CDC ACIP) into structured, source-attributed responses suitable for
agent reasoning.

> Educational use only. Not for clinical decisions.

## What's different

Most healthcare MCPs are thin wrappers around a single API. Neurophrax leans
into composition — the centerpiece tool, `medication_safety_review`,
cross-references four independent risk signals (pairwise interactions,
geriatric prescribing rules, duplicate therapeutic class, FDA boxed warnings)
against patient context to produce a single prioritized risk list. Every
tool returns both a human-readable summary and a structured `data` payload
plus explicit source attribution, and four of the eight emit FHIR R4
resources (`RiskAssessment`, `ImmunizationRecommendation`) ready to persist
back into a chart.

## Tools

| Tool | What it does |
|---|---|
| `drug_interaction_check` | All-pair check across DDInter 2.0 with OpenFDA-label fallback for missing pairs. |
| `medication_profile` | Synthesizes RxNorm + OpenFDA + DailyMed + RxClass + MedlinePlus into one drug brief (boxed warnings, classes, patient explanation, dosage). |
| `symptom_assessment` | Maps symptoms to FHIR-curated conditions and ICD-10-CM diagnosis codes via NLM Clinical Tables. |
| `patient_summary` | Validates conditions against the FHIR Conditions Value Set, resolves medications to RxCUIs, and computes a heuristic risk level. |
| `medication_safety_review` | **Composer.** Cross-references interactions + Beers Criteria + duplicate ATC class + FDA boxed warnings into a prioritized risk list. Emits a FHIR `RiskAssessment`. |
| `clinical_trial_search` | Matches the patient (or a free-text condition) to actively recruiting trials on ClinicalTrials.gov v2. |
| `clinical_evidence_search` | Composes PubMed E-Utilities queries with Humans + English filters and an optional study-type restriction (meta-analyses, RCTs, systematic reviews, guidelines). |
| `vaccine_recommendations` | Returns ACIP-curated routine, risk-based, and shared-decision-making vaccines for the patient context. Emits a FHIR `ImmunizationRecommendation`. |
| `prescription_safety_brief` | **Mega-composer.** Chains the safety review, per-finding evidence search, vaccine recommendations, and trial matching into one clinical brief. Emits a FHIR transaction `Bundle` (`RiskAssessment` + `ImmunizationRecommendation` + `DocumentReference`). |

## SHARP-on-MCP support

The server speaks the [SHARP-on-MCP](https://sharponmcp.com/) headers-based
context model. When a host propagates patient context via:

- `x-fhir-server-url` — base URL of the FHIR R4 server,
- `x-fhir-access-token` — bearer JWT (its `patient` claim auto-resolves the
  active patient),
- `x-patient-id` — fallback when no token is set,

every patient-aware tool can be invoked with no arguments — the server pulls
the patient, active medications, and active conditions from the FHIR server
directly. The advertised `ai.promptopinion/fhir-context` capability extension
declares the FHIR scopes the server consumes.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ MCP host  ─────────────►  FastAPI + streamable HTTP /mcp   │
│ (SHARP headers)                  │                         │
│                                  ▼                         │
│                          @mcp.tool wrappers                │
│                                  │                         │
│           ┌──────────────────────┼──────────────────────┐  │
│           ▼                      ▼                      ▼  │
│   live sources/           bundled rule packs    composer    │
│   (RxNorm, OpenFDA,       (Beers, ACIP,         (safety_    │
│    Clinical Tables,        DDInter 2.0)          review)    │
│    ClinicalTrials,                                         │
│    PubMed, ...)                                            │
└────────────────────────────────────────────────────────────┘
```

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# (optional, one-time) build the bundled DDInter SQLite from upstream
.venv/bin/python scripts/build_db.py
# start the MCP HTTP server
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
```

The MCP endpoint is at `http://localhost:8000/mcp`.

## Try the demo scenario

```bash
.venv/bin/python scripts/demo.py
```

The demo runs `medication_safety_review` for an 82-year-old on warfarin,
amiodarone, amoxicillin, lorazepam, and diphenhydramine — exercising every
risk signal at once. Then it calls `clinical_trial_search`, `clinical_evidence_search`,
and `vaccine_recommendations` to show the rest of the surface end-to-end.

## License & attribution

Code under this project's license (see repo). Bundled reference data and rule
packs each retain the license of their source — see [`data/LICENSES.md`](data/LICENSES.md).
