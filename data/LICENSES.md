# Bundled Reference Data — Licenses & Attribution

This directory holds curated reference datasets used for offline lookups
(drug interactions, clinical rule packs, code systems). No patient data is
stored here.

## Bundled datasets (in `neurophrax.sqlite`)

| Dataset | License | Redistribution | Status |
|---|---|---|---|
| DDInter 2.0 (drug-drug interactions) | Academic / non-commercial | Permitted with attribution | bundled |
| RxNorm RxCUI cross-walk | NIH/NLM — public domain | Permitted | bundled |

## Bundled rule packs (in `sources/`)

| Rule pack | Source | Redistribution |
|---|---|---|
| Beers Criteria 2023 (curated subset) | American Geriatrics Society — copyrighted | Software-flagging derivative; categories paraphrased with explicit attribution. Consult the published criteria for clinical decisions. |
| CDC ACIP Adult Immunization Schedule (curated subset) | Centers for Disease Control and Prevention — public domain (US) | Software-flagging derivative; full schedule consulted live for clinical decisions. |

## Live API sources (no bundled data)

These are queried at runtime; nothing is redistributed.

| Source | Endpoint | License |
|---|---|---|
| RxNorm | https://rxnav.nlm.nih.gov/REST | Public domain (NLM) |
| RxClass (ATC, EPC) | https://rxnav.nlm.nih.gov/REST/rxclass | Public domain (NLM) |
| OpenFDA Drug Label | https://api.fda.gov/drug | Public domain (FDA) |
| DailyMed | https://dailymed.nlm.nih.gov/dailymed/services/v2 | Public domain (NLM) |
| MedlinePlus Connect | https://connect.medlineplus.gov/service | Public domain (NLM) |
| NIH Clinical Tables (Conditions, ICD-10-CM, LOINC) | https://clinicaltables.nlm.nih.gov/api | Public domain |
| ClinicalTrials.gov v2 | https://clinicaltrials.gov/api/v2 | Public domain (NIH/NLM) |
| NCBI PubMed (E-Utilities) | https://eutils.ncbi.nlm.nih.gov/entrez/eutils | Public domain (NIH/NLM) |

## Not bundled (license restrictions)

| Source | Reason |
|---|---|
| SNOMED CT | Affiliate license required for redistribution |
| DrugBank (full) | Commercial license required |

When a dataset is added, its full license text is included alongside it and
its version is recorded in the `dataset_versions` table of
`neurophrax.sqlite`.
