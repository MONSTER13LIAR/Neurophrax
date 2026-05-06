# Bundled Reference Data — Licenses & Attribution

This directory holds curated reference datasets used for offline lookups
(drug interactions, clinical rule packs, code systems). No patient data is
stored here. No data is bundled yet — this directory will be populated in
later phases via `scripts/build_db.py`.

## Planned bundled datasets

| Dataset | License | Redistribution | Status |
|---|---|---|---|
| DDInter 2.0 (drug-drug interactions) | Academic / non-commercial | Permitted with attribution | pending |
| Beers Criteria 2023 (geriatric prescribing) | Encoded as derivative rule set | Rule encoding permitted | pending |
| ATC classification | WHO Collaborating Centre — non-commercial | Permitted with attribution | pending |
| ICD-10-CM | CMS — public domain (US) | Permitted | pending |

## Not bundled (license restrictions)

| Source | Reason |
|---|---|
| SNOMED CT | Affiliate license required; queried via Snowstorm API |
| DrugBank (full) | Commercial license required |

When a dataset is added, its full license text is included alongside it and
its version is recorded in the `dataset_versions` table of
`neurophrax.sqlite`.
