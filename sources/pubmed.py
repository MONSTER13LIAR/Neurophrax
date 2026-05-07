"""PubMed via NCBI E-Utilities (NIH/NLM, public domain).

Two-step retrieval: ``esearch`` returns matching PMIDs, ``esummary`` returns
the bibliographic detail for those PMIDs. We pre-filter for human studies in
English and prefer high-evidence study types so callers get something more
useful than a free-text PubMed query echo.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""
from __future__ import annotations

from typing import Any

from ._http import cached_get

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
WEB_BASE = "https://pubmed.ncbi.nlm.nih.gov"

# Map a friendly study-type label to a PubMed publication-type filter token.
STUDY_TYPE_FILTERS: dict[str, str] = {
    "meta-analysis": "Meta-Analysis[ptyp]",
    "systematic-review": "Systematic Review[ptyp]",
    "rct": "Randomized Controlled Trial[ptyp]",
    "review": "Review[ptyp]",
    "guideline": "Guideline[ptyp]",
    "practice-guideline": "Practice Guideline[ptyp]",
}


def _pubmed_url(pmid: str) -> str:
    return f"{WEB_BASE}/{pmid}/" if pmid else ""


def _build_query(
    question: str,
    *,
    max_age_years: int | None,
    study_types: list[str] | None,
) -> str:
    """Wrap the user's question in PubMed filter syntax.

    Adds Humans + English filters, optional study-type restriction, and a
    publication-date range. This composition is what makes the tool more than
    a thin proxy — it consistently returns clinically usable hits.
    """
    parts = [f"({question})"]
    parts.append("(humans[mh])")
    parts.append("(english[lang])")
    if study_types:
        ors = [STUDY_TYPE_FILTERS[s] for s in study_types if s in STUDY_TYPE_FILTERS]
        if ors:
            parts.append("(" + " OR ".join(ors) + ")")
    if max_age_years and max_age_years > 0:
        parts.append(f'("last {max_age_years} years"[dp])')
    return " AND ".join(parts)


async def search(
    *,
    question: str,
    max_age_years: int | None = 5,
    study_types: list[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search PubMed and return projected article summaries plus the query."""
    term = _build_query(question, max_age_years=max_age_years, study_types=study_types)
    try:
        es = await cached_get(
            f"{BASE}/esearch.fcgi",
            {
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "retmax": min(max(limit, 1), 30),
                "sort": "pub_date",
            },
        )
    except Exception:
        return {"query": term, "results": [], "total": 0}

    id_list: list[str] = ((es.get("esearchresult") or {}).get("idlist") or [])
    total = int((es.get("esearchresult") or {}).get("count") or 0)
    if not id_list:
        return {"query": term, "results": [], "total": total}

    try:
        summary = await cached_get(
            f"{BASE}/esummary.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
            },
        )
    except Exception:
        return {"query": term, "results": [], "total": total}

    docs = (summary.get("result") or {})
    results: list[dict[str, Any]] = []
    for pmid in id_list:
        d = docs.get(pmid)
        if not isinstance(d, dict):
            continue
        authors = d.get("authors") or []
        author_names = [a.get("name") for a in authors if a.get("name")]
        results.append({
            "pmid": pmid,
            "title": d.get("title") or "",
            "journal": d.get("fulljournalname") or d.get("source") or "",
            "pub_date": d.get("pubdate") or "",
            "authors": author_names[:6],
            "publication_types": d.get("pubtype") or [],
            "url": _pubmed_url(pmid),
        })
    return {"query": term, "results": results, "total": total}
