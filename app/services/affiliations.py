from __future__ import annotations

from datetime import date
import re


def short_id(value: str | None) -> str:
    return (value or "").rstrip("/").split("/")[-1]


def publication_date(work: dict) -> str:
    value = work.get("publication_date")
    if isinstance(value, str) and len(value) >= 10:
        try:
            return date.fromisoformat(value[:10]).isoformat()
        except ValueError:
            pass
    year = work.get("publication_year")
    try:
        return f"{int(year):04d}-01-01"
    except (TypeError, ValueError):
        return ""


def authorship_affiliation(authorship: dict) -> tuple[str, str, str]:
    institutions = authorship.get("institutions") or []
    countries = authorship.get("countries") or []
    for institution in institutions:
        name = (institution.get("display_name") or "").strip()
        if name:
            return name, institution.get("country_code") or (countries[0] if countries else ""), institution.get("id") or ""
    return "", countries[0] if countries else "", ""


def matching_raw_institution(authorship: dict, known_institutions: list[dict]) -> dict | None:
    raw = " ".join(authorship.get("raw_affiliation_strings") or [])
    normalized_raw = _normalized(raw)
    matches = [
        institution for institution in known_institutions
        if len((institution.get("display_name") or "").strip()) >= 8
        and f" {_normalized(institution['display_name'])} " in f" {normalized_raw} "
    ]
    return max(matches, key=lambda item: len(item["display_name"])) if matches else None


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def latest_affiliation(author_id: str, works: list[dict], known_institutions: list[dict] | None = None) -> dict | None:
    """Use only the named author's authorship, never a coauthor's affiliation."""
    observations = []
    for work in works:
        when = publication_date(work)
        for authorship in work.get("authorships") or []:
            if short_id((authorship.get("author") or {}).get("id")) != short_id(author_id):
                continue
            name, country, institution_id = authorship_affiliation(authorship)
            raw_match = matching_raw_institution(authorship, known_institutions or [])
            if raw_match:
                name = raw_match["display_name"]
                country = raw_match.get("country_code") or country
                institution_id = raw_match.get("id") or ""
            if name:
                observations.append({
                    "display_name": name,
                    "country_code": country,
                    "id": institution_id,
                    "publication_date": when,
                    "work_id": work.get("id") or "",
                    "source": "raw_affiliation_confirmed" if raw_match else "openalex_work_authorship",
                })
    return max(observations, key=lambda item: item["publication_date"]) if observations else None


def best_affiliation(author: dict, works: list[dict]) -> dict | None:
    recent = latest_affiliation(author.get("id", ""), works, author.get("last_known_institutions") or [])
    if recent:
        return recent
    for institution in author.get("last_known_institutions") or []:
        if institution.get("display_name"):
            return {
                "display_name": institution["display_name"],
                "country_code": institution.get("country_code") or "",
                "id": institution.get("id") or "",
                "publication_date": "",
                "work_id": "",
            }
    return None
