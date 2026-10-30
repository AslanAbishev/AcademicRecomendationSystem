from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re

WORK_ID = re.compile(r"^W\d+$")
AUTHOR_ID = re.compile(r"^A\d+$")
INSTITUTION_ID = re.compile(r"^I\d+$")
ORCID = re.compile(r"^\d{4}-\d{4}-\d{4}-[\dX]{4}$")
DOI = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


def _short_id(value: str | None) -> str:
    return (value or "").rstrip("/").split("/")[-1]


def _valid_orcid(value: str) -> bool:
    identifier = value.removeprefix("https://orcid.org/").upper()
    if not ORCID.fullmatch(identifier):
        return False
    digits = identifier.replace("-", "")
    total = 0
    for digit in digits[:-1]:
        total = (total + int(digit)) * 2
    check = (12 - total % 11) % 11
    return digits[-1] == ("X" if check == 10 else str(check))


def _edges(works: list[dict], excluded_authors: set[str] | None = None) -> set[tuple[str, str]]:
    pairs = set()
    excluded = excluded_authors or set()
    for work in works:
        if len(work.get("authorships") or []) > 20:
            continue
        authors = sorted({
            _short_id((item.get("author") or {}).get("id"))
            for item in work.get("authorships") or []
        })
        authors = [author for author in authors if AUTHOR_ID.fullmatch(author) and author not in excluded]
        if 1 < len(authors) <= 20:
            pairs.update(combinations(authors, 2))
    return pairs


def _write_evidence(by_year: dict[int, list[dict]], bucket_dir: Path, excluded_authors: set[str]) -> tuple[Path, Path, int, int]:
    events_path = bucket_dir / "coauthor_events.jsonl"
    review_path = bucket_dir / "affiliation_review.jsonl"
    events = 0
    flags = 0
    with events_path.open("w", encoding="utf-8") as event_file, review_path.open("w", encoding="utf-8") as review_file:
        for year, works in sorted(by_year.items()):
            for work in works:
                authorships = work.get("authorships") or []
                authors = sorted({
                    _short_id((item.get("author") or {}).get("id"))
                    for item in authorships
                    if AUTHOR_ID.fullmatch(_short_id((item.get("author") or {}).get("id")))
                    and _short_id((item.get("author") or {}).get("id")) not in excluded_authors
                })
                field = ((work.get("primary_topic") or {}).get("field") or {}).get("display_name") or "Unknown field"
                if len(authorships) <= 20:
                    for left, right in combinations(authors, 2):
                        event_file.write(json.dumps({
                            "work_id": work["id"], "publication_year": year,
                            "field": field, "author_a": left, "author_b": right,
                        }, ensure_ascii=False) + "\n")
                        events += 1
                for item in authorships:
                    raw = " ".join(item.get("raw_affiliation_strings") or [])
                    for institution in item.get("institutions") or []:
                        name = (institution.get("display_name") or "").strip()
                        if raw and name and name.casefold() not in raw.casefold():
                            review_file.write(json.dumps({
                                "work_id": work["id"], "publication_year": year,
                                "author_id": (item.get("author") or {}).get("id"),
                                "mapped_institution_id": institution.get("id"),
                                "mapped_name": name, "raw_affiliation_strings": item.get("raw_affiliation_strings") or [],
                                "reason": "mapped name absent from raw text; abbreviations may explain this",
                            }, ensure_ascii=False) + "\n")
                            flags += 1
    return events_path, review_path, events, flags


def _future_pairs_by_field(works: list[dict], prior: set[tuple[str, str]], train_authors: set[str], excluded_authors: set[str]) -> dict[str, int]:
    groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for work in works:
        field = ((work.get("primary_topic") or {}).get("field") or {}).get("display_name") or "Unknown field"
        groups[field].update(
            pair for pair in _edges([work], excluded_authors)
            if pair not in prior and pair[0] in train_authors and pair[1] in train_authors
        )
    return {field: len(pairs) for field, pairs in sorted(groups.items()) if pairs}


def prepare(base: Path, bucket_dir: Path, output: Path, report_path: Path, train_through: int = 2023, validation_year: int = 2024, test_year: int = 2025, extra_dir: Path | None = None) -> dict:
    if not train_through < validation_year < test_year:
        raise ValueError("Expected train year < validation year < test year")
    manifest_path = bucket_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bucket_paths = [Path(item["file"]) for item in manifest["buckets"]]
    extra_paths = []
    if extra_dir:
        extra_manifest_path = extra_dir / "manifest.json"
        if not extra_manifest_path.exists():
            raise FileNotFoundError(extra_manifest_path)
        extra_manifest = json.loads(extra_manifest_path.read_text(encoding="utf-8"))
        extra_paths = [Path(item["file"]) for item in extra_manifest["slices"]]
        if any(not path.with_suffix(".done").exists() for path in extra_paths):
            raise ValueError("Longitudinal collection contains unfinished author/year slices")
    inputs = [base, *bucket_paths, *extra_paths]
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Output must differ from every source file")
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    by_field_year: dict[str, Counter[str]] = defaultdict(Counter)
    by_year: dict[int, list[dict]] = defaultdict(list)
    author_ids: set[str] = set()
    orcid_authors: dict[str, set[str]] = defaultdict(set)
    author_orcids: dict[str, set[str]] = defaultdict(set)
    train_orcid_authors: dict[str, set[str]] = defaultdict(set)
    train_author_orcids: dict[str, set[str]] = defaultdict(set)
    doi_works: dict[str, set[str]] = defaultdict(set)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for path in inputs:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    counts["input_records"] += 1
                    work = json.loads(line)
                    work_id = _short_id(work.get("id"))
                    if not WORK_ID.fullmatch(work_id):
                        counts["invalid_work_ids"] += 1
                        continue
                    if work_id in ids:
                        counts["duplicates_removed"] += 1
                        continue
                    year = work.get("publication_year")
                    try:
                        year = int(year)
                    except (TypeError, ValueError):
                        counts["invalid_year"] += 1
                        continue
                    if not 2020 <= year <= test_year:
                        counts["outside_study_period"] += 1
                        continue
                    ids.add(work_id)
                    target.write(json.dumps(work, ensure_ascii=False) + "\n")
                    by_year[year].append(work)
                    doi = (work.get("doi") or "").removeprefix("https://doi.org/")
                    if not doi:
                        counts["missing_doi"] += 1
                    elif not DOI.fullmatch(doi):
                        counts["invalid_doi_format"] += 1
                    else:
                        doi_works[doi.casefold()].add(work_id)
                    field = ((work.get("primary_topic") or {}).get("field") or {}).get("display_name") or "Unknown field"
                    by_field_year[field][str(year)] += 1
                    counts["unique_works"] += 1
                    for authorship in work.get("authorships") or []:
                        author = authorship.get("author") or {}
                        author_id = _short_id(author.get("id"))
                        if not author_id:
                            counts["missing_author_ids"] += 1
                        elif not AUTHOR_ID.fullmatch(author_id):
                            counts["invalid_author_ids"] += 1
                        else:
                            author_ids.add(author_id)
                        orcid = author.get("orcid") or authorship.get("raw_orcid")
                        if orcid:
                            counts["authorships_with_orcid"] += 1
                            if not _valid_orcid(orcid):
                                counts["invalid_orcid_format_or_checksum"] += 1
                            elif AUTHOR_ID.fullmatch(author_id):
                                normalized_orcid = orcid.removeprefix("https://orcid.org/").upper()
                                orcid_authors[normalized_orcid].add(author_id)
                                author_orcids[author_id].add(normalized_orcid)
                                if year <= train_through:
                                    train_orcid_authors[normalized_orcid].add(author_id)
                                    train_author_orcids[author_id].add(normalized_orcid)
                        institutions = authorship.get("institutions") or []
                        raw = " ".join(authorship.get("raw_affiliation_strings") or []).casefold()
                        if not institutions:
                            counts["authorships_without_mapped_institution"] += 1
                        for institution in institutions:
                            institution_id = _short_id(institution.get("id"))
                            if not INSTITUTION_ID.fullmatch(institution_id):
                                counts["invalid_institution_ids"] += 1
                            name = (institution.get("display_name") or "").strip().casefold()
                            if raw and name and name not in raw:
                                counts["mapped_name_not_in_raw_affiliation"] += 1
                        if raw:
                            counts["authorships_with_raw_affiliation"] += 1
                        counts["authorships"] += 1

    conflicting_orcids = {orcid: authors for orcid, authors in orcid_authors.items() if len(authors) > 1}
    conflicting_authors = {author: orcids for author, orcids in author_orcids.items() if len(orcids) > 1}
    train_conflicting_orcids = {orcid: authors for orcid, authors in train_orcid_authors.items() if len(authors) > 1}
    train_conflicting_authors = {author: orcids for author, orcids in train_author_orcids.items() if len(orcids) > 1}
    excluded_authors = {
        author for authors in train_conflicting_orcids.values() for author in authors
    } | set(train_conflicting_authors)
    identity_review_path = bucket_dir / "identity_review.jsonl"
    with identity_review_path.open("w", encoding="utf-8") as identity_file:
        for orcid, authors in sorted(conflicting_orcids.items()):
            identity_file.write(json.dumps({"reason": "orcid_links_multiple_openalex_authors", "orcid": orcid, "author_ids": sorted(authors)}) + "\n")
        for author, orcids in sorted(conflicting_authors.items()):
            identity_file.write(json.dumps({"reason": "openalex_author_links_multiple_orcids", "author_id": author, "orcids": sorted(orcids)}) + "\n")
    train_identity_review_path = bucket_dir / f"identity_review_asof_{train_through}.jsonl"
    with train_identity_review_path.open("w", encoding="utf-8") as identity_file:
        for orcid, authors in sorted(train_conflicting_orcids.items()):
            identity_file.write(json.dumps({"reason": "orcid_links_multiple_openalex_authors", "orcid": orcid, "author_ids": sorted(authors)}) + "\n")
        for author, orcids in sorted(train_conflicting_authors.items()):
            identity_file.write(json.dumps({"reason": "openalex_author_links_multiple_orcids", "author_id": author, "orcids": sorted(orcids)}) + "\n")

    model_ready_path = output.with_name(f"{output.stem}_model_ready.jsonl")
    model_ready_count = 0
    model_ready_authorships_removed = 0
    model_ready_large_works_removed = 0
    model_ready_no_usable_authors = 0
    model_ready_by_year: dict[int, list[dict]] = defaultdict(list)
    with model_ready_path.open("w", encoding="utf-8") as model_file:
        for year, works in sorted(by_year.items()):
            for work in works:
                authorships = work.get("authorships") or []
                if len(authorships) > 20:
                    model_ready_large_works_removed += 1
                    continue
                usable = [
                    item for item in authorships
                    if AUTHOR_ID.fullmatch(_short_id((item.get("author") or {}).get("id")))
                    and _short_id((item.get("author") or {}).get("id")) not in excluded_authors
                ]
                model_ready_authorships_removed += len(authorships) - len(usable)
                if not usable:
                    model_ready_no_usable_authors += 1
                    continue
                ready_work = {**work, "authorships": usable}
                model_file.write(json.dumps(ready_work, ensure_ascii=False) + "\n")
                model_ready_by_year[year].append(ready_work)
                model_ready_count += 1

    train = [work for year, works in model_ready_by_year.items() if year <= train_through for work in works]
    validation = model_ready_by_year.get(validation_year, [])
    test = [work for year, works in model_ready_by_year.items() if year >= test_year for work in works]
    train_authors = {
        _short_id((item.get("author") or {}).get("id"))
        for work in train for item in work.get("authorships") or []
    }
    train_authors = {author for author in train_authors if AUTHOR_ID.fullmatch(author) and author not in excluded_authors}
    known = _edges(train, excluded_authors)
    valid = _edges(validation, excluded_authors)
    future = _edges(test, excluded_authors)
    eligible = lambda pair: pair[0] in train_authors and pair[1] in train_authors
    events_path, review_path, event_count, review_count = _write_evidence(by_year, bucket_dir, excluded_authors)
    report = {
        "source": "OpenAlex API records; field and institution mappings are algorithmic, not independently verified",
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "counts": dict(counts),
        "unique_authors": len(author_ids),
        "orcid_linked_to_multiple_openalex_authors": len(conflicting_orcids),
        "openalex_authors_linked_to_multiple_orcids": len(conflicting_authors),
        "train_cutoff_year": train_through,
        "orcid_conflicts_known_by_train_cutoff": len(train_conflicting_orcids),
        "author_conflicts_known_by_train_cutoff": len(train_conflicting_authors),
        "author_ids_excluded_from_derived_edges": len(excluded_authors),
        "identity_review": str(identity_review_path),
        "train_identity_review": str(train_identity_review_path),
        "model_ready": {
            "path": str(model_ready_path),
            "works": model_ready_count,
            "authorships_removed_due_to_missing_or_conflicting_ids": model_ready_authorships_removed,
            "large_author_works_excluded": model_ready_large_works_removed,
            "works_without_usable_authors_excluded": model_ready_no_usable_authors,
        },
        "doi_linked_to_multiple_openalex_works": sum(len(works) > 1 for works in doi_works.values()),
        "coauthor_events": {"path": str(events_path), "records": event_count},
        "affiliation_review": {"path": str(review_path), "records": review_count},
        "by_field_year": {field: dict(sorted(years.items())) for field, years in sorted(by_field_year.items())},
        "raw_works_by_year": {str(year): len(works) for year, works in sorted(by_year.items())},
        "temporal_evidence": {
            "train_works": len(train), "validation_works": len(validation), "test_works": len(test),
            "train_authors": len(train_authors),
            "train_coauthor_pairs": len(known),
            "new_validation_pairs_between_train_authors": sum(eligible(pair) for pair in valid - known),
            "new_test_pairs_between_train_authors": sum(eligible(pair) for pair in future - known - valid),
            "new_validation_pairs_by_primary_field": _future_pairs_by_field(validation, known, train_authors, excluded_authors),
            "new_test_pairs_by_primary_field": _future_pairs_by_field(test, known | valid, train_authors, excluded_authors),
            "large_author_works_excluded_from_pair_counts": model_ready_large_works_removed,
        },
        "caveats": [
            "OpenAlex author IDs are not independently identity-verified.",
            "Only identity conflicts observed in train-period publications exclude authors from derived pairs; full-corpus conflicts are reported separately.",
            "A mapped institution name absent from raw affiliation text is a review flag, not proof of an error.",
            "Future coauthorship is a proxy label, not measured profile-promotion impact.",
            "Works with more than 20 authors are retained but excluded from pair counts to avoid consortium cliques.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multi-field OpenAlex works and audit temporal collaboration labels.")
    parser.add_argument("--base", default="data/openalex_author_cohort_trainseed.jsonl")
    parser.add_argument("--bucket-dir", default="data/multifield")
    parser.add_argument("--output", default="data/openalex_multifield_dataset.jsonl")
    parser.add_argument("--report", default="data/evaluation/multifield_quality.json")
    parser.add_argument("--extra-dir", help="Optional completed longitudinal author/year slices to merge")
    args = parser.parse_args()
    report = prepare(Path(args.base), Path(args.bucket_dir), Path(args.output), Path(args.report), extra_dir=Path(args.extra_dir) if args.extra_dir else None)
    print(json.dumps({"counts": report["counts"], "temporal_evidence": report["temporal_evidence"]}, indent=2))


if __name__ == "__main__":
    main()
