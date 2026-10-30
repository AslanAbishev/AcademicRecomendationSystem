import json

import pytest

from app.services.affiliations import best_affiliation, latest_affiliation
from app.services.realtime_recommender import ECRRecommenderService
from app.services.experiment_service import ExperimentService
from scripts.enrich_semantic_scholar import enrich
from scripts.enrich_crossref import enrich as enrich_crossref
from scripts.evaluate_collaborators import evaluate, ranking_metrics
from scripts.expand_author_cohort import expand
from scripts.inspect_dataset import inspect
from scripts.collect_multifield_openalex import collect
from scripts.prepare_multifield_dataset import _valid_orcid, prepare
from scripts.expand_multifield_longitudinal import expand as expand_longitudinal, select_seeds
from scripts.freeze_temporal_cohort import freeze
from app.clients.openalex import OpenAlexClient


def _work(work_id, year, authors, title="Graph research"):
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": title,
        "publication_year": year,
        "topics": [{"display_name": "Graph Learning"}],
        "authorships": [
            {
                "author": {"id": f"https://openalex.org/{author_id}"},
                "institutions": [{"display_name": affiliation, "country_code": "KZ"}],
            }
            for author_id, affiliation in authors
        ],
    }


def test_latest_affiliation_uses_named_author_and_newest_work():
    works = [
        _work("W1", 2021, [("A1", "Old University"), ("A2", "Unrelated Lab")]),
        _work("W2", 2025, [("A1", "New University"), ("A3", "Another University")]),
    ]
    assert latest_affiliation("A1", works)["display_name"] == "New University"
    assert latest_affiliation("A2", works)["display_name"] == "Unrelated Lab"
    assert best_affiliation({"id": "https://openalex.org/A1", "last_known_institutions": [{"display_name": "Old University"}]}, works)["display_name"] == "New University"


def test_raw_affiliation_corrects_wrong_openalex_institution_mapping():
    work = _work("W1", 2026, [("A1", "Astana Medical University")])
    work["authorships"][0]["raw_affiliation_strings"] = ["School of Software Engineering, Astana IT University"]
    author = {
        "id": "https://openalex.org/A1",
        "last_known_institutions": [
            {"display_name": "Astana Medical University", "country_code": "KZ"},
            {"display_name": "Astana IT University", "country_code": "KZ"},
        ],
    }
    observation = best_affiliation(author, [work])
    assert observation["display_name"] == "Astana IT University"
    assert observation["source"] == "raw_affiliation_confirmed"


def test_collaborator_merge_does_not_restore_older_affiliation():
    service = ECRRecommenderService()
    old = service._extract_author_candidates_from_work(_work("W1", 2021, [("A1", "Old University")]))[0]
    new = service._extract_author_candidates_from_work(_work("W2", 2025, [("A1", "New University")]))[0]
    assert service._merge_candidate(new, old)["last_known_institutions"][0]["display_name"] == "New University"
    assert service._merge_candidate(old, new)["last_known_institutions"][0]["display_name"] == "New University"


def test_stale_candidate_work_is_replaced_by_author_profile():
    class Client:
        def get_author(self, author_id):
            return {"id": f"https://openalex.org/{author_id}", "last_known_institutions": [{"display_name": "Current University", "country_code": "KZ"}]}

    service = ECRRecommenderService(client=Client())
    candidate = service._extract_author_candidates_from_work(_work("W1", 2020, [("A1", "Old University")]))[0]
    refreshed = service._refresh_candidate_affiliation(candidate)
    assert refreshed["last_known_institutions"][0]["display_name"] == "Current University"
    assert refreshed["affiliation_source"] == "openalex_author_profile"


def test_candidate_raw_affiliation_corrects_mislinked_institution():
    class Client:
        def get_author(self, author_id):
            return {
                "id": f"https://openalex.org/{author_id}",
                "last_known_institutions": [
                    {"display_name": "Astana Medical University", "country_code": "KZ"},
                    {"display_name": "Astana IT University", "country_code": "KZ"},
                ],
            }

    service = ECRRecommenderService(client=Client())
    work = _work("W1", 2026, [("A1", "Astana Medical University")])
    work["authorships"][0]["raw_affiliation_strings"] = ["School of Software Engineering, Astana IT University"]
    candidate = service._extract_author_candidates_from_work(work)[0]
    refreshed = service._refresh_candidate_affiliation(candidate)
    assert refreshed["last_known_institutions"][0]["display_name"] == "Astana IT University"
    assert refreshed["affiliation_source"] == "raw_affiliation_confirmed"


def test_ranking_metrics_use_full_relevant_set_and_discounted_gain():
    metrics = ranking_metrics(["A1", "A2", "A3"], {"A2", "A4"}, 2)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 0.5
    assert 0 < metrics["ndcg_at_k"] < 1


def test_temporal_evaluation_rejects_overlapping_years(tmp_path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    work = _work("W1", 2025, [("A1", "University"), ("A2", "University")])
    train.write_text(json.dumps(work), encoding="utf-8")
    test.write_text(json.dumps(work), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly later"):
        evaluate(train, test, k=1, max_authors=3)


def test_temporal_evaluation_uses_future_links_with_shared_candidate_pool(tmp_path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text("\n".join(json.dumps(work) for work in [
        _work("W1", 2022, [("A1", "U"), ("A2", "U")], "Graph learning"),
        _work("W2", 2023, [("A3", "U"), ("A4", "U")], "Scientific discovery systems"),
    ]), encoding="utf-8")
    test.write_text(json.dumps(_work("W3", 2025, [("A1", "U"), ("A3", "U")])), encoding="utf-8")
    result = evaluate(train, test, k=1, max_authors=4, max_queries=4)
    assert result["evaluated_authors"] == 2
    assert result["held_out_links_in_evaluated_queries"] == 2
    assert set(result["models"]) >= {"random", "popularity", "tfidf", "bm25", "lsa_svd", "semantic_hashing", "hybrid", "fairness_reranked_hybrid"}


def test_frozen_pair_labels_exclude_validation_links_from_test_candidates(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text("\n".join(json.dumps(work) for work in [
        _work("W1", 2023, [("A1", "U")], "Graph learning"),
        _work("W2", 2023, [("A2", "U")], "Medical imaging"),
        _work("W3", 2023, [("A3", "U")], "Graph learning"),
    ]), encoding="utf-8")
    validation.write_text(json.dumps({"author_a": "A1", "author_b": "A2", "publication_year": 2024}), encoding="utf-8")
    test.write_text(json.dumps({"author_a": "A1", "author_b": "A3", "publication_year": 2025}), encoding="utf-8")

    result = evaluate(train, test, k=1, max_authors=3, prior_labels_path=validation)

    assert result["label_records"] is True
    assert result["held_out_links_in_evaluated_queries"] == 2
    assert result["models"]["tfidf"]["recall_at_k"] == 1.0


def test_graphsage_frozen_labels_only_use_training_authors(tmp_path):
    from scripts.train_graphsage import _label_edge_index

    labels = tmp_path / "pairs.jsonl"
    labels.write_text("\n".join(json.dumps(pair) for pair in [
        {"author_a": "A1", "author_b": "A2"},
        {"author_a": "A1", "author_b": "A9"},
    ]), encoding="utf-8")
    edges = _label_edge_index(labels, {"A1": 0, "A2": 1})
    assert edges.tolist() == [[0], [1]]


def test_graph_baseline_uses_only_two_hop_train_connections():
    from scripts.evaluate_collaborators import _two_hop_scores

    neighbors = {"A1": {"A2"}, "A2": {"A1", "A3"}, "A3": {"A2"}, "A4": set()}
    common, adamic = _two_hop_scores("A1", ["A2", "A3", "A4"], neighbors)
    assert common.tolist() == [0, 1, 0]
    assert adamic[1] > 0
    assert adamic[0] == adamic[2] == 0


def test_graph_diagnostics_reject_inspected_test_period(tmp_path):
    from scripts.diagnose_graphsage import diagnose

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "author_nodes.json").write_text('[{"author_id": "A1"}]', encoding="utf-8")
    (graph_dir / "author_edges.csv").write_text("source,target,weight\n", encoding="utf-8")
    labels = tmp_path / "future.jsonl"
    labels.write_text('{"author_a":"A1","author_b":"A2","publication_year":2025}', encoding="utf-8")
    with pytest.raises(ValueError, match="2024 validation"):
        diagnose(graph_dir, labels, tmp_path / "missing.npz")


def test_graph_text_ranker_rejects_test_period_labels(tmp_path):
    from scripts.train_graph_text_ranker import _pair_labels

    labels = tmp_path / "test.jsonl"
    labels.write_text('{"author_a":"A1","author_b":"A2","publication_year":2025}', encoding="utf-8")
    with pytest.raises(ValueError, match="2024 frozen labels"):
        _pair_labels(labels)


def test_graph_text_period_uses_only_supplied_past_works():
    from scripts.train_graph_text_ranker import Period

    past = [_work("W1", 2022, [("A1", "U")], "Graph learning")]
    future = _work("W2", 2023, [("A1", "U"), ("A2", "U")], "Future medicine")
    period = Period(past, max_authors=3, seed=42)
    assert period.authors == ["A1"]
    assert period.profiles["A1"]["titles"] == ["Graph learning"]
    assert "A2" not in period.neighbors
    assert future["id"] not in period.profiles


def test_graph_text_abstract_reconstruction_uses_positions():
    from scripts.train_graph_text_ranker import _abstract

    assert _abstract({"research": [1], "graph": [0, 2]}) == "graph research graph"


def test_prospective_holdout_seeds_use_only_2023_works(tmp_path):
    from scripts.collect_2026_holdout import select_train_seed_authors

    source = tmp_path / "train.jsonl"
    source.write_text("\n".join(json.dumps(work) for work in [
        _work("W1", 2022, [("A9", "U")]),
        _work("W2", 2023, [("A1", "U")]),
        _work("W3", 2023, [("A1", "U"), ("A2", "U")]),
    ]), encoding="utf-8")
    assert select_train_seed_authors(source, 2) == ["A1", "A2"]


def test_holdout_evaluation_rejects_model_changed_after_collection(tmp_path):
    from scripts.evaluate_2026_holdout import evaluate_holdout

    model = tmp_path / "model.json"
    model.write_text('{"model_name":"TemporalGraphTextRanker-v2"}', encoding="utf-8")
    training = tmp_path / "training.json"
    training.write_text('{"model_name":"TemporalGraphTextRanker-v2"}', encoding="utf-8")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "manifest.json").write_text('{"model_sha256":"not-the-current-model"}', encoding="utf-8")
    with pytest.raises(ValueError, match="model changed"):
        evaluate_holdout(model, training, frozen, tmp_path / "result.json")


def test_graph_text_json_coefficients_match_fitted_ranker():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scripts.train_graph_text_ranker import Period, _rank

    works = [
        _work("W1", 2023, [("A1", "U")], "Graph learning"),
        _work("W2", 2023, [("A2", "U")], "Medical imaging"),
        _work("W3", 2023, [("A3", "U")], "Graph learning"),
    ]
    period = Period(works, max_authors=3, seed=42)
    features = period.features("A1")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression().fit(scaler.transform(features), [0, 0, 1])
    artifact = {
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    labels = {"A1": {"A3"}}
    assert _rank(period, labels, ["A1"], model, scaler, 1, 0.0) == _rank(
        period, labels, ["A1"], artifact, None, 1, 0.0)


def test_semantic_scholar_enrichment_preserves_unmatched_and_provenance(tmp_path):
    class Response:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return [{"paperId": "S1", "citationCount": 12, "influentialCitationCount": 3, "isOpenAccess": True}]

    class Session:
        def post(self, endpoint, params, json, headers, timeout):
            assert json["ids"] == ["DOI:10.1000/example"]
            return Response()

    source = tmp_path / "source.jsonl"
    target = tmp_path / "enriched.jsonl"
    source.write_text('\n'.join(json.dumps(item) for item in [{"id": "W1", "doi": "https://doi.org/10.1000/example"}, {"id": "W2"}]), encoding="utf-8")
    result = enrich(source, target, session=Session(), delay=0)
    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert result["matched"] == 1
    assert records[0]["semantic_scholar_activity"]["citation_count"] == 12
    assert records[1] == {"id": "W2"}


def test_crossref_enrichment_keeps_funding_provenance(tmp_path):
    class Response:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"DOI": "10.1000/example", "is-referenced-by-count": 7, "funder": [{"name": "Research Council", "award": ["Grant-1"]}]}}

    class Session:
        headers = {}

        def get(self, url, timeout):
            assert url.endswith("10.1000%2Fexample")
            return Response()

    source = tmp_path / "works.jsonl"
    output = tmp_path / "crossref.jsonl"
    source.write_text(json.dumps({"id": "W1", "doi": "https://doi.org/10.1000/example"}), encoding="utf-8")
    result = enrich_crossref(source, output, limit=1, delay=0, session=Session())
    record = json.loads(output.read_text(encoding="utf-8"))
    assert result["matched"] == 1
    assert record["crossref_activity"]["awards"] == ["Grant-1"]
    assert record["crossref_activity"]["source"] == "Crossref REST API"


def test_author_cohort_expansion_deduplicates_real_work_ids(tmp_path):
    class Client:
        def iter_works(self, query, from_year, to_year, max_records, extra_filters):
            assert extra_filters == ["authorships.author.id:https://openalex.org/A1"]
            return [
                _work("W2", 2022, [("A1", "University")]),
                _work("W3", 2025, [("A1", "University")]),
            ]

    source = tmp_path / "seed.jsonl"
    output = tmp_path / "cohort.jsonl"
    source.write_text("\n".join(json.dumps(work) for work in [
        _work("W1", 2020, [("A1", "University")]),
        _work("W2", 2022, [("A1", "University")]),
    ]), encoding="utf-8")
    result = expand(source, output, max_authors=1, client=Client())
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["unique_works"] == 3
    assert [record["id"].split("/")[-1] for record in records] == ["W1", "W2", "W3"]


def test_author_cohort_seed_selection_ignores_future_years(tmp_path):
    class Client:
        def iter_works(self, query, from_year, to_year, max_records, extra_filters):
            assert extra_filters == ["authorships.author.id:https://openalex.org/A1"]
            return []

    source = tmp_path / "seed.jsonl"
    source.write_text("\n".join(json.dumps(work) for work in [
        _work("W1", 2021, [("A1", "U")]),
        _work("W2", 2022, [("A1", "U")]),
        _work("W3", 2025, [("A9", "U")]),
        _work("W4", 2025, [("A9", "U")]),
    ]), encoding="utf-8")
    result = expand(source, tmp_path / "cohort.jsonl", max_authors=2, client=Client())
    assert result["seed_authors"] == 1
    assert result["seed_train_through"] == 2023


def test_experiment_report_reads_actual_saved_metrics(tmp_path):
    report = tmp_path / "evaluation.json"
    report.write_text(json.dumps({"models": {"hybrid": {"ndcg_at_k": 0.4}}}), encoding="utf-8")
    assert ExperimentService().get_collaborator_evaluation(report)["models"]["hybrid"]["ndcg_at_k"] == 0.4


def test_dataset_audit_counts_duplicates_and_missing_fields(tmp_path):
    dataset = tmp_path / "works.jsonl"
    record = _work("W1", 2025, [("A1", "University")])
    dataset.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    report = inspect(dataset)
    assert report["records"] == 2
    assert report["unique_works"] == 1
    assert report["duplicate_id"] == 1
    assert report["with_doi"] == 0


def test_multifield_collector_uses_exact_field_year_and_resumes(tmp_path):
    class Client:
        def __init__(self):
            self.calls = 0

        def iter_works(self, query, from_year, to_year, max_records, extra_filters):
            self.calls += 1
            assert query == ""
            assert extra_filters == ["primary_topic.field.id:17"]
            return [
                {**_work("W1", 2023, [("A1", "University")]), "primary_topic": {"field": {"id": "https://openalex.org/fields/17"}}},
                {**_work("W2", 2022, [("A2", "University")]), "primary_topic": {"field": {"id": "https://openalex.org/fields/17"}}},
            ]

    client = Client()
    first = collect(tmp_path, {"computer_science": (17, "Computer Science")}, range(2023, 2024), 1, client)
    second = collect(tmp_path, {"computer_science": (17, "Computer Science")}, range(2023, 2024), 1, client)
    assert first["works_across_buckets"] == second["works_across_buckets"] == 1
    assert client.calls == 1


def test_multifield_collector_completes_partial_bucket_without_duplicates(tmp_path):
    class Client:
        def iter_works(self, query, from_year, to_year, max_records, extra_filters):
            assert max_records == 3
            return [
                {**_work(work_id, 2023, [("A1", "University")]), "primary_topic": {"field": {"id": "https://openalex.org/fields/17"}}}
                for work_id in ("W1", "W2", "W3")
            ]

    target = tmp_path / "computer_science_2023.jsonl"
    target.write_text(json.dumps({"id": "https://openalex.org/W1"}) + "\n", encoding="utf-8")
    report = collect(tmp_path, {"computer_science": (17, "Computer Science")}, range(2023, 2024), 2, Client())
    works = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert report["all_buckets_complete"]
    assert [work["id"].split("/")[-1] for work in works] == ["W1", "W2"]


def test_multifield_audit_counts_future_links_and_flags_affiliation(tmp_path):
    base = tmp_path / "base.jsonl"
    buckets = tmp_path / "buckets"
    buckets.mkdir()
    output = tmp_path / "merged.jsonl"
    report_path = tmp_path / "audit.json"
    train = _work("W1", 2023, [("A1", "Old University"), ("A2", "Old University")])
    test = _work("W2", 2025, [("A1", "Mapped University"), ("A3", "New University")])
    test["authorships"][0]["raw_affiliation_strings"] = ["Actual University"]
    base.write_text("\n".join(json.dumps(item) for item in [train, _work("W3", 2023, [("A3", "New University")])]), encoding="utf-8")
    bucket = buckets / "cs_2025.jsonl"
    bucket.write_text(json.dumps(test), encoding="utf-8")
    (buckets / "manifest.json").write_text(json.dumps({"buckets": [{"file": str(bucket)}]}), encoding="utf-8")
    report = prepare(base, buckets, output, report_path)
    assert report["counts"]["unique_works"] == 3
    assert report["temporal_evidence"]["new_test_pairs_between_train_authors"] == 1
    assert report["temporal_evidence"]["new_test_pairs_by_primary_field"] == {"Unknown field": 1}
    assert report["counts"]["mapped_name_not_in_raw_affiliation"] == 1
    assert report["coauthor_events"]["records"] == 2
    assert report["affiliation_review"]["records"] == 1
    assert report_path.exists()


def test_identity_conflicts_are_kept_in_source_but_excluded_from_pair_labels(tmp_path):
    base = tmp_path / "base.jsonl"
    buckets = tmp_path / "buckets"
    buckets.mkdir()
    first = _work("W1", 2023, [("A1", "University"), ("A2", "University")])
    second = _work("W2", 2025, [("A1", "University"), ("A3", "University")])
    for author in first["authorships"]:
        author["author"]["orcid"] = "https://orcid.org/0000-0002-1825-0097"
    base.write_text("\n".join(json.dumps(work) for work in [first, second]), encoding="utf-8")
    bucket = buckets / "empty.jsonl"
    bucket.write_text("", encoding="utf-8")
    (buckets / "manifest.json").write_text(json.dumps({"buckets": [{"file": str(bucket)}]}), encoding="utf-8")

    report = prepare(base, buckets, tmp_path / "merged.jsonl", tmp_path / "report.json")

    assert report["counts"]["unique_works"] == 2
    assert report["orcid_linked_to_multiple_openalex_authors"] == 1
    assert report["author_ids_excluded_from_derived_edges"] == 2
    assert report["coauthor_events"]["records"] == 0
    assert report["model_ready"]["works"] == 1
    assert report["temporal_evidence"]["train_works"] == 0
    assert report["temporal_evidence"]["test_works"] == 1
    ready = [json.loads(line) for line in (tmp_path / "merged_model_ready.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["author"]["id"].split("/")[-1] for item in ready[0]["authorships"]] == ["A3"]
    assert len((buckets / "identity_review.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_openalex_cursor_page_is_capped_at_supported_size():
    client = OpenAlexClient(api_key="test")
    requests = []

    def fake_get(path, params):
        requests.append(params)
        return {"results": [{"id": f"W{index}"} for index in range(100)], "meta": {"next_cursor": None}}

    client._get = fake_get
    assert len(list(client.iter_works("", per_page=200, max_records=100))) == 100
    assert requests[0]["per-page"] == 100


def test_orcid_checksum_is_checked_without_claiming_identity_verification():
    assert _valid_orcid("https://orcid.org/0000-0002-1825-0097")
    assert not _valid_orcid("0000-0002-1825-0098")


def test_longitudinal_seed_authors_are_selected_only_from_train_years(tmp_path):
    source = tmp_path / "works.jsonl"
    works = [
        _work("W1", 2022, [("A1", "University")]),
        _work("W2", 2023, [("A1", "University")]),
        _work("W3", 2025, [("A9", "University")]),
        _work("W4", 2025, [("A9", "University")]),
    ]
    for work in works:
        work["primary_topic"] = {"field": {"id": "https://openalex.org/fields/17"}}
    source.write_text("\n".join(json.dumps(work) for work in works), encoding="utf-8")
    assert select_seeds(source, per_field=2)["computer_science"] == ["A1"]


def test_multifield_prepare_requires_completed_longitudinal_slices(tmp_path):
    base = tmp_path / "base.jsonl"
    base.write_text(json.dumps(_work("W1", 2023, [("A1", "University")])), encoding="utf-8")
    buckets = tmp_path / "buckets"
    buckets.mkdir()
    (buckets / "manifest.json").write_text('{"buckets": []}', encoding="utf-8")
    extra = buckets / "longitudinal"
    extra.mkdir()
    slice_path = extra / "computer_science_A1_2025.jsonl"
    slice_path.write_text(json.dumps(_work("W2", 2025, [("A1", "University")])), encoding="utf-8")
    (extra / "manifest.json").write_text(json.dumps({"slices": [{"file": str(slice_path)}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unfinished"):
        prepare(base, buckets, tmp_path / "merged.jsonl", tmp_path / "report.json", extra_dir=extra)
    slice_path.with_suffix(".done").touch()
    report = prepare(base, buckets, tmp_path / "merged.jsonl", tmp_path / "report.json", extra_dir=extra)
    assert report["counts"]["unique_works"] == 2


def test_longitudinal_resume_freezes_seed_cohort(tmp_path):
    class Client:
        def __init__(self):
            self.calls = 0

        def iter_works(self, query, from_year, to_year, max_records, extra_filters):
            self.calls += 1
            return []

    source = tmp_path / "works.jsonl"
    works = [_work("W1", 2022, [("A1", "University")]), _work("W2", 2023, [("A1", "University")])]
    for work in works:
        work["primary_topic"] = {"field": {"id": "https://openalex.org/fields/17"}}
    source.write_text("\n".join(json.dumps(work) for work in works), encoding="utf-8")
    client = Client()
    first = expand_longitudinal(source, tmp_path / "longitudinal", per_field=1, per_author_year=2, client=client)
    source.write_text("", encoding="utf-8")
    second = expand_longitudinal(source, tmp_path / "longitudinal", per_field=1, per_author_year=2, client=client)
    assert first["seed_authors_by_field"] == second["seed_authors_by_field"]
    assert client.calls == 6


def test_future_orcid_conflict_cannot_change_train_snapshot(tmp_path):
    base = tmp_path / "works.jsonl"
    buckets = tmp_path / "buckets"
    buckets.mkdir()
    (buckets / "manifest.json").write_text('{"buckets": []}', encoding="utf-8")
    train = _work("W1", 2023, [("A1", "University"), ("A2", "University")])
    train["authorships"][0]["author"]["orcid"] = "https://orcid.org/0000-0002-1825-0097"
    train["cited_by_count"] = 999
    future = _work("W2", 2025, [("A2", "University"), ("A3", "University")])
    future["authorships"][0]["author"]["orcid"] = "https://orcid.org/0000-0002-1825-0097"
    works = [train, _work("W3", 2023, [("A3", "University")]), _work("W4", 2024, [("A1", "University"), ("A3", "University")]), future]
    base.write_text("\n".join(json.dumps(work) for work in works), encoding="utf-8")

    prepared = prepare(base, buckets, tmp_path / "merged.jsonl", tmp_path / "quality.json")
    frozen = freeze(tmp_path / "merged_model_ready.jsonl", tmp_path / "frozen")

    assert prepared["orcid_linked_to_multiple_openalex_authors"] == 1
    assert prepared["orcid_conflicts_known_by_train_cutoff"] == 0
    assert prepared["author_ids_excluded_from_derived_edges"] == 0
    assert (buckets / "identity_review_asof_2023.jsonl").read_text(encoding="utf-8") == ""
    snapshot = [json.loads(line) for line in (tmp_path / "frozen" / "train_asof_2023.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {item["id"].split("/")[-1] for item in snapshot} == {"W1", "W3"}
    assert "cited_by_count" not in snapshot[0]
    assert "orcid" not in snapshot[0]["authorships"][0]["author"]
    assert frozen["validation_new_pairs"] == 1
    assert frozen["test_new_pairs"] == 1
