import json
from argparse import Namespace

from app.services.research_database import ResearchDatabase
from app.services.scientific_embeddings import ScientificEmbeddingModel
from scripts.research_pipeline import build_graph
from scripts.split_openalex_temporal import split_dataset


def test_abstract_from_openalex_inverted_index():
    abstract = ResearchDatabase.abstract_from_inverted_index(
        {
            "machine": [0],
            "learning": [1],
            "profiles": [2],
        }
    )

    assert abstract == "machine learning profiles"


def test_hashing_embedding_model_returns_768_dimensions():
    model = ScientificEmbeddingModel("hashing")
    embeddings = model.encode(["scientific profile recommendation", "graph collaboration network"])

    assert embeddings.shape == (2, 768)


def test_build_graph_exports_graphsage_files(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "graphsage"
    work = {
        "id": "https://openalex.org/W1",
        "title": "Graph recommendation for young researchers",
        "topics": [{"display_name": "Recommender Systems"}],
        "primary_location": {"source": {"id": "https://openalex.org/S1", "display_name": "Test Journal"}},
        "authorships": [
            {
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Alice",
                    "works_count": 10,
                    "cited_by_count": 20,
                },
                "institutions": [{"display_name": "Astana IT University"}],
                "countries": ["KZ"],
            },
            {
                "author": {
                    "id": "https://openalex.org/A2",
                    "display_name": "Bob",
                    "works_count": 8,
                    "cited_by_count": 12,
                },
                "institutions": [{"display_name": "Astana IT University"}],
                "countries": ["KZ"],
            },
        ],
    }
    dataset_path.write_text(json.dumps(work), encoding="utf-8")

    build_graph(Namespace(input=str(dataset_path), output_dir=str(output_dir)))

    assert (output_dir / "author_nodes.json").exists()
    assert (output_dir / "author_edges.csv").exists()
    assert "A1,A2" in (output_dir / "author_edges.csv").read_text(encoding="utf-8")


def test_split_dataset_is_temporal_and_removes_duplicates(tmp_path):
    input_path = tmp_path / "works.jsonl"
    output_dir = tmp_path / "splits"
    records = [
        {"id": "W1", "publication_year": 2023, "title": "train"},
        {"id": "W1", "publication_year": 2023, "title": "duplicate"},
        {"id": "W2", "publication_year": 2024, "title": "validation"},
        {"id": "W3", "publication_year": 2025, "title": "test"},
    ]
    input_path.write_text(
        chr(10).join(json.dumps(item) for item in records),
        encoding="utf-8",
    )

    metadata = split_dataset(input_path, output_dir, 2023, 2024, 2025)

    assert metadata["duplicates_removed"] == 1
    assert metadata["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert json.loads((output_dir / "train.jsonl").read_text())["title"] == "train"
    assert json.loads((output_dir / "validation.jsonl").read_text())["title"] == "validation"
    assert json.loads((output_dir / "test.jsonl").read_text())["title"] == "test"
