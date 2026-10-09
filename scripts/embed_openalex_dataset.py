from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.services.research_database import ResearchDatabase
from app.services.scientific_embeddings import ScientificEmbeddingModel


def embed_dataset(args: argparse.Namespace) -> None:
    records = _read_records(Path(args.input), limit=args.limit)
    texts = [_work_text(record) for record in records]
    ids = [_short_id(record.get("id")) for record in records]

    model = ScientificEmbeddingModel(args.model, batch_size=args.batch_size)
    embeddings = model.encode(texts)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.array(ids),
        embeddings=embeddings,
        model=np.array([args.model]),
    )

    if args.db:
        db = ResearchDatabase(settings.database_url)
        db.init_schema()
        for work_id, text, embedding in zip(ids, texts, embeddings, strict=True):
            db.upsert_embedding(work_id, args.model, embedding.tolist(), text)

    print(
        json.dumps(
            {
                "status": "embedded",
                "model": args.model,
                "records": len(records),
                "output": str(output_path),
                "loaded_to_db": bool(args.db),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _read_records(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit else records


def _work_text(work: dict) -> str:
    topics = " ".join(topic.get("display_name", "") for topic in work.get("topics", []))
    source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name", "")
    abstract = ResearchDatabase.abstract_from_inverted_index(work.get("abstract_inverted_index"))
    return " ".join([work.get("title") or "", abstract, topics, source]).strip()


def _short_id(value: str | None) -> str:
    return value.split("/")[-1] if value else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate work embeddings for an OpenAlex JSONL dataset.")
    parser.add_argument("--input", default="data/openalex_temporal_dataset.jsonl")
    parser.add_argument("--output", default="data/embeddings/openalex_work_embeddings.npz")
    parser.add_argument("--model", default="hashing", help="hashing, scibert, specter2, or any HF model id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--db", action="store_true", help="Also store embeddings in PostgreSQL/pgvector.")
    args = parser.parse_args()
    embed_dataset(args)


if __name__ == "__main__":
    main()
