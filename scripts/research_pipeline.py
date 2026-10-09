from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.openalex import OpenAlexClient
from app.config import settings
from app.services.research_database import ResearchDatabase


DEFAULT_QUERY = "computer science artificial intelligence cybersecurity recommender systems"


def init_db() -> None:
    db = ResearchDatabase(settings.database_url)
    db.init_schema()
    print(json.dumps({"status": "initialized", "database_url": settings.database_url}, indent=2))


def collect_openalex(args: argparse.Namespace) -> None:
    client = OpenAlexClient(mailto=settings.openalex_mailto, api_key=settings.openalex_api_key)
    db = ResearchDatabase(settings.database_url)
    if args.db:
        db.init_schema()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for work in client.iter_works(
            args.query,
            per_page=args.per_page,
            max_records=args.max_works,
            from_year=args.from_year,
            to_year=args.to_year,
            extra_filters=args.filter,
        ):
            file.write(json.dumps(work, ensure_ascii=False) + "\n")
            if args.db:
                db.upsert_work(work)
            count += 1
            if count % 100 == 0:
                print(f"collected={count}")

    result = {
        "status": "collected",
        "works": count,
        "output": str(output_path),
        "loaded_to_db": bool(args.db),
        "query": args.query,
        "from_year": args.from_year,
        "to_year": args.to_year,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_graph(args: argparse.Namespace) -> None:
    works = _read_works(Path(args.input))
    graph_dir = Path(args.output_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    db = ResearchDatabase(settings.database_url) if getattr(args, "db", False) else None
    if db:
        db.init_schema()

    author_ids: dict[str, int] = {}
    author_features: dict[str, dict] = {}
    coauthor_edges: Counter[tuple[str, str]] = Counter()
    heterogeneous_edges: list[dict] = []

    for work in works:
        work_id = _short_id(work.get("id"))
        topics = [topic.get("display_name") for topic in work.get("topics", []) if topic.get("display_name")]
        source = ((work.get("primary_location") or {}).get("source") or {})
        source_id = _short_id(source.get("id"))
        author_list: list[str] = []

        for authorship in work.get("authorships", []):
            author = authorship.get("author") or {}
            author_id = _short_id(author.get("id"))
            if not author_id:
                continue
            author_ids.setdefault(author_id, len(author_ids))
            author_list.append(author_id)
            institutions = authorship.get("institutions") or []
            countries = authorship.get("countries") or []
            author_features.setdefault(
                author_id,
                {
                    "author_id": author_id,
                    "display_name": author.get("display_name", "Unknown author"),
                    "works_count": int(author.get("works_count", 0) or 0),
                    "cited_by_count": int(author.get("cited_by_count", 0) or 0),
                    "country_code": countries[0] if countries else "",
                    "affiliation": institutions[0].get("display_name", "") if institutions else "",
                },
            )
            if not author.get("works_count"):
                author_features[author_id]["works_count"] += 1
            if not author.get("cited_by_count"):
                author_features[author_id]["cited_by_count"] += int(work.get("cited_by_count", 0) or 0)
            heterogeneous_edges.append({"source": author_id, "target": work_id, "type": "authored"})

        for left, right in combinations(sorted(set(author_list)), 2):
            coauthor_edges[(left, right)] += 1
            coauthor_edges[(right, left)] += 1

        for topic in topics:
            topic_id = "topic:" + topic.lower().replace(" ", "_")
            heterogeneous_edges.append({"source": work_id, "target": topic_id, "type": "has_topic"})

        if source_id:
            heterogeneous_edges.append({"source": work_id, "target": source_id, "type": "published_in"})

    (graph_dir / "author_nodes.json").write_text(
        json.dumps(list(author_features.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (graph_dir / "author_edges.csv").write_text(
        "source,target,weight\n"
        + "\n".join(f"{left},{right},{weight}" for (left, right), weight in coauthor_edges.items()),
        encoding="utf-8",
    )
    (graph_dir / "heterogeneous_edges.jsonl").write_text(
        "\n".join(json.dumps(edge, ensure_ascii=False) for edge in heterogeneous_edges),
        encoding="utf-8",
    )
    (graph_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": str(uuid4()),
                "works": len(works),
                "authors": len(author_ids),
                "coauthor_edges": len(coauthor_edges),
                "heterogeneous_edges": len(heterogeneous_edges),
                "purpose": "GraphSAGE-ready co-author graph and heterogeneous author-work-topic-venue graph.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if db:
        db.upsert_graph_edges(
            [
                (
                    left,
                    right,
                    "coauthor",
                    float(weight),
                    {"dataset": str(args.input), "relation": "shared_openalex_work"},
                )
                for (left, right), weight in coauthor_edges.items()
            ]
        )

    print(json.dumps({"status": "graph_exported", "output_dir": str(graph_dir)}, ensure_ascii=False, indent=2))


def db_stats() -> None:
    db = ResearchDatabase(settings.database_url)
    print(json.dumps(db.stats(), indent=2))


def _read_works(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _short_id(value: str | None) -> str:
    return value.split("/")[-1] if value else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced OpenAlex + pgvector + graph dataset pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Initialize PostgreSQL + pgvector schema.")

    collect = subparsers.add_parser("collect-openalex", help="Collect a real OpenAlex temporal dataset.")
    collect.add_argument("--query", default=DEFAULT_QUERY)
    collect.add_argument("--output", default="data/openalex_temporal_dataset.jsonl")
    collect.add_argument("--max-works", type=int, default=1000)
    collect.add_argument("--per-page", type=int, default=200)
    collect.add_argument("--from-year", type=int, default=2020)
    collect.add_argument("--to-year", type=int, default=2026)
    collect.add_argument("--filter", action="append", default=[])
    collect.add_argument("--db", action="store_true", help="Also load records into PostgreSQL.")

    graph = subparsers.add_parser("build-graph", help="Export GraphSAGE-ready graph files from JSONL dataset.")
    graph.add_argument("--input", default="data/openalex_temporal_dataset.jsonl")
    graph.add_argument("--output-dir", default="data/graphsage")
    graph.add_argument("--db", action="store_true", help="Also load co-author edges into PostgreSQL.")

    subparsers.add_parser("db-stats", help="Show PostgreSQL dataset stats.")

    args = parser.parse_args()
    if args.command == "init-db":
        init_db()
    elif args.command == "collect-openalex":
        collect_openalex(args)
    elif args.command == "build-graph":
        build_graph(args)
    elif args.command == "db-stats":
        db_stats()


if __name__ == "__main__":
    main()
