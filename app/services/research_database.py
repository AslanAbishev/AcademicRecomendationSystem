from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row


BASE_DIR = Path(__file__).resolve().parents[2]
INIT_SQL_PATH = BASE_DIR / "scripts" / "sql" / "001_init_pgvector.sql"


class ResearchDatabase:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(INIT_SQL_PATH.read_text(encoding="utf-8"))
            conn.commit()

    def upsert_work(self, work: dict[str, Any]) -> None:
        summary = self._work_summary(work)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO openalex_works (
                    work_id, title, abstract, publication_year, cited_by_count,
                    source_id, source_name, source_type, doi, landing_page_url,
                    topics, raw
                )
                VALUES (
                    %(work_id)s, %(title)s, %(abstract)s, %(publication_year)s, %(cited_by_count)s,
                    %(source_id)s, %(source_name)s, %(source_type)s, %(doi)s, %(landing_page_url)s,
                    %(topics)s::jsonb, %(raw)s::jsonb
                )
                ON CONFLICT (work_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    abstract = EXCLUDED.abstract,
                    publication_year = EXCLUDED.publication_year,
                    cited_by_count = EXCLUDED.cited_by_count,
                    source_id = EXCLUDED.source_id,
                    source_name = EXCLUDED.source_name,
                    source_type = EXCLUDED.source_type,
                    doi = EXCLUDED.doi,
                    landing_page_url = EXCLUDED.landing_page_url,
                    topics = EXCLUDED.topics,
                    raw = EXCLUDED.raw,
                    collected_at = NOW()
                """,
                summary,
            )
            for position, authorship in enumerate(work.get("authorships", [])):
                author = self._author_summary(authorship)
                if not author:
                    continue
                conn.execute(
                    """
                    INSERT INTO openalex_authors (
                        author_id, display_name, works_count, cited_by_count,
                        affiliation, country_code, raw
                    )
                    VALUES (
                        %(author_id)s, %(display_name)s, %(works_count)s, %(cited_by_count)s,
                        %(affiliation)s, %(country_code)s, %(raw)s::jsonb
                    )
                    ON CONFLICT (author_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        works_count = GREATEST(openalex_authors.works_count, EXCLUDED.works_count),
                        cited_by_count = GREATEST(openalex_authors.cited_by_count, EXCLUDED.cited_by_count),
                        affiliation = COALESCE(EXCLUDED.affiliation, openalex_authors.affiliation),
                        country_code = COALESCE(EXCLUDED.country_code, openalex_authors.country_code),
                        raw = EXCLUDED.raw,
                        collected_at = NOW()
                    """,
                    author,
                )
                conn.execute(
                    """
                    INSERT INTO openalex_authorships (work_id, author_id, author_position, raw_affiliation)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (work_id, author_id) DO UPDATE SET
                        author_position = EXCLUDED.author_position,
                        raw_affiliation = EXCLUDED.raw_affiliation
                    """,
                    (
                        summary["work_id"],
                        author["author_id"],
                        position,
                        self._raw_affiliation(authorship),
                    ),
                )
            conn.commit()

    def upsert_embedding(self, work_id: str, model_name: str, embedding: list[float], text: str) -> None:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        vector_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO work_embeddings (work_id, model_name, embedding, text_hash)
                VALUES (%s, %s, %s::vector, %s)
                ON CONFLICT (work_id, model_name) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    text_hash = EXCLUDED.text_hash,
                    created_at = NOW()
                """,
                (work_id, model_name, vector_literal, text_hash),
            )
            conn.commit()

    def upsert_graph_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_edges (source_id, target_id, edge_type, weight, evidence)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    evidence = EXCLUDED.evidence
                """,
                (source_id, target_id, edge_type, weight, json.dumps(evidence or {}, ensure_ascii=False)),
            )
            conn.commit()

    def upsert_graph_edges(self, edges: list[tuple[str, str, str, float, dict[str, Any] | None]]) -> None:
        if not edges:
            return
        rows = [
            (source_id, target_id, edge_type, weight, json.dumps(evidence or {}, ensure_ascii=False))
            for source_id, target_id, edge_type, weight, evidence in edges
        ]
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO graph_edges (source_id, target_id, edge_type, weight, evidence)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                        weight = EXCLUDED.weight,
                        evidence = EXCLUDED.evidence
                    """,
                    rows,
                )
            conn.commit()

    def list_works(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT work_id, title, abstract, publication_year, cited_by_count,
                       source_name, source_type, topics, raw
                FROM openalex_works
                ORDER BY publication_year DESC NULLS LAST, cited_by_count DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return list(rows)

    def search_similar_works(
        self,
        embedding: list[float],
        model_name: str = "hashing",
        limit: int = 10,
        exclude_work_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        vector_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        excluded = exclude_work_ids or []
        with self._connect() as conn:
            rows = conn.execute(
                """
                WITH model_embeddings AS MATERIALIZED (
                    SELECT work_id, embedding
                    FROM work_embeddings
                    WHERE model_name = %s
                )
                SELECT
                    w.work_id,
                    w.title,
                    w.abstract,
                    w.publication_year,
                    w.cited_by_count,
                    w.source_name,
                    w.source_type,
                    w.landing_page_url,
                    w.topics,
                    1 - (e.embedding <=> %s::vector) AS similarity
                FROM model_embeddings e
                JOIN openalex_works w ON w.work_id = e.work_id
                WHERE NOT (w.work_id = ANY(%s))
                  AND w.title IS NOT NULL
                  AND w.title <> ''
                  AND LOWER(w.title) <> 'untitled work'
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (model_name, vector_literal, excluded, vector_literal, limit),
            ).fetchall()
        return list(rows)

    def embedding_counts_by_model(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT model_name, COUNT(*) AS count
                FROM work_embeddings
                GROUP BY model_name
                ORDER BY model_name
                """
            ).fetchall()
        return {str(row["model_name"]): int(row["count"]) for row in rows}

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM openalex_works) AS works,
                    (SELECT COUNT(*) FROM openalex_authors) AS authors,
                    (SELECT COUNT(*) FROM openalex_authorships) AS authorships,
                    (SELECT COUNT(*) FROM work_embeddings) AS embeddings,
                    (SELECT COUNT(*) FROM graph_edges) AS graph_edges
                """
            ).fetchone()
        return dict(row or {})

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _work_summary(self, work: dict[str, Any]) -> dict[str, Any]:
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        return {
            "work_id": self._short_id(work.get("id")),
            "title": work.get("title") or "Untitled work",
            "abstract": self.abstract_from_inverted_index(work.get("abstract_inverted_index")),
            "publication_year": work.get("publication_year"),
            "cited_by_count": int(work.get("cited_by_count", 0) or 0),
            "source_id": self._short_id(source.get("id")),
            "source_name": source.get("display_name"),
            "source_type": source.get("type"),
            "doi": work.get("doi"),
            "landing_page_url": location.get("landing_page_url"),
            "topics": json.dumps(self._topic_names(work), ensure_ascii=False),
            "raw": json.dumps(work, ensure_ascii=False),
        }

    def _author_summary(self, authorship: dict[str, Any]) -> dict[str, Any] | None:
        author = authorship.get("author") or {}
        author_id = self._short_id(author.get("id"))
        if not author_id:
            return None
        institutions = authorship.get("institutions") or []
        countries = authorship.get("countries") or []
        return {
            "author_id": author_id,
            "display_name": author.get("display_name") or "Unknown author",
            "works_count": int(author.get("works_count", 0) or 0),
            "cited_by_count": int(author.get("cited_by_count", 0) or 0),
            "affiliation": institutions[0].get("display_name") if institutions else None,
            "country_code": countries[0] if countries else (institutions[0].get("country_code") if institutions else None),
            "raw": json.dumps(authorship, ensure_ascii=False),
        }

    def _topic_names(self, work: dict[str, Any]) -> list[str]:
        return [topic.get("display_name") for topic in work.get("topics", []) if topic.get("display_name")]

    def _raw_affiliation(self, authorship: dict[str, Any]) -> str | None:
        raw = authorship.get("raw_affiliation_strings") or []
        return raw[0] if raw else None

    def _short_id(self, value: str | None) -> str | None:
        return value.split("/")[-1] if value else None

    @staticmethod
    def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
        if not index:
            return ""
        positioned: list[tuple[int, str]] = []
        for token, positions in index.items():
            positioned.extend((position, token) for position in positions)
        return " ".join(token for _, token in sorted(positioned))
