# ECR Academic Visibility MVP

MVP for the dissertation topic: "Machine learning model for personalized promotion of an early-career researcher's academic profile in digital academic networks."

## What changed

The original project was a single-file experiment that downloaded papers and compared embeddings. This MVP turns it into a small but extensible platform backend:

- FastAPI service with dashboard and collaborator recommendation endpoints.
- Simple frontend dashboard served by FastAPI.
- ECR-aware ranking logic with relevance, diversity, and early-career boost.
- Larger seed dataset for researchers and opportunities.
- OpenAlex ingestion script for generating richer academic seed data.
- Tests for API and recommendation behavior.
- Docker and docker-compose for reproducible startup.

## MVP scope

Current MVP focuses on a realistic baseline that is easy to run locally and easy to defend academically:

- Researcher profiling from structured metadata.
- Opportunity recommendation for journals, conferences, grants, and mentors.
- Collaborator recommendation with semantic similarity and complementary skills.
- Cold-start handling for researchers with very low publication counts.
- Transparent explanations for every recommendation.

## Architecture

```text
FastAPI API
  -> Recommender service
      -> TF-IDF baseline encoder
      -> ECR-aware re-ranker
      -> Researcher/opportunity seed datasets
```

This baseline is intentionally simpler than the final dissertation target. It creates a solid experimental platform where you can later swap in:

- SPECTER2 or SciBERT embeddings
- pgvector for ANN retrieval
- PyTorch Geometric HAN/HetGNN collaborator modeling
- fairness constraints and exposure metrics
- OpenAlex / ORCID ingestion pipelines

## API endpoints

- `GET /health`
- `GET /`
- `GET /researchers`
- `GET /opportunities`
- `GET /researchers/{id}/dashboard`
- `GET /researchers/{id}/collaborators`

## Run locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## Run with Docker

```bash
docker compose up --build
```

API:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## Tests

```bash
pytest
```

## Generate richer research data from OpenAlex

When you want to move beyond the built-in seed CSVs, run:

```bash
python scripts/fetch_openalex_seed.py
```

This generates:

- `data/openalex_researchers_generated.csv`
- `data/openalex_works_generated.csv`

## Why this is a better dissertation MVP

This version aligns much better with your literature review and technical concept:

- It models visibility promotion instead of only paper-to-paper similarity.
- It includes ECR-specific logic and cold-start handling.
- It exposes explainable ranking signals, which helps for thesis evaluation.
- It now has a lightweight frontend, so you can demo the system without Swagger only.
- It gives you a clean place to compare baseline vs advanced models.

## Recommended next milestones

1. Replace TF-IDF with SPECTER2 embeddings for profile/opportunity matching.
2. Add OpenAlex ingestion for real researcher and publication metadata.
3. Store embeddings in PostgreSQL + pgvector.
4. Add a heterogeneous graph module for collaborator recommendations.
5. Evaluate with Precision@K, NDCG, cold-start coverage, and exposure fairness metrics.
