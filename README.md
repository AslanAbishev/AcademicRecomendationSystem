# Academic Visibility Platform

Project for the dissertation topic: "Machine learning model for personalized promotion of an early-career researcher's academic profile in digital academic networks."

## Current scope

The app is now a personal cabinet for an early-career researcher. A user registers with minimal manual input, connects academic identifiers, and receives real OpenAlex-based profile enrichment, recommendations, and promotion actions.

- Registration and authorization with persisted local JSON storage.
- ORCID/OpenAlex enrichment for publications, citation metrics, topics, country, and affiliation.
- Personal cabinet with recent works, pgvector paper recommendations, venue/source recommendations, collaborator candidates, connected networks, and promotion actions.
- Visibility Twin: a gap model that compares the current profile with a target early-career visibility profile.
- Hybrid ML recommender with comparable model scores: TF-IDF cosine, BM25, LSA/SVD, semantic hashing fallback, SciBERT/SPECTER2 pgvector search, graph collaboration evidence, diversity, ECR boost, and fairness-aware reranking.
- Promotion progress tracker with snapshots, completed-action feedback, and metric deltas over time.
- OpenAlex dataset export script for future experiments.

No synthetic researchers or fake opportunity catalog items are used in runtime recommendations. Venues, works, and collaborator candidates come from OpenAlex.

## Architecture

```text
FastAPI API + static frontend
  -> Account service
  -> Registration/Profile enrichment service
      -> ORCID/OpenAlex author lookup
      -> derived metrics and domain inference
  -> Realtime recommender service
      -> TF-IDF baseline
      -> BM25 information retrieval baseline
      -> LSA/SVD latent semantic baseline
      -> semantic hashing fallback
      -> hybrid ensemble model comparison
      -> SciBERT/SPECTER2 pgvector paper recommendations
      -> OpenAlex co-author/work graph evidence
      -> ECR and fairness-aware reranking
  -> Promotion progress service
      -> snapshots
      -> completed action feedback
      -> deltas for visibility/promotion metrics
```

The dashboard can switch between `hashing`, `scibert`, and `specter2` for pgvector paper recommendations. Runtime recommendations expose per-model scores, while the separate `ml` Docker profile remains useful for batch embedding and GraphSAGE jobs.

## Web workspace

The frontend uses separate hash-routed screens with a persistent sidebar:
Overview, My publications, Discover research, Collaborators, Growth plan,
My profile, Find a researcher, and Recommendation settings. Browser Back/Forward
and refreshing a section preserve its URL.

- Sign in and registration are separate screens; optional onboarding fields are collapsed.
- Publications support local filtering and pagination over the records returned by the API.
- Papers and venues have separate tabs; ranking explanations expand on demand.
- Collaborator profiles open in an accessible dialog without replacing the current page.
- Public researcher browsing is labeled clearly and does not show personal progress or allow snapshots.
- Navigation adapts to a mobile drawer and supports keyboard navigation and reduced motion.

The frontend remains dependency-free: `index.html` defines the shell,
`workspace.js` handles navigation and presentation, and `app.js` handles API
requests and rendering. FastAPI serves all assets; no Node build step is needed.
With the Compose bind mount, frontend edits are visible after a browser refresh.

## API endpoints

- `GET /health`
- `GET /`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me/cabinet`
- `POST /auth/me/snapshots`
- `GET /auth/me/progress`
- `POST /auth/me/actions`
- `GET /experiments/ml-plan`
- `GET /experiments/dataset-stats`
- `GET /experiments/vector-search?q=...`
- `GET /researchers/search?q=...`
- `GET /researchers/{id}/dashboard?embedding_model=specter2`
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

The first API build installs transformer dependencies so SciBERT/SPECTER2 can run inside the dashboard. HuggingFace model files are shared through the `hf_cache` Docker volume.

API:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## Tests

```bash
python -m pytest
```

For manual UI checks without modifying real accounts, run
`python -m tests.ui_preview` and open `http://127.0.0.1:8001/`.
This separate test server uses explicitly labeled fixtures and temporary storage.
It does not test live OpenAlex data or transformer inference and is never loaded
by the normal application.

## Generate richer research data from OpenAlex

To export a real JSONL slice for offline experiments:

```bash
python scripts/build_openalex_dataset.py --query "computer science artificial intelligence cybersecurity" --per-page 200
```

This writes `data/openalex_dataset.jsonl`. Increase the page size or add pagination later for a dissertation-scale dataset.

## Advanced Research Pipeline

The project includes a separate advanced pipeline for dissertation experiments with PostgreSQL, pgvector, scientific embeddings, and GraphSAGE-ready graph data.

Start PostgreSQL + pgvector and initialize the schema:

```bash
docker compose up -d postgres
docker compose run --rm api python scripts/research_pipeline.py init-db
```

Collect a real temporal OpenAlex dataset and load it into PostgreSQL:

```bash
docker compose run --rm api python scripts/research_pipeline.py collect-openalex ^
  --query "computer science artificial intelligence cybersecurity recommender systems" ^
  --from-year 2020 ^
  --to-year 2025 ^
  --max-works 10000 ^
  --filter has_abstract:true ^
  --output data/openalex_temporal_dataset.jsonl ^
  --db
```

For a balanced temporal experiment, collect a separate slice for each year and merge them without duplicates:

```bash
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2020 --to-year 2020 --max-works 300 --output data/openalex_2020.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2021 --to-year 2021 --max-works 300 --output data/openalex_2021.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2022 --to-year 2022 --max-works 300 --output data/openalex_2022.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2023 --to-year 2023 --max-works 300 --output data/openalex_2023.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2024 --to-year 2024 --max-works 300 --output data/openalex_2024.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2025 --to-year 2025 --max-works 300 --output data/openalex_2025.jsonl
docker compose run --rm api python scripts/merge_openalex_yearly.py --inputs data/openalex_2020.jsonl data/openalex_2021.jsonl data/openalex_2022.jsonl data/openalex_2023.jsonl data/openalex_2024.jsonl data/openalex_2025.jsonl --output data/openalex_temporal_dataset.jsonl
```

For a balanced temporal experiment, collect a separate slice for each year and merge them without duplicates:

```bash
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2020 --to-year 2020 --max-works 300 --output data/openalex_2020.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2021 --to-year 2021 --max-works 300 --output data/openalex_2021.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2022 --to-year 2022 --max-works 300 --output data/openalex_2022.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2023 --to-year 2023 --max-works 300 --output data/openalex_2023.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2024 --to-year 2024 --max-works 300 --output data/openalex_2024.jsonl
docker compose run --rm api python scripts/research_pipeline.py collect-openalex --query "recommender systems" --from-year 2025 --to-year 2025 --max-works 300 --output data/openalex_2025.jsonl
docker compose run --rm api python scripts/merge_openalex_yearly.py --inputs data/openalex_2020.jsonl data/openalex_2021.jsonl data/openalex_2022.jsonl data/openalex_2023.jsonl data/openalex_2024.jsonl data/openalex_2025.jsonl --output data/openalex_temporal_dataset.jsonl
```

Current local sample already collected:

- `data/openalex_cs_ai_cyber_2020_2025_sample.jsonl`
- 1,000 real OpenAlex works
- 2,674 authors
- 2,721 authorships
- 1,000 stored embeddings
- 19,928 co-author graph edges

Generate lightweight Docker-safe embeddings:

```bash
docker compose run --rm api python scripts/embed_openalex_dataset.py ^
  --input data/openalex_temporal_dataset.jsonl ^
  --model hashing ^
  --output data/embeddings/openalex_hashing_embeddings.npz ^
  --db
```

For real SPECTER2 or SciBERT embeddings, use the heavier ML profile:

```bash
docker compose --profile ml build ml
docker compose --profile ml run --rm ml python scripts/embed_openalex_dataset.py --model scibert --limit 1000 --db
docker compose --profile ml run --rm ml python scripts/embed_openalex_dataset.py --model specter2 --limit 1000 --db
```

The `ml` profile uses a persistent `hf_cache` Docker volume so HuggingFace models are not downloaded again on every run.

Export a GraphSAGE-ready co-author graph:

```bash
docker compose run --rm api python scripts/research_pipeline.py build-graph ^
  --input data/openalex_temporal_dataset.jsonl ^
  --output-dir data/graphsage ^
  --db
```

Create a leakage-aware temporal split before evaluation. The default split uses publications through 2023 for training, 2024 for validation, and 2025+ for testing. The command intentionally fails if one of the periods is empty, because duplicating or inventing years would invalidate the experiment:

```bash
docker compose run --rm api python scripts/split_openalex_temporal.py ^
  --input data/openalex_temporal_dataset.jsonl ^
  --output-dir data/splits ^
  --train-through 2023 ^
  --validation-year 2024 ^
  --test-from 2025
```

Build one co-author graph per split:

```bash
docker compose run --rm api python scripts/research_pipeline.py build-graph --input data/splits/train.jsonl --output-dir data/graphsage/train
docker compose run --rm api python scripts/research_pipeline.py build-graph --input data/splits/validation.jsonl --output-dir data/graphsage/validation
docker compose run --rm api python scripts/research_pipeline.py build-graph --input data/splits/test.jsonl --output-dir data/graphsage/test
```

Train GraphSAGE on the training graph and evaluate temporal link prediction on validation and test graphs:

```bash
docker compose --profile ml run --rm ml python scripts/train_graphsage.py ^
  --graph-dir data/graphsage/train ^
  --validation-graph-dir data/graphsage/validation ^
  --test-graph-dir data/graphsage/test ^
  --output-dir data/graphsage/model ^
  --epochs 50 ^
  --seed 42
```

Check database stats:

```bash
docker compose run --rm api python scripts/research_pipeline.py db-stats
```

## Dissertation fit

This version aligns with the literature review and technical concept because it does more than recommend papers:

- It models profile promotion as measurable gaps and actions.
- It supports cold-start researchers through profile text and connected identifiers.
- It explains recommendations through comparable TF-IDF, BM25, LSA/SVD, semantic hashing, hybrid, graph, diversity, and fairness signals.
- It tracks whether promotion actions improve profile metrics over time.
- It provides multiple baselines for future SPECTER2/SciBERT/GraphSAGE experiments.

## Current experiment status

- Temporal splitting is implemented in `scripts/split_openalex_temporal.py`.
- GraphSAGE training exports author embeddings, a model checkpoint, and link-prediction AUC for train/validation/test.
- The online collaborator recommender loads `data/graphsage/model/graphsage_author_embeddings.npz`, uses the selected SciBERT/SPECTER2 embedding for candidate text, and combines transformer similarity, GraphSAGE similarity, OpenAlex graph evidence, diversity, ECR, and fairness signals.
- Known graph authors use stored GraphSAGE embeddings; unseen live authors use inductive inference over a local graph built from the current profile and OpenAlex evidence works.
- Each live collaborator explanation reports separate model scores; if an optional artifact or transformer cannot be loaded, the service reports the fallback reason instead of hiding it.
- The dataset must contain several publication years. The current one-year sample is suitable for a smoke test, but not for a valid temporal experiment.
