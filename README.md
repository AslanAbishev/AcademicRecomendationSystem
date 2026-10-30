# Academic Visibility Platform

[![CI and release image](https://github.com/AslanAbishev/AcademicRecomendationSystem/actions/workflows/ci.yml/badge.svg)](https://github.com/AslanAbishev/AcademicRecomendationSystem/actions/workflows/ci.yml)

Project for the dissertation topic: "Machine learning model for personalized promotion of an early-career researcher's academic profile in digital academic networks."

Repository: https://github.com/AslanAbishev/AcademicRecomendationSystem

## Technology rationale

| Technology | Role | Selection reason |
| --- | --- | --- |
| Python 3.12 | Main language | Mature data and ML ecosystem, rapid research prototyping, readable experiments |
| FastAPI | API and static application host | Typed request models, automatic OpenAPI documentation, simple testing |
| PostgreSQL + pgvector | Research data and vector search | Relational integrity with vector similarity in one reproducible service |
| OpenAlex | Publication and author metadata | Open scholarly graph with documented API and real identifiers |
| scikit-learn | TF-IDF, BM25 support, LSA/SVD and evaluation | Stable baselines that can be compared with transformer and graph methods |
| PyTorch + Transformers | SciBERT, SPECTER2 and GraphSAGE experiments | Established scientific ML tooling and reusable pretrained models |
| Docker Compose | Reproducible local environment | Consistent API, database, Redis, and optional ML profile |
| pytest | Automated tests | Lightweight unit and API testing with fixtures and temporary storage |
| GitHub Actions | CI and tagged release images | Native integration with the public repository and pull requests |

## Current scope

The app is now a personal cabinet for an early-career researcher. A user registers with minimal manual input, connects academic identifiers, and receives real OpenAlex-based profile enrichment, recommendations, and promotion actions.

- Registration and authorization with persisted local JSON storage.
- ORCID/OpenAlex enrichment for publications, citation metrics, topics, country, and affiliation.
- Personal cabinet with recent works, pgvector paper recommendations, venue/source recommendations, collaborator candidates, connected networks, and promotion actions.
- Collaborator affiliation is taken from the author's dated OpenAlex authorship where possible, checked against the current author profile when evidence is old, and shown with its source/date in the UI. It is not a live employment verification.
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
- `GET /experiments/collaborator-evaluation`
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

The automated suite covers the API, registration,
recommendation logic, temporal dataset splitting, embeddings, and graph export.

## CI/CD

The workflow is stored in `.github/workflows/ci.yml`.

- Every push to `main` and every pull request runs syntax verification, critical Ruff checks, and all pytest tests.
- After tests pass, GitHub Actions builds a lightweight API Docker image with `INSTALL_ML=false`.
- Tags matching `v*` trigger a delivery job that publishes the verified image to GitHub Container Registry.
- Heavy transformer and GraphSAGE experiments remain in the optional `ml` Compose profile and are not downloaded during the normal CI quality gate.

This design separates fast code validation from resource-intensive dissertation experiments while keeping tagged releases reproducible.

## Repository workflow

- Use short-lived `feature/*` and `fix/*` branches.
- Open a pull request for review before merging to `main`.
- Track defects and research improvements with the supplied GitHub issue templates.
- Follow the reproducibility and data-handling rules in `CONTRIBUTING.md`.

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

This writes a small `data/openalex_dataset.jsonl` preview. Use the cursor-based research pipeline below for larger collections.

## Advanced Research Pipeline

The project includes a separate advanced pipeline for dissertation experiments with PostgreSQL, pgvector, scientific embeddings, and GraphSAGE-ready graph data.

Start PostgreSQL + pgvector and initialize the schema:

```bash
docker compose up -d postgres
docker compose run --rm api python scripts/research_pipeline.py init-db
```

Collect a real temporal OpenAlex dataset. The yearly files keep the train/test period balanced; `--resume` adds only missing OpenAlex IDs after an interrupted run. The stored API key in `.env` is read locally and must never be committed.

```bash
for YEAR in 2020 2021 2022 2023 2024 2025; do
  docker compose run --rm api python scripts/research_pipeline.py collect-openalex \
    --query "recommender systems" --from-year "$YEAR" --to-year "$YEAR" \
    --max-works 2000 --per-page 200 --resume --output "data/openalex_${YEAR}.jsonl"
done
docker compose run --rm api python scripts/merge_openalex_yearly.py \
  --inputs data/openalex_2020.jsonl data/openalex_2021.jsonl data/openalex_2022.jsonl data/openalex_2023.jsonl data/openalex_2024.jsonl data/openalex_2025.jsonl \
  --output data/openalex_temporal_dataset.jsonl
```

For PowerShell, replace the shell loop with `foreach ($year in 2020..2025)` and use backticks for multiline commands. Both the original and enriched dataset are ignored by Git.

The broad yearly search alone produces few authors who appear in both training and test years. Build an author-linked cohort from repeat authors in the seed corpus. The output contains only works returned by OpenAlex and is deduplicated by work ID.

```bash
docker compose run --rm api python scripts/expand_author_cohort.py \
  --input data/openalex_temporal_dataset.jsonl \
  --output data/openalex_author_cohort_trainseed.jsonl \
  --max-authors 120 --max-works-per-author 100 --seed-train-through 2023
docker compose run --rm api python scripts/inspect_dataset.py \
  --input data/openalex_author_cohort_trainseed.jsonl \
  --output data/evaluation/dataset_quality.json
```

Additional scientific activity can be attached by DOI from Crossref (citation, funder/award and license metadata) and optionally from the Semantic Scholar Academic Graph API (citations, influential citations and open access). Citation totals from different providers are kept in separate fields and are never silently combined. Semantic Scholar may limit anonymous calls; set `SEMANTIC_SCHOLAR_API_KEY` locally if needed.

```bash
docker compose run --rm api python scripts/enrich_crossref.py \
  --input data/openalex_author_cohort_trainseed.jsonl \
  --output data/openalex_crossref_trainseed_enriched.jsonl --limit 200
docker compose run --rm api python scripts/enrich_semantic_scholar.py \
  --input data/openalex_author_cohort_trainseed.jsonl \
  --output data/openalex_semantic_scholar_enriched.jsonl --limit 500
```

The collaborator benchmark uses earlier publications to build author texts and tests whether new coauthor links appear later. It compares random and popularity controls with TF-IDF, BM25, LSA/SVD, semantic hashing, their fixed-weight hybrid, and a simple low-visibility fairness reranking of that hybrid. Models share the same candidate pool and the same held-out positives. `fairness_exposure_gap` is the absolute difference in mean discounted top-K exposure per eligible author between low- and high-publication groups; it is **not** a validated early-career label. Coverage is the share of eligible authors recommended at least once.

```bash
docker compose run --rm api python scripts/split_openalex_temporal.py \
  --input data/openalex_author_cohort_trainseed.jsonl --output-dir data/splits \
  --train-through 2023 --validation-year 2024 --test-from 2025
docker compose run --rm api python scripts/evaluate_collaborators.py \
  --train data/splits/train.jsonl --test data/splits/test.jsonl \
  --k 10 --max-authors 3000 --max-queries 100 \
  --output data/evaluation/collaborators.json
```

The benchmark evaluates local text baselines. It does not claim that SciBERT, SPECTER2 or GraphSAGE have been evaluated by these five ranking metrics. To compare them under the same protocol, recompute their embeddings from train-only inputs and rank the identical candidate set.

### Multi-field data preparation

The initial corpus was collected with a recommender-systems query and is not representative of other disciplines. The multi-field collector uses the OpenAlex **primary field** classification for Computer Science, Medicine, Physics and Astronomy, Engineering, Social Sciences, and Agricultural and Biological Sciences. It saves separate files per field/year, resumes incomplete buckets, and uses cursor paging with at most 100 works per page. The selection is recent-first within each year, so it is a deliberately bounded research sample, not a random or population-representative sample. OpenAlex topic and affiliation mappings are algorithmic, not independently verified.

```bash
python scripts/collect_multifield_openalex.py --works-per-field-year 300
python scripts/prepare_multifield_dataset.py
python scripts/freeze_temporal_cohort.py
```

If one-time field/year sampling leaves too few future collaborations in a discipline, collect additional longitudinal works for authors selected **only from training years**. The author/year slices have completion markers and can resume after interruption. This is an intentional active-author cohort, not a population sample; report its selection when interpreting results.

```bash
python scripts/expand_multifield_longitudinal.py --input data/openalex_multifield_dataset.jsonl --per-field 5 --per-author-year 25
python scripts/prepare_multifield_dataset.py --extra-dir data/multifield/longitudinal
python scripts/freeze_temporal_cohort.py
```

The preparation step merges the collected works with the existing real corpus by OpenAlex ID and writes `data/evaluation/multifield_quality.json`. Its audit checks work/author/institution ID structure, DOI format, ORCID checksum, missing mapped institutions, and mapped names that do not occur in raw affiliation text. The latter are **review flags**, not proven errors; a valid ORCID checksum does not verify the identity match. `data/multifield/coauthor_events.jsonl` records dated author pairs with work evidence, while `data/multifield/affiliation_review.jsonl` records review cases for triage. `data/multifield/identity_review.jsonl` lists retrospective ORCID/OpenAlex conflicts; only conflicts visible in works published by the training cutoff (`data/multifield/identity_review_asof_2023.jsonl`) exclude authors from derived graph labels. The raw merged corpus remains intact. The model-ready file removes missing or train-conflicting author IDs and works with more than 20 authors, but it still contains present-day OpenAlex metadata and **must not be used directly as historical training features**.

For a predictive experiment, train on `data/multifield/frozen/train_asof_2023.jsonl` only. It retains publication text, dated authorship IDs, and raw affiliation strings while stripping current citation totals, author ORCIDs and topic assignments. Use `validation_new_coauthors.jsonl` for 2024 tuning and `test_new_coauthors.jsonl` for a final 2025 evaluation; both are under `data/multifield/frozen/`. The manifest records counts and SHA-256 hashes. Do not fit text vectorizers, graph embeddings, feature scalers or candidate selection using future works. This is a **publication-year proxy**, not a true historical API snapshot: OpenAlex records were retrieved later and may have been corrected since publication. Future coauthorship is a proxy label, not measured promotion impact. All collected records, API keys and reports remain outside Git.

Run the frozen baseline comparison and build the train-only graph:

```bash
python -m scripts.run_frozen_benchmark
python scripts/research_pipeline.py build-graph --input data/multifield/frozen/train_asof_2023.jsonl --output-dir data/multifield/frozen/graphsage_train
```

The benchmark fits TF-IDF and LSA on train-period author profiles, compares all rankers on the same sampled train-author candidate pool, selects by 2024 NDCG@10, and reports 2025 metrics once. It excludes 2024 coauthor pairs from 2025 candidates. Its JSON report is `data/evaluation/frozen_benchmark.json`. In the October 2026 local run (3,000 candidate authors, 100 positive-link query authors, seed 42), hybrid won validation with NDCG@10 0.257310 and achieved test NDCG@10 0.273897, Precision@10 0.061, Recall@10 0.4675, coverage 0.221, and exposure gap 0.001588. LSA/SVD scored higher than hybrid on the test, but was not selected by validation; changing the choice after seeing test metrics would invalidate the protocol. These conditional-query results do not generalize to every new user.

GraphSAGE was trained in the Docker ML image on the frozen graph (56,316 train-period authors, 302,526 directed coauthor edges, 30 epochs, seed 42). Reproduce the run and compare it against the same candidate pool:

```bash
docker compose --profile ml run --rm --no-deps -e OMP_NUM_THREADS=4 ml python scripts/train_graphsage.py --graph-dir data/multifield/frozen/graphsage_train --validation-labels data/multifield/frozen/validation_new_coauthors.jsonl --test-labels data/multifield/frozen/test_new_coauthors.jsonl --output-dir data/multifield/frozen/graphsage_model --epochs 30 --seed 42
python -m scripts.run_frozen_benchmark --graphsage-embeddings data/multifield/frozen/graphsage_model/graphsage_author_embeddings.npz --output data/evaluation/frozen_benchmark_graphsage.json
```

This run's GraphSAGE NDCG@10 was 0.048464 on 2024 validation and 0.028508 on 2025 test, versus hybrid 0.257310 and 0.273897 respectively. GraphSAGE also had lower test Precision@10 (0.009 versus 0.061) and Recall@10 (0.073333 versus 0.4675). Its sampled-negative link-prediction AUC was 0.2381/0.2204 on validation/test. The graph-only features are sparse (training-period work count, no contemporaneous citation totals, no inferred country for many authors); these results do not justify deployment. The trained checkpoint and embeddings remain local research artifacts, and live recommendations continue to use the hybrid formula. Live per-request candidates and fitted text representations still differ from the offline benchmark, so these scores are not online performance claims.

Graph diagnostics should use **2024 validation labels only** while developing a revised model:

```bash
python -m scripts.diagnose_graphsage
python scripts/evaluate_collaborators.py --train data/multifield/frozen/train_asof_2023.jsonl --test data/multifield/frozen/validation_new_coauthors.jsonl --graphsage-embeddings data/multifield/frozen/graphsage_model/graphsage_author_embeddings.npz --train-graph-edges data/multifield/frozen/graphsage_train/author_edges.csv --output data/evaluation/frozen_validation_graph_diagnostics.json
```

The validation audit found 832 of 1,024 new coauthor pairs already had a common train-period neighbor. In the shared 3,000-author candidate pool, common neighbors reached NDCG@10 0.475444 and Adamic-Adar 0.481179, above hybrid 0.257310 and GraphSAGE 0.048464; graph methods had lower coverage and a higher low/high-work-count exposure gap. The graph checkpoint shows feature collapse: 731 of 1,024 globally sampled author pairs had cosine similarity at least 0.999. This explains why global random-negative AUC and query-matched comparisons disagree; neither is a substitute for top-K evaluation. Do not tune a revised model using the already-inspected 2025 test results. Obtain a new untouched holdout or define a new prospective temporal protocol before claiming an improvement.

A first supervised graph-text research model is trained without the inspected 2025 test:

```bash
python -m scripts.train_graph_text_ranker
```

`TemporalGraphTextRanker-v1` fits a regularized logistic ranker on 2023 new coauthor pairs using **only 2020-2022** publication text, coauthor topology and work counts as features. It recomputes features from works through 2023 for 2024 validation, compares the same candidate/query pool with TF-IDF, hybrid and Adamic-Adar, and chooses a small low-work-count bonus on validation. A safe JSON coefficient artifact is written to `data/models/temporal_graph_text_v1.json`; a protocol/metric report is written to `data/evaluation/temporal_graph_text_v1.json`. In the local run, it used 8,281 pair examples from 100 eligible 2023 query authors. On the 2024 validation cohort its NDCG@10 was 0.451223, coverage 0.148 and exposure gap 0.003094: better NDCG than the text hybrid (0.257310), but worse than Adamic-Adar (0.481179), with lower coverage and worse exposure gap. The fitted text coefficient is small relative to the graph terms, so this is a research prototype, **not evidence of a novel or superior production model**. Validation is used for bonus selection and is not an untouched final estimate. The 2025 labels are not read by this training script; get a new prospective holdout before deployment claims.

`TemporalGraphTextRanker-v2` adds up to four title-and-abstract excerpts per author and reserves a small number of topical candidates in the recommendation list. Its coefficients still fit 2023 links using features through 2022; the low-work-count bonus and topical slots are selected on 2024 validation under an Adamic-Adar coverage/exposure constraint. The 2025 labels are not loaded. Reproduce its frozen training, partial-2026 collection, and one-time holdout evaluation in this order:

```bash
python -m scripts.train_graph_text_ranker_v2
python -m scripts.collect_2026_holdout
python -m scripts.prepare_2026_holdout
python -m scripts.evaluate_2026_holdout
```

The collection locks the v2 model hash before calling OpenAlex and selects 60 seed authors solely from their 2023 work counts. It collects publications dated January 1-September 30, 2026 (481 distinct works in the local run); the preparation step strips current citations, ORCIDs and topics from the pre-2026 feature snapshot and identifies 398 new undirected links between previously observed authors. On 2024 validation, the selected v2 variant had NDCG@10 0.414625, coverage 0.165 and exposure gap 0.002389, versus Adamic-Adar 0.481179/0.158667/0.002695. On the **locked partial-2026 holdout**, 94 queries and 156 directed relevant links were evaluable in the fixed 3,000-author candidate pool: v2 NDCG@10 0.334526, coverage 0.149333, exposure gap 0.002446; Adamic-Adar 0.374808/0.143/0.002743; and the previous hybrid 0.238053/0.164/0.002142. The [local JSON report](data/evaluation/holdout_2026_v2.json) records all metrics and hashes. This is a newly collected **retrospective**, conditional, author-seeded, partial-year research test, not a prospective field trial or proof of broad promotion impact. V2 improves on the text hybrid in this cohort but does not beat the strongest graph-only baseline on ranking quality, so it is **not deployed**. Do not use the inspected 2026 labels to tune another version and then cite the same holdout as independent evidence.

October 2026 local collection: all 36 field/year buckets completed with 10,800 fetched works. Together with the earlier author cohort this gave 27,198 unique works. Longitudinal enrichment for 30 train-period seed authors returned 2,669 author/year records, most overlapping existing works; the final raw corpus contains 27,665 unique works and 95,699 distinct nonempty OpenAlex author IDs. The retrospective audit found 161 ORCIDs linked to multiple author IDs; 76 conflicts were visible by the 2023 training cutoff, excluding 153 implicated author IDs from derived graph labels. The model-ready corpus has 26,755 works: 17,444 train, 4,705 validation and 4,606 test works. The frozen benchmark has 1,024 novel validation-period and 567 novel test-period undirected coauthor pairs between train-observed authors, with test pairs also excluding those seen in validation. Computer Science has 27 test-period pairs; this is still too small for strong field-specific conclusions. There are 8,012 authorships without an OpenAlex author ID and 18,336 without a mapped institution; these remain missing rather than invented. The 46,874 raw/mapped affiliation text discrepancies are review flags, many of which may simply be abbreviations or alternative institution names.

### Collected corpus and benchmark (October 2026)

- OpenAlex seed: 12,003 unique works from 2020-2025; author-linked expansion using 120 seed authors selected from 2020-2023 only: 17,677 unique works and 53,197 distinct author IDs, all with publication years and no duplicate work IDs.
- Data completeness in the expanded corpus: 17,625 titles, 17,420 DOIs, 16,844 indexed abstracts, 17,640 topic lists and 17,557 records with authorships. Missing values remain missing; they are not fabricated.
- Temporal split: 11,350 train works (2020-2023), 3,267 validation works (2024), 3,060 test works (2025).
- Crossref DOI enrichment: 200 of 200 attempted records matched in the local enriched export. Semantic Scholar returned HTTP 429 without its own key; no Semantic Scholar values were fabricated.
- Ranking evaluation: 100 authors with at least one held-out positive in the selected train-only candidate pool, 169 held-out links, 3,000 candidate authors, seed 42, K=10. The query cohort is conditional on having evaluable future links; scores should not be generalized to all users.

| Ranker | Precision@10 | Recall@10 | NDCG@10 | Coverage | Exposure gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.000 | 0.000 | 0.000 | 0.285 | 0.00005 |
| Popularity | 0.002 | 0.012 | 0.012 | 0.006 | 0.00356 |
| TF-IDF | 0.072 | 0.409 | 0.228 | 0.216 | 0.00191 |
| BM25 | 0.077 | 0.432 | 0.256 | 0.209 | 0.00252 |
| LSA/SVD | 0.061 | 0.365 | 0.243 | 0.215 | 0.00076 |
| Semantic hashing | 0.071 | 0.409 | 0.246 | 0.212 | 0.00206 |
| Hybrid | 0.071 | 0.409 | 0.263 | 0.218 | 0.00189 |
| Fairness reranked hybrid | 0.046 | 0.273 | 0.156 | 0.207 | 0.00076 |

Fairness reranking reduced the measured exposure gap but also reduced ranking quality in this experiment, so its weight needs calibration on validation data before being recommended as a default. These are offline collaborator metrics, not measured profile promotion or citation uplift. The fully reproducible JSON report is written locally to `data/evaluation/collaborators.json` and excluded from Git because it depends on collected data.

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

## License

This project is released under the MIT License. See `LICENSE`.

## Current experiment status

- Temporal splitting is implemented in `scripts/split_openalex_temporal.py`.
- GraphSAGE training exports author embeddings, a model checkpoint, and link-prediction AUC for train/validation/test.
- The online collaborator recommender loads `data/graphsage/model/graphsage_author_embeddings.npz`, uses the selected SciBERT/SPECTER2 embedding for candidate text, and combines transformer similarity, GraphSAGE similarity, OpenAlex graph evidence, diversity, ECR, and fairness signals.
- Known graph authors use stored GraphSAGE embeddings; unseen live authors use inductive inference over a local graph built from the current profile and OpenAlex evidence works.
- Each live collaborator explanation reports separate model scores; if an optional artifact or transformer cannot be loaded, the service reports the fallback reason instead of hiding it.
- GraphSAGE temporal training now uses train-period authors only and zeroes current citation totals when validation/test graphs are provided. Existing saved checkpoints predate this correction and must be retrained before citing temporal AUC.
