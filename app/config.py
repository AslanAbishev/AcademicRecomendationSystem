from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "ECR Academic Visibility MVP"
    environment: str = "development"
    data_dir: Path = BASE_DIR / "data"
    openalex_mailto: str = "researcher@example.com"
    openalex_api_key: str | None = None
    database_url: str = "postgresql://ecr:ecr@localhost:5432/ecr_mvp"
    registration_store_path: Path = BASE_DIR / "data" / "registered_profiles.json"
    user_store_path: Path = BASE_DIR / "data" / "users.json"
    promotion_snapshot_store_path: Path = BASE_DIR / "data" / "promotion_snapshots.json"
    promotion_action_store_path: Path = BASE_DIR / "data" / "promotion_actions.json"
    graphsage_embeddings_path: Path = BASE_DIR / "data" / "graphsage" / "model" / "graphsage_author_embeddings.npz"
    graphsage_model_path: Path = BASE_DIR / "data" / "graphsage" / "model" / "graphsage_model.pt"
    relevance_weight: float = 0.5
    diversity_weight: float = 0.3
    ecr_weight: float = 0.2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
