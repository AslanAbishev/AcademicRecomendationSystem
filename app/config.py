from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "ECR Academic Visibility MVP"
    environment: str = "development"
    data_dir: Path = BASE_DIR / "data"
    openalex_mailto: str = "researcher@example.com"
    relevance_weight: float = 0.5
    diversity_weight: float = 0.3
    ecr_weight: float = 0.2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
