from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import settings


@lru_cache(maxsize=1)
def load_researchers() -> pd.DataFrame:
    return pd.read_csv(settings.data_dir / "researchers.csv")


@lru_cache(maxsize=1)
def load_opportunities() -> pd.DataFrame:
    return pd.read_csv(settings.data_dir / "opportunities.csv")


@lru_cache(maxsize=1)
def load_openalex_researchers() -> pd.DataFrame:
    path = settings.data_dir / "openalex_researchers_generated.csv"
    if not Path(path).exists():
        return pd.DataFrame(
            columns=["author_id", "name", "works_count", "topics", "institutions", "countries"]
        )
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_openalex_works() -> pd.DataFrame:
    path = settings.data_dir / "openalex_works_generated.csv"
    if not Path(path).exists():
        return pd.DataFrame(
            columns=["work_id", "title", "year", "type", "cited_by_count", "topics", "abstract_excerpt", "source_query"]
        )
    return pd.read_csv(path)
