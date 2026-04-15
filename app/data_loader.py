from functools import lru_cache

import pandas as pd

from app.config import settings


@lru_cache(maxsize=1)
def load_researchers() -> pd.DataFrame:
    return pd.read_csv(settings.data_dir / "researchers.csv")


@lru_cache(maxsize=1)
def load_opportunities() -> pd.DataFrame:
    return pd.read_csv(settings.data_dir / "opportunities.csv")
