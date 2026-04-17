from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STABILITY_PATH = PROJECT_ROOT / "data" / "raw" / "scorecard" / "raw_residual_FL_stable_programs.csv"

STABILITY_COLUMNS = [
    "unit_id",
    "code",
    "credential_level",
    "school_name",
    "title",
    "1_year_earning",
    "1_year_pred",
    "1_year_error",
    "4_year_earning",
    "4_year_pred",
    "4_year_error",
    "5_year_earning",
    "5_year_pred",
    "5_year_error",
    "confidence",
    "1_year_score",
    "4_year_score",
    "5_year_score",
    "rank_1",
    "rank_4",
    "rank_5",
    "rank_std_pct",
    "mean_rank_std_pct",
    "median_score",
]

MATCH_COLUMNS = ["unit_id_key", "code_key", "credential_level_key"]


def _normalize_code(series: pd.Series) -> pd.Series:
    values = (
        series.astype("string")
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(4)
    )
    return values.replace({"0000": pd.NA, "<NA>": pd.NA})


def _normalize_integer_key(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")
    return values.replace({"<NA>": pd.NA})


def normalize_program_keys(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    empty = pd.Series(pd.NA, index=normalized.index, dtype="object")
    normalized["code_key"] = _normalize_code(normalized.get("code", empty))
    normalized["credential_level_key"] = _normalize_integer_key(
        normalized.get("credential_level", empty)
    )
    normalized["unit_id_key"] = _normalize_integer_key(
        normalized.get("unit_id", empty)
    )
    return normalized


@lru_cache(maxsize=1)
def load_stability_data(path: str | None = None) -> pd.DataFrame:
    stability_path = Path(path) if path is not None else STABILITY_PATH
    stability = pd.read_csv(stability_path, usecols=STABILITY_COLUMNS)
    stability = normalize_program_keys(stability)
    stability["stability_status"] = "No multi-year signal"
    stability.loc[stability["median_score"] > 0, "stability_status"] = "Above expected across years"
    stability.loc[stability["median_score"] < 0, "stability_status"] = "Below expected across years"
    return stability


def merge_stability_data(frame: pd.DataFrame, stability: pd.DataFrame | None = None) -> pd.DataFrame:
    stability_frame = load_stability_data() if stability is None else stability
    normalized = normalize_program_keys(frame)
    merged = normalized.merge(
        stability_frame,
        on=MATCH_COLUMNS,
        how="left",
        suffixes=("", "_stability"),
    )
    return merged
