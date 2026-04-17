from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "clean" / "scorecard" / "clean_scorecard_FL_programs.csv"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "florida_scorecard_multiyear_earnings_xgb_model.joblib"

DEFAULT_YEAR = 4
YEARS = (1, 4, 5)
TARGET_COLUMNS = {
    1: "1_yr_median_earnings",
    4: "4_yr_median_earnings",
    5: "5_yr_median_earnings",
}
PREDICTION_COLUMNS = {
    1: "predicted_1_yr_median_earnings",
    4: "predicted_4_yr_median_earnings",
    5: "predicted_5_yr_median_earnings",
}
TARGET_COLUMN = TARGET_COLUMNS[DEFAULT_YEAR]
PREDICTION_COLUMN = PREDICTION_COLUMNS[DEFAULT_YEAR]
MODEL_NAME = "florida-scorecard-multiyear-earnings-xgboost"
SCOPE_NAME = "Florida"
ARTIFACT_VERSION = 2

CATEGORICAL_FEATURES = [
    "code",
    "school_type",
    "locale",
    "carnegie_size_setting",
    "open_admissions_policy",
    "title_iv_eligibility_type",
    "credential_level",
    "distance",
    "selectivity_bucket",
]

NUMERIC_FEATURES = [
    "admission_rate_overall",
    "location_lat",
    "location_lon",
    "median_family_income",
    "students_with_pell_grant",
    "age_entry",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

XGB_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 10,
    "n_estimators": 300,
    "subsample": 0.8,
}


def normalize_text(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text or np.nan


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()

    for column in FEATURE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA

    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].apply(normalize_text)

    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if "code" in prepared.columns:
        prepared["code"] = (
            prepared["code"]
            .astype("string")
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(4)
        )
        prepared["code"] = prepared["code"].replace({"0000": pd.NA, "<NA>": pd.NA})

    return prepared


def get_xgb_regressor_class():
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required to train or load the Florida earnings deployment model."
        ) from exc
    return XGBRegressor


def build_pipeline() -> Pipeline:
    xgb_regressor = get_xgb_regressor_class()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
    )
    model = xgb_regressor(
        random_state=42,
        n_jobs=1,
        verbosity=0,
        objective="reg:squarederror",
        **XGB_PARAMS,
    )
    return Pipeline(steps=[("preprocessor", transformer), ("model", model)])


def build_feature_defaults(prepared: pd.DataFrame) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for column in CATEGORICAL_FEATURES:
        series = prepared[column].dropna().astype(str)
        defaults[column] = series.mode().iat[0] if not series.empty else "Missing"
    for column in NUMERIC_FEATURES:
        value = pd.to_numeric(prepared[column], errors="coerce").median()
        defaults[column] = 0.0 if pd.isna(value) else float(value)
    return defaults


def train_year_model(prepared: pd.DataFrame, year: int) -> dict[str, object]:
    target_column = TARGET_COLUMNS[year]
    prediction_column = PREDICTION_COLUMNS[year]

    training_frame = prepared[FEATURE_COLUMNS].copy()
    training_frame[target_column] = pd.to_numeric(prepared[target_column], errors="coerce")
    training_frame = training_frame[training_frame[target_column].notna() & (training_frame[target_column] > 0)].copy()

    X = training_frame[FEATURE_COLUMNS]
    y = training_frame[target_column].astype(float)
    y_log = np.log(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=0.2,
        random_state=42,
    )

    evaluation_pipeline = build_pipeline()
    evaluation_pipeline.fit(X_train, y_train_log)
    predicted_log = evaluation_pipeline.predict(X_test)

    y_test = np.exp(y_test_log)
    predicted = np.clip(np.exp(predicted_log), a_min=0, a_max=None)
    metrics = {
        "mae_log": float(mean_absolute_error(y_test_log, predicted_log)),
        "r2_log": float(r2_score(y_test_log, predicted_log)),
        "mae": float(mean_absolute_error(y_test, predicted)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "available_rows": int(len(training_frame)),
    }

    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y_log)

    return {
        "target_column": target_column,
        "prediction_column": prediction_column,
        "metrics": metrics,
        "pipeline": final_pipeline,
    }


def train_and_save_model(
    dataset_path: Path | str = DATA_PATH,
    artifact_path: Path | str = ARTIFACT_PATH,
) -> "EarningsPredictor":
    raw = pd.read_csv(dataset_path)
    prepared = prepare_feature_frame(raw)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": MODEL_NAME,
        "model_type": "XGBoost",
        "scope": SCOPE_NAME,
        "default_year": DEFAULT_YEAR,
        "years": YEARS,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "prediction_columns": PREDICTION_COLUMNS,
        "models": {year: train_year_model(prepared, year) for year in YEARS},
        "feature_options": {
            column: sorted(prepared[column].dropna().astype(str).unique().tolist())
            for column in CATEGORICAL_FEATURES
        },
        "feature_defaults": build_feature_defaults(prepared),
        "example_records": prepared[FEATURE_COLUMNS].dropna(how="all").head(5).to_dict(orient="records"),
    }
    artifact["metrics"] = artifact["models"][DEFAULT_YEAR]["metrics"]
    artifact["metrics_by_year"] = {
        year: artifact["models"][year]["metrics"] for year in YEARS
    }

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    return EarningsPredictor(artifact=artifact, artifact_path=artifact_path)


def is_current_artifact(artifact: object) -> bool:
    if not isinstance(artifact, dict):
        return False
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        return False
    if artifact.get("model_type") != "XGBoost":
        return False
    if tuple(artifact.get("years", ())) != YEARS:
        return False
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        return False
    models = artifact.get("models", {})
    return all(year in models for year in YEARS)


def load_predictor(artifact_path: Path | str = ARTIFACT_PATH) -> "EarningsPredictor":
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. "
            "Run `python scripts/train_model.py` to create it."
        )
    artifact = joblib.load(artifact_path)
    return EarningsPredictor(artifact=artifact, artifact_path=artifact_path)


def load_or_train_predictor(
    artifact_path: Path | str = ARTIFACT_PATH,
    dataset_path: Path | str = DATA_PATH,
    force_retrain: bool = False,
) -> "EarningsPredictor":
    artifact_path = Path(artifact_path)
    if artifact_path.exists() and not force_retrain:
        artifact = joblib.load(artifact_path)
        if is_current_artifact(artifact):
            return EarningsPredictor(artifact=artifact, artifact_path=artifact_path)
    return train_and_save_model(dataset_path=dataset_path, artifact_path=artifact_path)


class EarningsPredictor:
    def __init__(self, artifact: dict[str, object], artifact_path: Path | None = None):
        self.artifact = artifact
        self.artifact_path = artifact_path
        self.default_year = int(self.artifact["default_year"])
        self.models = {
            int(year): details for year, details in self.artifact["models"].items()
        }

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(int(year) for year in self.artifact["years"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.artifact["feature_columns"])

    @property
    def prediction_column(self) -> str:
        return self.prediction_columns_by_year[self.default_year]

    @property
    def prediction_columns_by_year(self) -> dict[int, str]:
        return {
            int(year): column
            for year, column in self.artifact["prediction_columns"].items()
        }

    @property
    def metrics_by_year(self) -> dict[int, dict[str, object]]:
        return {
            int(year): details["metrics"]
            for year, details in self.models.items()
        }

    def metadata(self) -> dict[str, object]:
        metadata = {
            key: value
            for key, value in self.artifact.items()
            if key not in {"models"}
        }
        metadata["metrics"] = self.metrics_by_year[self.default_year]
        metadata["metrics_by_year"] = self.metrics_by_year
        return metadata

    def evaluation_table(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for year in self.years:
            metrics = self.metrics_by_year[year]
            rows.append(
                {
                    "Year": f"{year}-year",
                    "R2 (log scale)": round(float(metrics["r2_log"]), 3),
                    "MAE ($)": round(float(metrics["mae"]), 2),
                    "Train rows": int(metrics["train_rows"]),
                    "Test rows": int(metrics["test_rows"]),
                }
            )
        return pd.DataFrame(rows)

    def prepare_inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = prepare_feature_frame(frame)
        return data[self.feature_columns]

    def predict_frame(
        self,
        frame: pd.DataFrame,
        years: tuple[int, ...] | list[int] | None = None,
    ) -> pd.DataFrame:
        selected_years = tuple(self.years if years is None else years)
        inputs = self.prepare_inputs(frame)
        output = frame.copy()

        for year in selected_years:
            pipeline = self.models[int(year)]["pipeline"]
            prediction_column = self.prediction_columns_by_year[int(year)]
            predicted = np.clip(np.exp(pipeline.predict(inputs)), a_min=0, a_max=None)
            output[prediction_column] = np.round(predicted, 2)

        return output

    def predict_records(
        self,
        records: list[dict[str, object]],
        years: tuple[int, ...] | list[int] | None = None,
    ) -> list[dict[str, object]]:
        frame = pd.DataFrame(records)
        results = self.predict_frame(frame, years=years)
        return results.to_dict(orient="records")

    def template_frame(self, rows: int = 1) -> pd.DataFrame:
        if rows <= 0:
            return pd.DataFrame(columns=self.feature_columns)

        defaults = self.artifact.get("feature_defaults", {})
        example_records = self.artifact.get("example_records") or []
        template = pd.DataFrame(example_records, columns=self.feature_columns)

        if template.empty:
            template = pd.DataFrame([defaults], columns=self.feature_columns)

        template = prepare_feature_frame(template)[self.feature_columns].copy()
        for column in CATEGORICAL_FEATURES:
            template[column] = template[column].fillna(defaults.get(column, "Missing"))
        for column in NUMERIC_FEATURES:
            template[column] = template[column].fillna(defaults.get(column, 0.0))

        repeats = int(np.ceil(rows / len(template)))
        expanded = pd.concat([template] * repeats, ignore_index=True)
        return expanded.head(rows).reset_index(drop=True)


def main() -> None:
    predictor = load_or_train_predictor()
    metadata = predictor.metadata()
    print(f"Saved model artifact to {predictor.artifact_path}")
    for year in predictor.years:
        metrics = metadata["metrics_by_year"][year]
        print(
            f"{year}-year XGBoost | R2(log): {metrics['r2_log']:.3f} | "
            f"MAE(log): {metrics['mae_log']:.3f} | MAE($): {metrics['mae']:.2f}"
        )


if __name__ == "__main__":
    main()
