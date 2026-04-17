from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ira.modeling import load_predictor

predictor = load_predictor()
app = FastAPI(title="Florida Earnings Predictor", version="0.1.0")


class ProgramFeatures(BaseModel):
    code: str | None = None
    unit_id: int | None = None
    title: str | None = None
    school_name: str | None = None
    distance: str | None = None
    school_type: str | None = None
    credential_level: str | None = None
    locale: str | None = None
    carnegie_size_setting: str | None = None
    open_admissions_policy: str | None = None
    title_iv_eligibility_type: str | None = None
    selectivity_bucket: str | None = None
    admission_rate_overall: float | None = None
    location_lat: float | None = None
    location_lon: float | None = None
    median_family_income: float | None = None
    students_with_pell_grant: float | None = None
    age_entry: float | None = None


class BatchPredictionRequest(BaseModel):
    records: list[ProgramFeatures]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: ProgramFeatures) -> dict[str, object]:
    record = payload.model_dump()
    prediction = predictor.predict_records([record])[0]
    return {
        "prediction_columns": predictor.prediction_columns_by_year,
        "prediction_column": predictor.prediction_column,
        "result": prediction,
    }


@app.post("/predict-batch")
def predict_batch(payload: BatchPredictionRequest) -> dict[str, object]:
    records = [record.model_dump() for record in payload.records]
    predictions = predictor.predict_records(records)
    return {
        "count": len(predictions),
        "prediction_columns": predictor.prediction_columns_by_year,
        "prediction_column": predictor.prediction_column,
        "results": predictions,
    }
