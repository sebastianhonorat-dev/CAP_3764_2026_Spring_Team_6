from fastapi.testclient import TestClient

from ira.api import app, predictor


client = TestClient(app)


def test_predict_returns_all_years() -> None:
    payload = predictor.template_frame(rows=1).iloc[0].to_dict()

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_column"] == predictor.prediction_column
    assert set(map(int, body["prediction_columns"].keys())) == set(predictor.years)
    for column in predictor.prediction_columns_by_year.values():
        assert column in body["result"]
        assert body["result"][column] >= 0


def test_predict_batch_returns_count_and_results() -> None:
    payload = {
        "records": predictor.template_frame(rows=2).to_dict(orient="records"),
    }

    response = client.post("/predict-batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    for result in body["results"]:
        for column in predictor.prediction_columns_by_year.values():
            assert column in result
            assert result[column] >= 0
