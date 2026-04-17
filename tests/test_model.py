from pathlib import Path

from ira.modeling import load_predictor
from ira.modeling.earnings import ARTIFACT_PATH


def test_model_artifact_loads_and_has_all_years() -> None:
    predictor = load_predictor()

    assert Path(ARTIFACT_PATH).exists()
    assert predictor.default_year == 4
    assert predictor.years == (1, 4, 5)
    assert set(predictor.prediction_columns_by_year.keys()) == {1, 4, 5}
