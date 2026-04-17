from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ira.modeling.earnings import DATA_PATH, TARGET_COLUMNS, load_predictor


FEATURE_LABELS = {
    "code": "Program field (CIP code)",
    "school_type": "School type",
    "locale": "Area type",
    "carnegie_size_setting": "School size and setting",
    "open_admissions_policy": "Open admissions policy",
    "title_iv_eligibility_type": "Title IV eligibility",
    "credential_level": "Degree or certificate type",
    "distance": "Distance learning",
    "selectivity_bucket": "Selectivity group",
    "admission_rate_overall": "Admission rate",
    "location_lat": "North/South location",
    "location_lon": "East/West location",
    "median_family_income": "Median family income",
    "students_with_pell_grant": "Students with Pell grant",
    "age_entry": "Average age at entry",
}


def get_source_feature(feature_name: str, feature_columns: list[str]) -> str:
    if feature_name.startswith("num__"):
        raw = feature_name[len("num__") :]
    elif feature_name.startswith("cat__"):
        raw = feature_name[len("cat__") :]
    else:
        raw = feature_name

    for column in sorted(feature_columns, key=len, reverse=True):
        if raw == column or raw.startswith(f"{column}_"):
            return column
    return raw


def main() -> None:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is not installed. Add the project environment first, then run this script again."
        ) from exc

    predictor = load_predictor()
    frame = pd.read_csv(DATA_PATH)
    year = predictor.default_year
    target_column = TARGET_COLUMNS[year]

    usable = frame[pd.to_numeric(frame[target_column], errors="coerce").notna()].copy()
    sample = usable.sample(n=min(len(usable), 250), random_state=42).reset_index(drop=True)

    model_details = predictor.models[year]
    pipeline = model_details["pipeline"]
    prepared_inputs = predictor.prepare_inputs(sample)
    transformed = pipeline.named_steps["preprocessor"].transform(prepared_inputs)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(transformed)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance["source_feature"] = importance["feature"].map(
        lambda name: get_source_feature(name, predictor.feature_columns)
    )
    grouped_importance = (
        importance.groupby("source_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    grouped_importance["label"] = grouped_importance["source_feature"].map(
        lambda name: FEATURE_LABELS.get(name, name)
    )

    output_dir = PROJECT_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "shap_summary_4_year.png"
    top_features = grouped_importance.head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features["label"], top_features["mean_abs_shap"], color="#2f6ca3")
    ax.set_title("Most Important Features in the 4-Year Earnings Model", pad=12)
    ax.set_xlabel("Average impact on the prediction")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center")

    fig.text(
        0.01,
        0.01,
        "Higher values mean the feature had more influence on the model's predictions overall.",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary plot to {png_path}")
    print("\nTop grouped features:")
    print(grouped_importance[["label", "mean_abs_shap"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
