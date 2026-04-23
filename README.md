# Florida Program Earnings Prediction

**Course:** Advanced Data Science - Spring 2026  
**Instructor:** Prof. Gustavo Garcia Melero  
**Team:** Sebastian Honorat, Frank Vela

## Project Goal

Develop a machine learning system to predict Florida program earnings and identify programs that perform above or below expected outcomes using multi-year earnings data and XGBoost.

## Problem

Students, schools, and instructors often look at earnings data without enough context. A program can have high earnings in raw dollar terms and still underperform compared with similar programs, or it can have modest earnings and still do better than expected once student and school context is considered.

This project focuses on predicting Florida program earnings and then comparing reported earnings with those predictions. That gives us a clearer way to talk about whether a program at a specific school is above or below expected outcomes.

## Stakeholder and Use Case

This project is most useful for:

- students comparing Florida programs
- schools reviewing program outcomes
- instructors reviewing a full data science workflow

The main use case is simple: enter a program or upload a batch of rows, generate predicted earnings, and compare those predictions with the saved multi-year results.

## Data Sources

### College Scorecard

We use College Scorecard program-level data for:

- 1-year, 4-year, and 5-year median earnings
- program information such as CIP code and credential level
- school context such as admissions, income, Pell share, and location

### Saved Multi-Year Comparison File

We also use a saved Florida comparison file to show whether a program-school record is above or below expected outcomes across multiple years.

## Project Workflow

The project covers the main parts of the data science lifecycle:

1. data collection from College Scorecard
2. data cleaning and preprocessing
3. exploratory data analysis
4. multi-year model training and evaluation
5. deployment through FastAPI and Streamlit

## Preprocessing Summary

The modeling pipeline:

- keeps Florida program-level records
- uses categorical and numeric school and program features
- fills in missing values
- turns categorical fields into a model-ready format
- scales numeric features
- trains separate models for 1-year, 4-year, and 5-year earnings

## Modeling

The deployed model is XGBoost. The project predicts three earnings horizons:

- 1-year earnings
- 4-year earnings
- 5-year earnings

We use the 4-year model as the main comparison view in the app, while the saved 1-year, 4-year, and 5-year results help show whether a program is consistently above or below expected outcomes.

### Why XGBoost

XGBoost was chosen because it gave strong performance across the earnings targets while still being practical to deploy in a simple prediction pipeline.

### Model Results

| Year   | R^2 (log scale) |   MAE ($) | Train rows | Test rows |
|:-------|----------------:|----------:|-----------:|----------:|
| 1-year |           0.858 |  5,918.64 |      1,680 |       420 |
| 4-year |           0.786 |  7,970.84 |      2,000 |       501 |
| 5-year |           0.861 |  7,258.93 |      1,430 |       358 |

## Model Interpretation

The repo includes a simple SHAP script for the deployed 4-year model:

```powershell
python scripts/run_shap_summary.py
```

That script saves:

- `artifacts/shap_summary_4_year.png`

The saved image is a simple grouped feature-importance chart that can be used in the report or slides to explain which features had the strongest effect on predictions.

## Assumptions and Limitations

- The current deployment is Florida-only.
- The analysis is at the program-at-school level, not overall school quality.
- Earnings are for graduates who are working and not enrolled.
- Some programs have missing data or limited coverage.
- Above or below expected compares reported earnings with the model and saved multi-year results. It does not prove that a school caused the outcome.

## Deployment

### FastAPI

Run the API:

```powershell
uvicorn ira.api:app --reload
```

Available endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict-batch`

Example single prediction request:

```json
{
  "code": "5138",
  "distance": "1",
  "school_type": "Public",
  "credential_level": "2",
  "locale": "11",
  "carnegie_size_setting": "Large four-year, primarily nonresidential",
  "open_admissions_policy": "2",
  "title_iv_eligibility_type": "1",
  "selectivity_bucket": "Inclusive",
  "admission_rate_overall": 0.82,
  "location_lat": 28.54,
  "location_lon": -81.38,
  "median_family_income": 42000,
  "students_with_pell_grant": 0.43,
  "age_entry": 22.0
}
```

### Streamlit

Run the dashboard:

```powershell
streamlit run src/ira/streamlit_app.py
```

The dashboard supports:

- program search
- side-by-side comparison of predicted and reported earnings
- 1-, 4-, and 5-year earnings view
- batch CSV upload for predictions

## Tests

Run the minimal test suite with:

```powershell
pytest -q
```

The restored tests cover:

- `/predict`
- `/predict-batch`
- model artifact loading

## Environment Setup

Create the Conda environment:

```powershell
conda env create -f environment.yml
conda activate florida-program-earnings
pip install -e .[dev]
```

## Repository Structure

```text
.
|-- artifacts/
|-- data/
|   |-- clean/
|   `-- raw/
|-- docs/
|   `-- final_report_outline.md
|-- notebooks/
|   `-- README.md
|-- scripts/
|   |-- run_ingest
|   |-- run_shap_summary.py
|   `-- train_model.py
|-- src/
|   `-- ira/
|       |-- api.py
|       |-- stability.py
|       |-- streamlit_app.py
|       `-- modeling/
|-- tests/
|-- environment.yml
|-- pyproject.toml
|-- README.md
```

## Notebook Order

The notebooks should be read in this order:

1. `01_data_collection/01_scorecard_ingestion.ipynb`
2. `02_cleaning/02_scorecard_cleaning.ipynb`
3. `03_eda/03_eda_outcomes_debt_earnings.ipynb`
4. `04_modeling/04_xgboost_multi_year_modeling.ipynb`
5. `05_deployment/05_deployment_walkthrough.ipynb`


