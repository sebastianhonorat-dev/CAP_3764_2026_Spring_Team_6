# Institutional ROI Analysis

**Course:** Advanced Data Science — Spring 2026
**Instructor:** Prof. Gustavo García Melero
**Team:** Sebastian Honorat, Frank Vela

---

## Overview

This project examines **institutional return on investment (ROI)** in U.S. higher education by analyzing economic outcomes alongside institutional characteristics at the degree level.

Rather than relying solely on raw earnings or debt metrics, the analysis aims to evaluate institutional performance **relative to expectations**, accounting for factors such as selectivity, enrollment scale, and student composition.

The current phase centers on **data ingestion and exploratory analysis** using the U.S. Department of Education’s College Scorecard API. Future iterations will incorporate research productivity data from OpenAlex to support a broader institutional performance framework.

---

## Research Focus

The project is guided by the following questions:

* Which institutions generate **above-expected economic outcomes** for graduates within specific degree programs?
* How do earnings and debt metrics shift when contextualized by institutional constraints?
* To what extent do economic ROI and research productivity diverge among otherwise comparable institutions?

> **Note:** A formal ROI methodology has not yet been finalized. Model design will evolve iteratively as data coverage and exploratory insights mature.

---

## Data Sources

### College Scorecard API

Primary source for degree-level outcomes and institutional attributes, including:

* Median post-graduation earnings
* Student debt levels
* Completion and retention rates
* Admissions metrics and selectivity indicators

### OpenAlex Scholarly Knowledge Graph *(Planned)*

Future integration will support analysis of:

* Institutional publication volume
* Citation-based impact measures
* Field-normalized research output

---

## Current Project Status

### Implemented

* College Scorecard API ingestion pipeline
* Secure environment-based API key management
* CSV-based persistence of retrieved datasets
* Comprehensive pagination for full API coverage


### In Progress
* Data cleaning and validation


### In Progress
* Data cleaning and validation
* Definition of a degree-level ROI / value-added metric
* Exploratory data analysis (EDA) notebooks
* Exploratory data analysis (EDA) notebooks
* OpenAlex data integration
* Cross-institutional comparative analysis

### Explicitly Out of Scope

* Machine learning models
* Predictive systems beyond baseline expectation frameworks

---

## Tech Stack

**Python:** 3.11

**Core Libraries:**

* `requests`
* `python-dotenv`
* `pandas`
* `numpy`
* `plotly`

**Environment:** Conda
**APIs:** College Scorecard, OpenAlex

---

## Repository Structure

```text
.
├── data/
│   └── raw/
│       └── scorecard/
│           └── test.csv
├── notebooks/
│   ├── setup.ipynb
│   └── eda_scorecard.ipynb
├── src/
│   └── ingest_scorecard.py
├── Misc/
│   └── CollegeScorecardDataDictionary.xlsx
├── deps.py
├── requirements.in
├── requirements.txt
├── .env
└── README.md
```

---

## Setup & Usage

### Environment Configuration

Create a `.env` file in the project root:

```env
COLLEGE_SCORECARD_API_KEY=your_key_here
OPENALEX_API_KEY=your_key_here
```

Create Conda environment:
Create Conda environment:

```powershell
conda env create -f environment.yml
conda activate institutional-roi
pip install -e .
```

---

## Data Ingestion

Run the ingestion pipeline:

```bash
python src/ingest_scorecard.py
```

Retrieved data is written to:

```
data/raw/scorecard/
```

The project intentionally separates ingestion and analysis to maintain modularity and support iterative development.

---

## Notes & Limitations

* API pagination is not yet fully implemented; current ingestion may not capture the complete dataset.
* ROI methodology remains under active development.
* Findings at this stage should be interpreted as exploratory rather than definitive institutional rankings.

---

## Academic & Portfolio Context

This repository is structured to support:

* Reproducible academic research
* Transparent methodological iteration
* Incremental expansion toward a comprehensive institutional performance model