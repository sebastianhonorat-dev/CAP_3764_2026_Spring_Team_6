Institutional ROI Analysis

Course: Advanced Data Science — Spring 2026
Instructor: Prof. Gustavo García Melero
Team: Sebastian Honorat, Frank Vela

Overview

This project investigates institutional return on investment (ROI) in U.S. higher education by comparing economic outcomes to institutional constraints at the degree level.

The core goal is to move beyond raw earnings or debt figures and instead evaluate how effectively institutions perform relative to expectations, given factors such as selectivity, size, and student composition.

At its current stage, the project focuses on data ingestion and exploratory analysis using the U.S. Department of Education’s College Scorecard API, with plans to integrate OpenAlex research output data in later phases.

Research Focus

Key questions guiding the project:

Which institutions deliver higher-than-expected economic outcomes for graduates of a specific degree?

How do earnings and debt outcomes vary when normalized against institutional constraints?

How might economic ROI and research productivity differ across institutions with similar profiles?

Note: A final ROI or ranking equation has not yet been defined and will be developed iteratively based on data quality and exploratory findings.

Data Sources

College Scorecard API
Used for degree-level outcomes and institutional characteristics, including:

Median earnings

Student debt

Completion and retention rates

Admissions and selectivity proxies

OpenAlex Scholarly Knowledge Graph (planned)
Intended for future integration of:

Publication counts by institution

Citation impact metrics

Field-normalized research output

Current Project Status
Implemented

College Scorecard API ingestion

Environment-based API key management

Initial exploratory data analysis (EDA) notebooks

CSV persistence of ingested data

In Progress / Planned

Robust pagination handling for full API coverage

Definition of a degree-level institutional ROI or value-added metric

Integration of OpenAlex research data

Comparative institutional analysis

Explicitly Out of Scope

Machine learning models

Predictive modeling beyond baseline expectation frameworks

Tech Stack

Python: 3.11

Libraries:

requests

python-dotenv

pandas

numpy

plotly

pathlib

Environment: Conda

APIs: College Scorecard, OpenAlex

Repository Structure
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

Setup & Usage
Environment Setup

Create a .env file in the project root with:

COLLEGE_SCORECARD_API_KEY=your_key_here
OPENALEX_API_KEY=your_key_here


Install dependencies:

pip install -r requirements.txt

Data Ingestion

Run the College Scorecard ingestion script:

python src/ingest_scorecard.py


This script retrieves data from the API and writes CSV files to data/raw/scorecard/.

There is currently no unified main script. Ingestion and analysis are intentionally modular.

Notes & Limitations

API pagination is not yet fully implemented; current ingestion may not capture all available pages.

ROI and ranking methodologies are under active design and have not been finalized.

Results at this stage should be considered exploratory, not definitive rankings.

Academic & Portfolio Use

This repository is structured to support:

Reproducible academic analysis

Transparent methodological development

Incremental expansion into a broader institutional performance framework