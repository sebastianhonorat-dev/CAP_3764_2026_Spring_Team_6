# ingest_scorecard.py

import os
import requests
import pandas as pd
from pathlib import Path
from ira.config import SCORECARD_KEY, BASE_URL
import math


def collect(state: str = "FL", per_page: int = 100) -> pd.DataFrame:
    fields = ",".join([
    "school.name",
    "latest.programs.cip_4_digit.unit_id",
    "latest.programs.cip_4_digit.school.type",
    "latest.programs.cip_4_digit.school.main_campus",
    "programs.cip_4_digit.code",
    "programs.cip_4_digit.title",
    "programs.cip_4_digit.credential.level",
    "programs.cip_4_digit.distance",
    "latest.programs.cip_4_digit.earnings.4_yr.overall_median_earnings",
    "latest.programs.cip_4_digit.earnings.4_yr.working_not_enrolled.overall_count",
    "latest.programs.cip_4_digit.debt.staff_grad_plus.all.eval_inst.median",
    "latest.programs.cip_4_digit.debt.staff_grad_plus.all.eval_inst.median_payment"
    ])

    params = {
        "api_key": SCORECARD_KEY,
        "school.state": "FL",
        "fields": fields,
        "per_page": "100"
    }

    dfs = []


    params["page"] = "0"
    response = requests.get(BASE_URL, params=params)

    data = response.json()
    total = int(data["metadata"]["total"])
    per_page = int(data["metadata"]["per_page"])
    total_pages = math.ceil(total / per_page)

    results = [r for r in data.get("results", []) if r.get("latest.programs.cip_4_digit")]
    df_page = pd.json_normalize(
        results,
        record_path=["latest.programs.cip_4_digit"],
        meta="school.name",
        errors="ignore"
    )
    dfs.append(df_page)


    for page in range(1, total_pages):
        params["page"] = str(page)
        response = requests.get(BASE_URL, params=params)
        
        data = response.json()
        results = [r for r in data.get("results", []) if r.get("latest.programs.cip_4_digit")]
        df_page = pd.json_normalize(
            results,
            record_path=["latest.programs.cip_4_digit"],
            meta="school.name",
            errors="ignore"
        )
        dfs.append(df_page)

        df = pd.concat(dfs, ignore_index=True)


    df = df[
        df["earnings.4_yr.overall_median_earnings"].notna()
        & df["debt.staff_grad_plus.all.eval_inst.median"].notna()
    ]

    df["debt.staff_grad_plus.all.eval_inst.median"] = df["debt.staff_grad_plus.all.eval_inst.median"].astype(float)
    df["earnings.4_yr.working_not_enrolled.overall_count"] = df["earnings.4_yr.working_not_enrolled.overall_count"].astype(float)
    df["earnings.4_yr.overall_median_earnings"] = df["earnings.4_yr.overall_median_earnings"].astype(float)

    print(f"Total pages fetched: {total_pages}")
    print(f"Total Rows/Programs ingested: {len(df)}")
    return df

def save(df: pd.DataFrame, filename: str = "scorecard_FL_programs.csv") -> None :
    root = Path(__file__).resolve().parents[3]
    target_path = root / "data" / "raw" / "scorecard" / "scorecard_FL_programs.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)