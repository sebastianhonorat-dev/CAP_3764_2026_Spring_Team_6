# ingest_scorecard.py

import requests
import pandas as pd
from pathlib import Path
from ira.config import SCORECARD_KEY, BASE_URL
import math
import time
import random


def collect(state: str = "FL", per_page: int = 100) -> pd.DataFrame:
    fields = ",".join([
        "id",
        "school.name",

        "latest.programs.cip_4_digit.code",
        "latest.programs.cip_4_digit.unit_id",
        "latest.programs.cip_4_digit.title",
        "latest.programs.cip_4_digit.school.type",
        "latest.programs.cip_4_digit.credential.level",
        "latest.programs.cip_4_digit.distance",

        "latest.school.locale",
        "latest.school.carnegie_size_setting",
        "latest.admissions.admission_rate.overall",
        "latest.student.demographics.median_family_income",
        "latest.student.students_with_pell_grant",
        "latest.school.open_admissions_policy",
        "latest.student.demographics.age_entry",
        "latest.school.title_iv.eligibility_type",
        "latest.programs.cip_4_digit.earnings.4_yr.overall_median_earnings",
        "latest.programs.cip_4_digit.earnings.4_yr.working_not_enrolled.overall_count"
    ])

    params = {
        "api_key": SCORECARD_KEY,
        "school.state": state,
        "fields": fields,
        "per_page": str(per_page),

        # only programs with earnings reported
        "latest.programs.cip_4_digit.earnings.4_yr.overall_median_earnings__range": "1.."
    }

    dfs = []

    params["page"] = "0"
    response = get_with_retries(BASE_URL, params=params, timeout=30)
    data = get_json_or_raise(response)

    total = int(data["metadata"]["total"])
    per_page_actual = int(data["metadata"]["per_page"])
    total_pages = math.ceil(total / per_page_actual)

    META = [
        "id",
        "school.name",
        "latest.school.locale",
        "latest.school.carnegie_size_setting",
        "latest.admissions.admission_rate.overall",
        "latest.student.demographics.median_family_income",
        "latest.student.students_with_pell_grant",
        "latest.school.open_admissions_policy",
        "latest.student.demographics.age_entry",
        "latest.school.title_iv.eligibility_type",
    ]

    def page_to_df(data):
        results = [r for r in data.get("results", []) if r.get("latest.programs.cip_4_digit")]
        return pd.json_normalize(
            results,
            record_path=["latest.programs.cip_4_digit"],
            meta=META,
            errors="ignore",
        )

    dfs.append(page_to_df(data))

    for page in range(1, total_pages):
        params["page"] = str(page)
        response = get_with_retries(BASE_URL, params=params, timeout=30)
        data = get_json_or_raise(response)
        dfs.append(page_to_df(data))

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    print(f"Total pages fetched: {total_pages}")
    print(f"Total Rows/Programs ingested: {len(df)}")
    return df

def get_json_or_raise(response: requests.Response):
    # Raise for HTTP errors early (4xx/5xx)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        ct = response.headers.get("Content-Type", "")
        body_preview = (response.text or "")[:800]
        raise RuntimeError(
            f"HTTP {response.status_code} for {response.url}\n"
            f"Content-Type: {ct}\n"
            f"Body preview:\n{body_preview}"
        ) from e

    # Check content-type sanity (helps catch HTML responses)
    ct = response.headers.get("Content-Type", "")
    if "json" not in ct.lower():
        body_preview = (response.text or "")[:800]
        raise RuntimeError(
            f"Expected JSON but got Content-Type: {ct}\n"
            f"URL: {response.url}\n"
            f"Body preview:\n{body_preview}"
        )

    # Parse JSON with a clearer error if it fails
    try:
        return response.json()
    except json.JSONDecodeError as e:
        body_preview = (response.text or "")[:800]
        raise RuntimeError(
            f"JSON decode failed for {response.url}\n"
            f"Body preview:\n{body_preview}"
        ) from e
    

def get_with_retries(url, params, tries=5, timeout=30):
    last = None
    for i in range(tries):
        r = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
        if r.status_code < 500:
            return r
        last = r
        time.sleep((2 ** i) + random.random())
    return last

def save(df: pd.DataFrame, file_type: str="scorecard",clean: str = 0, file_name: str = "scorecard_FL_programs.csv") -> None :
    root = Path(__file__).resolve().parents[3]
    if clean:
        target_path = root/"data"/"clean"/file_type/f"clean_{file_name}.csv"
    else:
        target_path = root/"data"/"raw"/file_type/f"raw_{file_name}.csv"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)