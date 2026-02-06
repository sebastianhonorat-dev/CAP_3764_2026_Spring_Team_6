# ingest_scorecard.py

import os
import requests
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

root = Path.cwd()

pd.set_option("display.max_columns", None)

load_dotenv()

SCORECARD_KEY = os.getenv("COLLEGE_SCORECARD_API_KEY")

BASE_URL = "https://api.data.gov/ed/collegescorecard/v1/schools"

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

params= {
    "api_key": SCORECARD_KEY,
    "school.state":"FL",
    "fields": fields,
    "latest.programs.cip_4_digit.earnings.4_yr.overall_median_earnings__not":"null",
    "latest.programs.cip_4_digit.debt.staff_grad_plus.all.eval_inst.median__not":"null",
    "page":"1",
    "per_page":"100"
}
response = requests.get(
    BASE_URL,
    params = params,
)
data = response.json()
df = pd.json_normalize(
    data["results"],
    record_path=["latest.programs.cip_4_digit"],
    meta="school.name",
    errors="ignore"
)

df["debt.staff_grad_plus.all.eval_inst.median"] = df["debt.staff_grad_plus.all.eval_inst.median"].astype(float)
df["earnings.4_yr.working_not_enrolled.overall_count"] = df["earnings.4_yr.working_not_enrolled.overall_count"].astype(float)
df["earnings.4_yr.overall_median_earnings"] = df["earnings.4_yr.overall_median_earnings"].astype(float)

target_path = root/"data"/"raw"/"scorecard"/"test.csv"
df.to_csv(target_path)