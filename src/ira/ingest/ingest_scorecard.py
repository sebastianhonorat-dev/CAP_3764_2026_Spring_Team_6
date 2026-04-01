# ingest_scorecard.py

import json
import math
import random
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

from ira.config import BASE_URL, SCORECARD_KEY
from ira.logging_utils import get_logger


logger = get_logger(__name__)


def redact_url(value: str) -> str:
    if not value:
        return value

    parts = urlsplit(value)
    query = parse_qsl(parts.query, keep_blank_values=True)
    if not query:
        return value

    redacted_query = []
    for key, query_value in query:
        key_lower = key.lower()
        if key_lower == "api_key" or key_lower.endswith("_key"):
            redacted_query.append((key, "***REDACTED***"))
        else:
            redacted_query.append((key, query_value))

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(redacted_query, doseq=True),
            parts.fragment,
        )
    )


def build_request_url(url: str, params: dict) -> str:
    prepared = requests.Request("GET", url, params=params).prepare()
    return redact_url(prepared.url)


def collect(state: str = "FL", per_page: int = 100) -> pd.DataFrame:
    logger.info("Starting College Scorecard collection for state=%s with per_page=%s", state, per_page)

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
    total_api_rows = 0
    total_filtered_rows = 0

    params["page"] = "0"
    response = get_with_retries(BASE_URL, params=params, timeout=30)
    data = get_json_or_raise(response)

    total = int(data["metadata"]["total"])
    per_page_actual = int(data["metadata"]["per_page"])
    total_pages = math.ceil(total / per_page_actual)
    logger.info("API reports %s total rows across %s pages", total, total_pages)

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

    def page_to_df(data, page_number: int):
        nonlocal total_api_rows, total_filtered_rows

        raw_results = data.get("results", [])
        total_api_rows += len(raw_results)

        results = [r for r in raw_results if r.get("latest.programs.cip_4_digit")]
        total_filtered_rows += len(results)

        page_df = pd.json_normalize(
            results,
            record_path=["latest.programs.cip_4_digit"],
            meta=META,
            errors="ignore",
        )
        logger.info(
            "Fetched page %s/%s: %s raw rows, %s kept rows, %s program rows",
            page_number + 1,
            total_pages,
            len(raw_results),
            len(results),
            len(page_df),
        )
        return page_df

    dfs.append(page_to_df(data, 0))

    for page in range(1, total_pages):
        params["page"] = str(page)
        response = get_with_retries(BASE_URL, params=params, timeout=30)
        data = get_json_or_raise(response)
        dfs.append(page_to_df(data, page))

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    logger.info("Total rows collected before filtering: %s", total_api_rows)
    logger.info("Total rows after filtering: %s", total_filtered_rows)
    logger.info("Total program rows ingested: %s", len(df))
    return df


def get_json_or_raise(response: requests.Response):
    response_url = redact_url(response.url)

    # Raise for HTTP errors early (4xx/5xx)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        ct = response.headers.get("Content-Type", "")
        body_preview = (response.text or "")[:800]
        raise RuntimeError(
            f"HTTP {response.status_code} for {response_url}\n"
            f"Content-Type: {ct}\n"
            f"Body preview:\n{body_preview}"
        ) from e

    # Check content-type sanity (helps catch HTML responses)
    ct = response.headers.get("Content-Type", "")
    if "json" not in ct.lower():
        body_preview = (response.text or "")[:800]
        raise RuntimeError(
            f"Expected JSON but got Content-Type: {ct}\n"
            f"URL: {response_url}\n"
            f"Body preview:\n{body_preview}"
        )

    # Parse JSON with a clearer error if it fails
    try:
        return response.json()
    except json.JSONDecodeError as e:
        body_preview = (response.text or "")[:800]
        raise RuntimeError(
            f"JSON decode failed for {response_url}\n"
            f"Body preview:\n{body_preview}"
        ) from e


def get_with_retries(url, params, tries=5, timeout=30):
    last = None
    safe_url = build_request_url(url, params)
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
        except requests.RequestException as exc:
            logger.warning(
                "Request attempt %s/%s failed for %s: %s",
                i + 1,
                tries,
                safe_url,
                exc.__class__.__name__,
            )
            if i == tries - 1:
                raise RuntimeError(
                    f"Request failed after {tries} attempts for {safe_url}: {exc.__class__.__name__}"
                ) from exc
            time.sleep((2 ** i) + random.random())
            continue

        if r.status_code < 500:
            return r
        last = r
        logger.warning(
            "Server error on attempt %s/%s for %s: HTTP %s",
            i + 1,
            tries,
            redact_url(r.url),
            r.status_code,
        )
        time.sleep((2 ** i) + random.random())

    if last is not None:
        logger.error("Exhausted retries for %s after %s attempts", redact_url(last.url), tries)
    return last


def save(df: pd.DataFrame, file_type: str="raw", file_name: str = "scorecard_FL_programs.csv") -> None :
    root = Path(__file__).resolve().parents[3]
    target_path = root/"data"/file_type/"scorecard"/f"{file_type}_{file_name}"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
