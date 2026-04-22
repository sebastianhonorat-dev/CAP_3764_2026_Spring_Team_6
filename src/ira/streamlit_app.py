from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ira.modeling import TARGET_COLUMNS, load_predictor
from ira.stability import load_stability_data, merge_stability_data

DATA_PATH = PROJECT_ROOT / "data" / "clean" / "scorecard" / "clean_scorecard_FL_programs.csv"
PRIMARY_YEAR = 4
PRIMARY_TARGET_COLUMN = TARGET_COLUMNS[PRIMARY_YEAR]
PRIMARY_PREDICTION_COLUMN = "predicted_4_year_display"
PRIMARY_ACTUAL_COLUMN = "actual_4_year_display"
PREDICTION_LABEL = "Predicted 4-year earnings (XGBoost)"
YEAR_LABELS = {
    1: "1 year after completion",
    4: "4 years after completion",
    5: "5 years after completion",
}

FEATURE_LABELS = {
    "code": "Program CIP code",
    "distance": "Online availability",
    "school_type": "School ownership",
    "credential_level": "Award type",
    "locale": "Community type",
    "carnegie_size_setting": "School size and campus setting",
    "open_admissions_policy": "Open admissions",
    "title_iv_eligibility_type": "Federal aid category",
    "selectivity_bucket": "Admissions selectivity",
    "admission_rate_overall": "Admission rate",
    "location_lat": "Latitude",
    "location_lon": "Longitude",
    "median_family_income": "Median family income",
    "students_with_pell_grant": "Students receiving Pell grants",
    "age_entry": "Average age at entry",
}

VALUE_LABELS = {
    "distance": {
        "0": "Not reported to IPEDS",
        "1": "No credential in this field can be completed fully online",
        "2": "Some credentials in this field can be completed fully online",
        "3": "All credentials in this field can be completed fully online",
    },
    "locale": {
        "11": "City: Large",
        "12": "City: Midsize",
        "13": "City: Small",
        "21": "Suburb: Large",
        "22": "Suburb: Midsize",
        "23": "Suburb: Small",
        "31": "Town: Fringe",
        "32": "Town: Distant",
        "33": "Town: Remote",
        "41": "Rural: Fringe",
        "42": "Rural: Distant",
        "43": "Rural: Remote",
    },
    "carnegie_size_setting": {
        "-2": "Not applicable",
        "0": "Not classified",
        "1": "Two-year, very small",
        "2": "Two-year, small",
        "3": "Two-year, medium",
        "4": "Two-year, large",
        "5": "Two-year, very large",
        "6": "Four-year, very small, primarily nonresidential",
        "7": "Four-year, very small, primarily residential",
        "8": "Four-year, very small, highly residential",
        "9": "Four-year, small, primarily nonresidential",
        "10": "Four-year, small, primarily residential",
        "11": "Four-year, small, highly residential",
        "12": "Four-year, medium, primarily nonresidential",
        "13": "Four-year, medium, primarily residential",
        "14": "Four-year, medium, highly residential",
        "15": "Four-year, large, primarily nonresidential",
        "16": "Four-year, large, primarily residential",
        "17": "Four-year, large, highly residential",
        "18": "Exclusively graduate or professional",
    },
    "open_admissions_policy": {
        "1": "Yes",
        "2": "No",
        "3": "Does not enroll first-time students",
    },
    "title_iv_eligibility_type": {
        "1": "Participates in federal Title IV aid programs",
        "2": "Branch campus of a Title IV participating institution",
        "3": "Limited Title IV participation",
        "5": "Not currently participating, but has an OPE ID",
        "6": "Not currently participating and has no OPE ID",
        "7": "Stopped participating during the collection year",
        "8": "Became eligible during the collection year",
        "19": "Not eligible",
    },
    "selectivity_bucket": {
        "elite": "More selective admissions",
        "mid": "Moderately selective admissions",
        "open": "Broad-access admissions",
    },
}


@st.cache_resource(show_spinner=False)
def get_predictor():
    return load_predictor()


@st.cache_data(show_spinner=False)
def load_scored_programs(data_path: str, _predictor) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["code"] = (
        df["code"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .str.zfill(4)
    )
    df["program_label"] = (
        df["code"]
        + " | "
        + df["title"].fillna("Unknown program")
        + " | "
        + df["school_name"].fillna("Unknown school")
    )
    df["search_text"] = (
        df["code"].fillna("")
        + " "
        + df["title"].fillna("")
        + " "
        + df["school_name"].fillna("")
        + " "
        + df["program_label"].fillna("")
    ).map(normalize_search_text)

    scored = _predictor.predict_frame(df)
    scored = merge_stability_data(scored, load_stability_data())
    scored[PRIMARY_PREDICTION_COLUMN] = scored["4_year_pred"].fillna(scored[_predictor.prediction_column])
    scored[PRIMARY_ACTUAL_COLUMN] = scored["4_year_earning"].fillna(scored[PRIMARY_TARGET_COLUMN])
    scored["gap"] = scored["4_year_error"]
    fallback_gap = scored[PRIMARY_ACTUAL_COLUMN] - scored[PRIMARY_PREDICTION_COLUMN]
    scored["gap"] = scored["gap"].fillna(fallback_gap)
    scored["performance"] = np.where(
        scored["gap"].notna(),
        np.where(scored["gap"] >= 0, "Above expected", "Below expected"),
        "No reported earnings",
    )
    scored["stability_status"] = scored["stability_status"].fillna("No multi-year signal")
    return scored.sort_values(["school_name", "title"]).reset_index(drop=True)


def configure_page(page_title: str) -> None:
    st.set_page_config(
        page_title=page_title,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()


def load_app_state():
    predictor = get_predictor()
    metadata = predictor.metadata()
    metadata["evaluation_table"] = predictor.evaluation_table().rename(
        columns={"R2 (log scale)": "R^2 (log scale)"}
    )
    programs = load_scored_programs(str(DATA_PATH), predictor)
    actual_programs = programs.dropna(subset=["gap"]).copy()
    return predictor, metadata, programs, actual_programs


def normalize_search_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def format_currency(value: object) -> str:
    if pd.isna(value):
        return "Not available"
    return f"${float(value):,.0f}"


def format_signed_currency(value: object) -> str:
    if pd.isna(value):
        return "Not available"
    amount = float(value)
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):,.0f}"


def format_percent(value: object) -> str:
    if pd.isna(value):
        return "Not available"
    return f"{float(value) * 100:.1f}%"


def format_number(value: object) -> str:
    if pd.isna(value):
        return "Not available"
    amount = float(value)
    if amount.is_integer():
        return f"{int(amount):,}"
    return f"{amount:,.1f}"


def format_credential_level(value: object) -> str:
    if pd.isna(value):
        return "Not available"

    try:
        level = int(float(value))
    except (TypeError, ValueError):
        return str(value)

    credential_map = {
        1: "Certificate program",
        2: "Associate degree",
        3: "Bachelor's degree",
        4: "Post-baccalaureate certificate",
        5: "Master's degree",
        6: "Doctoral degree",
        7: "Professional doctorate (for example law or medicine)",
        8: "Graduate certificate",
        99: "Other or unclassified award level",
    }
    return credential_map.get(level, f"Credential level {level}")


def format_feature_value(column: str, value: object) -> str:
    if pd.isna(value) or value == "missing":
        return "Not available"
    if column == "credential_level":
        return format_credential_level(value)
    if column in {
        "distance",
        "locale",
        "carnegie_size_setting",
        "open_admissions_policy",
        "title_iv_eligibility_type",
        "selectivity_bucket",
    }:
        return get_value_label(column, value)
    if column in {"admission_rate_overall", "students_with_pell_grant"}:
        return format_percent(value)
    if column == "median_family_income":
        return format_currency(value)
    if column == "age_entry":
        return format_number(value)
    return str(value)


def shorten_text(value: object, limit: int = 36) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_form_value(value: object, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    return str(value)


def normalize_code_label(value: object, fallback: str = "") -> str:
    text = normalize_form_value(value, fallback=fallback).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        return text[:-2]
    return text


def get_value_label(column: str, value: object) -> str:
    normalized = normalize_code_label(value)
    label = VALUE_LABELS.get(column, {}).get(normalized)
    if label:
        return label
    return normalize_form_value(value, fallback="Not available")


def get_choice_label(column: str, value: object) -> str:
    if column == "credential_level":
        return format_credential_level(value)
    return format_feature_value(column, value)


def render_hero(title: str, description: str, chips: list[str]) -> None:
    chip_html = "".join(f"<span class='chip'>{chip}</span>" for chip in chips)
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
            <div class="chip-row">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_chart_label(row: pd.Series) -> str:
    return (
        f"{row['code']} | "
        f"{shorten_text(row['title'], 28)} | "
        f"{shorten_text(row['school_name'], 20)}"
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-top: #07111c;
            --bg-bottom: #0d1826;
            --text: #e7eef7;
            --muted: #98aabc;
            --panel: rgba(16, 26, 40, 0.94);
            --border: #22364a;
            --primary: #dbe7f5;
            --secondary: #7fb7d8;
            --highlight: #f3b64c;
            --good: #32b67a;
            --bad: #f06a5c;
        }
        .stApp {
            background: linear-gradient(180deg, var(--bg-top) 0%, #0a1521 50%, var(--bg-bottom) 100%);
            color: var(--text);
        }
        html, body, [class*="css"] {
            font-family: "Aptos", "Segoe UI", sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #0a1623 0%, #102235 52%, #163954 100%);
            border-radius: 22px;
            padding: 2rem 2.1rem;
            color: white;
            border: 1px solid rgba(127, 183, 216, 0.16);
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
            margin-bottom: 1.1rem;
        }
        .hero h1 {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2.4rem;
            line-height: 1.15;
            margin: 0 0 0.7rem 0;
            letter-spacing: 0.01em;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.18);
        }
        .hero p {
            margin: 0;
            max-width: 900px;
            line-height: 1.65;
            color: rgba(255, 255, 255, 0.98);
            font-size: 1rem;
        }
        .chip-row {
            margin-top: 1rem;
        }
        .chip {
            display: inline-block;
            margin: 0 0.45rem 0.45rem 0;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(127, 183, 216, 0.10);
            border: 1px solid rgba(127, 183, 216, 0.28);
            color: #eaf2fb;
            font-weight: 600;
            font-size: 0.86rem;
        }
        .metric-card {
            background: linear-gradient(180deg, #0f1a29 0%, #142132 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
            min-height: 122px;
        }
        .metric-label {
            color: #9db2c6;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: var(--highlight);
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.1;
            text-shadow: 0 0 18px rgba(243, 182, 76, 0.12);
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.35rem;
            line-height: 1.45;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.2);
            margin-bottom: 0.8rem;
        }
        .panel h3 {
            color: var(--primary);
            margin: 0 0 0.55rem 0;
            font-size: 1.08rem;
            font-weight: 800;
        }
        .panel p,
        .panel li {
            color: var(--text);
            line-height: 1.6;
            font-size: 0.98rem;
        }
        .panel ul {
            margin: 0.4rem 0 0 1.1rem;
            padding: 0;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.28rem 0.72rem;
            font-weight: 600;
            font-size: 0.84rem;
            margin-top: 0.5rem;
        }
        .status-good {
            background: rgba(30, 138, 91, 0.12);
            color: var(--good);
        }
        .status-bad {
            background: rgba(195, 74, 54, 0.12);
            color: var(--bad);
        }
        .status-neutral {
            background: rgba(47, 111, 143, 0.10);
            color: var(--secondary);
        }
        .section-note {
            color: #aab9c9;
            font-size: 0.95rem;
            font-weight: 500;
            margin-top: -0.2rem;
            margin-bottom: 0.9rem;
        }
        .guide-card,
        .spotlight-card,
        .summary-card {
            background: linear-gradient(180deg, rgba(18, 30, 46, 0.98) 0%, rgba(14, 24, 38, 0.98) 100%);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
            padding: 1.2rem 1.3rem;
            margin-bottom: 0.9rem;
        }
        .guide-card h3,
        .spotlight-card h3,
        .summary-card h3 {
            color: var(--primary);
            margin: 0 0 0.55rem 0;
            font-size: 1.1rem;
            font-weight: 800;
        }
        .guide-card p,
        .summary-card p {
            color: var(--text);
            line-height: 1.6;
            margin: 0.35rem 0;
            font-size: 0.97rem;
        }
        .spotlight-kicker {
            color: var(--secondary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .spotlight-value {
            color: var(--highlight);
            font-size: 2.6rem;
            line-height: 1.05;
            font-weight: 900;
            margin-bottom: 0.4rem;
            text-shadow: 0 0 18px rgba(243, 182, 76, 0.12);
        }
        .spotlight-card p {
            color: #dbe7f5;
            line-height: 1.6;
            margin: 0;
        }
        .summary-line {
            color: var(--text);
            line-height: 1.55;
            margin: 0.42rem 0;
            font-size: 0.97rem;
        }
        .summary-line strong {
            color: var(--primary);
        }
        [data-testid="stSidebar"] {
            background: rgba(12, 21, 32, 0.98);
            border-left: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(20, 33, 50, 0.96);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.5rem 1rem;
            color: #d8e3ef;
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] {
            background: #17324b;
            color: #ffffff;
            border-color: #2f5677;
        }
        .stButton > button,
        .stDownloadButton > button {
            background: #17324b;
            color: #f7fbff;
            border: 1px solid #2f5677;
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: #20405d;
            border-color: #3f6c92;
            color: white;
        }
        .stRadio [role="radiogroup"] {
            gap: 0.55rem;
            display: flex;
            flex-wrap: wrap;
        }
        .stRadio [role="radiogroup"] label {
            background: rgba(20, 33, 50, 0.96);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.5rem 0.85rem;
            min-height: 2.9rem;
            transition: all 0.18s ease;
        }
        .stRadio [role="radiogroup"] label:hover {
            border-color: #3f6c92;
            background: #17324b;
        }
        .stRadio [role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(180deg, #17324b 0%, #20425f 100%);
            border-color: #4d7aa3;
            box-shadow: 0 0 0 1px rgba(127, 183, 216, 0.16);
        }
        div[data-testid="stMetricValue"] {
            color: var(--highlight);
            font-weight: 800;
        }
        div[data-testid="stMetricLabel"] {
            color: #dbe7f5;
            font-weight: 700;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            background: rgba(16, 26, 40, 0.92);
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] label,
        .stCaption,
        .st-emotion-cache-10trblm,
        .st-emotion-cache-16idsys {
            color: var(--text);
        }
        .stSelectbox label,
        .stTextInput label,
        .stFileUploader label,
        .stCheckbox label {
            color: #dbe7f5 !important;
            font-weight: 600;
        }
        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stFileUploader section {
            background: #101a28;
            color: #e7eef7;
            border-color: var(--border);
        }
        .stAlert {
            background: rgba(16, 26, 40, 0.96);
            border: 1px solid var(--border);
        }
        hr {
            border-color: var(--border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(cards: list[dict[str, str]]) -> None:
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        note = f"<div class='metric-note'>{card['note']}</div>" if card.get("note") else ""
        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{card['label']}</div>
                <div class="metric-value">{card['value']}</div>
                {note}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_text_panel(title: str, items: list[str], body: str | None = None) -> None:
    list_items = "".join(f"<li>{item}</li>" for item in items)
    body_html = f"<p>{body}</p>" if body else ""
    st.markdown(
        f"""
        <div class="panel">
            <h3>{title}</h3>
            {body_html}
            <ul>{list_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(performance: str, gap: object = np.nan) -> str:
    gap_text = format_signed_currency(gap)
    if performance == "Above expected":
        return f"<span class='status-pill status-good'>Above expected ({gap_text})</span>"
    if performance == "Below expected":
        return f"<span class='status-pill status-bad'>Below expected ({gap_text})</span>"
    return "<span class='status-pill status-neutral'>No reported earnings</span>"


def stability_badge(status: str) -> str:
    if status == "Above expected across years":
        return "<span class='status-pill status-good'>Above expected across years</span>"
    if status == "Below expected across years":
        return "<span class='status-pill status-bad'>Below expected across years</span>"
    return "<span class='status-pill status-neutral'>No multi-year signal</span>"


def apply_chart_theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            labelColor="#dbe7f5",
            titleColor="#dbe7f5",
            labelFontSize=12,
            titleFontSize=13,
            gridColor="#27394a",
            tickColor="#6f869c",
        )
        .configure_legend(
            labelColor="#dbe7f5",
            titleColor="#dbe7f5",
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_title(
            color="#f2f7fb",
            fontSize=16,
            fontWeight="bold",
        )
        .configure(background="#0f1a29")
        .configure_view(strokeWidth=0, fill="#0f1a29")
    )


def build_scatter_chart(frame: pd.DataFrame, prediction_column: str) -> alt.Chart:
    scatter_sample = frame.sample(n=min(len(frame), 700), random_state=42).copy()
    ceiling = float(frame[[prediction_column, PRIMARY_ACTUAL_COLUMN]].max().max()) * 1.05

    scatter = (
        alt.Chart(scatter_sample)
        .mark_circle(size=95, opacity=0.9, stroke="#0f1a29", strokeWidth=1)
        .encode(
            x=alt.X(
                f"{prediction_column}:Q",
                title="Predicted 4-year earnings",
                axis=alt.Axis(format="$,.0f", tickCount=6),
            ),
            y=alt.Y(
                f"{PRIMARY_ACTUAL_COLUMN}:Q",
                title="Actual 4-year earnings",
                axis=alt.Axis(format="$,.0f", tickCount=6),
            ),
            color=alt.Color(
                "performance:N",
                title="Compared with expected",
                scale=alt.Scale(
                    domain=["Above expected", "Below expected"],
                    range=["#32b67a", "#f06a5c"],
                ),
            ),
            tooltip=[
                alt.Tooltip("code:N", title="CIP"),
                alt.Tooltip("title:N", title="Program"),
                alt.Tooltip("school_name:N", title="School"),
                alt.Tooltip(f"{prediction_column}:Q", title="Predicted", format="$,.0f"),
                alt.Tooltip(f"{PRIMARY_ACTUAL_COLUMN}:Q", title="Actual", format="$,.0f"),
                alt.Tooltip("gap:Q", title="Gap vs expected", format="$,.0f"),
            ],
        )
        .properties(height=420)
    )

    baseline = (
        alt.Chart(pd.DataFrame({"earnings": [0, ceiling]}))
        .mark_line(color="#8aa0b7", strokeDash=[5, 5], size=2.2)
        .encode(x="earnings:Q", y="earnings:Q")
    )

    return apply_chart_theme(scatter + baseline)


def build_extreme_gap_chart(frame: pd.DataFrame) -> alt.Chart:
    top_positive = frame.nlargest(5, "gap").copy()
    top_negative = frame.nsmallest(5, "gap").copy()
    chart_df = pd.concat([top_positive, top_negative], ignore_index=True).drop_duplicates()
    chart_df["short_label"] = chart_df.apply(build_chart_label, axis=1)
    domain_min = float(chart_df["gap"].min()) * 1.1
    domain_max = float(chart_df["gap"].max()) * 1.1

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=6, stroke="#0f1a29", strokeWidth=1)
        .encode(
            x=alt.X(
                "gap:Q",
                title="Actual minus expected",
                axis=alt.Axis(format="$,.0f"),
                scale=alt.Scale(domain=[domain_min, domain_max]),
            ),
            y=alt.Y(
                "short_label:N",
                sort=alt.EncodingSortField(field="gap", order="descending"),
                title="Program",
            ),
            color=alt.Color(
                "performance:N",
                title="Compared with expected",
                scale=alt.Scale(
                    domain=["Above expected", "Below expected"],
                    range=["#32b67a", "#f06a5c"],
                ),
            ),
            tooltip=[
                alt.Tooltip("code:N", title="CIP"),
                alt.Tooltip("title:N", title="Program"),
                alt.Tooltip("school_name:N", title="School"),
                alt.Tooltip("gap:Q", title="Gap vs expected", format="$,.0f"),
                alt.Tooltip(f"{PRIMARY_ACTUAL_COLUMN}:Q", title="Actual", format="$,.0f"),
            ],
        )
        .properties(height=380)
    )
    zero_line = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color="#8aa0b7", size=2)
        .encode(x="zero:Q")
    )
    return apply_chart_theme(bars + zero_line)


def build_program_comparison_chart(prediction: float, actual: float) -> alt.Chart:
    comparison_df = pd.DataFrame(
        {
            "Measure": ["Predicted", "Actual"],
            "Earnings": [prediction, actual],
        }
    )

    bars = (
        alt.Chart(comparison_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=68, stroke="#0f1a29", strokeWidth=1)
        .encode(
            x=alt.X("Measure:N", title=None),
            y=alt.Y("Earnings:Q", title="4-year earnings", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "Measure:N",
                legend=None,
                scale=alt.Scale(domain=["Predicted", "Actual"], range=["#58a6ff", "#f3b64c"]),
            ),
            tooltip=[
                alt.Tooltip("Measure:N"),
                alt.Tooltip("Earnings:Q", format="$,.0f"),
            ],
        )
    )

    labels = bars.mark_text(dy=-12, color="#eef5fb", fontWeight="bold").encode(
        text=alt.Text("Earnings:Q", format="$,.0f")
    )

    return apply_chart_theme((bars + labels).properties(height=320))


def filter_programs(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    query = query.strip()
    if not query:
        return frame

    terms = re.findall(r"[a-z0-9]+", query.lower())
    if not terms:
        return frame

    mask = pd.Series(True, index=frame.index)
    for term in terms:
        mask &= frame["search_text"].str.contains(re.escape(term), regex=True, na=False)

    filtered = frame[mask].copy()
    if filtered.empty:
        return filtered

    filtered["match_score"] = filtered.apply(
        lambda row: score_program_match(row, query, terms),
        axis=1,
    )
    return filtered.sort_values(
        ["match_score", "school_name", "title"],
        ascending=[False, True, True],
    )


def score_program_match(row: pd.Series, query: str, terms: list[str]) -> float:
    normalized_query = normalize_search_text(query)
    digit_query = "".join(ch for ch in query if ch.isdigit())
    code = str(row.get("code", ""))
    title_search = normalize_search_text(row.get("title", ""))
    school_search = normalize_search_text(row.get("school_name", ""))
    label_search = normalize_search_text(row.get("program_label", ""))
    search_text = str(row.get("search_text", ""))

    score = 0.0

    if digit_query:
        padded = digit_query.zfill(4)
        if code == padded:
            score += 500
        elif code.startswith(digit_query):
            score += 300
        elif digit_query in code:
            score += 180

    if normalized_query:
        if label_search.startswith(normalized_query):
            score += 260
        if title_search.startswith(normalized_query):
            score += 240
        if school_search.startswith(normalized_query):
            score += 220
        if f" {normalized_query}" in f" {title_search}":
            score += 120
        if f" {normalized_query}" in f" {school_search}":
            score += 110

        score += 60 * SequenceMatcher(None, normalized_query, title_search[: max(len(normalized_query), 1)]).ratio()
        score += 45 * SequenceMatcher(None, normalized_query, school_search[: max(len(normalized_query), 1)]).ratio()

    for term in terms:
        if title_search.startswith(term):
            score += 80
        elif school_search.startswith(term):
            score += 70
        elif code.startswith(term):
            score += 90
        elif f" {term}" in f" {search_text}":
            score += 25
        elif term in search_text:
            score += 10

    return score


def build_highlight_table(frame: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    top = frame.sort_values("gap", ascending=ascending).head(5).copy()
    return pd.DataFrame(
        {
            "CIP": top["code"],
            "Program": top["title"],
            "School": top["school_name"],
            "Actual": top[PRIMARY_ACTUAL_COLUMN].map(format_currency),
            "Predicted": top[PRIMARY_PREDICTION_COLUMN].map(format_currency),
            "Gap vs expected": top["gap"].map(format_signed_currency),
            "Multi-year pattern": top["stability_status"],
        }
    )


def render_sidebar(metadata: dict[str, object], scored_programs: pd.DataFrame) -> None:
    metrics = metadata["metrics"]
    actual_programs = scored_programs[PRIMARY_ACTUAL_COLUMN].notna().sum()
    stable_programs = scored_programs["median_score"].notna().sum()
    friendly_feature_labels = [FEATURE_LABELS.get(column, column) for column in metadata["feature_columns"]]

    st.sidebar.title("About this app")
    st.sidebar.write(
        "Use the page list above to switch between the dashboard, an individual prediction form, and batch scoring."
    )
    st.sidebar.metric("4-year model fit (R^2, log scale)", f"{metrics['r2_log']:.3f}")
    st.sidebar.metric("Typical 4-year error", format_currency(metrics["mae"]))
    st.sidebar.write(f"**Training rows:** {metrics['train_rows']:,}")
    st.sidebar.write(f"**Test rows:** {metrics['test_rows']:,}")
    st.sidebar.write(f"**Programs with reported earnings:** {actual_programs:,}")
    st.sidebar.write(f"**Programs with a multi-year pattern:** {stable_programs:,}")

    with st.sidebar.expander("What the model uses", expanded=False):
        st.write(", ".join(friendly_feature_labels))
    with st.sidebar.expander("Model results by year", expanded=False):
        st.dataframe(metadata["evaluation_table"], width="stretch", hide_index=True)


def render_dashboard() -> None:
    configure_page("Florida Program Earnings Dashboard")
    _, metadata, programs, actual_programs = load_app_state()

    total_programs = len(programs)
    total_schools = programs["school_name"].nunique()
    reported_programs = len(actual_programs)
    coverage_rate = reported_programs / total_programs if total_programs else 0
    median_gap = actual_programs["gap"].median()
    median_actual = actual_programs[PRIMARY_ACTUAL_COLUMN].median()
    above_count = int((actual_programs["gap"] >= 0).sum())
    below_count = int((actual_programs["gap"] < 0).sum())

    render_sidebar(metadata, programs)

    render_hero(
        "Florida Program Earnings Dashboard",
        "Compare Florida programs' reported earnings with what the model expected, then use the sidebar pages for individual or batch predictions.",
        ["XGBoost model", "Statewide comparison", "Individual predictions", "Batch predictions"],
    )

    render_metric_cards(
        [
            {
                "label": "Programs",
                "value": f"{total_programs:,}",
                "note": "Programs included in this dashboard.",
            },
            {
                "label": "Schools",
                "value": f"{total_schools:,}",
                "note": "Schools represented in this dashboard.",
            },
            {
                "label": "Reported 4-year earnings",
                "value": f"{reported_programs:,}",
                "note": f"{coverage_rate:.1%} of all programs can be compared with the 4-year prediction.",
            },
            {
                "label": "4-year XGBoost R2",
                "value": f"{metadata['metrics']['r2_log']:.3f}",
                "note": "Held-out accuracy for the 4-year model.",
            },
        ]
    )

    st.write("")
    overview_col, chart_col = st.columns([1, 1.55])

    with overview_col:
        render_text_panel(
            "How to read this",
            [
                "Green means reported 4-year earnings were above the model's prediction. Red means they were below.",
                "The multi-year label summarizes the saved 1-, 4-, and 5-year results.",
                f"Typical reported 4-year earnings: {format_currency(median_actual)}. Typical gap: {format_signed_currency(median_gap)}.",
            ],
            body=(
                "Built from College Scorecard data and the team's saved multi-year results. "
                f"{above_count:,} programs are above expected and {below_count:,} are below expected."
            ),
        )

    with chart_col:
        st.subheader("Statewide earnings pattern")
        st.markdown(
            "<div class='section-note'>Each dot is one program at one school. Dots above the dashed line had higher actual 4-year earnings than the model expected.</div>",
            unsafe_allow_html=True,
        )
        st.altair_chart(build_scatter_chart(actual_programs, PRIMARY_PREDICTION_COLUMN), width="stretch")

    st.subheader("Programs with the largest gaps")
    st.markdown(
        "<div class='section-note'>These are strong examples for the presentation because the difference between actual and expected earnings is easy to explain.</div>",
        unsafe_allow_html=True,
    )

    highlight_column_config = {
        "CIP": st.column_config.TextColumn("CIP", width="small"),
        "Program": st.column_config.TextColumn("Program", width="large"),
        "School": st.column_config.TextColumn("School", width="medium"),
        "Actual": st.column_config.TextColumn("Actual", width="small"),
        "Predicted": st.column_config.TextColumn("Predicted", width="small"),
        "Gap vs expected": st.column_config.TextColumn("Gap vs expected", width="small"),
        "Multi-year pattern": st.column_config.TextColumn("Multi-year pattern", width="medium"),
    }

    st.markdown("**Programs most above expected**")
    st.dataframe(
        build_highlight_table(actual_programs, ascending=False),
        width="stretch",
        hide_index=True,
        column_config=highlight_column_config,
        height=216,
    )

    st.markdown("**Programs most below expected**")
    st.dataframe(
        build_highlight_table(actual_programs, ascending=True),
        width="stretch",
        hide_index=True,
        column_config=highlight_column_config,
        height=216,
    )

    st.caption("This chart shows the biggest positive and negative gaps in one place.")
    st.altair_chart(build_extreme_gap_chart(actual_programs), width="stretch")


if __name__ == "__main__":
    render_dashboard()
