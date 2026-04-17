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
    "code": "CIP code",
    "distance": "Distance learning code",
    "school_type": "School type",
    "credential_level": "Credential level",
    "locale": "Locale code",
    "carnegie_size_setting": "Carnegie size and setting",
    "open_admissions_policy": "Open admissions policy",
    "title_iv_eligibility_type": "Title IV eligibility",
    "selectivity_bucket": "Selectivity group",
    "admission_rate_overall": "Admission rate",
    "location_lat": "Latitude",
    "location_lon": "Longitude",
    "median_family_income": "Median family income",
    "students_with_pell_grant": "Students with Pell grant",
    "age_entry": "Average age at entry",
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


def build_chart_label(row: pd.Series) -> str:
    return (
        f"{row['code']} | "
        f"{shorten_text(row['title'], 28)} | "
        f"{shorten_text(row['school_name'], 20)}"
    )


def build_program_option_label(row: pd.Series) -> str:
    actual_text = format_currency(row.get(PRIMARY_ACTUAL_COLUMN, row.get("4_yr_median_earnings")))
    gap_text = format_signed_currency(row.get("gap"))
    return (
        f"{row['code']} | "
        f"{shorten_text(row['title'], 50)} | "
        f"{shorten_text(row['school_name'], 30)} | "
        f"Actual: {actual_text} | Gap: {gap_text}"
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


def build_results_table(frame: pd.DataFrame, max_rows: int = 12) -> pd.DataFrame:
    preview = frame.copy()
    if "match_score" in preview.columns:
        preview = preview.head(max_rows)
    else:
        preview["sort_gap"] = preview["gap"].abs().fillna(-1)
        preview = preview.sort_values(["sort_gap", "title"], ascending=[False, True]).head(max_rows)
    return pd.DataFrame(
        {
            "CIP": preview["code"],
            "Program": preview["title"],
            "School": preview["school_name"],
            "Actual": preview[PRIMARY_ACTUAL_COLUMN].map(format_currency),
            "Predicted": preview[PRIMARY_PREDICTION_COLUMN].map(format_currency),
            "Gap vs expected": preview["gap"].map(format_signed_currency),
            "Compared with expected": preview["performance"],
            "Multi-year pattern": preview["stability_status"],
        }
    )


def build_feature_table(selected_row: pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS.get(column, column) for column in feature_columns],
            "Value": [format_feature_value(column, selected_row.get(column)) for column in feature_columns],
        }
    )


def build_year_summary_table(selected_row: pd.Series, predictor) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for year in predictor.years:
        actual = selected_row.get(f"{year}_year_earning")
        if pd.isna(actual):
            actual = selected_row.get(TARGET_COLUMNS[year])

        predicted = selected_row.get(f"{year}_year_pred")
        if pd.isna(predicted):
            predicted = selected_row.get(predictor.prediction_columns_by_year[year])

        gap = selected_row.get(f"{year}_year_error")
        if pd.isna(gap) and pd.notna(actual) and pd.notna(predicted):
            gap = float(actual) - float(predicted)

        rows.append(
            {
                "Time after completion": YEAR_LABELS[year],
                "Actual": format_currency(actual),
                "Predicted": format_currency(predicted),
                "Gap": format_signed_currency(gap),
            }
        )

    return pd.DataFrame(rows)


def build_demo_examples(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}

    examples: dict[str, int] = {}

    selections = [
        ("Recommended: Above expected", frame.nlargest(1, "gap").iloc[0]),
        ("Recommended: Below expected", frame.nsmallest(1, "gap").iloc[0]),
    ]

    for prefix, row in selections:
        label = (
            f"{prefix} | {row['code']} | "
            f"{shorten_text(row['title'], 30)} | {shorten_text(row['school_name'], 22)}"
        )
        examples[label] = int(row.name)

    return examples


def render_sidebar(metadata: dict[str, object], scored_programs: pd.DataFrame) -> None:
    metrics = metadata["metrics"]
    actual_programs = scored_programs[PRIMARY_ACTUAL_COLUMN].notna().sum()
    stable_programs = scored_programs["median_score"].notna().sum()
    friendly_feature_labels = [FEATURE_LABELS.get(column, column) for column in metadata["feature_columns"]]

    st.sidebar.title("About this dashboard")
    st.sidebar.write(
        "This dashboard predicts Florida program earnings and shows which programs are above or below expected earnings across 1, 4, and 5 years."
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
        st.dataframe(metadata["evaluation_table"], use_container_width=True, hide_index=True)


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Florida Program Earnings Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    predictor = get_predictor()
    metadata = predictor.metadata()
    metadata["evaluation_table"] = predictor.evaluation_table().rename(
        columns={"R2 (log scale)": "R^2 (log scale)"}
    )
    programs = load_scored_programs(str(DATA_PATH), predictor)
    actual_programs = programs.dropna(subset=["gap"]).copy()

    total_programs = len(programs)
    total_schools = programs["school_name"].nunique()
    reported_programs = len(actual_programs)
    coverage_rate = reported_programs / total_programs if total_programs else 0
    median_gap = actual_programs["gap"].median()
    median_actual = actual_programs[PRIMARY_ACTUAL_COLUMN].median()
    above_count = int((actual_programs["gap"] >= 0).sum())
    below_count = int((actual_programs["gap"] < 0).sum())

    render_sidebar(metadata, programs)

    st.markdown(
        """
        <div class="hero">
            <h1>Florida Program Earnings Dashboard</h1>
            <p>
                Compare Florida programs' reported earnings with what the model expected. Search by school
                or program, review the 1-, 4-, and 5-year results, and upload a CSV to score new rows.
            </p>
            <div class="chip-row">
                <span class="chip">XGBoost model</span>
                <span class="chip">1-, 4-, and 5-year view</span>
                <span class="chip">Program search</span>
                <span class="chip">Batch upload</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        st.altair_chart(build_scatter_chart(actual_programs, PRIMARY_PREDICTION_COLUMN), use_container_width=True)

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
        use_container_width=True,
        hide_index=True,
        column_config=highlight_column_config,
        height=216,
    )

    st.markdown("**Programs most below expected**")
    st.dataframe(
        build_highlight_table(actual_programs, ascending=True),
        use_container_width=True,
        hide_index=True,
        column_config=highlight_column_config,
        height=216,
    )

    st.caption("This chart shows the biggest positive and negative gaps in one place.")
    st.altair_chart(build_extreme_gap_chart(actual_programs), use_container_width=True)

    explore_tab, batch_tab = st.tabs(["Explore programs", "Batch upload"])

    with explore_tab:
        st.subheader("Find a program")
        st.markdown(
            "<div class='section-note'>Search by CIP code, school name, or program title. For the presentation, the simplest flow is: search, pick a match, and explain the gap.</div>",
            unsafe_allow_html=True,
        )

        demo_examples = build_demo_examples(actual_programs)
        demo_choice = st.selectbox(
            "Quick demo pick",
            options=["Type my own search"] + list(demo_examples.keys()),
            index=1 if demo_examples else 0,
        )

        search = st.text_input(
            "Search programs",
            value="",
            placeholder="Example: 1205, Atlantic Technical, nursing, culinary",
            disabled=demo_choice != "Type my own search",
        )

        if demo_choice == "Type my own search":
            filtered_programs = filter_programs(programs, search)
        else:
            filtered_programs = programs.loc[[demo_examples[demo_choice]]].copy()
            st.caption("Showing a preselected example to keep the demo quick and reliable.")

        st.write(f"Matches found: {len(filtered_programs):,}")

        if filtered_programs.empty:
            st.warning("No matches found. Try a CIP code, school name, or a few words from the program title.")
        else:
            preview_table = build_results_table(filtered_programs)
            preview_count = len(preview_table)
            total_matches = len(filtered_programs)
            if demo_choice == "Type my own search" and search.strip():
                st.caption("Best matches appear first.")
            if total_matches > preview_count:
                st.caption(f"Showing the first {preview_count} matches.")
            st.dataframe(preview_table, use_container_width=True)

            option_ids = filtered_programs.index.tolist()
            selected_index = 0
            search_code = "".join(ch for ch in search if ch.isdigit())
            if search_code:
                exact_match = filtered_programs[filtered_programs["code"] == search_code.zfill(4)]
                if len(exact_match) == 1:
                    selected_index = option_ids.index(int(exact_match.index[0]))

            selected_id = st.selectbox(
                "Choose a program",
                options=option_ids,
                index=selected_index,
                format_func=lambda row_id: build_program_option_label(filtered_programs.loc[row_id]),
            )
            selected_row = filtered_programs.loc[selected_id]

            selected_prediction = selected_row[PRIMARY_PREDICTION_COLUMN]
            selected_actual = selected_row[PRIMARY_ACTUAL_COLUMN]
            selected_gap = selected_row["gap"]
            selected_year_table = build_year_summary_table(selected_row, predictor)
            confidence_text = str(selected_row.get("confidence", "Not available"))
            median_score_text = (
                f"{float(selected_row['median_score']):.3f}"
                if pd.notna(selected_row.get("median_score"))
                else "Not available"
            )
            rank_volatility_text = (
                f"{float(selected_row['mean_rank_std_pct']):.3f}"
                if pd.notna(selected_row.get("mean_rank_std_pct"))
                else "Not available"
            )

            detail_col, metric_col = st.columns([1.2, 1])
            with detail_col:
                st.markdown(
                    f"""
                    <div class="panel">
                        <h3>{selected_row['title']}</h3>
                        <p>
                            <strong>School:</strong> {selected_row['school_name']}<br>
                            <strong>CIP code:</strong> {selected_row['code']}<br>
                            <strong>Award type:</strong> {format_credential_level(selected_row.get('credential_level'))}<br>
                            <strong>Selectivity group:</strong> {format_feature_value('selectivity_bucket', selected_row.get('selectivity_bucket'))}<br>
                            <strong>Admission rate:</strong> {format_feature_value('admission_rate_overall', selected_row.get('admission_rate_overall'))}<br>
                            <strong>Median family income:</strong> {format_feature_value('median_family_income', selected_row.get('median_family_income'))}<br>
                            <strong>How strong this pattern looks:</strong> {confidence_text}<br>
                            <strong>Overall multi-year score:</strong> {median_score_text}<br>
                            <strong>How much the ranking changes across years:</strong> {rank_volatility_text}
                        </p>
                        {status_badge(selected_row['performance'], selected_gap)}
                        <div style="margin-top:0.45rem;">{stability_badge(selected_row['stability_status'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with metric_col:
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Predicted 4-year earnings", format_currency(selected_prediction))
                metric_b.metric("Actual 4-year earnings", format_currency(selected_actual))
                metric_c.metric("4-year gap", format_signed_currency(selected_gap))

                if pd.notna(selected_gap):
                    if selected_gap >= 0:
                        st.success(
                            f"Actual 4-year earnings were {format_signed_currency(selected_gap)} above the model's prediction."
                        )
                    else:
                        st.error(
                            f"Actual 4-year earnings were {format_signed_currency(selected_gap)} below the model's prediction."
                        )
                else:
                    st.info("There is no saved 4-year comparison for this program, so only the new prediction is shown.")

            if pd.notna(selected_actual):
                st.altair_chart(
                    build_program_comparison_chart(float(selected_prediction), float(selected_actual)),
                    use_container_width=True,
                )

            st.markdown("**1-, 4-, and 5-year earnings**")
            st.caption(
                "These values represent median earnings of graduates who are working and not enrolled, "
                "measured 1, 4, and 5 years after completion. Each year is a separate snapshot, "
                "so the numbers do not always increase step by step."
            )
            st.dataframe(selected_year_table, use_container_width=True, hide_index=True)
            st.caption(
                "A higher number later on is common, but it is not guaranteed. The 1-, 4-, and 5-year values come from separate reported follow-up periods."
            )

            with st.expander("What the model used for this prediction", expanded=False):
                st.dataframe(
                    build_feature_table(selected_row, predictor.feature_columns),
                    use_container_width=True,
                )

    with batch_tab:
        st.subheader("Upload a file to score programs")
        st.markdown(
            "<div class='section-note'>Use this during the presentation: download the template, upload a CSV, and then download the results.</div>",
            unsafe_allow_html=True,
        )

        batch_left, batch_right = st.columns([1, 1.2])
        template_df = programs[predictor.feature_columns].head(10).copy()

        with batch_left:
            st.markdown(
                """
                <div class="panel">
                    <h3>Step 1: Download the template</h3>
                    <p>This sample file already has the columns the model needs, in the right order.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download template CSV",
                data=template_df.to_csv(index=False).encode("utf-8"),
                file_name="sample_program_input.csv",
                mime="text/csv",
            )

            with st.expander("Columns the model needs", expanded=False):
                st.dataframe(
                    pd.DataFrame({"Required column": predictor.feature_columns}),
                    use_container_width=True,
                )

        with batch_right:
            st.markdown(
                """
                <div class="panel">
                    <h3>Step 2: Upload your file</h3>
                    <p>Upload a CSV with those columns. The app will preview the predictions and let you download the full scored file.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

            if uploaded_file is not None:
                batch = pd.read_csv(uploaded_file)
                missing_columns = [column for column in predictor.feature_columns if column not in batch.columns]

                if missing_columns:
                    st.error(f"Your file is missing these columns: {', '.join(missing_columns)}")
                else:
                    scored_batch = predictor.predict_frame(batch)
                    preview_columns = [
                        column
                        for column in [
                            "code",
                            "title",
                            "school_name",
                            "credential_level",
                            predictor.prediction_columns_by_year[1],
                            predictor.prediction_columns_by_year[4],
                            predictor.prediction_columns_by_year[5],
                        ]
                        if column in scored_batch.columns
                    ]
                    preview_frame = scored_batch[preview_columns].head(10).rename(
                        columns={
                            "code": "CIP",
                            "title": "Program",
                            "school_name": "School",
                            "credential_level": "Credential level",
                            predictor.prediction_columns_by_year[1]: "Predicted 1-year earnings",
                            predictor.prediction_columns_by_year[4]: "Predicted 4-year earnings",
                            predictor.prediction_columns_by_year[5]: "Predicted 5-year earnings",
                        }
                    )

                    preview_a, preview_b = st.columns(2)
                    preview_a.metric("Rows scored", f"{len(scored_batch):,}")
                    preview_b.metric("Predictions returned", "1-, 4-, and 5-year earnings")

                    st.dataframe(preview_frame, use_container_width=True)
                    st.download_button(
                        "Download scored CSV",
                        data=scored_batch.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    render_dashboard()
