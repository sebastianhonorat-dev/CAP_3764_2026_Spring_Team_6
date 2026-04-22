from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ira.streamlit_app import (
    FEATURE_LABELS,
    configure_page,
    format_currency,
    load_app_state,
    render_metric_cards,
    render_sidebar,
)


def inject_batch_compact_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1.2rem;
        }
        .hero {
            padding: 1.1rem 1.25rem;
            border-radius: 18px;
            margin-bottom: 0.7rem;
        }
        .hero h1 {
            font-size: 1.8rem;
            margin-bottom: 0.35rem;
        }
        .hero p {
            font-size: 0.94rem;
            line-height: 1.45;
        }
        .chip-row {
            margin-top: 0.55rem;
        }
        .chip {
            padding: 0.32rem 0.62rem;
            margin-bottom: 0.25rem;
            font-size: 0.8rem;
        }
        .metric-card,
        .panel,
        .guide-card,
        .summary-card,
        .spotlight-card {
            padding: 0.9rem 1rem;
            margin-bottom: 0.55rem;
            border-radius: 16px;
        }
        .metric-card {
            min-height: 92px;
        }
        .metric-label {
            font-size: 0.82rem;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            font-size: 1.45rem;
        }
        .metric-note,
        .guide-card p,
        .summary-card p,
        .panel p {
            font-size: 0.9rem;
            line-height: 1.42;
        }
        .guide-card h3,
        .summary-card h3,
        .spotlight-card h3,
        .panel h3 {
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }
        .spotlight-value {
            font-size: 2rem;
            margin-bottom: 0.2rem;
        }
        .section-note {
            margin-bottom: 0.55rem;
            font-size: 0.9rem;
        }
        div[data-testid="stFileUploader"] {
            margin-bottom: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_required_columns_table(predictor) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Required column": predictor.feature_columns,
            "What it means": [FEATURE_LABELS.get(column, column) for column in predictor.feature_columns],
        }
    )


def build_batch_preview_table(scored_batch: pd.DataFrame, predictor) -> pd.DataFrame:
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

    preview_frame = scored_batch[preview_columns].head(5).copy().rename(
        columns={
            "code": "CIP",
            "title": "Program",
            "school_name": "School",
            "credential_level": "Award type",
            predictor.prediction_columns_by_year[1]: "Predicted 1-year earnings",
            predictor.prediction_columns_by_year[4]: "Predicted 4-year earnings",
            predictor.prediction_columns_by_year[5]: "Predicted 5-year earnings",
        }
    )

    for column in [
        "Predicted 1-year earnings",
        "Predicted 4-year earnings",
        "Predicted 5-year earnings",
    ]:
        if column in preview_frame.columns:
            preview_frame[column] = preview_frame[column].map(format_currency)

    return preview_frame


def render_batch_prediction_workspace(predictor) -> None:
    info_col, stats_col = st.columns([1.25, 1])
    with info_col:
        st.markdown(
            """
            <div class="guide-card">
                <h3>Quick flow</h3>
                <p>Download the template, fill one row per program, upload the file, and download the scored CSV.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stats_col:
        render_metric_cards(
            [
                {
                    "label": "Required fields",
                    "value": f"{len(predictor.feature_columns)}",
                    "note": "Use the template columns as-is.",
                },
                {
                    "label": "Predictions",
                    "value": "1, 4, 5",
                    "note": "Years added to each row.",
                },
            ]
        )

    batch_left, batch_right = st.columns([0.92, 1.08], gap="large")
    template_df = predictor.template_frame(rows=25)
    required_columns_table = build_required_columns_table(predictor)

    with batch_left:
        st.markdown(
            """
            <div class="panel">
                <h3>Step 1: Download the template</h3>
                <p>Start here so your file has the right columns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download template CSV",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="sample_program_input.csv",
            mime="text/csv",
            width="stretch",
        )
        st.markdown(
            """
            <div class="summary-card">
                <h3>Checklist</h3>
                <p>One row = one program.</p>
                <p>Keep the same column names.</p>
                <p>Extra columns can stay.</p>
                <p>The template includes 25 sample rows, so the results preview only shows part of the file.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Columns the model needs", expanded=False):
            st.dataframe(
                required_columns_table,
                width="stretch",
                hide_index=True,
                height=240,
            )
        with st.expander("Preview the template", expanded=False):
            st.dataframe(template_df.head(4), width="stretch", hide_index=True, height=180)

    with batch_right:
        st.markdown(
            """
            <div class="panel">
                <h3>Step 2: Upload your file</h3>
                <p>Upload the CSV and the app will score every row.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload a CSV",
            type=["csv"],
            help="The easiest path is to fill in the template and upload it here.",
        )

        if uploaded_file is None:
            st.markdown(
                """
                <div class="summary-card">
                    <h3>Output</h3>
                    <p>Your original file stays the same, and the app adds predicted 1-, 4-, and 5-year earnings.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        try:
            batch = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error("We could not read that file as a CSV.")
            st.caption(str(exc))
            return

        missing_columns = [column for column in predictor.feature_columns if column not in batch.columns]

        if missing_columns:
            st.error("Your file is missing required columns.")
            st.dataframe(
                required_columns_table[required_columns_table["Required column"].isin(missing_columns)],
                width="stretch",
                hide_index=True,
                height=220,
            )
            st.caption("Download the template and keep the same column names. Extra columns are okay.")
            return

        scored_batch = predictor.predict_frame(batch)
        preview_frame = build_batch_preview_table(scored_batch, predictor)
        summary_left, summary_right = st.columns([1.2, 0.8])
        with summary_left:
            st.success(f"Done. {len(scored_batch):,} rows were scored.")
        with summary_right:
            st.download_button(
                "Download scored CSV",
                data=scored_batch.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                width="stretch",
            )

        st.markdown(
            f"""
            <div class="summary-card">
                <h3>Preview</h3>
                <p>Showing the first 5 rows only. The download includes all <strong>{len(scored_batch):,}</strong> rows.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(preview_frame, width="stretch", hide_index=True, height=260)
        st.download_button(
            "Download template again",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="sample_program_input.csv",
            mime="text/csv",
            width="stretch",
        )


def render_batch_predictions_page() -> None:
    configure_page("Florida Program Earnings | Batch Predictions")
    inject_batch_compact_styles()
    predictor, metadata, programs, _ = load_app_state()
    render_sidebar(metadata, programs)
    st.markdown(
        """
        <div class="hero">
            <h1>Batch Predictions</h1>
            <p>Upload one CSV, score every row, and download the finished file with predicted 1-, 4-, and 5-year earnings.</p>
            <div class="chip-row">
                <span class="chip">CSV upload</span>
                <span class="chip">Template download</span>
                <span class="chip">Bulk scoring</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_batch_prediction_workspace(predictor)


render_batch_predictions_page()
