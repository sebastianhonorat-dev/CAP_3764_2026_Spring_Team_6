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
    PRIMARY_ACTUAL_COLUMN,
    PRIMARY_PREDICTION_COLUMN,
    build_program_comparison_chart,
    configure_page,
    filter_programs,
    format_currency,
    format_credential_level,
    format_feature_value,
    format_signed_currency,
    get_choice_label,
    load_app_state,
    normalize_code_label,
    normalize_form_value,
    render_hero,
    render_metric_cards,
    render_sidebar,
    shorten_text,
    stability_badge,
    status_badge,
    TARGET_COLUMNS,
    YEAR_LABELS,
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


def build_prediction_only_table(selected_row: pd.Series, predictor) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for year in predictor.years:
        predicted = selected_row.get(predictor.prediction_columns_by_year[year])
        rows.append(
            {
                "Time after completion": YEAR_LABELS[year],
                "Estimated earnings": format_currency(predicted),
            }
        )
    return pd.DataFrame(rows)


def closest_numeric_option_index(options: list[dict[str, object]], default_value: float) -> int:
    if not options:
        return 0
    return min(
        range(len(options)),
        key=lambda index: abs(float(options[index]["value"]) - float(default_value)),
    )


def option_index_by_value(options: list[dict[str, object]], default_value: object) -> int:
    for index, option in enumerate(options):
        if option["value"] == default_value:
            return index
    return 0


def build_profile_choice_options(programs: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    admission = programs["admission_rate_overall"].dropna()
    pell = programs["students_with_pell_grant"].dropna()
    family_income = programs["median_family_income"].dropna()

    admission_quantiles = admission.quantile([0.2, 0.5, 0.8]).to_dict()
    pell_quantiles = pell.quantile([0.2, 0.5, 0.8]).to_dict()
    income_quantiles = family_income.quantile([0.2, 0.5, 0.8]).to_dict()

    return {
        "selectivity_bucket": [
            {"label": "Easier to get into", "value": "open"},
            {"label": "Somewhat selective", "value": "mid"},
            {"label": "More selective", "value": "elite"},
        ],
        "admission_rate_overall": [
            {"label": f"Lower admit rate | {admission_quantiles[0.2]:.0%}", "value": float(admission_quantiles[0.2])},
            {"label": f"Typical | {admission_quantiles[0.5]:.0%}", "value": float(admission_quantiles[0.5])},
            {"label": f"Higher admit rate | {admission_quantiles[0.8]:.0%}", "value": float(admission_quantiles[0.8])},
        ],
        "students_with_pell_grant": [
            {"label": f"Lower Pell share | {pell_quantiles[0.2]:.0%}", "value": float(pell_quantiles[0.2])},
            {"label": f"Typical | {pell_quantiles[0.5]:.0%}", "value": float(pell_quantiles[0.5])},
            {"label": f"Higher Pell share | {pell_quantiles[0.8]:.0%}", "value": float(pell_quantiles[0.8])},
        ],
        "median_family_income": [
            {"label": f"Lower family income | {format_currency(income_quantiles[0.2])}", "value": float(income_quantiles[0.2])},
            {"label": f"Typical | {format_currency(income_quantiles[0.5])}", "value": float(income_quantiles[0.5])},
            {"label": f"Higher family income | {format_currency(income_quantiles[0.8])}", "value": float(income_quantiles[0.8])},
        ],
    }


def build_cip_reference(programs: pd.DataFrame) -> pd.DataFrame:
    def first_common_title(series: pd.Series) -> str:
        clean = series.dropna().astype(str).str.strip()
        if clean.empty:
            return "Unknown program area"
        modes = clean.mode()
        if not modes.empty:
            return str(modes.iat[0])
        return str(clean.iat[0])

    cip_reference = (
        programs.assign(code=programs["code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(4))
        .dropna(subset=["code"])
        .groupby("code", as_index=False)
        .agg(
            sample_title=("title", first_common_title),
            school_count=("school_name", "nunique"),
            program_count=("title", "size"),
        )
    )
    cip_reference["option_label"] = (
        cip_reference["sample_title"].map(lambda value: shorten_text(value, 68))
        + " | CIP "
        + cip_reference["code"]
    )
    return cip_reference.sort_values(["sample_title", "code"]).reset_index(drop=True)


def build_single_prediction_summary_lines(scored_row: pd.Series) -> list[str]:
    return [
        (
            f"<strong>Program and school:</strong> "
            f"{format_feature_value('credential_level', scored_row.get('credential_level'))} "
            f"at a {format_feature_value('school_type', scored_row.get('school_type'))} school"
        ),
        (
            f"<strong>How hard it is to get in:</strong> "
            f"{format_feature_value('selectivity_bucket', scored_row.get('selectivity_bucket'))}; "
            f"admission rate {format_feature_value('admission_rate_overall', scored_row.get('admission_rate_overall'))}"
        ),
        (
            f"<strong>Student background:</strong> "
            f"{format_feature_value('students_with_pell_grant', scored_row.get('students_with_pell_grant'))} "
            f"of students receive Pell grants, with median family income "
            f"{format_feature_value('median_family_income', scored_row.get('median_family_income'))}"
        ),
        "<strong>Other details:</strong> Anything not shown here uses a typical Florida default.",
    ]


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


def render_program_explorer(programs: pd.DataFrame, actual_programs: pd.DataFrame, predictor) -> None:
    st.subheader("Program examples")
    st.markdown(
        "<div class='section-note'>Search by CIP code, school name, or program title. This tab shows real programs and how their reported earnings compared with the model.</div>",
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
        return

    preview_table = build_results_table(filtered_programs)
    preview_count = len(preview_table)
    total_matches = len(filtered_programs)
    if demo_choice == "Type my own search" and search.strip():
        st.caption("Best matches appear first.")
    if total_matches > preview_count:
        st.caption(f"Showing the first {preview_count} matches.")
    st.dataframe(preview_table, width="stretch")

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
            width="stretch",
        )

    st.markdown("**1-, 4-, and 5-year earnings**")
    st.caption(
        "These values represent median earnings of graduates who are working and not enrolled, "
        "measured 1, 4, and 5 years after completion. Each year is a separate snapshot, "
        "so the numbers do not always increase step by step."
    )
    st.dataframe(selected_year_table, width="stretch", hide_index=True)
    st.caption(
        "A higher number later on is common, but it is not guaranteed. The 1-, 4-, and 5-year values come from separate reported follow-up periods."
    )

    with st.expander("See the inputs behind this result", expanded=False):
        st.dataframe(
            build_feature_table(selected_row, predictor.feature_columns),
            width="stretch",
        )


def render_individual_prediction_form(predictor, metadata: dict[str, object], programs: pd.DataFrame) -> None:
    st.subheader("Estimate earnings for one program")
    st.markdown(
        "<div class='section-note'>We kept the main questions on the page so this stays easy to use. If you want, you can open More options and change a few extra details.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="guide-card">
            <h3>What you will enter</h3>
            <p>Program type, award type, school type, how selective the school is, Pell Grant share, and family income.</p>
            <p>For most quick estimates, that is enough.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_row = predictor.template_frame(rows=1).iloc[0]
    feature_options = metadata.get("feature_options", {})
    profile_options = build_profile_choice_options(programs)
    cip_reference = build_cip_reference(programs)
    cip_options = cip_reference["option_label"].tolist()
    cip_label_to_code = dict(zip(cip_reference["option_label"], cip_reference["code"]))
    cip_label_to_title = dict(zip(cip_reference["option_label"], cip_reference["sample_title"]))
    default_cip_code = normalize_code_label(template_row.get("code"))
    default_cip_label = next(
        (
            label
            for label, code in cip_label_to_code.items()
            if code == default_cip_code
        ),
        cip_options[0],
    )

    def select_input(label: str, column: str, help_text: str | None = None):
        options = [normalize_form_value(option) for option in feature_options.get(column, [])]
        default_value = normalize_form_value(template_row.get(column))
        if default_value and default_value not in options:
            options = [default_value] + options
        if not options:
            options = [default_value] if default_value else [""]
        default_index = options.index(default_value) if default_value in options else 0
        return st.selectbox(
            label,
            options=options,
            index=default_index,
            format_func=lambda option: get_choice_label(column, option),
            help=help_text,
        )

    with st.form("single_prediction_form"):
        title = st.text_input(
            "Program name (optional)",
            value="",
            placeholder="Example: Registered Nursing",
        )
        school_name = st.text_input(
            "School name (optional)",
            value="",
            placeholder="Example: Miami Dade College",
        )
        st.markdown("**Program type**")
        st.caption("Search by program name or CIP code. Try terms like nursing, business, culinary, welding, or 5138.")
        selected_cip_label = st.selectbox(
            "Program type",
            options=cip_options,
            index=cip_options.index(default_cip_label),
            help="This searchable list helps match program names to CIP codes.",
            label_visibility="collapsed",
        )
        code = cip_label_to_code[selected_cip_label]
        st.caption(f"Chosen program: {cip_label_to_title[selected_cip_label]} (CIP {code})")

        input_col_a, input_col_b = st.columns(2)
        with input_col_a:
            credential_level = select_input(
                "Award type",
                "credential_level",
                help_text="The type of credential students earn in this program.",
            )
            school_type = select_input(
                "School ownership",
                "school_type",
                help_text="Whether the school is public, nonprofit, or for-profit.",
            )
            st.markdown("**How selective the school is**")
            st.caption("Pick the option that feels closest.")
            selectivity_labels = [option["label"] for option in profile_options["selectivity_bucket"]]
            selectivity_choice = st.radio(
                "How selective the school is",
                options=selectivity_labels,
                index=option_index_by_value(
                    profile_options["selectivity_bucket"],
                    normalize_form_value(template_row.get("selectivity_bucket")),
                ),
                horizontal=True,
                label_visibility="collapsed",
            )
            selectivity_bucket = next(
                option["value"]
                for option in profile_options["selectivity_bucket"]
                if option["label"] == selectivity_choice
            )
        with input_col_b:
            st.markdown("**Admission rate**")
            st.caption("Choose the option that feels closest.")
            admission_labels = [option["label"] for option in profile_options["admission_rate_overall"]]
            admission_choice = st.radio(
                "Admission rate",
                options=admission_labels,
                index=closest_numeric_option_index(
                    profile_options["admission_rate_overall"],
                    float(template_row.get("admission_rate_overall", 0.0)),
                ),
                horizontal=True,
                label_visibility="collapsed",
            )
            admission_rate_overall = next(
                option["value"]
                for option in profile_options["admission_rate_overall"]
                if option["label"] == admission_choice
            )

            st.markdown("**Pell Grant share**")
            st.caption("This gives the model a sense of student financial need.")
            pell_labels = [option["label"] for option in profile_options["students_with_pell_grant"]]
            pell_choice = st.radio(
                "Pell Grant share",
                options=pell_labels,
                index=closest_numeric_option_index(
                    profile_options["students_with_pell_grant"],
                    float(template_row.get("students_with_pell_grant", 0.0)),
                ),
                horizontal=True,
                label_visibility="collapsed",
            )
            students_with_pell_grant = next(
                option["value"]
                for option in profile_options["students_with_pell_grant"]
                if option["label"] == pell_choice
            )

            st.markdown("**Median family income**")
            st.caption("Pick the option that feels closest.")
            income_labels = [option["label"] for option in profile_options["median_family_income"]]
            income_choice = st.radio(
                "Median family income",
                options=income_labels,
                index=closest_numeric_option_index(
                    profile_options["median_family_income"],
                    float(template_row.get("median_family_income", 0.0)),
                ),
                horizontal=True,
                label_visibility="collapsed",
            )
            median_family_income = next(
                option["value"]
                for option in profile_options["median_family_income"]
                if option["label"] == income_choice
            )

        with st.expander("More options (optional)", expanded=False):
            st.caption("Only open this if you want to change a few extra details.")
            fine_tune_a, fine_tune_b = st.columns(2)
            with fine_tune_a:
                age_entry = st.number_input(
                    "Average age at entry",
                    min_value=0.0,
                    value=float(template_row.get("age_entry", 0.0)),
                    step=0.5,
                    format="%.1f",
                    help="Typical age when students start at the institution.",
                )
                locale = select_input(
                    "Community type",
                    "locale",
                    help_text="The school's city, suburb, town, or rural setting.",
                )
                admission_rate_override = st.number_input(
                    "Exact admission rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(admission_rate_overall),
                    step=0.01,
                    format="%.2f",
                    help="Optional exact override for the preset admissions band.",
                )
                pell_override = st.number_input(
                    "Exact Pell share",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(students_with_pell_grant),
                    step=0.01,
                    format="%.2f",
                    help="Optional exact override for the preset Pell band.",
                )
            with fine_tune_b:
                distance = select_input(
                    "Online availability",
                    "distance",
                    help_text="How fully students can complete the field online.",
                )
                income_override = st.number_input(
                    "Exact median family income",
                    min_value=0.0,
                    value=float(median_family_income),
                    step=1000.0,
                    format="%.0f",
                    help="Optional exact override for the preset income band.",
                )
                carnegie_size_setting = select_input(
                    "School size and campus setting",
                    "carnegie_size_setting",
                    help_text="A general label for the size of the school and whether students mostly commute or live on campus.",
                )

        submitted = st.form_submit_button("Estimate earnings", width="stretch")

    if not submitted:
        return

    admission_rate_overall = admission_rate_override
    students_with_pell_grant = pell_override
    median_family_income = income_override
    carnegie_size_setting = carnegie_size_setting or normalize_form_value(template_row.get("carnegie_size_setting")) or None
    open_admissions_policy = normalize_form_value(template_row.get("open_admissions_policy")) or None
    title_iv_eligibility_type = normalize_form_value(template_row.get("title_iv_eligibility_type")) or None
    location_lat = float(template_row.get("location_lat", 0.0))
    location_lon = float(template_row.get("location_lon", 0.0))

    record = {
        "title": title or None,
        "school_name": school_name or None,
        "code": code or None,
        "credential_level": credential_level or None,
        "school_type": school_type or None,
        "distance": distance or None,
        "locale": locale or None,
        "carnegie_size_setting": carnegie_size_setting or None,
        "open_admissions_policy": open_admissions_policy or None,
        "title_iv_eligibility_type": title_iv_eligibility_type or None,
        "selectivity_bucket": selectivity_bucket or None,
        "admission_rate_overall": admission_rate_overall,
        "students_with_pell_grant": students_with_pell_grant,
        "median_family_income": median_family_income,
        "age_entry": age_entry,
        "location_lat": location_lat,
        "location_lon": location_lon,
    }
    scored_single = predictor.predict_frame(pd.DataFrame([record])).iloc[0]

    display_program = title.strip() if title.strip() else cip_label_to_title[selected_cip_label]
    display_school = school_name.strip() if school_name.strip() else "the school details you entered"
    summary_lines = "".join(
        f"<div class='summary-line'>{line}</div>"
        for line in build_single_prediction_summary_lines(scored_single)
    )

    st.markdown(
        f"""
        <div class="spotlight-card">
            <div class="spotlight-kicker">Best estimate</div>
            <div class="spotlight-value">{format_currency(scored_single[predictor.prediction_columns_by_year[4]])}</div>
            <p>
                Estimated median earnings about 4 years after finishing <strong>{display_program}</strong>,
                based on the details entered for <strong>{display_school}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_metric_cards(
        [
            {
                "label": "1 year after completion",
                "value": format_currency(scored_single[predictor.prediction_columns_by_year[1]]),
                "note": "Early-career estimate.",
            },
            {
                "label": "4 years after completion",
                "value": format_currency(scored_single[predictor.prediction_columns_by_year[4]]),
                "note": "Main number to focus on.",
            },
            {
                "label": "5 years after completion",
                "value": format_currency(scored_single[predictor.prediction_columns_by_year[5]]),
                "note": "A later-career estimate.",
            },
        ]
    )

    summary_col, table_col = st.columns([1.05, 0.95])
    with summary_col:
        st.markdown(
            f"""
            <div class="summary-card">
                <h3>What we used</h3>
                {summary_lines}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with table_col:
        st.markdown("**Estimated earnings over time**")
        st.dataframe(
            build_prediction_only_table(scored_single, predictor),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "These are model estimates, not actual reported earnings. If you want to compare real programs with the model, open the Real programs tab."
    )

    with st.expander("See all the details used for this estimate", expanded=False):
        st.dataframe(
            build_feature_table(scored_single, predictor.feature_columns),
            width="stretch",
            hide_index=True,
        )


def render_individual_predictions_page() -> None:
    configure_page("Florida Program Earnings | Individual Predictions")
    predictor, metadata, programs, actual_programs = load_app_state()
    render_sidebar(metadata, programs)
    render_hero(
        "Estimate Program Earnings",
        "Use the quick form to estimate earnings, or open the Real programs tab to compare actual programs with the model.",
        ["Quick form", "1-, 4-, and 5-year estimates"],
    )

    manual_tab, explorer_tab = st.tabs(["Quick estimate", "Examples"])
    with manual_tab:
        render_individual_prediction_form(predictor, metadata, programs)
    with explorer_tab:
        render_program_explorer(programs, actual_programs, predictor)


render_individual_predictions_page()
