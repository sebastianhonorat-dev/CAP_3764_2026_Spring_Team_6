import pandas as pd
import numpy as np

from ira.logging_utils import get_logger


logger = get_logger(__name__)

def clean (df: pd.DataFrame) -> pd.DataFrame:
    original_shape = df.shape
    logger.info("Starting clean() with shape=%s", original_shape)

    #----------------------------------------- Standardizing column names
    df.columns = df.columns.str.replace(".", "_")

    df.columns = (
        df.columns
            .str.replace(
                r"(earnings_|overall_|staff_grad_plus_all_eval_inst_|not_enrolled_|latest_school_|latest_student_demographics_|latest_student_|latest_school_|latest_admissions_)"
                ,""
                ,regex=True
            )
    )

    df.columns = df.columns.str.strip()
    #----------------------------------------- Enforcing schema
    for col in ["code", "unit_id"]: # explicit string changes
        if col in df.columns:
            df[col] = df[col].astype("string")        

    for col in ["students_with_pell_grant", "admission_rate_overall"]: # explicit numeric changes
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
            )

            df[col] = pd.to_numeric(df[col], errors="coerce")

    median_cols = df.columns[df.columns.str.contains("median", case=False, na=False)]
    df[median_cols] = df[median_cols].apply(pd.to_numeric, errors="coerce") 
    # ----------------------------------------- Text cleaning (object + string)
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    sentinel = r"^(NA|N/A|na|Na|n/a|nan|NAN|Nan|Null|NULL|null|None|NONE|none)$"

    if len(str_cols) > 0:
        df[str_cols] = df[str_cols].astype("string")
        df[str_cols] = df[str_cols].replace(sentinel, pd.NA, regex=True)
        df[str_cols] = df[str_cols].replace(r"^\s*$", pd.NA, regex=True)
        df[str_cols] = df[str_cols].apply(
            lambda s: s.str.strip().str.replace("  ", " ", regex=False)
        )
    # ----------------------------------------- Numeric cleanup
    numeric_cols = df.select_dtypes(include="number").columns
    logger.info("Applying numeric cleanup to %s numeric columns", len(numeric_cols))
    df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < 0, np.nan)

    #----------------------------------------- De-duplicating rows
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        logger.warning("Removing %s duplicated rows", duplicate_count)
        df = df.drop_duplicates()

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if not missing_counts.empty:
        top_missing = ", ".join(
            f"{column}={count}" for column, count in missing_counts.head(5).items()
        )
        logger.warning(
            "Missing values remain in %s columns after cleaning; top columns: %s",
            len(missing_counts),
            top_missing,
        )

    logger.info("Completed clean() with shape=%s", df.shape)
    logger.info("Cleaning shape summary: %s -> %s", original_shape, df.shape)

    return df
