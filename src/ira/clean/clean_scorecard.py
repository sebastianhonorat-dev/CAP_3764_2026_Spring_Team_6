import pandas as pd
import numpy as np
import time

def clean (df: pd.DataFrame) -> pd.DataFrame:

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
    print(f"Numeric columns: {numeric_cols}")
    df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < 0, np.nan)

    #----------------------------------------- De-duplicating rows
    if df.duplicated().sum():
        print(f"{df.duplicated().sum()} duplicated rows found.\n{df[df.duplicated()==True]}")
        # df = df.drop_duplicates()

    return df