import pandas as pd
import numpy as np
import time

def clean (df: pd.DataFrame) -> pd.DataFrame:

    #----------------------------------------- Standardizing column names
    df.columns = df.columns.str.replace(".", "_")
    df.columns = (
        df.columns
            .str.replace(r"(earnings_|overall_|staff_grad_plus_all_eval_inst_|not_enrolled_)","",regex=True)
    )
    df.columns = df.columns.str.strip()
    before = df.copy()

    #----------------------------------------- Enforcing schema
    df["code"]=df["code"].astype(str)
    df["unit_id"]=df["unit_id"].astype(str)

    #----------------------------------------- Trimming + normalizing text fields
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    df = df.apply(lambda col: col.str.replace("  "," ") if col.dtype == "object" else col) #replaces double spaces

    #----------------------------------------- Handling missing and sentinel values
    df = df.apply(lambda col: col.fillna(pd.NA) if col.dtype == "object" else col) #replaces empty cells with null
    df = df.apply(lambda col: col.str.replace(r"(NA|N/A|na|Na|n/a|nan|NAN|Nan|Null|NULL)",pd.NA, regex = True) if col.dtype == "object" else col) #replaces sentinel values with null
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < 0, np.nan)

    #----------------------------------------- De-duplicating rows
    if df.duplicated().sum():
        print(f"{df.duplicated().sum()} duplicated rows found.")
        df = df.drop_duplicates()

    return df