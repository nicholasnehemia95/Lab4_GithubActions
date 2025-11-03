import argparse
import os
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

#add comment to enable push

def load_data(input_path: str | None) -> pd.DataFrame:
    if input_path:
        df = pd.read_csv(input_path)
        return df
    # Fallback: California Housing (features in X, target y)
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    # Make the target column name explicit
    df.rename(columns={"MedHouseVal": "target_MedHouseVal"}, inplace=True)
    return df

def build_transformer(df: pd.DataFrame, add_poly: bool) -> ColumnTransformer:
    # choose numeric columns only; drop target from transform
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("target_")]

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if add_poly:
        # keep polynomial simple to avoid explosion
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    num_pipe = Pipeline(steps)
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre
c
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Example feature on California Housing columns (works only if present)
    # If user provides a CSV without these columns, we guard with try/except.
    df = df.copy()
    try:
        if {"AveRooms", "AveOccup"}.issubset(df.columns):
            df["rooms_per_occupant"] = (df["AveRooms"] / df["AveOccup"]).replace([float("inf"), -float("inf")], pd.NA)
    except Exception:
        # Silently skip if columns aren't present or division issue
        pass
    return df

def transform_dataframe(df: pd.DataFrame, add_poly: bool) -> pd.DataFrame:
    df_eng = engineer_features(df)

    # separate target(s) if present
    target_cols = [c for c in df_eng.columns if c.startswith("target_")]
    X = df_eng.drop(columns=target_cols, errors="ignore")

    pre = build_transformer(df_eng, add_poly=add_poly)
    Xt = pre.fit_transform(X)

    # Get nice column names back from ColumnTransformer
    out_cols = pre.get_feature_names_out()
    X_out = pd.DataFrame(Xt, columns=out_cols)

    # reattach targets (untransformed) at the end for convenience
    for t in target_cols:
        X_out[t] = df_eng[t].values

    return X_out

def main():
    parser = argparse.ArgumentParser(description="Tiny data transformation CLI (replaces calculator).")
    parser.add_argument("--input", type=str, default=None, help="Optional CSV path. If omitted, use California Housing.")
    parser.add_argument("--out", type=str, default="data/processed/housing_transformed.csv", help="Output CSV path.")
    parser.add_argument("--poly", action="store_true", help="Include polynomial features (degree=2).")
    args = parser.parse_args()

    # Ensure output folder exists
    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    df = load_data(args.input)
    df_out = transform_dataframe(df, add_poly=args.poly)
    df_out.to_csv(out_path, index=False)

    print(f"✅ Wrote {len(df_out):,} rows × {df_out.shape[1]} columns to {out_path}")

if __name__ == "__main__":
    main()
