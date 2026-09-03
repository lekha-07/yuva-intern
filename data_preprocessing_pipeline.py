"""
Data Collection, Cleaning, and Preprocessing for Logistics Analysis
----------------------------------------------------------------------
Week 2 Task — standalone illustration of the preprocessing pipeline
described in the report. Runs on synthetic sample data modeled on
public logistics datasets (e.g., Kaggle's DataCo Smart Supply Chain).

Pipeline stages:
    1. Simulated data collection (with intentionally injected quality issues)
    2. Missing value handling
    3. Duplicate removal
    4. Text/date standardization
    5. Outlier detection (IQR method) + domain-rule filtering
    6. Normalization (Min-Max scaling)

Requires: pandas, numpy, scikit-learn
    pip install pandas numpy scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# 1. Simulated raw data collection (deliberately messy, like real source data)
# ---------------------------------------------------------------------------
def generate_raw_orders(n_orders=1500, seed=11):
    rng = np.random.default_rng(seed)

    dcs = [" dc_north", "DC_Central", "dc_south "]
    regions = ["region a", "Region_B", " Region C ", "Region D"]

    order_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 180, n_orders), unit="D"
    )
    promised_lag = rng.integers(2, 5, n_orders)
    delivery_lag = promised_lag + rng.integers(-1, 3, n_orders)

    df = pd.DataFrame(
        {
            "order_id": np.arange(1, n_orders + 1),
            "dc_id": rng.choice(dcs, n_orders),
            "region": rng.choice(regions, n_orders),
            "quantity": rng.integers(1, 15, n_orders).astype(float),
            "distance_km": rng.uniform(2, 60, n_orders).round(1),
            "order_date": order_dates,
        }
    )
    df["promised_date"] = df["order_date"] + pd.to_timedelta(promised_lag, unit="D")
    df["delivery_date"] = df["order_date"] + pd.to_timedelta(delivery_lag, unit="D")

    hub_coords = {
        "region a": (40.71, -74.00), "Region_B": (41.88, -87.63),
        " Region C ": (29.76, -95.37), "Region D": (34.05, -118.24),
    }
    df["delivery_lat"] = df["region"].map(lambda r: hub_coords[r][0]) + rng.normal(0, 0.15, n_orders)
    df["delivery_lon"] = df["region"].map(lambda r: hub_coords[r][1]) + rng.normal(0, 0.15, n_orders)

    # --- Inject realistic data quality issues ---
    # Missing coordinates / delivery dates (~3%)
    missing_idx = rng.choice(df.index, size=int(0.03 * n_orders), replace=False)
    df.loc[missing_idx, ["delivery_lat", "delivery_lon"]] = np.nan

    # Duplicate a handful of orders (system retry simulation)
    dup_idx = rng.choice(df.index, size=15, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Inject outliers: a few negative quantities and absurd distances
    outlier_idx = rng.choice(df.index, size=10, replace=False)
    df.loc[outlier_idx[:5], "quantity"] = -1
    df.loc[outlier_idx[5:], "distance_km"] = rng.uniform(500, 900, 5)

    return df


# ---------------------------------------------------------------------------
# 2-6. Preprocessing pipeline
# ---------------------------------------------------------------------------
def flag_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def preprocess_orders(raw_orders: pd.DataFrame) -> pd.DataFrame:
    df = raw_orders.copy()
    report = {"starting_rows": len(df)}

    # Missing values: drop rows missing structurally required fields
    df = df.dropna(subset=["delivery_lat", "delivery_lon", "delivery_date"])
    report["after_missing_value_drop"] = len(df)

    # Duplicates
    df = df.drop_duplicates(subset="order_id", keep="first")
    report["after_dedup"] = len(df)

    # Standardize text
    df["region"] = df["region"].str.strip().str.replace("_", " ", regex=False).str.title()
    df["dc_id"] = df["dc_id"].str.strip().str.upper()

    # Domain-rule + statistical outlier removal
    df["delay_days"] = (df["delivery_date"] - df["promised_date"]).dt.days
    df = df[(df["quantity"] > 0) & (df["distance_km"] > 0)]
    df = df[~flag_outliers_iqr(df["distance_km"]) & ~flag_outliers_iqr(df["delay_days"])]
    report["after_outlier_removal"] = len(df)

    # Normalization
    scaler = MinMaxScaler()
    num_cols = ["distance_km", "quantity", "delay_days"]
    df[[f"{c}_scaled" for c in num_cols]] = scaler.fit_transform(df[num_cols])

    return df.reset_index(drop=True), report


if __name__ == "__main__":
    print("=== 1. Simulating raw data collection ===")
    raw = generate_raw_orders()
    print(f"Raw rows: {len(raw)}")
    print(f"Missing values per column:\n{raw.isna().sum()[raw.isna().sum() > 0]}")

    print("\n=== 2-6. Running preprocessing pipeline ===")
    clean, report = preprocess_orders(raw)
    for step, count in report.items():
        print(f"{step}: {count}")

    print("\n=== Sample of cleaned, scaled data ===")
    print(clean[["order_id", "dc_id", "region", "distance_km", "distance_km_scaled",
                  "quantity", "quantity_scaled"]].head(10).to_string(index=False))

    print(f"\nFinal clean dataset: {len(clean)} rows (from {len(raw)} raw rows)")
