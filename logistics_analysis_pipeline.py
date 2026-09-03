"""
Strategic Planning and Data Exploration in Logistics
------------------------------------------------------
Standalone illustration of the analysis pipeline described in the strategic
planning report. Runs end-to-end on a small synthetic dataset so it can be
executed without access to the company's real systems.

Pipeline stages:
    1. Data generation / loading
    2. Data cleaning
    3. Exploratory data analysis
    4. Delivery-zone clustering (K-Means)
    5. Demand forecasting (Random Forest regression)
    6. Route optimization (pseudocode -> simplified nearest-neighbor demo)
    7. Inventory rebalancing recommendation

Requires: pandas, numpy, scikit-learn  (matplotlib optional, for EDA plots)
    pip install pandas numpy scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------------------
# 1. Data generation (stand-in for pulling real order/inventory/delivery data)
# ---------------------------------------------------------------------------
def generate_sample_data(n_orders=2000, seed=42):
    rng = np.random.default_rng(seed)

    dcs = ["DC_North", "DC_Central", "DC_South"]
    regions = ["Region_A", "Region_B", "Region_C", "Region_D"]
    skus = [f"SKU_{i:03d}" for i in range(1, 21)]

    order_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 180, n_orders), unit="D"
    )
    promised_lag = rng.integers(2, 5, n_orders)
    delivery_lag = promised_lag + rng.integers(-1, 3, n_orders)  # some late, some early

    df = pd.DataFrame(
        {
            "order_id": np.arange(1, n_orders + 1),
            "dc_id": rng.choice(dcs, n_orders),
            "region": rng.choice(regions, n_orders),
            "sku": rng.choice(skus, n_orders),
            "quantity": rng.integers(1, 15, n_orders),
            "distance_km": rng.uniform(2, 60, n_orders).round(1),
            "order_date": order_dates,
        }
    )
    df["promised_date"] = df["order_date"] + pd.to_timedelta(promised_lag, unit="D")
    df["delivery_date"] = df["order_date"] + pd.to_timedelta(delivery_lag, unit="D")

    # Approximate delivery coordinates around fictional regional hubs
    hub_coords = {
        "Region_A": (40.71, -74.00),
        "Region_B": (41.88, -87.63),
        "Region_C": (29.76, -95.37),
        "Region_D": (34.05, -118.24),
    }
    df["delivery_lat"] = df["region"].map(lambda r: hub_coords[r][0]) + rng.normal(0, 0.15, n_orders)
    df["delivery_lon"] = df["region"].map(lambda r: hub_coords[r][1]) + rng.normal(0, 0.15, n_orders)

    return df


def generate_inventory_snapshot(dcs=("DC_North", "DC_Central", "DC_South"), skus=None, seed=7):
    rng = np.random.default_rng(seed)
    if skus is None:
        skus = [f"SKU_{i:03d}" for i in range(1, 21)]
    rows = [
        {"dc_id": dc, "sku": sku, "qty": int(rng.integers(0, 200))}
        for dc in dcs
        for sku in skus
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------------------------
def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.drop_duplicates(subset="order_id")
    orders = orders.dropna(subset=["delivery_lat", "delivery_lon", "delivery_date"])
    orders = orders[orders["delivery_date"] >= orders["order_date"]]
    orders["region"] = orders["region"].str.strip().str.title()
    return orders


# ---------------------------------------------------------------------------
# 3. Exploratory data analysis
# ---------------------------------------------------------------------------
def summarize_kpis(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    orders["delay_days"] = (orders["delivery_date"] - orders["promised_date"]).dt.days
    otd_by_dc = (orders["delay_days"] <= 0).groupby(orders["dc_id"]).mean().rename("on_time_rate")
    avg_distance = orders.groupby("dc_id")["distance_km"].mean().rename("avg_distance_km")
    return pd.concat([otd_by_dc, avg_distance], axis=1).reset_index()


# ---------------------------------------------------------------------------
# 4. Delivery-zone clustering
# ---------------------------------------------------------------------------
def cluster_delivery_zones(orders: pd.DataFrame, n_clusters=8) -> pd.DataFrame:
    orders = orders.copy()
    coords = orders[["delivery_lat", "delivery_lon"]].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    orders.loc[coords.index, "delivery_zone"] = kmeans.fit_predict(coords)
    return orders


# ---------------------------------------------------------------------------
# 5. Demand forecasting
# ---------------------------------------------------------------------------
def forecast_weekly_demand(orders: pd.DataFrame):
    weekly = (
        orders.set_index("order_date")
        .groupby(["region", pd.Grouper(freq="W")])["quantity"]
        .sum()
        .reset_index()
    )
    weekly["week_of_year"] = weekly["order_date"].dt.isocalendar().week.astype(int)
    weekly = pd.get_dummies(weekly, columns=["region"], drop_first=True)

    feature_cols = [c for c in weekly.columns if c not in ("order_date", "quantity")]
    X = weekly[feature_cols]
    y = weekly["quantity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return model, mae


# ---------------------------------------------------------------------------
# 6. Route optimization (simplified nearest-neighbor demo standing in for a
#    full OR-Tools VRP solve, which is described as pseudocode in the report)
# ---------------------------------------------------------------------------
def simple_route_for_zone(orders: pd.DataFrame, zone_id: float):
    stops = orders[orders["delivery_zone"] == zone_id][["order_id", "delivery_lat", "delivery_lon"]]
    if stops.empty:
        return []
    stops = stops.reset_index(drop=True)
    visited = [stops.iloc[0]["order_id"]]
    remaining = stops.iloc[1:].copy()
    current = stops.iloc[0]

    while not remaining.empty:
        remaining["dist"] = np.hypot(
            remaining["delivery_lat"] - current["delivery_lat"],
            remaining["delivery_lon"] - current["delivery_lon"],
        )
        nxt = remaining.loc[remaining["dist"].idxmin()]
        visited.append(nxt["order_id"])
        current = nxt
        remaining = remaining.drop(nxt.name)

    return visited


# ---------------------------------------------------------------------------
# 7. Inventory rebalancing recommendation
# ---------------------------------------------------------------------------
def recommend_rebalance(inventory_df: pd.DataFrame, avg_daily_demand: dict, safety_stock_days=7):
    plan = []
    for sku, daily_demand in avg_daily_demand.items():
        target = daily_demand * safety_stock_days
        for dc in inventory_df["dc_id"].unique():
            on_hand = inventory_df.query("sku == @sku and dc_id == @dc")["qty"].sum()
            if on_hand < target:
                plan.append({"sku": sku, "dc_id": dc, "transfer_in_needed": round(target - on_hand, 1)})
    return pd.DataFrame(plan)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 1-2. Generating and cleaning sample data ===")
    orders_raw = generate_sample_data()
    orders = clean_orders(orders_raw)
    inventory = generate_inventory_snapshot()
    print(f"Orders after cleaning: {len(orders)} rows")

    print("\n=== 3. KPI summary by distribution center ===")
    print(summarize_kpis(orders).to_string(index=False))

    print("\n=== 4. Clustering delivery zones ===")
    orders = cluster_delivery_zones(orders)
    print(orders["delivery_zone"].value_counts().sort_index())

    print("\n=== 5. Forecasting weekly demand ===")
    model, mae = forecast_weekly_demand(orders)
    print(f"Random Forest demand forecast MAE: {mae:.2f} units/week")

    print("\n=== 6. Example route for delivery zone 0 ===")
    route = simple_route_for_zone(orders, 0.0)
    print(f"Visit order: {route[:10]}{'...' if len(route) > 10 else ''}")

    print("\n=== 7. Inventory rebalancing recommendation (sample SKUs) ===")
    sample_demand = {sku: rate for sku, rate in
                      orders.groupby("sku")["quantity"].mean().div(7).head(5).items()}
    rebalance_plan = recommend_rebalance(inventory, sample_demand)
    print(rebalance_plan.to_string(index=False) if not rebalance_plan.empty else "No rebalancing needed.")
