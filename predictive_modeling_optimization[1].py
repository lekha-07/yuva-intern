"""
Week 4 Task — Predictive Modeling and Optimization in Logistics Systems
Builds a delivery-time forecasting model and a downstream resource-
allocation optimization on top of it. Produces the metrics and charts
used in the report, and prints everything needed for the write-up.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.optimize import linprog

sns.set_theme(style="whitegrid", palette="deep")
OUT = "./charts"
import os
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Problem Definition and Data Simulation
# ---------------------------------------------------------------------------
def generate_dataset(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    dcs = ["DC_North", "DC_Central", "DC_South"]
    regions = ["Region_A", "Region_B", "Region_C", "Region_D"]

    order_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 240, n), unit="D")
    dc_id = rng.choice(dcs, n, p=[0.38, 0.34, 0.28])
    region = rng.choice(regions, n)

    region_base_dist = {"Region_A": 12, "Region_B": 22, "Region_C": 35, "Region_D": 45}
    distance_km = np.array([region_base_dist[r] for r in region]) + rng.normal(0, 6, n)
    distance_km = np.clip(distance_km, 1, None)

    shipment_volume = rng.integers(5, 250, n).astype(float)
    day_of_week = order_dates.dayofweek  # 0=Mon
    is_weekend = (day_of_week >= 5).astype(int)

    dc_efficiency = {"DC_North": 1.0, "DC_Central": 0.85, "DC_South": 1.15}
    weekend_penalty = is_weekend * rng.uniform(2, 6, n)
    volume_effect = shipment_volume * 0.02

    delivery_time_hr = (
        np.array([dc_efficiency[d] for d in dc_id]) * (distance_km * 0.9 + rng.normal(6, 3, n))
        + weekend_penalty + volume_effect
    )
    delivery_time_hr = np.clip(delivery_time_hr, 1, None)

    df = pd.DataFrame({
        "order_date": order_dates,
        "dc_id": dc_id,
        "region": region,
        "distance_km": distance_km.round(1),
        "shipment_volume": shipment_volume,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "delivery_time_hr": delivery_time_hr.round(2),
    })
    return df


df = generate_dataset()
df.to_csv("./logistics_modeling_dataset.csv", index=False)
print(f"Dataset shape: {df.shape}")
print(df.describe(include="all").T[["count", "mean", "std", "min", "max"]] if False else "")

# ---------------------------------------------------------------------------
# 2. Feature Preparation
# ---------------------------------------------------------------------------
feature_cols_num = ["distance_km", "shipment_volume", "day_of_week", "is_weekend"]
feature_cols_cat = ["dc_id", "region"]
target_col = "delivery_time_hr"

X = df[feature_cols_num + feature_cols_cat]
y = df[target_col]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), feature_cols_cat)],
    remainder="passthrough",
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# 3. Model Selection and Training
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
}

results = []
fitted_pipelines = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")

    results.append({
        "model": name, "RMSE": rmse, "MAE": mae, "R2": r2,
        "CV_RMSE_mean": -cv_scores.mean(), "CV_RMSE_std": cv_scores.std(),
    })
    fitted_pipelines[name] = pipe

results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n=== Model comparison (test set) ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["model"]
best_pipe = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------------------------
# 4. Hyperparameter Tuning (Random Forest)
# ---------------------------------------------------------------------------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 10, None],
    "model__min_samples_leaf": [1, 5],
}
rf_pipe = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(random_state=42))])
grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)

tuned_preds = grid.predict(X_test)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))
tuned_mae = mean_absolute_error(y_test, tuned_preds)
tuned_r2 = r2_score(y_test, tuned_preds)

print(f"\n=== Tuned Random Forest ===")
print(f"Best params: {grid.best_params_}")
print(f"Tuned RMSE: {tuned_rmse:.3f}, MAE: {tuned_mae:.3f}, R2: {tuned_r2:.4f}")

final_model = grid.best_estimator_
final_preds = tuned_preds

# ---------------------------------------------------------------------------
# 5. Charts: predicted vs actual, residuals, feature importance, model comparison
# ---------------------------------------------------------------------------
plt.figure(figsize=(6.2, 6))
plt.scatter(y_test, final_preds, alpha=0.35, s=18, color="#2E75B6")
lims = [min(y_test.min(), final_preds.min()), max(y_test.max(), final_preds.max())]
plt.plot(lims, lims, color="#C00000", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Delivery Time (hr)")
plt.ylabel("Predicted Delivery Time (hr)")
plt.title(f"Predicted vs. Actual Delivery Time\n(Tuned Random Forest, R\u00b2 = {tuned_r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/01_predicted_vs_actual.png", dpi=160)
plt.close()

residuals = y_test.values - final_preds
plt.figure(figsize=(7, 4.2))
sns.histplot(residuals, bins=30, kde=True, color="#2E75B6")
plt.axvline(0, color="#C00000", linestyle="--")
plt.title("Distribution of Prediction Residuals (Actual - Predicted)")
plt.xlabel("Residual (hours)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT}/02_residuals.png", dpi=160)
plt.close()

# Feature importance (from tuned RF)
ohe = final_model.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(feature_cols_cat))
all_feature_names = cat_feature_names + feature_cols_num
importances = final_model.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=True)

plt.figure(figsize=(7, 5))
plt.barh(fi.index, fi.values, color="#2E75B6")
plt.title("Feature Importance (Tuned Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/03_feature_importance.png", dpi=160)
plt.close()

plt.figure(figsize=(7, 4.2))
plt.bar(results_df["model"], results_df["RMSE"], color="#8FAADC")
plt.bar(["Random Forest (Tuned)"], [tuned_rmse], color="#2E75B6")
plt.title("Model Comparison — Test RMSE (lower is better)")
plt.ylabel("RMSE (hours)")
plt.tight_layout()
plt.savefig(f"{OUT}/04_model_comparison.png", dpi=160)
plt.close()

print("\nFeature importances:\n", fi.sort_values(ascending=False))

# ---------------------------------------------------------------------------
# 6. Optimization: shipment-to-DC assignment minimizing total predicted time
#    subject to daily capacity constraints (linear programming)
# ---------------------------------------------------------------------------
# Build a small scenario: 4 regions' next-day shipments, 3 DCs, each DC has
# a daily capacity. Predict delivery time for every (region, DC) pair, then
# solve an LP to assign shipments to DCs minimizing total predicted delivery
# hours, subject to each DC's capacity.

regions_list = ["Region_A", "Region_B", "Region_C", "Region_D"]
dcs_list = ["DC_North", "DC_Central", "DC_South"]
next_day_volume = {"Region_A": 80, "Region_B": 65, "Region_C": 50, "Region_D": 40}   # shipments to assign
dc_capacity = {"DC_North": 90, "DC_Central": 80, "DC_South": 70}                      # max shipments/day

region_base_dist = {"Region_A": 12, "Region_B": 22, "Region_C": 35, "Region_D": 45}
scenario_rows = []
for r in regions_list:
    for d in dcs_list:
        scenario_rows.append({
            "region": r, "dc_id": d,
            "distance_km": region_base_dist[r],
            "shipment_volume": next_day_volume[r],
            "day_of_week": 1, "is_weekend": 0,
        })
scenario_df = pd.DataFrame(scenario_rows)
scenario_df["predicted_hr"] = final_model.predict(
    scenario_df[feature_cols_num + feature_cols_cat]
)

# LP: minimize sum(x_rd * predicted_hr_rd) subject to
#   sum_d x_rd = demand_r  (all shipments from region r assigned)
#   sum_r x_rd <= capacity_d (DC daily capacity)
#   x_rd >= 0
n_r, n_d = len(regions_list), len(dcs_list)
cost = scenario_df.pivot(index="region", columns="dc_id", values="predicted_hr").loc[regions_list, dcs_list].values
c = cost.flatten()  # variables ordered region-major, dc-minor

A_eq, b_eq = [], []
for i, r in enumerate(regions_list):
    row = np.zeros(n_r * n_d)
    row[i * n_d:(i + 1) * n_d] = 1
    A_eq.append(row)
    b_eq.append(next_day_volume[r])

A_ub, b_ub = [], []
for j, d in enumerate(dcs_list):
    row = np.zeros(n_r * n_d)
    for i in range(n_r):
        row[i * n_d + j] = 1
    A_ub.append(row)
    b_ub.append(dc_capacity[d])

res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
assignment = res.x.reshape(n_r, n_d)
assignment_df = pd.DataFrame(assignment, index=regions_list, columns=dcs_list).round(1)

print("\n=== Optimized region -> DC assignment (shipments/day) ===")
print(assignment_df)
print(f"\nTotal predicted delivery hours (optimized): {res.fun:.1f}")

# Baseline: naive "always nearest region-default DC" assignment for comparison
naive_map = {"Region_A": "DC_North", "Region_B": "DC_North", "Region_C": "DC_Central", "Region_D": "DC_South"}
naive_total = sum(
    next_day_volume[r] * scenario_df[(scenario_df.region == r) & (scenario_df.dc_id == naive_map[r])]["predicted_hr"].values[0]
    for r in regions_list
)
print(f"Total predicted delivery hours (naive fixed assignment): {naive_total:.1f}")
print(f"Improvement: {naive_total - res.fun:.1f} hours ({(naive_total - res.fun) / naive_total * 100:.1f}%)")

# Chart: optimized assignment heatmap
plt.figure(figsize=(6.5, 4.5))
sns.heatmap(assignment_df, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Shipments assigned"})
plt.title("Optimized Shipment Assignment: Region \u2192 Distribution Center")
plt.tight_layout()
plt.savefig(f"{OUT}/05_optimized_assignment.png", dpi=160)
plt.close()

print("\nAll charts saved to", OUT)
