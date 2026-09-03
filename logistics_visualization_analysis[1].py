"""
Week 3 Task — Advanced Data Analysis and Visualization in Logistics
Generates a synthetic (already-cleaned) logistics dataset and produces
the visualizations used in the report.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
OUT = "./charts"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulate a hypothetical, already-cleaned logistics dataset
# ---------------------------------------------------------------------------
def generate_dataset(n=2200, seed=42):
    rng = np.random.default_rng(seed)
    dcs = ["DC_North", "DC_Central", "DC_South"]
    regions = ["Region_A", "Region_B", "Region_C", "Region_D"]

    order_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 210, n), unit="D")
    dc_id = rng.choice(dcs, n, p=[0.38, 0.34, 0.28])
    region = rng.choice(regions, n)

    # distance correlated with region (some regions are just farther from hubs)
    region_base_dist = {"Region_A": 12, "Region_B": 22, "Region_C": 35, "Region_D": 45}
    distance_km = np.array([region_base_dist[r] for r in region]) + rng.normal(0, 6, n)
    distance_km = np.clip(distance_km, 1, None)

    # shipment volume
    shipment_volume = rng.integers(5, 250, n).astype(float)

    # transportation cost driven by distance + volume + noise
    transport_cost = 4.5 * distance_km + 0.8 * shipment_volume + rng.normal(0, 15, n)
    transport_cost = np.clip(transport_cost, 10, None)

    # delivery time (hours) driven mostly by distance, with DC efficiency differences
    dc_efficiency = {"DC_North": 1.0, "DC_Central": 0.85, "DC_South": 1.15}
    delivery_time_hr = np.array([dc_efficiency[d] for d in dc_id]) * (
        distance_km * 0.9 + rng.normal(6, 3, n)
    )
    delivery_time_hr = np.clip(delivery_time_hr, 1, None)

    promised_hr = distance_km * 0.9 + 8
    delay_hr = delivery_time_hr - promised_hr

    df = pd.DataFrame({
        "order_date": order_dates,
        "dc_id": dc_id,
        "region": region,
        "distance_km": distance_km.round(1),
        "shipment_volume": shipment_volume,
        "transport_cost": transport_cost.round(2),
        "delivery_time_hr": delivery_time_hr.round(2),
        "delay_hr": delay_hr.round(2),
    })
    return df


df = generate_dataset()
df.to_csv("./logistics_dataset.csv", index=False)

# ---------------------------------------------------------------------------
# Central tendency / summary stats (printed, also used in report narrative)
# ---------------------------------------------------------------------------
summary = df[["distance_km", "shipment_volume", "transport_cost", "delivery_time_hr", "delay_hr"]].describe()
summary.to_csv("./summary_stats.csv")
print(summary)

# ---------------------------------------------------------------------------
# Chart 1: Distribution of delivery time (histogram + KDE)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.2))
sns.histplot(df["delivery_time_hr"], bins=30, kde=True, color="#2E75B6")
plt.axvline(df["delivery_time_hr"].mean(), color="#C00000", linestyle="--", label=f"Mean = {df['delivery_time_hr'].mean():.1f} hr")
plt.axvline(df["delivery_time_hr"].median(), color="#548235", linestyle="--", label=f"Median = {df['delivery_time_hr'].median():.1f} hr")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (hours)")
plt.ylabel("Number of Shipments")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/01_delivery_time_distribution.png", dpi=160)
plt.close()

# ---------------------------------------------------------------------------
# Chart 2: Transportation cost by distribution center (boxplot)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.2))
order = df.groupby("dc_id")["transport_cost"].median().sort_values().index
sns.boxplot(data=df, x="dc_id", y="transport_cost", order=order, palette="Blues")
plt.title("Transportation Cost by Distribution Center")
plt.xlabel("Distribution Center")
plt.ylabel("Transportation Cost ($)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_cost_by_dc_boxplot.png", dpi=160)
plt.close()

# ---------------------------------------------------------------------------
# Chart 3: Weekly shipment volume trend (line chart)
# ---------------------------------------------------------------------------
weekly = df.set_index("order_date").resample("W")["shipment_volume"].sum()
plt.figure(figsize=(8, 4.2))
plt.plot(weekly.index, weekly.values, color="#2E75B6", marker="o", markersize=3)
plt.title("Weekly Total Shipment Volume Over Time")
plt.xlabel("Week")
plt.ylabel("Total Shipment Volume (units)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT}/03_weekly_volume_trend.png", dpi=160)
plt.close()

# ---------------------------------------------------------------------------
# Chart 4: Correlation heatmap
# ---------------------------------------------------------------------------
corr = df[["distance_km", "shipment_volume", "transport_cost", "delivery_time_hr", "delay_hr"]].corr()
plt.figure(figsize=(6.5, 5.2))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Key Logistics Variables")
plt.tight_layout()
plt.savefig(f"{OUT}/04_correlation_heatmap.png", dpi=160)
plt.close()

# ---------------------------------------------------------------------------
# Chart 5: Distance vs. transport cost (scatter with regression line)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.5))
sns.regplot(data=df, x="distance_km", y="transport_cost", scatter_kws={"alpha": 0.3, "s": 15, "color": "#2E75B6"}, line_kws={"color": "#C00000"})
plt.title("Transportation Cost vs. Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost ($)")
plt.tight_layout()
plt.savefig(f"{OUT}/05_distance_vs_cost_scatter.png", dpi=160)
plt.close()

# ---------------------------------------------------------------------------
# Chart 6: Average delay by region (bar chart)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.2))
region_delay = df.groupby("region")["delay_hr"].mean().sort_values()
colors = ["#548235" if v <= 0 else "#C00000" for v in region_delay.values]
plt.bar(region_delay.index, region_delay.values, color=colors)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Average Delivery Delay by Region")
plt.xlabel("Region")
plt.ylabel("Average Delay (hours, negative = early)")
plt.tight_layout()
plt.savefig(f"{OUT}/06_avg_delay_by_region.png", dpi=160)
plt.close()

print("\nAll charts saved to", OUT)
print("\nRegion delay summary:\n", region_delay)
print("\nDC cost summary:\n", df.groupby('dc_id')['transport_cost'].median())
print("\nCorrelation distance vs cost:", corr.loc['distance_km', 'transport_cost'])
print("Correlation distance vs delivery_time_hr:", corr.loc['distance_km', 'delivery_time_hr'])
