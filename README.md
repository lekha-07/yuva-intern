# Strategic Planning and Data Exploration in Logistics

**Week 1 Task — Yuva Intern | Logistics Data Analyst Internship**

## Overview

This repository contains the deliverables for the Week 1 task: a strategic planning report and a supporting Python pipeline for a logistics data analysis project. The scenario simulates a regional e-commerce logistics provider operating three distribution centers, and covers route optimization, inventory management, and supply chain integration.

## Contents

| File | Description |
|---|---|
| `Strategic_Planning_Logistics_Report.docx` | Full strategic planning report — logistics scenario, KPIs, literature/data research, roadmap, code illustrations, and conclusion. |
| `logistics_analysis_pipeline.py` | Standalone, runnable Python script implementing the analysis pipeline end-to-end on synthetic sample data. |
| `README.md` | This file. |

## Key Performance Indicators (KPIs)

- **On-Time Delivery Rate (OTD%)**
- **Inventory Turnover Ratio**
- **Freight Cost per Unit Shipped**
- **Order Fill Rate**

## Pipeline Stages (`logistics_analysis_pipeline.py`)

1. **Data generation / loading** — synthetic orders and inventory data standing in for real company data.
2. **Data cleaning** — deduplication, missing-value handling, date/label standardization.
3. **Exploratory data analysis** — KPI summary by distribution center.
4. **Delivery-zone clustering** — K-Means on delivery coordinates to group stops into planning zones.
5. **Demand forecasting** — Random Forest regression on weekly demand by region.
6. **Route optimization (demo)** — nearest-neighbor stop sequencing per delivery zone (the full report also outlines an OR-Tools VRP approach).
7. **Inventory rebalancing** — flags distribution centers below target safety stock per SKU.

## How to Run

```bash
pip install pandas numpy scikit-learn
python logistics_analysis_pipeline.py
```

The script prints KPI summaries, cluster sizes, forecast accuracy (MAE), an example delivery route, and inventory rebalancing recommendations to the console.

## Notes

- Data is synthetically generated (`generate_sample_data`, `generate_inventory_snapshot`) for demonstration purposes, since real company data was not available for this exercise.
- See `Strategic_Planning_Logistics_Report.docx` for the full write-up, including the business rationale behind each KPI and roadmap phase.
