"""Deep EDA: melt weekly panel, business KPIs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("Data")
OUT = Path("deep_eda_output.json")


def load_melted():
    frames = []
    for year_file in ["2023.csv", "2024.csv", "2025.csv"]:
        df = pd.read_csv(DATA / year_file)
        year = year_file.replace(".csv", "")
        week_cols = [c for c in df.columns if c not in ("Channel", "Material Description", "Category")]
        long = df.melt(
            id_vars=["Channel", "Material Description", "Category"],
            value_vars=week_cols,
            var_name="week_id",
            value_name="qty",
        )
        long["calendar_year"] = int(year)
        long["week_num"] = long["week_id"].astype(str).str[-2:].astype(int)
        long["qty"] = pd.to_numeric(long["qty"], errors="coerce").fillna(0)
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def main():
    print("Loading melted panel...")
    m = load_melted()

    r = {}
    r["panel_rows"] = len(m)
    r["sparsity_zero_pct"] = round(100 * (m["qty"] == 0).mean(), 2)
    r["sparsity_positive_pct"] = round(100 * (m["qty"] > 0).mean(), 2)

    r["category_value_counts"] = m["Category"].value_counts().to_dict()
    r["n_channels"] = int(m["Channel"].nunique())
    r["n_products"] = int(m["Material Description"].nunique())
    r["n_channel_product_pairs"] = int(m.groupby(["Channel", "Material Description"]).ngroups)

    # pivot categories wide per observation
    cat_pivot = (
        m.groupby(["Channel", "Material Description", "week_id", "calendar_year", "week_num"], observed=True)[
            ["Category", "qty"]
        ]
        .apply(lambda x: x.set_index("Category")["qty"].to_dict())
        .reset_index(name="metrics")
    )
    # simpler approach
    wide = m.pivot_table(
        index=["Channel", "Material Description", "week_id", "calendar_year", "week_num"],
        columns="Category",
        values="qty",
        aggfunc="first",
        fill_value=0,
    ).reset_index()

    for c in ["Sell-in", "Cust. Sales", "Channel Inv."]:
        if c in wide.columns:
            wide[c] = wide[c].astype(float)

    r["panel_wide_rows"] = len(wide)

    # Only rows with any activity
    active = wide[(wide.get("Sell-in", 0) + wide.get("Cust. Sales", 0) + wide.get("Channel Inv.", 0)) > 0]
    r["active_panel_rows"] = len(active)
    r["active_panel_pct"] = round(100 * len(active) / len(wide), 2)

    # Correlations on active subset
    if all(c in wide.columns for c in ["Sell-in", "Cust. Sales", "Channel Inv."]):
        sub = active.sample(min(50000, len(active)), random_state=42) if len(active) > 50000 else active
        corr = sub[["Sell-in", "Cust. Sales", "Channel Inv."]].corr().round(4).to_dict()
        r["category_correlations_active"] = corr

    # Sell-through proxy: cust sales / (channel inv lag?) simplified
    if "Cust. Sales" in wide.columns and "Channel Inv." in wide.columns:
        wide["sell_through"] = np.where(
            wide["Channel Inv."] > 0,
            wide["Cust. Sales"] / wide["Channel Inv."],
            np.nan,
        )
        r["sell_through_median"] = round(float(wide["sell_through"].median(skipna=True)), 4)
        r["stockout_signal"] = int(((wide["Channel Inv."] == 0) & (wide["Cust. Sales"] > 0)).sum())
        r["overstock_signal"] = int(((wide["Channel Inv."] > wide["Channel Inv."].quantile(0.95)) & (wide["Cust. Sales"] == 0)).sum())

    # Top customers by total cust sales
    if "Cust. Sales" in wide.columns:
        by_ch = wide.groupby("Channel")["Cust. Sales"].sum().sort_values(ascending=False)
        r["top10_customers_cust_sales"] = by_ch.head(10).astype(int).to_dict()
        r["bottom10_customers_cust_sales"] = by_ch.tail(10).astype(int).to_dict()
        r["customer_concentration_top10_pct"] = round(
            100 * by_ch.head(10).sum() / by_ch.sum(), 2
        )

    # Top products
    if "Cust. Sales" in wide.columns:
        by_prod = wide.groupby("Material Description")["Cust. Sales"].sum().sort_values(ascending=False)
        r["top10_products_cust_sales"] = {
            k[:60]: int(v) for k, v in by_prod.head(10).items()
        }
        r["n_products_with_sales"] = int((by_prod > 0).sum())

    # Weekly seasonality (aggregate cust sales)
    if "Cust. Sales" in wide.columns:
        by_week = wide.groupby(["calendar_year", "week_num"])["Cust. Sales"].sum()
        r["peak_weeks_2023"] = (
            by_week.loc[2023].sort_values(ascending=False).head(5).astype(int).to_dict()
            if 2023 in by_week.index.get_level_values(0)
            else {}
        )
        r["peak_weeks_2024"] = (
            by_week.loc[2024].sort_values(ascending=False).head(5).astype(int).to_dict()
            if 2024 in by_week.index.get_level_values(0)
            else {}
        )

    # YoY same week comparison
    yoy = (
        wide.groupby(["Channel", "Material Description", "week_num"])["Cust. Sales"]
        .sum()
        .reset_index()
    )

    # Duplicate channel-product-category rows per file
    dup_check = []
    for yf in ["2023.csv", "2024.csv", "2025.csv"]:
        df = pd.read_csv(DATA / yf)
        dup = df.duplicated(subset=["Channel", "Material Description", "Category"]).sum()
        dup_check.append({yf: int(dup)})
    r["duplicates_channel_product_category"] = dup_check

    # Unique products per customer
    prod_per_cust = wide.groupby("Channel")["Material Description"].nunique()
    r["avg_products_per_customer"] = round(float(prod_per_cust.mean()), 2)
    r["max_products_per_customer"] = int(prod_per_cust.max())

    # Weeks coverage per year file
    for yf in ["2023.csv", "2024.csv", "2025.csv"]:
        df = pd.read_csv(DATA / yf, nrows=1)
        weeks = [c for c in df.columns if c not in ("Channel", "Material Description", "Category")]
        r[f"weeks_in_{yf}"] = len(weeks)

    # 2025 partial year check - sum by week
    w2025 = wide[wide["calendar_year"] == 2025]
    if len(w2025) and "Cust. Sales" in wide.columns:
        last_active_week = int(
            w2025.groupby("week_num")["Cust. Sales"].sum().pipe(lambda s: s[s > 0].index.max())
            if (w2025["Cust. Sales"] > 0).any()
            else 0
        )
        r["2025_last_week_with_aggregate_sales"] = last_active_week

    # Churn proxy: customers with declining 2024 vs 2023 H2
    if "Cust. Sales" in wide.columns:
        s23 = wide[wide["calendar_year"] == 2023].groupby("Channel")["Cust. Sales"].sum()
        s24 = wide[wide["calendar_year"] == 2024].groupby("Channel")["Cust. Sales"].sum()
        joined = pd.DataFrame({"y2023": s23, "y2024": s24}).fillna(0)
        joined["delta_pct"] = np.where(
            joined["y2023"] > 0,
            100 * (joined["y2024"] - joined["y2023"]) / joined["y2023"],
            np.nan,
        )
        r["customers_down_50pct_yoy"] = int((joined["delta_pct"] < -50).sum())
        r["customers_up_50pct_yoy"] = int((joined["delta_pct"] > 50).sum())
        r["customers_zero_2024_but_active_2023"] = int(
            ((joined["y2023"] > 0) & (joined["y2024"] == 0)).sum()
        )

    # ML-ready stats: series length per sku-customer
    series_len = (
        wide.groupby(["Channel", "Material Description"])["Cust. Sales"]
        .apply(lambda s: (s > 0).sum())
        .describe()
        .round(2)
        .to_dict()
    )
    r["nonzero_weeks_cust_sales_per_series"] = series_len

    # Category totals by year
    if all(c in wide.columns for c in ["Sell-in", "Cust. Sales", "Channel Inv."]):
        r["totals_by_year"] = (
            wide.groupby("calendar_year")[["Sell-in", "Cust. Sales", "Channel Inv."]]
            .sum()
            .astype(int)
            .to_dict()
        )

    OUT.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
