"""Comprehensive EDA for RetailMind hackathon data."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATA_DIR = Path("Data")
FILES = ["2023.csv", "2024.csv", "2025.csv"]
OUT = Path("analysis_output.json")


def load_all():
    dfs = {}
    for f in FILES:
        dfs[f] = pd.read_csv(DATA_DIR / f, low_memory=False)
    combined = pd.concat(
        [dfs[f].assign(_source_year=f.replace(".csv", "")) for f in FILES],
        ignore_index=True,
    )
    return dfs, combined


def schema_report(df, name):
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        null_pct = float(s.isna().mean() * 100)
        nunique = int(s.nunique(dropna=True))
        sample = s.dropna().head(3).tolist()
        if len(sample) > 0 and isinstance(sample[0], str) and len(str(sample[0])) > 80:
            sample = [str(x)[:80] + "..." for x in sample]
        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "null_pct": round(null_pct, 4),
                "nunique": nunique,
                "sample": sample,
            }
        )
    return {
        "file": name,
        "n_rows": n,
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "schema": rows,
    }


def quality_report(df, name):
    dup_rows = int(df.duplicated().sum())
    dup_keys = None
    key_cols = [c for c in df.columns if any(k in c.lower() for k in ["id", "sku", "cliente", "producto", "fecha", "date"])]
    if len(key_cols) >= 2:
        dup_keys = int(df.duplicated(subset=key_cols[:5], keep=False).sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers = {}
    for col in numeric_cols[:15]:
        q1, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
        below = int((df[col] < q1).sum())
        above = int((df[col] > q99).sum())
        if below + above > 0:
            outliers[col] = {"p01": float(q1), "p99": float(q99), "below_p01": below, "above_p99": above}

    negatives = {}
    for col in numeric_cols:
        neg = int((df[col] < 0).sum())
        if neg > 0:
            negatives[col] = neg

    return {
        "file": name,
        "duplicate_rows": dup_rows,
        "duplicate_rows_pct": round(100 * dup_rows / len(df), 4),
        "duplicate_on_key_cols_approx": dup_keys,
        "outliers_1_99": outliers,
        "negative_values": negatives,
    }


def parse_dates(df):
    date_cols = []
    for col in df.columns:
        if "fecha" in col.lower() or "date" in col.lower():
            date_cols.append(col)
    parsed = {}
    for col in date_cols:
        try:
            s = pd.to_datetime(df[col], errors="coerce")
            parsed[col] = {
                "parse_fail_pct": round(100 * s.isna().mean(), 4),
                "min": str(s.min()) if s.notna().any() else None,
                "max": str(s.max()) if s.notna().any() else None,
            }
        except Exception as e:
            parsed[col] = {"error": str(e)}
    return parsed


def business_eda(combined):
    r = {}
    # detect key columns heuristically
    cols_lower = {c: c.lower() for c in combined.columns}
    col_map = {v: k for k, v in cols_lower.items()}

    def find_col(*keywords):
        for c in combined.columns:
            cl = c.lower()
            if all(k in cl for k in keywords):
                return c
        for c in combined.columns:
            cl = c.lower()
            if any(k in cl for k in keywords):
                return c
        return None

    cliente = find_col("cliente") or find_col("customer")
    producto = find_col("producto") or find_col("sku") or find_col("item")
    fecha = find_col("fecha") or find_col("date")
    ventas = find_col("venta") or find_col("sales") or find_col("cantidad") or find_col("qty")
    inventario = find_col("invent") or find_col("stock")

    r["detected_cols"] = {
        "cliente": cliente,
        "producto": producto,
        "fecha": fecha,
        "ventas": ventas,
        "inventario": inventario,
    }

    if fecha:
        combined["_fecha"] = pd.to_datetime(combined[fecha], errors="coerce")
        combined["_year"] = combined["_fecha"].dt.year
        combined["_month"] = combined["_fecha"].dt.month
        r["date_range"] = {
            "min": str(combined["_fecha"].min()),
            "max": str(combined["_fecha"].max()),
        }
        r["rows_by_year"] = combined["_year"].value_counts().sort_index().astype(int).to_dict()
        r["rows_by_month"] = combined["_month"].value_counts().sort_index().astype(int).to_dict()

    numeric = combined.select_dtypes(include=[np.number]).columns.tolist()
    r["numeric_summary"] = combined[numeric].describe().round(4).to_dict()

    if producto:
        top_prod = combined[producto].value_counts().head(15)
        bot_prod = combined[producto].value_counts().tail(15)
        r["top_products"] = top_prod.astype(int).to_dict()
        r["bottom_products"] = bot_prod.astype(int).to_dict()
        r["n_products"] = int(combined[producto].nunique())

    if cliente:
        top_cli = combined[cliente].value_counts().head(15)
        r["top_clients"] = top_cli.astype(int).to_dict()
        r["n_clients"] = int(combined[cliente].nunique())

    if ventas and inventario:
        corr = combined[[ventas, inventario]].corr().iloc[0, 1]
        r["sales_inventory_corr"] = round(float(corr), 4) if not np.isnan(corr) else None

    if numeric:
        corr_mat = combined[numeric].corr()
        # top correlations
        pairs = []
        for i, c1 in enumerate(numeric):
            for c2 in numeric[i + 1 :]:
                v = corr_mat.loc[c1, c2]
                if not np.isnan(v):
                    pairs.append((c1, c2, round(float(v), 4)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        r["top_correlations"] = pairs[:20]

    # per source year stats
    if "_source_year" in combined.columns and ventas:
        r["sales_by_file_year"] = (
            combined.groupby("_source_year")[ventas].agg(["sum", "mean", "count"]).round(4).to_dict()
        )

    return r


def cross_file_analysis(dfs):
    r = {}
    schemas = {f: set(dfs[f].columns) for f in FILES}
    r["columns_per_file"] = {f: list(dfs[f].columns) for f in FILES}
    all_cols = set.union(*schemas.values())
    r["common_columns"] = sorted(set.intersection(*schemas.values()))
    r["union_columns"] = sorted(all_cols)
    for f in FILES:
        r[f"only_in_{f}"] = sorted(schemas[f] - set.intersection(*[schemas[g] for g in FILES if g != f]))

    # row counts
    r["row_counts"] = {f: len(dfs[f]) for f in FILES}

    # overlap of keys if detectable
    for f in FILES:
        dfs[f].columns.tolist()

    # Try cliente+producto+fecha overlap between years
    def key_frame(df):
        cols = df.columns.tolist()
        cl = [c.lower() for c in cols]
        keys = []
        for kw in ["cliente", "producto", "sku", "fecha"]:
            for i, c in enumerate(cl):
                if kw in c and cols[i] not in keys:
                    keys.append(cols[i])
        return keys[:3]

    keys = key_frame(dfs[FILES[0]])
    r["inferred_join_keys"] = keys
    if len(keys) >= 2:
        for i, f1 in enumerate(FILES):
            for f2 in FILES[i + 1 :]:
                k1 = dfs[f1][keys].drop_duplicates()
                k2 = dfs[f2][keys].drop_duplicates()
                merge = k1.merge(k2, on=keys, how="inner")
                r[f"key_overlap_{f1}_{f2}"] = {
                    "unique_keys_f1": len(k1),
                    "unique_keys_f2": len(k2),
                    "overlap": len(merge),
                }
    return r


def main():
    print("Loading data...")
    dfs, combined = load_all()
    report = {"schemas": [], "quality": [], "dates": {}, "cross_file": {}, "business_eda": {}}

    for f in FILES:
        df = dfs[f]
        print(f"Schema {f}...")
        report["schemas"].append(schema_report(df, f))
        print(f"Quality {f}...")
        report["quality"].append(quality_report(df, f))
        report["dates"][f] = parse_dates(df)

    print("Cross file...")
    report["cross_file"] = cross_file_analysis(dfs)

    print("Business EDA...")
    report["business_eda"] = business_eda(combined)
    report["combined_rows"] = len(combined)
    report["combined_cols"] = list(combined.columns)

    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
