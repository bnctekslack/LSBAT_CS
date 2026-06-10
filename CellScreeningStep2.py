import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis_config import STEP2_COLUMNS, MIN_CLUSTER_SIZE


DEFAULT_OUTPUT_DIR = "Results"
DEFAULT_CLUSTER_INDEX = 1
ANALYSIS_COLUMNS = STEP2_COLUMNS

def _calculate_ranges(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    ranges = {}
    for col in cols:
        q1 = df[col].quantile(0.33)
        q2 = df[col].quantile(0.66)
        vmin, vmax = df[col].min(), df[col].max()
        if q1 == q2:
            q1 = vmin + (vmax - vmin) / 3
            q2 = vmin + (vmax - vmin) * 2 / 3
        ranges[col] = {"low": vmin, "mid1": q1, "mid2": q2, "high": vmax}
    return ranges


def _membership_percentile(x, col, ranges, df_col):
    low, mid1, mid2, high = (
        ranges[col]["low"],
        ranges[col]["mid1"],
        ranges[col]["mid2"],
        ranges[col]["high"],
    )

    if low <= x < mid1:
        low_pct = (df_col[(df_col >= low) & (df_col < mid1)] <= x).mean() * 100
    else:
        low_pct = 0.0

    if mid1 <= x <= mid2:
        med_pct = (df_col[(df_col >= mid1) & (df_col <= mid2)] <= x).mean() * 100
    else:
        med_pct = 0.0

    if mid2 < x <= high:
        high_pct = (df_col[(df_col > mid2) & (df_col <= high)] <= x).mean() * 100
    else:
        high_pct = 0.0

    return low_pct, med_pct, high_pct


def _select_top_pack_num(df_fis: pd.DataFrame) -> pd.DataFrame:
    priority_order = ["Low", "Med", "High"]
    priority_map = {cls: i for i, cls in enumerate(priority_order)}

    selected_list = []
    remaining = MIN_CLUSTER_SIZE

    for cls in priority_order:
        if remaining <= 0:
            break
        cls_df = df_fis[df_fis["FIS_Class"] == cls].copy()
        pct_col = f"{cls}_mean(%)"
        if pct_col not in cls_df.columns or cls_df.empty:
            continue
        cls_df = cls_df.sort_values(by=pct_col, ascending=True)
        take_n = min(remaining, len(cls_df))
        if take_n > 0:
            selected_list.append(cls_df.head(take_n))
            remaining -= take_n

    if remaining > 0:
        already_selected = (
            pd.concat(selected_list, ignore_index=True)["Lot Number"].tolist()
            if selected_list
            else []
        )
        leftover = df_fis[~df_fis["Lot Number"].isin(already_selected)].copy()
        if not leftover.empty:
            def get_class_pct_val(row):
                cls = row["FIS_Class"]
                pct_col = f"{cls}_mean(%)"
                val = row.get(pct_col, np.nan)
                return val if pd.notna(val) else 1e9

            leftover["_class_priority"] = leftover["FIS_Class"].map(priority_map)
            leftover["_class_pct"] = leftover.apply(get_class_pct_val, axis=1)
            leftover = leftover.sort_values(by=["_class_priority", "_class_pct"], ascending=[True, True])
            to_take = leftover.head(remaining)
            selected_list.append(to_take)

    if selected_list:
        return pd.concat(selected_list, ignore_index=True).reset_index(drop=True)
    return pd.DataFrame(columns=df_fis.columns)


def _cluster_sheet_indices(cs1_file: str) -> List[int]:
    xls = pd.ExcelFile(cs1_file)
    cluster_indices = []
    for sheet in xls.sheet_names:
        match = re.fullmatch(r"Cluster(\d+)", sheet)
        if match:
            cluster_indices.append(int(match.group(1)))
    if not cluster_indices:
        raise ValueError(f"[Step2] No Cluster# sheets found in {cs1_file}.")
    return sorted(cluster_indices)


def _available_analysis_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    available_cols = [c for c in cols if c in df.columns]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        print(f"[Step2][WARN] Missing columns skipped: {missing_cols}")
    if available_cols:
        return available_cols

    candidate_cols = [c for c in df.columns if c != "Lot Number"]
    numeric_cols = []
    for c in candidate_cols:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().any():
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError(f"[Step2] No analysis columns found. Requested: {cols}")
    fallback_cols = numeric_cols[:2]
    print(f"[Step2][WARN] Using fallback columns: {fallback_cols}")
    return fallback_cols


def _select_best_raw_cells(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if len(df) < MIN_CLUSTER_SIZE:
        raise ValueError(
            f"[Step2] Not enough rows ({len(df)}) to select {MIN_CLUSTER_SIZE} cells."
        )
    if "Lot Number" not in df.columns:
        raise ValueError("[Step2] Missing required column: Lot Number")

    df = df.copy()
    cols = _available_analysis_columns(df, cols)
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    print(f"[Step2] Loaded {len(df)} rows for analysis columns: {cols}")

    print("[Step2] Calculating percentile ranges and membership scores...")
    ranges = _calculate_ranges(df, cols)
    df_fis = df[["Lot Number"] + cols].copy()
    for col in cols:
        df_fis[[f"{col}_Low_pct(%)", f"{col}_Med_pct(%)", f"{col}_High_pct(%)"]] = df_fis[col].apply(
            lambda x: pd.Series(_membership_percentile(x, col, ranges, df[col]))
        )

    df_fis["Low_mean(%)"] = df_fis[[f"{c}_Low_pct(%)" for c in cols]].mean(axis=1)
    df_fis["Med_mean(%)"] = df_fis[[f"{c}_Med_pct(%)" for c in cols]].mean(axis=1)
    df_fis["High_mean(%)"] = df_fis[[f"{c}_High_pct(%)" for c in cols]].mean(axis=1)

    def pick_fis_class(row):
        vals = [row["Low_mean(%)"], row["Med_mean(%)"], row["High_mean(%)"]]
        idx = int(np.argmax(vals))
        return ["Low", "Med", "High"][idx]

    df_fis["FIS_Class"] = df_fis.apply(pick_fis_class, axis=1)

    print(f"[Step2] Selecting top {MIN_CLUSTER_SIZE} cells based on FIS results...")
    df_selected_pack_num = _select_top_pack_num(df_fis.copy())
    selected_lot_nums = df_selected_pack_num["Lot Number"].unique().tolist()
    selected_order = pd.DataFrame({"Lot Number": selected_lot_nums})
    df_selected_raw = selected_order.merge(df, on="Lot Number", how="left")
    print(f"[Step2] Selected {len(df_selected_raw)} cells for Best_{MIN_CLUSTER_SIZE}cells_raw sheet.")
    return df_selected_raw


def run_step2(
    cs1_file: str,
    cluster_index: Optional[int] = DEFAULT_CLUSTER_INDEX,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cols: Optional[List[str]] = None,
    worst_cluster: Optional[int] = None,
) -> str:
    if worst_cluster is not None:
        print("[Step2][WARN] worst_cluster is ignored. Worst cells sheets are no longer generated.")
    cols = cols or ANALYSIS_COLUMNS
    cluster_index = cluster_index if cluster_index is not None else DEFAULT_CLUSTER_INDEX
    sheet_name = f"Cluster{cluster_index}"

    print(f"[Step2] Loading '{cs1_file}' sheet '{sheet_name}'...")
    df = pd.read_excel(cs1_file, sheet_name=sheet_name)
    df_selected_raw = _select_best_raw_cells(df, cols)
    best_raw_sheet_name = f"Best_{MIN_CLUSTER_SIZE}cells_raw({cluster_index})"

    output_path = os.path.join(output_dir, "Step2_Results.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_selected_raw.to_excel(writer, sheet_name=best_raw_sheet_name, index=False)

    print(f"[Step2] 저장 완료: {output_path}")
    return output_path


def run_step2_all(
    cs1_file: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cols: Optional[List[str]] = None,
) -> str:
    cols = cols or ANALYSIS_COLUMNS
    cluster_indices = _cluster_sheet_indices(cs1_file)
    output_path = os.path.join(output_dir, "Step2_Results.xlsx")

    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for cluster_index in cluster_indices:
            sheet_name = f"Cluster{cluster_index}"
            print(f"[Step2] Loading '{cs1_file}' sheet '{sheet_name}'...")
            df = pd.read_excel(cs1_file, sheet_name=sheet_name)
            df_selected_raw = _select_best_raw_cells(df, cols)
            best_raw_sheet_name = f"Best_{MIN_CLUSTER_SIZE}cells_raw({cluster_index})"
            df_selected_raw.to_excel(writer, sheet_name=best_raw_sheet_name, index=False)

    print(f"[Step2] 저장 완료: {output_path}")
    return output_path


__all__ = ["run_step2", "run_step2_all"]
