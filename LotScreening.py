import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from CellScreeningStep0 import DEFAULT_DATA_FILE, DEFAULT_SHEET_NAME
from analysis_config import ANALYSIS_ITEMS, LOT_SCREEN_LIMITS

DEFAULT_OUTPUT_DIR = "Results"
METRIC_DETAIL_COLUMNS = [
    "Metric",
    "Column",
    "Raw Samples",
    "Used Samples",
    "Removed Outliers",
    "Lower Bound (+/-3sigma)",
    "Upper Bound (+/-3sigma)",
    "Mean",
    "Std Dev",
    "Six Sigma",
    "Six Sigma %",
    "Limit %",
    "Metric Pass",
    "Outlier Lots",
]


def _detect_lot_column(df: pd.DataFrame) -> str:
    for col in ["Lot Number", "LOT", "Unnamed: 0"]:
        if col in df.columns:
            return col
    return df.columns[0]


def _build_lot_items() -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for entry in ANALYSIS_ITEMS:
        if len(entry) < 5:
            continue
        column, start, end, title, save_col, *_ = entry
        if str(save_col).strip().lower() == "used":
            continue

        metric_name = title
        limit_pct = LOT_SCREEN_LIMITS.get(save_col)
        if limit_pct is None:
            # LotScreening 제외 항목은 건너뜀
            continue

        column_candidates = []
        for cand in (column, save_col, title):
            if cand and cand not in column_candidates:
                column_candidates.append(str(cand))

        items.append(
            {
                "metric": metric_name,
                "column_candidates": column_candidates,
                "label": save_col,
                "limit_pct": limit_pct,
                "start": start,
                "end": end,
            }
        )
    return items


LOT_SCREEN_TARGETS = _build_lot_items()
LOT_SCREEN_ITEMS = LOT_SCREEN_TARGETS


def _extract_metric_frame(
    df: pd.DataFrame, lot_col: str, item: Dict[str, object], column_name: str
) -> pd.DataFrame:
    start_idx = int(item.get("start", 1)) - 2
    end_idx = int(item.get("end", len(df))) - 2
    value_series = pd.to_numeric(
        df.loc[start_idx:end_idx, column_name], errors="coerce"
    )
    lot_series = df.loc[start_idx:end_idx, lot_col]
    result = pd.DataFrame(
        {"Lot Number": lot_series, "value": value_series}, copy=False
    )
    return result.dropna(subset=["Lot Number"])


def _evaluate_metric(values: pd.Series, limit_pct: Optional[float]) -> Tuple[Dict[str, float], bool, List[str]]:
    clean_values = values.dropna()
    raw_count = int(clean_values.shape[0])
    if raw_count == 0:
        stats = {
            "raw_count": 0,
            "used_count": 0,
            "removed": 0,
            "mean": np.nan,
            "std": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "six_sigma": np.nan,
            "ratio_pct": np.nan,
        }
        return stats, False, []

    raw_mean = float(clean_values.mean())
    raw_std = float(np.std(clean_values.values, ddof=0))
    lower = raw_mean - 3 * raw_std
    upper = raw_mean + 3 * raw_std
    mask = (clean_values >= lower) & (clean_values <= upper)
    filtered = clean_values[mask]
    filtered_count = int(filtered.shape[0])
    removed_lots = clean_values.index[~mask].tolist()
    removed = raw_count - filtered_count

    if filtered_count == 0:
        stats = {
            "raw_count": raw_count,
            "used_count": 0,
            "removed": removed,
            "mean": np.nan,
            "std": np.nan,
            "lower": lower,
            "upper": upper,
            "six_sigma": np.nan,
            "ratio_pct": np.nan,
        }
        # 기준이 없으면 PASS 처리, 있으면 FAIL
        metric_pass = limit_pct is None
        return stats, metric_pass, removed_lots

    mean = float(filtered.mean())
    std = float(np.std(filtered.values, ddof=0))
    six_sigma = 6 * std
    ratio_pct = np.inf if mean == 0 else (six_sigma / mean) * 100
    if limit_pct is None:
        metric_pass = True
    else:
        metric_pass = bool(np.isfinite(ratio_pct) and ratio_pct <= limit_pct)
    stats = {
        "raw_count": raw_count,
        "used_count": filtered_count,
        "removed": removed,
        "mean": mean,
        "std": std,
        "lower": lower,
        "upper": upper,
        "six_sigma": six_sigma,
        "ratio_pct": ratio_pct,
    }
    return stats, metric_pass, removed_lots


def _resolve_column_name(df_columns: pd.Index, candidates: List[str]) -> Optional[str]:
    normalized_map = {str(col).strip().lower(): col for col in df_columns}
    for cand in candidates:
        if not cand:
            continue
        cand_str = str(cand).strip()
        if cand_str in df_columns:
            return cand_str
        cand_norm = cand_str.lower()
        if cand_norm in normalized_map:
            return normalized_map[cand_norm]
        match = re.match(r"unnamed:\s*(\d+)", cand_norm)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(df_columns):
                return df_columns[idx]
    return None


def _sanitize_sheet_name(base: str, existing: set) -> str:
    sanitized = re.sub(r'[\\*?:/\[\]]', "_", base) if base else "Metric"
    sanitized = sanitized.strip() or "Metric"
    sanitized = sanitized[:31]
    candidate = sanitized
    idx = 2
    while candidate in existing:
        suffix = f"_{idx}"
        candidate = (sanitized[: 31 - len(suffix)] + suffix) if len(sanitized) + len(suffix) > 31 else sanitized + suffix
        idx += 1
    existing.add(candidate)
    return candidate


def run_lot_screen(
    input_path: str = DEFAULT_DATA_FILE,
    sheet_name: str = DEFAULT_SHEET_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> tuple[str, bool]:
    os.makedirs(output_dir, exist_ok=True)
    print("[LotScreen] Loading source data...")
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    lot_col = _detect_lot_column(df)
    print(f"[LotScreen] Lot identifier column: {lot_col}")

    metric_records: List[Dict[str, object]] = []
    limit_check_records: List[Dict[str, object]] = []
    overall_pass = True

    for item in LOT_SCREEN_ITEMS:
        column_name = _resolve_column_name(df.columns, item["column_candidates"])
        if column_name is None:
            print(f"[LotScreen][WARN] Columns {item['column_candidates']} not found. Skipping.")
            continue
        print(f"[LotScreen] Processing metric '{item['metric']}' ({item['label']})...")
        metric_frame = _extract_metric_frame(df, lot_col, item, column_name)
        if metric_frame.empty:
            print(f"[LotScreen][WARN] No data found for '{item['metric']}'.")
            continue
        metric_series = (
            metric_frame.dropna(subset=["value"])
            .set_index("Lot Number")["value"]
        )
        limit_value = item["limit_pct"]
        stats, metric_pass, removed_lots = _evaluate_metric(metric_series, limit_value)
        outlier_list = ""
        if removed_lots:
            outlier_list = ", ".join(str(lot)[-4:] for lot in removed_lots)

        if limit_value is not None and np.isfinite(stats["ratio_pct"]):
            print(
                f"[LotScreen]   → {item['label']}: Six Sigma % = {stats['ratio_pct']:.4f}% "
                f"(limit {limit_value:.2f}%) => {'PASS' if metric_pass else 'FAIL'}"
            )

        metric_records.append(
            {
                "Metric": item["metric"],
                "Column": item["label"],
                "Raw Samples": stats["raw_count"],
                "Used Samples": stats["used_count"],
                "Removed Outliers": stats["removed"],
                "Lower Bound (+/-3sigma)": stats["lower"],
                "Upper Bound (+/-3sigma)": stats["upper"],
                "Mean": stats["mean"],
                "Std Dev": stats["std"],
                "Six Sigma": stats["six_sigma"],
                "Six Sigma %": stats["ratio_pct"],
                "Limit %": limit_value if limit_value is not None else np.nan,
                "Metric Pass": "PASS" if metric_pass else "FAIL",
                "Outlier Lots": outlier_list,
            }
        )
        if limit_value is not None:
            limit_check_records.append(
                {
                    "Metric": item["metric"],
                    "Column": item["label"],
                    "Six Sigma %": stats["ratio_pct"],
                    "Limit %": limit_value,
                    "Metric Pass": "PASS" if metric_pass else "FAIL",
                }
            )
            overall_pass = overall_pass and metric_pass

    metric_df = pd.DataFrame(metric_records)
    if metric_df.empty:
        metric_df = pd.DataFrame(columns=METRIC_DETAIL_COLUMNS)
    limit_summary_columns = ["Metric", "Column", "Six Sigma %", "Limit %", "Metric Pass"]
    limit_summary = pd.DataFrame(limit_check_records, columns=limit_summary_columns)

    output_path = os.path.join(output_dir, "LotScreening.xlsx")
    print("[LotScreen] Writing Excel output...")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        metric_df.to_excel(writer, sheet_name="Metric_Detail", index=False)
        limit_summary.to_excel(writer, sheet_name="Metric_Summary", index=False)

    failed = int((limit_summary["Metric Pass"] == "FAIL").sum()) if not limit_summary.empty else 0
    print(f"[LotScreen] Result File: {output_path}")
    print(f"[LotScreen] Metrics Checked: {len(limit_summary)} | Failed Metrics: {failed}")
    if failed:
        failing_metrics = limit_summary[limit_summary["Metric Pass"] == "FAIL"]["Metric"]
        print(f"[LotScreen] Failed Metrics: {', '.join(map(str, failing_metrics))}")

    lot_pass = overall_pass
    return output_path, lot_pass


__all__ = ["run_lot_screen"]
