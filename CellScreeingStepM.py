import os
import re
from typing import Optional

import pandas as pd

from analysis_config import BAT_PACK_SERIES_SIZE, BAT_PACK_PARALLEL_SIZE, MIN_CLUSTER_SIZE


DEFAULT_OUTPUT_DIR = "Results"
DEFAULT_GROUP_SIZE = BAT_PACK_PARALLEL_SIZE

CAPACITY_COL = "Capacity(Ah)"
DCIR_SECOND_COL = "DCIR10_10s(mΩ)"


def _group_cells(
    df: pd.DataFrame,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("[StepM] Input sheet is empty.")

    if CAPACITY_COL not in df.columns:
        raise ValueError(f"[StepM] Missing required column: {CAPACITY_COL}")
    if DCIR_SECOND_COL not in df.columns:
        raise ValueError(f"[StepM] Missing required column: {DCIR_SECOND_COL}")

    total = len(df)
    if total < group_size:
        raise ValueError(f"[StepM] Not enough rows ({total}) to form a group of {group_size}.")
    if total % group_size != 0:
        raise ValueError(
            f"[StepM] Row count ({total}) is not divisible by group size ({group_size})."
        )

    expected_total = BAT_PACK_SERIES_SIZE * BAT_PACK_PARALLEL_SIZE
    if total != expected_total:
        print(
            f"[StepM][WARN] Expected {expected_total} rows (Series*Parallel), got {total}."
        )

    df_sorted = df.copy()
    df_sorted[CAPACITY_COL] = pd.to_numeric(df_sorted[CAPACITY_COL], errors="coerce")
    df_sorted[DCIR_SECOND_COL] = pd.to_numeric(df_sorted[DCIR_SECOND_COL], errors="coerce")
    if df_sorted[CAPACITY_COL].isna().all():
        raise ValueError(f"[StepM] {CAPACITY_COL} has no numeric values.")
    if df_sorted[DCIR_SECOND_COL].isna().all():
        raise ValueError(f"[StepM] {DCIR_SECOND_COL} has no numeric values.")
    df_sorted = df_sorted.sort_values(by=CAPACITY_COL).reset_index(drop=True)

    # 1) Capacity 기준으로 Series 개수만큼 밴드 생성
    df_sorted["Band"] = pd.qcut(
        df_sorted.index + 1,
        q=BAT_PACK_SERIES_SIZE,
        labels=[i + 1 for i in range(BAT_PACK_SERIES_SIZE)],
    )

    # 2) 각 Band 안에서 Parallel 위치(1P~9P) 번호 부여
    df_grouped = df_sorted.reset_index(drop=True).copy()
    df_grouped["Band"] = df_grouped["Band"].astype(int)
    df_grouped["Group"] = df_grouped.groupby("Band").cumcount() + 1
    return df_grouped


def run_stepM(
    step2_file: str,
    cluster_index: Optional[int] = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    suffix = f"({cluster_index})" if cluster_index is not None else ""
    best_raw_sheet = (
        f"Best_{MIN_CLUSTER_SIZE}cells_raw{suffix}"
        if suffix
        else f"Best_{MIN_CLUSTER_SIZE}cells_raw"
    )

    print(f"[StepM] Loading '{step2_file}' sheet '{best_raw_sheet}'...")
    df = pd.read_excel(step2_file, sheet_name=best_raw_sheet)
    df_grouped = _group_cells(df, group_size=group_size)
    output_sheet = f"Cluster{cluster_index}_Grouped" if cluster_index is not None else "Grouped"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "StepM_Results.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_grouped.to_excel(writer, sheet_name=output_sheet, index=False)

    print(f"[StepM] 저장 완료: {output_path}")
    return output_path


def run_stepM_all(
    step2_file: str,
    group_size: int = DEFAULT_GROUP_SIZE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    xls = pd.ExcelFile(step2_file)
    sheet_items = []
    pattern = re.compile(rf"Best_{MIN_CLUSTER_SIZE}cells_raw\((\d+)\)")
    for sheet in xls.sheet_names:
        match = pattern.fullmatch(sheet)
        if match:
            sheet_items.append((int(match.group(1)), sheet))
    if not sheet_items:
        raise ValueError(f"[StepM] No Best_{MIN_CLUSTER_SIZE}cells_raw(#) sheets found in {step2_file}.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "StepM_Results.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for cluster_index, sheet in sorted(sheet_items):
            print(f"[StepM] Loading '{step2_file}' sheet '{sheet}'...")
            df = pd.read_excel(step2_file, sheet_name=sheet)
            df_grouped = _group_cells(df, group_size=group_size)
            df_grouped.to_excel(writer, sheet_name=f"Cluster{cluster_index}_Grouped", index=False)

    print(f"[StepM] 저장 완료: {output_path}")
    return output_path


__all__ = ["run_stepM", "run_stepM_all"]
