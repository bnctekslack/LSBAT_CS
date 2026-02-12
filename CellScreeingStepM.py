import os
from typing import List, Optional

import pandas as pd
from k_means_constrained import KMeansConstrained

from analysis_config import BAT_PACK_SERIES_SIZE, BAT_PACK_PARALLEL_SIZE, MIN_CLUSTER_SIZE
from CellScreeningStep1 import DEFAULT_WEIGHTS


DEFAULT_OUTPUT_DIR = "Results"
DEFAULT_GROUP_SIZE = BAT_PACK_PARALLEL_SIZE

CAPACITY_COL = "Capacity(Ah)"
DCIR_SECOND_COL = "DCIR10_10s(mΩ)"


def _top_weight_columns() -> List[str]:
    if not DEFAULT_WEIGHTS:
        raise ValueError("[StepM] DEFAULT_WEIGHTS is empty.")
    return [
        col for col, _ in sorted(DEFAULT_WEIGHTS.items(), key=lambda x: x[1], reverse=True)
    ]


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
    if df.empty:
        raise ValueError(f"[StepM] '{best_raw_sheet}' sheet is empty.")

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

    # 2) 밴드별로 Parallel 단위(9개) 그룹 부여 (밴드 내 추가 정렬 없음)
    df_grouped = df_sorted.reset_index(drop=True).copy()
    df_grouped["Group"] = df_grouped["Band"].astype(int)
    n_clusters = df_grouped["Group"].nunique()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "StepM_Results.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_grouped.to_excel(writer, sheet_name=f"{best_raw_sheet}_Grouped", index=False)
        for group_id in range(1, n_clusters + 1):
            group_df = df_grouped[df_grouped["Group"] == group_id]
            sheet_name = f"{best_raw_sheet}_Group{group_id:02d}"
            group_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[StepM] 저장 완료: {output_path}")
    return output_path


__all__ = ["run_stepM"]
