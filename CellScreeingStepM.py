import os
from typing import List, Optional

import pandas as pd
from k_means_constrained import KMeansConstrained

from CellScreeningStep1 import DEFAULT_WEIGHTS


DEFAULT_OUTPUT_DIR = "Results"
DEFAULT_GROUP_SIZE = 8


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
    best_raw_sheet = f"Best_72cells_raw{suffix}" if suffix else "Best_72cells_raw"

    print(f"[StepM] Loading '{step2_file}' sheet '{best_raw_sheet}'...")
    df = pd.read_excel(step2_file, sheet_name=best_raw_sheet)
    if df.empty:
        raise ValueError(f"[StepM] '{best_raw_sheet}' sheet is empty.")

    weight_cols = _top_weight_columns()
    similarity_cols = [col for col in weight_cols if col in df.columns]
    if not similarity_cols:
        raise ValueError(
            f"[StepM] No weighted similarity columns found in '{best_raw_sheet}'."
        )
    df_numeric = df[similarity_cols].apply(pd.to_numeric, errors="coerce")
    if df_numeric.isna().all().all():
        raise ValueError("[StepM] Similarity columns have no numeric values.")
    df_numeric = df_numeric.fillna(df_numeric.mean(numeric_only=True))

    total = len(df)
    if total < group_size:
        raise ValueError(f"[StepM] Not enough rows ({total}) to form a group of {group_size}.")
    if total % group_size != 0:
        raise ValueError(
            f"[StepM] Row count ({total}) is not divisible by group size ({group_size})."
        )

    n_clusters = total // group_size
    X = df_numeric.values

    print(f"[StepM] Grouping {total} rows into {n_clusters} groups of {group_size}...")
    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=group_size,
        size_max=group_size,
        random_state=42,
    )
    labels = kmeans.fit_predict(X)

    df_grouped = df.copy()
    df_grouped["Group"] = labels + 1
    df_grouped = df_grouped.sort_values(by="Group").reset_index(drop=True)

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
