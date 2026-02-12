import os
import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.drawing.image import Image as XLImage

from analysis_config import ANALYSIS_ITEMS, DESIRED_ORDER, DEFAULT_WEIGHTS

# SDI 21700 배터리 사양서
#- 공칭 용량: 5.0Ah ± 4%  → (4.80 ~ 5.20 Ah)
#- 공칭 전압: 3.6V, 초기 전압: 3.45V ± 0.015V → (3.435 ~ 3.465 V)
#- 무게: 70.5g ± 1.7%  → (69.3 ~ 71.7 g)
#- 초기 ACIR: 11.4mΩ ± 8.8%  → (10.4 ~ 12.4 mΩ)

DEFAULT_DATA_FILE = "RawData/SDI_21700_50S_특성데이터(260212-2).xlsx"
DEFAULT_SHEET_NAME = "Raw Data"
DEFAULT_OUTPUT_DIR = "Results"
DEFAULT_IQR_FACTOR = 2.5 #1.5 ~ 3.0


def detect_lot_column(df: pd.DataFrame) -> str:
    possible_cols = ["Lot Number", "LOT", "Unnamed: 0"]
    return next((c for c in possible_cols if c in df.columns), df.columns[0])


def _resolve_column_name(columns: pd.Index, *candidates: str) -> str | None:
    normalized_map = {str(col).strip().lower(): col for col in columns}
    for cand in candidates:
        if cand is None:
            continue
        cand_str = str(cand).strip()
        if not cand_str:
            continue
        if cand_str in columns:
            return cand_str
        cand_norm = cand_str.lower()
        if cand_norm in normalized_map:
            return normalized_map[cand_norm]
        match = re.match(r"unnamed:\s*(\d+)", cand_norm)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(columns):
                return columns[idx]
    return None


def get_outliers(series: pd.Series, manual_range=None, factor: float = 1.5):
    Q1 = series.quantile(0.25)
    Q2 = series.median()
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    if manual_range is not None:
        lower, upper = manual_range
    else:
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

    outliers = series[(series < lower) | (series > upper)]
    non_outliers = series[(series >= lower) & (series <= upper)]
    whisker_low = series[series >= lower].min()
    whisker_high = series[series <= upper].max()
    return outliers, non_outliers, Q1, Q2, Q3, lower, upper, whisker_low, whisker_high


def run_step0(
    input_path: str = DEFAULT_DATA_FILE,
    sheet_name: str = DEFAULT_SHEET_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    iqr_factor: float = DEFAULT_IQR_FACTOR,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Step0] 결과 저장 폴더 생성 완료: {output_dir}")

    df = pd.read_excel(input_path, sheet_name=sheet_name)
    lot_col = detect_lot_column(df)

    # Used 컬럼 자동 감지: ANALYSIS_ITEMS 정의와 컬럼명 패턴을 모두 활용
    remark_columns = set()
    # ANALYSIS_ITEMS에 save_col이 Used인 항목의 원본 컬럼명 우선 사용
    used_from_items = [
        item[0] for item in ANALYSIS_ITEMS
        if len(item) >= 5 and str(item[4]).strip().lower() == "used"
    ]
    for col in used_from_items:
        resolved = _resolve_column_name(df.columns, col)
        if resolved:
            remark_columns.add(resolved)
        else:
            print(f"[Step0][WARN] Used column '{col}' not found in source data.")
    remark_columns = list(remark_columns)

    outlier_dict_total = {}
    non_outlier_dict_total = {}
    outlier_dict_temp = {}
    non_outlier_dict_temp = {}
    missing_data_dict = {}  # 측정값 누락 추적용
    graph_images = {}
    summary_records = []
    remark_map = {}
    lots_with_outlier = set()

    # 전체 Lot 리스트 추출 (ANALYSIS_ITEMS 범위 우선 사용)
    lot_series = df[lot_col]
    lot_rows = lot_series.notna()
    start_idx = min(item[1] for item in ANALYSIS_ITEMS) - 2
    end_idx = max(item[2] for item in ANALYSIS_ITEMS) - 2
    if 0 <= start_idx <= end_idx < len(df):
        range_index = df.loc[start_idx:end_idx].index
        data_index = range_index[lot_rows.loc[range_index]]
        all_lots = lot_series.loc[data_index].unique()
    elif lot_rows.sum() > 0:
        data_index = lot_series[lot_rows].index
        all_lots = lot_series[lot_rows].unique()
    else:
        # fallback: 기존 고정 범위
        data_index = df.loc[4:1043].index
        all_lots = df.loc[4:1043, lot_col].dropna().unique()
    
    # Used에 내용이 있는 Lot 수집 (내용이 있을 때만 Outliers로 이동)
    if remark_columns:
        remark_series = df.loc[data_index, [lot_col] + remark_columns]
        for _, row in remark_series.iterrows():
            lot_number = row[lot_col]
            if pd.notna(lot_number):
                remarks = [
                    str(row[col]).strip()
                    for col in remark_columns
                    if col in row and pd.notna(row[col]) and str(row[col]).strip()
                ]
                if remarks:
                    remark_map[lot_number] = " | ".join(remarks)
    
    # Used가 ANALYSIS_ITEMS에 들어 있더라도 분석 대상에서 제외
    analysis_items = [
        item for item in ANALYSIS_ITEMS
        if len(item) >= 5 and str(item[4]).strip().lower() != "used"
    ]
    
    for item in analysis_items:
        col, start, end, title, save_col, *rest = item
        ylim = rest[0] if len(rest) > 0 else None
        ystep = rest[1] if len(rest) > 1 else 0.1

        resolved_col = _resolve_column_name(df.columns, col, save_col, title)
        if resolved_col is None:
            print(f"[Step0][WARN] Column candidates {[col, save_col, title]} not found. Skipping {title}.")
            continue

        # 원본 데이터 (변환 전)
        raw_series = df.loc[data_index, resolved_col]
        
        # 숫자로 변환
        data = pd.to_numeric(raw_series, errors='coerce')
        
        # 측정값 없는 Lot 추적
        missing_indices = data[data.isna()].index
        for idx in missing_indices:
            lot_number = df.loc[idx, lot_col]
            if pd.notna(lot_number):  # Lot Number가 유효한 경우만
                missing_data_dict.setdefault(lot_number, []).append(save_col)
        
        # NaN 제거 후 분석
        data = data.dropna()
        
        if data.empty:
            continue

        manual_range = ylim if ylim else None
        outliers, non_outliers, Q1, Q2, Q3, lower, upper, _, _ = get_outliers(
            data, manual_range, iqr_factor
        )

        outlier_ratio = len(outliers) / len(data) * 100
        print(
            f"[Step0] [{title}] 이상치 {len(outliers)}개 / 전체 {len(data)}개 "
            f"({outlier_ratio:.2f}%) | 기준: [{lower:.3f} ~ {upper:.3f}]"
        )

        summary_records.append(
            {
                "항목명": save_col,
                "전체 개수": len(data),
                "이상치 개수": len(outliers),
                "이상치 비율(%)": round(outlier_ratio, 2),
                "하한(lower)": round(lower, 3),
                "상한(upper)": round(upper, 3),
                "Q1": round(Q1, 3),
                "Q2": round(Q2, 3),
                "Q3": round(Q3, 3),
                "IQR": round(Q3 - Q1, 3),
                "평균값": round(data.mean(), 3),
                "표준편차": round(data.std(ddof=1), 3),
                "가중치": round(float(DEFAULT_WEIGHTS.get(save_col, 1.0)), 3),
            }
        )

        for idx, value in outliers.items():
            lot_number = df.loc[idx, lot_col]
            outlier_dict_temp.setdefault(lot_number, {})[save_col] = value
            lots_with_outlier.add(lot_number)

        for idx, value in non_outliers.items():
            lot_number = df.loc[idx, lot_col]
            non_outlier_dict_temp.setdefault(lot_number, {})[save_col] = value

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        ax.set_title(title)
        ax.set_ylabel(save_col)
        if ylim:
            ax.set_ylim(*ylim)
            ax.set_yticks(np.arange(ylim[0], ylim[1] + ystep, ystep))
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        graph_images[title] = buf

    # ========== 제거 순서 적용 ==========
    # 1) 측정값이 누락된 Lot을 Outliers로 이동
    #print("\n📊 측정값 누락 처리 중...")
    for lot, missing_cols in missing_data_dict.items():
        #print(f"[Step0]    {lot}: {len(missing_cols)}개 항목 누락 → Outliers로 이동")
        outlier_dict_total.setdefault(lot, {})

        # Non_Outliers에 있던 데이터를 Outliers로 이동 후 제거
        if lot in non_outlier_dict_temp:
            outlier_dict_total[lot].update(non_outlier_dict_temp.pop(lot))

        # 누락된 항목 표시 (NaN으로)
        for col_name in missing_cols:
            outlier_dict_total.setdefault(lot, {})[col_name] = np.nan

    # 2) Used가 있는 Lot을 Outliers로 이동시키고 Used 컬럼으로 표시
    if remark_map:
        print("\n[Step0] Used에 사용 표시된 Lot를 Outliers로 이동 중...")
        for lot, remark_text in remark_map.items():
            outlier_dict_total.setdefault(lot, {})
            if lot in non_outlier_dict_temp:
                outlier_dict_total[lot].update(non_outlier_dict_temp.pop(lot))
            outlier_dict_total[lot]["Used"] = remark_text

    # 3) 이상치값 Lot을 Outliers로 이동
    for lot, values in outlier_dict_temp.items():
        outlier_dict_total.setdefault(lot, {}).update(values)
        if lot in non_outlier_dict_temp:
            non_outlier_dict_temp.pop(lot, None)

    # Non_Outliers 최종 확정
    non_outlier_dict_total = non_outlier_dict_temp
    # ====================================

    outlier_df = pd.DataFrame([{lot_col: lot, **v} for lot, v in outlier_dict_total.items()])
    non_outlier_df = pd.DataFrame(
        [
            {lot_col: lot, **v}
            for lot, v in non_outlier_dict_total.items()
        ]
    )

    rename_map = {lot_col: "Lot Number", "Unnamed: 0": "Lot Number"}
    outlier_df.rename(columns=rename_map, inplace=True)
    non_outlier_df.rename(columns=rename_map, inplace=True)
    outlier_df = outlier_df[[c for c in DESIRED_ORDER if c in outlier_df.columns]]
    non_outlier_df = non_outlier_df[[c for c in DESIRED_ORDER if c in non_outlier_df.columns]]

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df[
        [
            "항목명",
            "전체 개수",
            "이상치 개수",
            "이상치 비율(%)",
            "하한(lower)",
            "상한(upper)",
            "Q1",
            "Q2",
            "Q3",
            "IQR",
            "평균값",
            "표준편차",
            "가중치",
        ]
    ]

    # 실행 현황 출력
    initial_lot_count = len(all_lots)
    missing_lots = set(missing_data_dict.keys())
    used_lots = set(remark_map.keys())
    missing_lot_count = len(missing_lots)
    used_lot_count = len(used_lots)
    final_non_outlier_count = len(non_outlier_dict_total)
    final_outlier_count = len(outlier_dict_total)
    final_total_lot_count = final_non_outlier_count
    outlier_lots = set(outlier_dict_total.keys())
    outlier_union_count = len(missing_lots | used_lots | outlier_lots)
    print(f"[Step0] 초기 Lot 개수: {initial_lot_count}")
    print(f"[Step0] 1. 측정값 누락 Lot 개수: {missing_lot_count}")
    print(f"[Step0] 2. Used 표시 Lot 개수: {used_lot_count}")
    print(f"[Step0] 3. 측정값 이상치 개수: {len(outlier_dict_temp)}")
    print(f"[Step0] 이상치 제거 사유 합산(중복 제거): {outlier_union_count}")
    print(f"[Step0] 최종 남아있는 Lot 개수: {final_total_lot_count} (Outliers: {final_outlier_count}, Non_Outliers: {final_non_outlier_count})")

    output_file = os.path.join(output_dir, "Step0_Results.xlsx")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        outlier_df.to_excel(writer, sheet_name="Outliers_List", index=False)
        non_outlier_df.to_excel(writer, sheet_name="Non_Outliers_List", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        wb = writer.book
        ws = wb.create_sheet("Boxplot Graph")
        row = 1
        for title, buf in graph_images.items():
            buf.seek(0)
            img = XLImage(buf)
            ws.add_image(img, f"A{row}")
            row += 25

    print(f"\n[Step0] 저장 완료: {output_file}")
    print("[Step0] Summary 시트에서 항목별 이상치 비율과 IQR 확인 가능.")
    print(f"[Step0] 측정값 누락된 Lot: {len(missing_data_dict)}개 → Outliers_List로 분류됨")
    return output_file


__all__ = ["run_step0"]
