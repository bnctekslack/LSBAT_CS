from __future__ import annotations

from typing import Dict, List, Tuple, Optional


ItemSpec = Dict[str, object]


"""
ITEM_SPECS 작성 방법:
- source: 원본 엑셀 컬럼명(예: "Unnamed: 35")
- start/end: 데이터 행 범위 (1-based, 엑셀 행 기준)
- title: 그래프/로그에 표시될 제목
- save_col: 결과 엑셀 컬럼명 (단위 포함 권장)
- ylim/ystep: 박스플롯 y축 범위/간격 (없으면 None)
- lot_screen: LotScreening 포함 여부 (기본 True)
- step0: Step0 분석 포함 여부 (기본 True)
- step1: Step1 클러스터링 포함 여부 (기본 True)
- step2: Step2 분석 포함 여부 (기본 False)
- weight: Step1 가중치 (step1=True일 때만 사용)
- lot_limit: LotScreening limit(%) (없으면 None)
- is_used: Used 표시용 컬럼이면 True
"""

# 예시 템플릿 (필요 시 복사 후 사용)
# {
#     "source": "Unnamed: 40",
#     "start": 6,
#     "end": 1045,
#     "title": "New Metric",
#     "save_col": "New Metric(unit)",
#     "ylim": None,
#     "ystep": 0.1,
#     "lot_screen": True,
#     "step0": True,
#     "step1": True,
#     "step2": False,
#     "weight": 1.0,
#     "lot_limit": 0.5,
# }

ITEM_SPECS: List[ItemSpec] = [
    {
        "source": "Unnamed: 23", # Initial ACIR
        "start": 6,
        "end": 1045,
        "title": "Initial ACIR",
        "save_col": "Initial ACIR(mΩ)",
        "ylim": (10.4, 12.4),
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.5,
        "lot_limit": 15.0,
    },
    {
        "source": "Unnamed: 26", # 100% ACIR
        "start": 6,
        "end": 1045,
        "title": "ACIR100",
        "save_col": "ACIR100(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": False,
        "step0": False,
        "step1": False,
        "lot_limit": 15.0,
    },
    {
        "source": "Unnamed: 29", # 0% ACIR
        "start": 6,
        "end": 1045,
        "title": "ACIR0",
        "save_col": "ACIR0(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": False,
        "step0": False,
        "step1": False,
        "lot_limit": 15.0,
    },
    {
        "source": "Unnamed: 33", # 50% ACIR
        "start": 6,
        "end": 1045,
        "title": "ACIR50",
        "save_col": "ACIR50(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": False,
        "step0": False,
        "step1": False,
        "lot_limit": 15.0,
    },
    {
        "source": "Unnamed: 7", # Weight
        "start": 6,
        "end": 1045,
        "title": "Weight",
        "save_col": "Weight(g)",
        "ylim": (70.0, 71.2),
        "ystep": 0.1,
        "step1": False,
        "lot_limit": 2.0,
    },
    {
        "source": "Unnamed: 12", # Height
        "start": 6,
        "end": 1045,
        "title": "Height",
        "save_col": "Height(mm)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": False,
        "lot_limit": 2.0,
    },
    {
        "source": "Unnamed: 14", # Diameter
        "start": 6,
        "end": 1045,
        "title": "Diameter",
        "save_col": "Diameter(mm)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": False,
        "lot_limit": 2.0,
    },
    {
        "source": "Unnamed: 30",  # Capacity
        "start": 6,
        "end": 1045,
        "title": "Capacity",
        "save_col": "Capacity(Ah)",
        "ylim": (4.78, 5.20),
        "ystep": 0.01,
        "lot_screen": False,
        "step0": True,
        "step1": True,
        "step2": True,
        "weight": 4.0,
        "lot_limit": 3.0,
    },
    {
        "source": "Unnamed: 22", # Initial Voltage
        "start": 6,
        "end": 1045,
        "title": "Initial Voltage",
        "save_col": "Initial Voltage(V)",
        "ylim": (3.44, 3.47),
        "ystep": 0.001,
        "lot_screen": False,
        "step0": False,
        "step1": False,
        "lot_limit": 0.5,
    },
    {
        "source": "Unnamed: 25",
        "start": 6,
        "end": 1045,
        "title": "Voltage4.2",   #4.2V
        "save_col": "Voltage4.2(V)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": False,
        "step0": False,
        "step1": False,
        "lot_limit": 0.5,
    },
    {
        "source": "Unnamed: 28",
        "start": 6,
        "end": 1045,
        "title": "Voltage2.5",        #2.5V
        "save_col": "Voltage2.5(V)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": False,
        "step0": False,
        "step1": False,
        "lot_limit": 4.0,
    },
    {
        "source": "Unnamed: 32",
        "start": 6,
        "end": 1045,
        "title": "Voltage3.6",      #3.6V
        "save_col": "Voltage3.6(V)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 0.2,
        "lot_limit": 0.8,
    },
    {
        "source": "Unnamed: 35", # OCV100
        "start": 6,
        "end": 1045,
        "title": "OCV100",
        "save_col": "OCV100(V)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 0.5,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 36", # DCIR100_start
        "start": 6,
        "end": 1045,
        "title": "DCIR100_start",
        "save_col": "DCIR100_start(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 0.8,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 37", # DCIR100_1s
        "start": 6,
        "end": 1045,
        "title": "DCIR100_1s",
        "save_col": "DCIR100_1s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.2,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 38", # DCIR100_10s
        "start": 6,
        "end": 1045,
        "title": "DCIR100_10s",
        "save_col": "DCIR100_10s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.5,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 40", # OCV90
        "start": 6,
        "end": 1045,
        "title": "OCV90",
        "save_col": "OCV90(V)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 0.3,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 41", # DCIR90_start
        "start": 6,
        "end": 1045,
        "title": "DCIR90_start",
        "save_col": "DCIR90_start(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.0,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 42", # DCIR90_1s
        "start": 6,
        "end": 1045,
        "title": "DCIR90_1s",
        "save_col": "DCIR90_1s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.3,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 43", # DCIR90_10s
        "start": 6,
        "end": 1045,
        "title": "DCIR90_10s",
        "save_col": "DCIR90_10s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.6,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 45", # OCV50
        "start": 6,
        "end": 1045,
        "title": "OCV50",
        "save_col": "OCV50(V)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.0,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 46", # DCIR50_start
        "start": 6,
        "end": 1045,
        "title": "DCIR50_start",
        "save_col": "DCIR50_start(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.4,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 47", # DCIR50_1s
        "start": 6,
        "end": 1045,
        "title": "DCIR50_1s",
        "save_col": "DCIR50_1s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.8,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 48", # DCIR50_10s
        "start": 6,
        "end": 1045,
        "title": "DCIR50_10s",
        "save_col": "DCIR50_10s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "step2": True,
        "weight": 2.5,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 50", # OCV10
        "start": 6,
        "end": 1045,
        "title": "OCV10",
        "save_col": "OCV10(V)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.2,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 51", # DCIR10_start
        "start": 6,
        "end": 1045,
        "title": "DCIR10_start",
        "save_col": "DCIR10_start(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 1.8,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 52", # DCIR10_1s
        "start": 6,
        "end": 1045,
        "title": "DCIR10_1s",
        "save_col": "DCIR10_1s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 2.6,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 53", # DCIR10_10s
        "start": 6,
        "end": 1045,
        "title": "DCIR10_10s",
        "save_col": "DCIR10_10s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "step2": True,
        "weight": 3.5,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 55", # OCV0
        "start": 6,
        "end": 1045,
        "title": "OCV0",
        "save_col": "OCV0(V)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 0.2,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 56", # DCIR0_start
        "start": 6,
        "end": 1045,
        "title": "DCIR0_start",
        "save_col": "DCIR0_start(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 2.2,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 57", # DCIR0_1s
        "start": 6,
        "end": 1045,
        "title": "DCIR0_1s",
        "save_col": "DCIR0_1s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "step2": True,
        "weight": 3.0,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 58", # DCIR0_10s
        "start": 6,
        "end": 1045,
        "title": "DCIR0_10s",
        "save_col": "DCIR0_10s(mΩ)",
        "ylim": None,
        "ystep": 0.1,
        "lot_screen": True,
        "step0": True,
        "step1": True,
        "weight": 0.5,
        "lot_limit": None,
    },
    {
        "source": "Unnamed: 59",
        "start": 6,
        "end": 1045,
        "title": "Used",
        "save_col": "Used",
        "ylim": None,
        "ystep": 0.1,
        "step1": False,
        "weight": None,
        "lot_limit": None,
        "is_used": True,
    },
]


def build_step0_items() -> List[Tuple[object, int, int, str, str, Optional[tuple], float]]:
    items: List[Tuple[object, int, int, str, str, Optional[tuple], float]] = []
    for spec in ITEM_SPECS:
        if spec.get("step0", True) is False:
            continue
        items.append(
            (
                spec["source"],
                int(spec["start"]),
                int(spec["end"]),
                str(spec["title"]),
                str(spec["save_col"]),
                spec.get("ylim"),
                float(spec.get("ystep", 0.1)),
            )
        )
    return items


def build_desired_order() -> List[str]:
    return ["Lot Number"] + [str(spec["save_col"]) for spec in ITEM_SPECS]


def build_step1_columns() -> List[str]:
    return [
        str(spec["save_col"])
        for spec in ITEM_SPECS
        if spec.get("step1", True)
    ]


def build_step2_columns() -> List[str]:
    return [
        str(spec["save_col"])
        for spec in ITEM_SPECS
        if spec.get("step2", False)
    ]


def build_step1_weights() -> Dict[str, float]:
    return {
        str(spec["save_col"]): float(spec["weight"])
        for spec in ITEM_SPECS
        if spec.get("step1", True) and spec.get("weight") is not None
    }


def build_lot_screen_limits() -> Dict[str, float]:
    return {
        str(spec["save_col"]): float(spec["lot_limit"])
        for spec in ITEM_SPECS
        if spec.get("lot_limit") is not None
        and spec.get("lot_screen", True)
        and not spec.get("is_used", False)
    }


ANALYSIS_ITEMS = build_step0_items()
DESIRED_ORDER = build_desired_order()
STEP1_COLUMNS = build_step1_columns()
STD_COLS = build_step1_columns()
DEFAULT_WEIGHTS = build_step1_weights()
LOT_SCREEN_LIMITS = build_lot_screen_limits()
STEP2_COLUMNS = build_step2_columns()

# Size of Battery Pack for clustering
BAT_PACK_SERIES_SIZE = 8
BAT_PACK_PARALLEL_SIZE = 9
MIN_CLUSTER_SIZE = (BAT_PACK_SERIES_SIZE * BAT_PACK_PARALLEL_SIZE)
MAX_CLUSTER_SIZE = MIN_CLUSTER_SIZE * 1.5  # Allow some flexibility
