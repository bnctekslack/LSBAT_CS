### 개요
이 프로그램은 LotScreening → Step0 → Step1 → Step2 순서로 배터리 셀을 선별합니다.
LotScreening은 Lot 단위의 품질 판정, Step0은 셀 단위 이상치 제거, Step1은 클러스터링, Step2는 최종 72개 셀 선정입니다.

### 분석 항목 설정 (중앙 설정)
모든 분석 항목은 `analysis_config.py`의 `ITEM_SPECS`에서 관리합니다.
여기서 항목을 추가/삭제하거나 사용 여부를 조정하면 Step0/Step1/LotScreening/Step2에 모두 반영됩니다.

`ITEM_SPECS` 주요 필드:
1. `source`: 원본 엑셀 컬럼명 (예: `Unnamed: 35`)
2. `save_col`: 결과/분석에서 사용하는 컬럼명 (단위 포함 권장)
3. `step0`, `step1`, `step2`, `lot_screen`: 각 단계 포함 여부
4. `weight`: Step1 가중치 (step1=True일 때만 사용)
5. `lot_limit`: LotScreening limit(%)
6. `is_used`: Used 표시용 컬럼이면 True

### LotScreening (Lot 단위 판정)
Lot. NASA JSC-20793 Rev.D 문서의 Lot Acceptance Testing 로직을 기반으로 합니다.
1. `analysis_config.py`의 `LOT_SCREEN_LIMITS`에 따라 항목별 3-sigma outlier 제거
2. outlier 제거 후 평균(μ), 표준편차(σ) 재계산
3. 변동성(%) = 6σ/μ * 100
4. 변동성이 limit(%)를 넘으면 Lot 불합격

### Step0 (셀 단위 이상치 제거)
Step0는 원본 엑셀에서 항목별 이상치를 탐지하고 Outliers/Non_Outliers를 분리합니다.
1. Used 표시 셀은 Outliers로 이동
2. 측정값 누락 셀은 Outliers로 이동
3. 나머지 항목에 대해 IQR 방식으로 이상치 판단
4. 결과: `Results/Step0_Results.xlsx`

IQR 기준:
1. Q1 (25%), Q2 (중앙값), Q3 (75%)
2. IQR = Q3 - Q1
3. 하한 = Q1 - (factor × IQR)
4. 상한 = Q3 + (factor × IQR)

### Step1 (클러스터링)
Step1은 Step0의 Non_Outliers를 대상으로 K-Means 클러스터링을 수행합니다.
1. `analysis_config.py`의 `STEP1_COLUMNS`, `DEFAULT_WEIGHTS` 사용
2. K는 DBI/CHI 점수를 이용해 최적 선택
3. 클러스터 번호는 1부터 시작
4. 결과: `Results/Step1_Results.xlsx`

### Step2 (최종 72개 선정)
Step2는 특정 클러스터 시트를 읽어 퍼지 멤버십/백분위수 기반으로 72개 셀을 선택합니다.
1. 분석 컬럼은 `analysis_config.py`의 `STEP2_COLUMNS` 사용
2. 컬럼이 없으면 경고 후 자동 fallback(숫자형 컬럼 중 첫 2개)
3. 결과: `Results/Step2_Results.xlsx`

### StepM (팩 구성용 그룹핑)
StepM은 Step2에서 선택된 Best 셀을 **Parallel 크기 기준으로 그룹핑**합니다.
1. Capacity(Ah)로 전체를 정렬
2. Capacity를 `BAT_PACK_SERIES_SIZE` 등분해 밴드 생성
3. 밴드당 `BAT_PACK_PARALLEL_SIZE`개씩 그룹으로 묶음
4. 결과: `Results/StepM_Results.xlsx`

### Pack 구성
1. 9개의 밴드를 round-robin 방식으로 1P~9P로 구성하면 됨.

### 자주 조정하는 항목
1. Step0 이상치 기준: `analysis_config.py`의 `ylim/ystep` 및 `DEFAULT_IQR_FACTOR`
2. Step1 가중치: `analysis_config.py`의 `weight`
3. LotScreening 기준: `analysis_config.py`의 `lot_limit`
4. Step2 분석 컬럼: `analysis_config.py`의 `step2=True` 설정

### 실행 방법 예시
1. 전체 파이프라인 실행
   - `python main.py`
2. 개별 실행
   - LotScreening: `run_lot_screen()`  
   - Step0: `run_step0()`  
   - Step1: `run_step1(cs0_path)`  
   - Step2: `run_step2(cs1_path, cluster_index=best_cluster)`

### 결과 파일 위치
모든 결과는 기본적으로 `Results/` 폴더에 저장됩니다.
