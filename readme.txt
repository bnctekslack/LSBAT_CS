Lot. NASA JSC-20793 Rev.D 문서의 Lot Acceptance Testing을 참조
    1. OCV, 질량, 용량 등에서 ±3σ 밖에 있는 셀(3-sigma outlier) 을 로트에서 제외
    2. 이상치를 제거한 “남은 셀들”로 평균(μ)과 표준편차(σ)를 다시 계산
    3. 그 결과 분포의 ±3σ 범위가 평균 대비 너무 크면(=흩어짐이 크면) 로트 자체를 불합격
        - 기준값(%) : 6σ/μ * 100%
        - 허용치
            OCV  1%
            무게  2%
            용량  5%
            ACIR 15% (DCIR 대체) 

Step0. 배터리 셀 특성 데이터의 이상치(outlier)를 탐지하고 분석
    1. Used 항목 추가 : 기존 시험에 사용되었던 배터리 셀을 제고하고 다음단계 진행
    2. ANALYSIS_ITEMS : ("Unnamed: 23", 6, 1045, "Initial ACIR Result", "Initial ACIR(mΩ)", (10.4, 12.4), 0.1)
        - **컬럼명**: Excel의 어느 열을 읽을지
        - **시작/끝 행**: 6행부터 1045행까지 데이터
        - **제목**: 그래프 제목
        - **저장명**: 결과 파일의 컬럼명
        - **수동 범위**: 이상치 판단 기준 (선택사항)
        - **Y축 간격**: 그래프 눈금 간격
    3. get_outliers :  IQR(사분위 범위) 방식으로 이상치를 찾음
        Q1 (25%) ─── Q2 (중앙값) ─── Q3 (75%)
            └─ IQR = Q3 - Q1 ─┘

        이상치 기준:
        - 하한: Q1 - (factor × IQR)  [기본 2.5배]
        - 상한: Q3 + (factor × IQR)

Step1. 정상으로 판정된 배터리들을 K-Means 클러스터링으로 그룹화
    1. 12개 특성을 기준으로 배터리를 분류
    2. compute_k_metrics : 최적의 K 찾기  => 9
        DBI (Davies-Bouldin Index) - 낮을수록 좋음
        CHI (Calinski-Harabasz Index) - 높을수록 좋음
    3. 데이터 정규화
    4. 최고의 클러스트 선정 : 각 클러스터의 표준편차 계산 -> 표준편차를 순위로 변환 -> 순위 합계가 가장 낮음 (가장 안정적)

Step2. 엑셀 파일에서 특정 클러스터 데이터를 읽어와 퍼지 멤버십(Fuzzy Membership) 및 백분위수(Percentile) 기반으로 
    1. 데이터를 분류하고, 최종적으로 72개의 샘플을 우선순위에 따라 선정하여 결과를 엑셀 파일로 저장
    2. 분석에 사용할 컬럼 이름들 
    3. _calculate_ranges() : 각 분석 컬럼의 데이터 범위를 계산하고, 데이터를 3분위수(33%, 66%)를 기준으로 
        Low, Mid1, Mid2, High 4개 지점으로 나눕니다
    4. _membership_percentile() : 특정 데이터 포인트가 Low, Med1-Mid2, High 범위에 속할 때, 
        해당 범위 내에서의 백분위수를 계산하여 반환합니다.
    5. _select_top_72() : 계산된 FIS(Fuzzy Inference System) 등급과 백분위수 평균을 기준으로 
        총 72개의 데이터 포인트를 선택합니다
    
사용자 결정사항
    Step0의 이상치 판단 기준 : "Initial ACIR(mΩ)","Weight(g)", "Capacity(Ah)","Initial Voltage(V)"
    Step1의 K 범위 : "9~15", 가중치 : "DEFAULT_WEIGHTS"
    Step2의 분석에 사용할 항목들 : ANALYSIS_COLUMNS = ["Initial ACIR(mΩ)","Capacity(Ah)"]