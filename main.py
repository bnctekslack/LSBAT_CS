from LotScreening import run_lot_screen
from CellScreeningStep0 import run_step0
from CellScreeningStep1 import run_step1
from CellScreeningStep2 import run_step2
from CellScreeingStepM import run_stepM

def main():
    ############################### LOT Screen ##################################
    lot_screen_path, lot_pass = run_lot_screen()
    if not lot_pass:
        print("[Main] Lot screen failed. Check LotScreening.xlsx before proceeding.")
        return

    ############################### STEP0 ##################################
    cs0_path = run_step0()

    ############################### STEP1 ##################################
    # 1. 권장 가중치 사용 (기본)
    #cs1_path, best_cluster, worst_cluster = run_step1(cs0_path)

    # 2. 균등 가중치 사용 (기존 방식)
    #cs1_path, best_cluster, worst_cluster = run_step1(cs0_path, use_equal_weights=True)

    # 3. 권장 가중치 사용
    cs1_path, best_cluster, worst_cluster = run_step1(cs0_path, use_equal_weights=False)

    # 4. 커스텀 가중치 사용
    # custom_weights = {
    #     "Capacity(Ah)": 5.0,        # 용량 최우선
    #     "Initial ACIR(mΩ)": 3.0,    # 저항 중요
    #     "Weight(g)": 0.5,           # 무게 덜 중요
    # }

    # cs1_path, best_cluster, worst_cluster = run_step1(cs0_path, weights=custom_weights)

    ############################### STEP2 ##################################    
    cluster_index = best_cluster if best_cluster is not None else 1
    cs2_path = run_step2(cs1_path, cluster_index=cluster_index, worst_cluster=worst_cluster)

    ############################### STEP Modulation ############################
    run_stepM(cs2_path, cluster_index=cluster_index)

if __name__ == "__main__":
    main()
