from LotScreening import run_lot_screen
from CellScreeningStep0 import run_step0
from CellScreeningStep1 import run_step1
from CellScreeningStep2 import run_step2_all
from CellScreeingStepM import run_stepM_all

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
    cs2_path = run_step2_all(cs1_path)

    ############################### STEP Modulation ############################
    run_stepM_all(cs2_path)

if __name__ == "__main__":
    main()
