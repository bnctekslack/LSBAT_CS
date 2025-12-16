import os
from io import BytesIO
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.drawing.image import Image as XLImage
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained


COLUMNS_TO_ANALYZE = {
    "All item": [
        "Capacity(Ah)",
        "Weight(g)",
        "Height(mm)",
        "Width(mm)",
        "Initial Voltage(V)",
        "100% Voltage(V)",
        "0% Voltage(V)",
        "50% Voltage(V)",
        "Initial ACIR(mΩ)",
        "100% ACIR(mΩ)",
        "0% ACIR(mΩ)",
        "50% ACIR(mΩ)",
    ]
}

STD_COLS = [
    "Capacity(Ah)",
    "Weight(g)",
    "Height(mm)",
    "Width(mm)",
    "Initial Voltage(V)",
    "100% Voltage(V)",
    "0% Voltage(V)",
    "50% Voltage(V)",
    "Initial ACIR(mΩ)",
    "100% ACIR(mΩ)",
    "0% ACIR(mΩ)",
    "50% ACIR(mΩ)",
]

DEFAULT_OUTPUT_DIR = "Results"

# 권장 가중치 설정
DEFAULT_WEIGHTS = {
    # Tier 1: 성능/안전 직결
    "Capacity(Ah)": 3.0,
    "Initial Voltage(V)": 3.0,
    "Initial ACIR(mΩ)": 2.5,
    
    # Tier 2: 운영 특성
    "100% ACIR(mΩ)": 2.0,
    "0% ACIR(mΩ)": 2.0,
    "50% ACIR(mΩ)": 2.0,
    
    "100% Voltage(V)": 1.0,
    "0% Voltage(V)": 1.0,
    "50% Voltage(V)": 1.0,
    
    # Tier 3: 부가 특성
    "Weight(g)": 1.5,
    
    # Tier 4: 물리적 호환성
    "Height(mm)": 0.5,
    "Width(mm)": 0.5,
}

class BalancedKMeans:
    def __init__(self, n_clusters=10, max_iter=100, random_state=42, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None

    def _kmeans_plus_plus_init(self, X):
        n, d = X.shape
        centers = np.empty((self.n_clusters, d), dtype=float)
        idx = self.random_state.randint(0, n)
        centers[0] = X[idx]
        closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for c in range(1, self.n_clusters):
            probs = closest_dist_sq / closest_dist_sq.sum()
            idx = self.random_state.choice(n, p=probs)
            centers[c] = X[idx]
            new_dist_sq = np.sum((X - centers[c]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
        return centers

    def _balanced_assignment(self, X, centers, caps):
        n = X.shape[0]
        k = centers.shape[0]
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        order = np.argmin(dists, axis=1)
        base = dists[np.arange(n), order]
        idx_order = np.argsort(base)
        labels = -np.ones(n, dtype=int)
        caps_left = caps.copy()
        nearest_rank = np.argsort(dists, axis=1)
        for i in idx_order:
            for j in range(k):
                c = nearest_rank[i, j]
                if caps_left[c] > 0:
                    labels[i] = c
                    caps_left[c] -= 1
                    break
        if np.any(labels == -1):
            for i in np.where(labels == -1)[0]:
                c = np.argmax(caps_left)
                labels[i] = c
                caps_left[c] -= 1
        return labels

    def fit_predict(self, X):
        n = X.shape[0]
        k = self.n_clusters
        base = n // k
        r = n % k
        capacities = np.array([base + 1 if i < r else base for i in range(k)], dtype=int)
        centers = self._kmeans_plus_plus_init(X)
        labels_prev = None
        for _ in range(self.max_iter):
            labels = self._balanced_assignment(X, centers, capacities)
            new_centers = np.array([X[labels == c].mean(axis=0) for c in range(k)])
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if labels_prev is not None and np.all(labels == labels_prev):
                break
            if shift < self.tol:
                break
            labels_prev = labels
        self.cluster_centers_ = centers
        self.labels_ = labels
        return labels


def _normalize_scores(scores: List[float], mode: str) -> np.ndarray:
    arr = np.array(scores)
    if arr.size == 0:
        return arr
    span = np.max(arr) - np.min(arr)
    if span == 0:
        return np.ones_like(arr)
    if mode == "dbi":
        return 1 - (arr - np.min(arr)) / span
    return (arr - np.min(arr)) / span


def compute_k_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    results = {}
    for name, cols in COLUMNS_TO_ANALYZE.items():
        if any(col not in df.columns for col in cols):
            continue
        data = df[cols].dropna().values
        if data.size == 0:
            continue

        K_range = range(9, 15) ################## K ####################
        dbi_scores, chi_scores = [], []
        for k in K_range:
            kmeans = KMeansConstrained(
                n_clusters=k,
                size_min=int(data.shape[0] / k * 0.8),
                size_max=int(data.shape[0] / k * 1.2),
                random_state=42,                #의미없는 시작값 0~100
            )
            labels = kmeans.fit_predict(data)
            dbi_scores.append(davies_bouldin_score(data, labels))       #DBI 점수 계산 (낮을수록 좋음 ↓)
            chi_scores.append(calinski_harabasz_score(data, labels))    #CHI 점수 계산 (높을수록 좋음 ↑)

        dbi_norm = _normalize_scores(dbi_scores, "dbi")
        chi_norm = _normalize_scores(chi_scores, "chi")
        combined_score = 0.5 * dbi_norm + 0.5 * chi_norm
        optimal_k_final = K_range[int(np.argmax(combined_score))]
        print(f"[{name}] Optimal K (Combined): {optimal_k_final}")
        results[name] = {
            "k_values": list(K_range),
            "dbi_scores": dbi_scores,
            "chi_scores": chi_scores,
            "optimal_k_dbi": K_range[int(np.argmin(dbi_scores))],
            "optimal_k_chi": K_range[int(np.argmax(chi_scores))],
            "optimal_k_final": optimal_k_final,
        }
    return results


def _add_k_selection_sheet(workbook, k_results: Dict[str, Dict[str, object]]):
    ws = workbook.create_sheet("K_Selection_Result")
    ws.append(["분석 대상", "DBI k", "CHI k", "최종 추천 k"])

    for name, result in k_results.items():
        ws.append(
            [
                name,
                result["optimal_k_dbi"],
                result["optimal_k_chi"],
                result["optimal_k_final"],
            ]
        )
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(result["k_values"], result["dbi_scores"], "o-", label="DBI (↓)")
        axs[0].axvline(result["optimal_k_dbi"], color="r", linestyle="--", label="Optimal DBI")
        axs[0].set_title("Davies-Bouldin Index")
        axs[0].set_xlabel("k")
        axs[0].legend()

        axs[1].plot(result["k_values"], result["chi_scores"], "o-", label="CHI (↑)")
        axs[1].axvline(result["optimal_k_chi"], color="r", linestyle="--", label="Optimal CHI")
        axs[1].set_title("Calinski-Harabasz Index")
        axs[1].set_xlabel("k")
        axs[1].legend()

        fig.suptitle(f"Optimal K Determination for {name}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        img_data = BytesIO()
        plt.savefig(img_data, format="png", bbox_inches="tight")
        plt.close(fig)
        img = XLImage(img_data)
        img.width, img.height = 600, 250
        ws.add_image(img, f"G{ws.max_row}")


def run_step1(
    cs0_file: str, 
    output_dir: str = DEFAULT_OUTPUT_DIR,
    weights: dict = None,  # 새 파라미터
    use_equal_weights: bool = False  # 균등 가중치 옵션
):
    #     
    df = pd.read_excel(cs0_file, sheet_name="Non_Outliers_List")
    k_results = compute_k_metrics(df)
    if "All item" not in k_results:
        raise ValueError("필요한 분석 항목 데이터를 찾을 수 없습니다.")

    k = k_results["All item"]["optimal_k_final"]
    id_col = "Lot Number"
    feature_cols = COLUMNS_TO_ANALYZE["All item"]

    df_use = df[[id_col] + feature_cols].dropna().reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_use[feature_cols])

    bkm = BalancedKMeans(n_clusters=k, random_state=42)
    labels = bkm.fit_predict(X_scaled)
    df_use["cluster"] = labels
    df_out = df.merge(df_use[[id_col, "cluster"]], on=id_col, how="left")

    df_scaled = df_use.copy()
    df_scaled[feature_cols] = X_scaled
    df_std = df_scaled.copy()
    df_std["cluster"] = labels

    df_counts = df_use["cluster"].value_counts().sort_index().reset_index()
    df_counts.columns = ["cluster", "count"]

    # 표준편차 계산
    std_cols_use = [c for c in STD_COLS if c in df_out.columns]
    df_cluster_std = df_out.groupby("cluster")[std_cols_use].std().reset_index()
    
    # 순위 변환
    df_rank = df_cluster_std.copy()
    for col in std_cols_use:
        df_rank[col] = df_cluster_std[col].rank(method="min", ascending=True)
    
    # ========== 가중치 적용 부분 ==========
    if use_equal_weights:
        # 균등 가중치 (기존 방식)
        print("⚖️  균등 가중치 사용 (모든 항목 1.0)")
        weights_to_use = {col: 1.0 for col in std_cols_use}
    else:
        # 가중치 적용
        if weights is None:
            weights_to_use = DEFAULT_WEIGHTS
            print("🎯 권장 가중치 적용:")
        else:
            weights_to_use = weights
            print("🎯 사용자 정의 가중치 적용:")
        
        # 가중치 출력
        for col in std_cols_use:
            w = weights_to_use.get(col, 1.0)
            print(f"   {col:25s}: {w:.1f}")
    
    # 가중치 적용 순위 합산
    df_rank["total_rank"] = sum(
        df_rank[col] * weights_to_use.get(col, 1.0)
        for col in std_cols_use
    )
    # =====================================
    
    # 최고 / 최악 클러스터 선정
    best_cluster = int(df_rank.loc[df_rank["total_rank"].idxmin(), "cluster"]) if len(df_rank) else None
    worst_cluster = int(df_rank.loc[df_rank["total_rank"].idxmax(), "cluster"]) if len(df_rank) else None
    print(f"🌟 가장 안정적인 클러스터: {best_cluster}")
    print(f"⚠️ 가장 변동성이 큰 클러스터: {worst_cluster}")
    
    # 가중치 정보 추가 저장
    df_weights = pd.DataFrame([
        {"항목": col, "가중치": weights_to_use.get(col, 1.0)}
        for col in std_cols_use
    ])
    
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "Step1_Results.xlsx")
    cs1_path = base_path
    if os.path.exists(base_path):
        try:
            os.remove(base_path)
        except PermissionError:
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            cs1_path = os.path.join(output_dir, f"Step1_Results_{ts}.xlsx")
            print(f"[WARN] {base_path} 파일을 덮어쓸 수 없어 새 파일로 저장합니다: {cs1_path}")

    with pd.ExcelWriter(cs1_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Original_Data", index=False)
        df_std.to_excel(writer, sheet_name="Clustered_StandardScaler", index=False)
        df_counts.to_excel(writer, sheet_name="Cluster_Counts", index=False)
        
        for c in sorted(df_out["cluster"].dropna().unique()):
            cluster_members = df_out[df_out["cluster"] == c][["Lot Number"] + feature_cols]
            cluster_members.to_excel(writer, sheet_name=f"Cluster{int(c)}", index=False)
        
        df_cluster_std.to_excel(writer, sheet_name="Cluster_STD", index=False)
        df_rank.to_excel(writer, sheet_name="Cluster_STD_Rank", index=False)
        df_weights.to_excel(writer, sheet_name="Applied_Weights", index=False)  # 새 시트
        _add_k_selection_sheet(writer.book, k_results)
    
    print("✅ 저장 완료:", cs1_path)
    return cs1_path, best_cluster, worst_cluster


__all__ = ["run_step1"]
