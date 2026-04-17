"""
experiment_runner.py - Fixed Version for Numba CUDA AMSR-PSO
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone

from cuda_amsr_pso import (
    CUDA_AMSR_PSO, 
    AMSRPSOConfig, 
    compute_dependency_matrix
)


# ─── DATASET LOADERS ─────────────────────────────────────────────────────────
print("RUNNING THIS EXPERIMENT RUNNER FILE")
def load_kdd99(max_rows=20000):
    from sklearn.datasets import fetch_kddcup99
    print("[Dataset] Loading KDD Cup 99...")
    data = fetch_kddcup99(percent10=True)
    X = np.array(data.data)
    y = np.array(data.target)

    for i in range(X.shape[1]):
        try:
            X[:, i] = X[:, i].astype(float)
        except:
            X[:, i] = LabelEncoder().fit_transform(X[:, i].astype(str))

    X = X.astype(np.float32)
    y = LabelEncoder().fit_transform(y.astype(str))

    if len(X) > max_rows:
        idx = np.random.choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]

    print(f"KDD99: X={X.shape}, classes={len(np.unique(y))}")
    return X, y, "KDD Cup 1999"


def load_unsw_nb15(path: str = "data/UNSW_NB15_training-set.csv", max_rows: int = 20000):
    if not os.path.exists(path):
        print(f"[Dataset] UNSW-NB15 not found. Using synthetic data.")
        np.random.seed(42)
        X = np.random.randn(10000, 49).astype(np.float32)
        y = np.random.randint(0, 10, 10000)
        return X, y, "UNSW-NB15 (synthetic)"

    print(f"[Dataset] Loading UNSW-NB15...")
    df = pd.read_csv(path, nrows=max_rows)
    drop_cols = [c for c in ["id", "attack_cat"] if c in df.columns]
    label_col = "label" if "label" in df.columns else df.columns[-1]
    y = LabelEncoder().fit_transform(df[label_col].values.astype(str))
    df = df.drop(columns=drop_cols + [label_col], errors="ignore")
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    X = df.values.astype(np.float32)
    print(f"UNSW-NB15: X={X.shape}, classes={len(np.unique(y))}")
    return X, y, "UNSW-NB15"


def load_higgs(path: str = "data/HIGGS.csv", max_rows: int = 30000):
    if not os.path.exists(path):
        print(f"[Dataset] HIGGS not found. Using synthetic data.")
        np.random.seed(99)
        X = np.random.randn(10000, 28).astype(np.float32)
        y = np.random.randint(0, 2, 10000)
        return X, y, "HIGGS (synthetic)"

    print(f"[Dataset] Loading HIGGS...")
    df = pd.read_csv(path, nrows=max_rows, header=None)
    y = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values.astype(np.float32)
    print(f"HIGGS: X={X.shape}, classes={len(np.unique(y))}")
    return X, y, "HIGGS"


# ─── BASELINES ───────────────────────────────────────────────────────────────

def baseline_info_gain(X_tr, y_tr, X_te, y_te, clf, k_ratio=0.5):
    k = max(1, int(X_tr.shape[1] * k_ratio))
    selector = SelectKBest(mutual_info_classif, k=k)
    X_tr_s = selector.fit_transform(X_tr, y_tr)
    X_te_s = selector.transform(X_te)
    clf2 = clone(clf)
    clf2.fit(X_tr_s, y_tr)
    preds = clf2.predict(X_te_s)
    return preds, k


def baseline_rfe(X_tr, y_tr, X_te, y_te, clf, k_ratio=0.5):
    k = max(1, int(X_tr.shape[1] * k_ratio))
    base = LogisticRegression(max_iter=200, random_state=42, solver="saga")
    rfe = RFE(base, n_features_to_select=k, step=0.1)
    X_tr_s = rfe.fit_transform(X_tr, y_tr)
    X_te_s = rfe.transform(X_te)
    clf2 = clone(clf)
    clf2.fit(X_tr_s, y_tr)
    preds = clf2.predict(X_te_s)
    return preds, k


def baseline_pso_standard(X_tr, y_tr, X_te, y_te, clf):
    cfg_std = AMSRPSOConfig(n_particles=20, n_swarms=1, n_iter=50,
                            w_max=0.7, w_min=0.7, eta=0.0,
                            alpha=0.9, beta=0.05, gamma=0.0,
                            exchange_interval=9999)
    pso = CUDA_AMSR_PSO(config=cfg_std, classifier=clone(clf))
    pso.fit(X_tr, y_tr)
    X_tr_s = pso.transform(X_tr)
    X_te_s = pso.transform(X_te)
    clf2 = clone(clf)
    clf2.fit(X_tr_s, y_tr)
    preds = clf2.predict(X_te_s)
    return preds, len(pso.best_features_)


def clone_clf(clf):
    return clone(clf)


# ─── METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, n_selected, n_total, runtime, train_acc=None):
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    metrics = {
        "Accuracy (%)": round(accuracy_score(y_true, y_pred) * 100, 2),
        "F1-score": round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "Reduction (%)": round((1 - n_selected / n_total) * 100, 2),
        "Runtime (s)": round(runtime, 2),
        "N_features": n_selected,
    }
    if train_acc is not None:
        metrics["Train_Acc (%)"] = round(train_acc * 100, 2)
        test_acc = accuracy_score(y_true, y_pred)
        metrics["Gap (%)"] = round((train_acc - test_acc) * 100, 2)
    return metrics


# ─── FULL EXPERIMENT ─────────────────────────────────────────────────────────
def run_experiment(X, y, dataset_name: str, n_runs: int = 3, output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'#'*75}")
    print(f"  EXPERIMENT: {dataset_name} (Numba CUDA Version)")
    print(f"{'#'*75}")

    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    all_rows = []
    cuda_convergences = []

    for run in range(1, n_runs + 1):
        print(f"\n─── Run {run}/{n_runs} ───────────────────────────────────────")

        # Safe stratified split
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=run*7, stratify=y
            )
        except ValueError:
            print("[WARN] Stratified split failed, using random split")
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=run*7
            )

        # Scale only on train
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        D = X_tr.shape[1]
        row = {"Run": run}

        # Baselines
        t0 = time.time()
        preds, k = baseline_info_gain(X_tr, y_tr, X_te, y_te, clf)
        row["Information Gain"] = compute_metrics(y_te, preds, k, D, time.time()-t0)

        t0 = time.time()
        try:
            preds, k = baseline_rfe(X_tr, y_tr, X_te, y_te, clf)
            row["RFE"] = compute_metrics(y_te, preds, k, D, time.time()-t0)
        except Exception as ex:
            print(f"  RFE failed: {ex}")
            row["RFE"] = None

        t0 = time.time()
        preds, k = baseline_pso_standard(X_tr, y_tr, X_te, y_te, clf)
        row["Standard BPSO"] = compute_metrics(y_te, preds, k, D, time.time()-t0)

        # ── CUDA-AMSR-PSO with Numba CUDA ─────────────────────────────────
        t0 = time.time()

        cfg = AMSRPSOConfig()

        # Force settings — especially for HIGGS
        if "HIGGS" in dataset_name.upper():
            cfg.n_particles = 30
            cfg.n_swarms    = 3
            cfg.n_iter      = 100
            cfg.cv_folds    = 8
            print("DEBUG: [HIGGS] Forcing ENHANCED settings → particles=30, iter=100, cv_folds=8")
        else:
            cfg.n_particles = 30
            cfg.n_swarms    = 3
            cfg.n_iter      = 80
            cfg.cv_folds    = 5

        print(f"DEBUG: Final config → N={cfg.n_particles}, T={cfg.n_iter}, CV={cfg.cv_folds}")

        pso = CUDA_AMSR_PSO(config=cfg, classifier=clone_clf(clf))
        dep = compute_dependency_matrix(X_tr)
        pso.fit(X_tr, y_tr, dep_matrix=dep)
        runtime = time.time() - t0

        X_tr_s = pso.transform(X_tr)
        X_te_s = pso.transform(X_te)
        final_clf = clone_clf(clf)
        final_clf.fit(X_tr_s, y_tr)

        preds = final_clf.predict(X_te_s)
        train_preds = final_clf.predict(X_tr_s)
        train_acc = accuracy_score(y_tr, train_preds)

        row["CUDA-AMSR-PSO"] = compute_metrics(
            y_te, preds, len(pso.best_features_), D, runtime, train_acc=train_acc
        )

        cuda_convergences.append(pso.convergence_)
        all_rows.append(row)

    # Aggregate
    methods = ["Information Gain", "RFE", "Standard BPSO", "CUDA-AMSR-PSO"]
    table_rows = []

    for method in methods:
        values = [r[method] for r in all_rows if r.get(method) is not None]
        if not values:
            continue
        mean_row = {"Method": method}
        metric_keys = list(values[0].keys())
        for metric in metric_keys:
            vals = [v[metric] for v in values if metric in v]
            if vals:
                mean_row[metric] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
        table_rows.append(mean_row)

    df_table = pd.DataFrame(table_rows)
    print(f"\n{'='*75}")
    print(f"  TABLE II — Performance Comparison: {dataset_name}")
    print(f"{'='*75}")
    print(df_table.to_string(index=False))

    csv_path = os.path.join(output_dir, f"{dataset_name.replace(' ','_')}_results.csv")
    df_table.to_csv(csv_path, index=False)

    conv_path = os.path.join(output_dir, f"{dataset_name.replace(' ','_')}_convergence.npy")
    np.save(conv_path, np.array(cuda_convergences))

    print(f"\n  Results saved to {output_dir}/")
    return df_table, cuda_convergences


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("CUDA-AMSR-PSO Experimental Runner (Numba CUDA Version)")
    print("=" * 70 + "\n")

    datasets = []
    #X_kdd, y_kdd, name_kdd = load_kdd99(max_rows=20000)
    #X_unsw, y_unsw, name_unsw = load_unsw_nb15(max_rows=20000)
    X_higgs, y_higgs, name_higgs = load_higgs(max_rows=30000)

    #datasets.append((X_kdd, y_kdd, name_kdd))
    #datasets.append((X_unsw, y_unsw, name_unsw))
    datasets.append((X_higgs, y_higgs, name_higgs))

    for X, y, name in datasets:
        print(f"\n=== Starting {name} ===")
        run_experiment(X, y, name, n_runs=3)

    print("\n\n All experiments completed successfully!")
    
