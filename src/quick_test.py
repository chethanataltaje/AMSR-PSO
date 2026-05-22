"""
quick_test.py
─────────────
Run this FIRST to verify your installation is correct.
Uses a tiny synthetic dataset — completes in ~30 seconds.

    python quick_test.py
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

print("=" * 55)
print("  CUDA-AMSR-PSO Quick Smoke Test")
print("=" * 55)

# ── Check GPU ────────────────────────────────────────────────
try:
    import cupy as cp
    arr = cp.array([1.0, 2.0, 3.0])
    print(f"✔ CuPy OK | GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except Exception as e:
    print(f"✘ CuPy not available ({e}) — will run on CPU")

# ── Load a real small dataset ─────────────────────────────────
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n Dataset: Breast Cancer | D={X.shape[1]} features | N_samples={len(X)}")

# ── Import and run ────────────────────────────────────────────
from cuda_amsr_pso import CUDA_AMSR_PSO, AMSRPSOConfig, compute_dependency_matrix

cfg = AMSRPSOConfig(
    n_particles=10,
    n_swarms=2,
    n_iter=20,        # small for quick test
    cv_folds=3,
    alpha=0.9, beta=0.05, gamma=0.05
)

clf     = KNeighborsClassifier(n_neighbors=5)
dep_mat = compute_dependency_matrix(X_tr)
selector = CUDA_AMSR_PSO(config=cfg, classifier=clf)
selector.fit(X_tr, y_tr, dep_matrix=dep_mat)

# ── Evaluate ──────────────────────────────────────────────────
from sklearn.base import clone
X_tr_s = selector.transform(X_tr)
X_te_s = selector.transform(X_te)

final_clf = clone(clf)
final_clf.fit(X_tr_s, y_tr)
acc_selected = accuracy_score(y_te, final_clf.predict(X_te_s))

baseline = clone(clf)
baseline.fit(X_tr, y_tr)
acc_all = accuracy_score(y_te, baseline.predict(X_te))

print(f"\n{'─'*45}")
print(f"  All features   : {X.shape[1]} → Acc = {acc_all*100:.2f}%")
print(f"  AMSR-PSO subset: {len(selector.best_features_)} → Acc = {acc_selected*100:.2f}%")
print(f"  Feature indices: {selector.best_features_}")
print(f"  Best fitness   : {selector.best_fitness_:.4f}")
print(f"{'─'*45}")
print("\n✔ Smoke test passed! Ready to run experiment_runner.py\n")
