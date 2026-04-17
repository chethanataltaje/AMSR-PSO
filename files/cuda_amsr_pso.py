"""
CUDA-AMSR-PSO with Numba CUDA Kernel
Real CUDA programming for fused velocity + position update
"""

import numpy as np
import time
import warnings
import math
from typing import Tuple, List
from dataclasses import dataclass

# ─── GPU SETUP ───────────────────────────────────────────────────────────────
USE_GPU = True

try:
    from numba import cuda
    import numba
    print(f"[INFO] Numba CUDA initialized successfully (v{numba.__version__})")
except ImportError:
    print("[ERROR] Numba not installed. Run: conda install numba -c conda-forge")
    USE_GPU = False

try:
    import cupy as cp
    xp = cp
    print("[INFO] CuPy available for array handling.")
except ImportError:
    import numpy as cp
    xp = np
    print("[INFO] CuPy not found → using NumPy fallback.")

from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")


# ─── CONFIGURATION ───────────────────────────────────────────────────────────
@dataclass
class AMSRPSOConfig:
    n_particles: int = 30
    n_swarms: int = 3
    n_iter: int = 80
    c1: float = 2.0
    c2: float = 2.0
    w_max: float = 0.9
    w_min: float = 0.4
    eta: float = 0.05
    alpha: float = 0.9
    beta: float = 0.05
    gamma: float = 0.05
    exchange_interval: int = 10
    cv_folds: int = 5
    random_state: int = 42


# ─── NUMBA CUDA KERNEL ───────────────────────────────────────────────────────
@cuda.jit
def fused_velocity_position_kernel(V, X, Pbest, Gbest, w, c1, c2, X_new, V_new, N, D):
    idx = cuda.grid(1)
    if idx >= N:
        return
    for j in range(D):
        seed = (idx + 1) * (j + 1)
        r1 = ((seed * 17 + 13) % 100) / 100.0
        r2 = ((seed * 31 + 7) % 100) / 100.0

        v = (w * V[idx, j] +
             c1 * r1 * (Pbest[idx, j] - X[idx, j]) +
             c2 * r2 * (Gbest[j] - X[idx, j]))

        v = max(-6.0, min(6.0, v))
        V_new[idx, j] = v

        prob = 1.0 / (1.0 + math.exp(-v))
        u = ((seed * 97 + 19) % 100) / 100.0
        X_new[idx, j] = 1 if prob > u else 0


def fused_velocity_position_update(V, X, Pbest, Gbest, w: float, c1: float, c2: float):
    if not USE_GPU:
        N, D = V.shape
        r1 = np.random.rand(N, D).astype(np.float32)
        r2 = np.random.rand(N, D).astype(np.float32)
        u = np.random.rand(N, D).astype(np.float32)
        V_new = np.clip(w * V + c1 * r1 * (Pbest - X) + c2 * r2 * (Gbest - X), -6.0, 6.0)
        prob = 1.0 / (1.0 + np.exp(-V_new))
        X_new = (prob > u).astype(np.int8)
        return V_new, X_new

    N, D = V.shape

    V_d = cuda.to_device(V.astype(np.float32))
    X_d = cuda.to_device(X.astype(np.int32))
    Pbest_d = cuda.to_device(Pbest.astype(np.int32))
    Gbest_d = cuda.to_device(Gbest.astype(np.int32))

    V_new_d = cuda.device_array((N, D), dtype=np.float32)
    X_new_d = cuda.device_array((N, D), dtype=np.int32)

    threads = 256
    blocks = (N + threads - 1) // threads

    fused_velocity_position_kernel[blocks, threads](
        V_d, X_d, Pbest_d, Gbest_d, w, c1, c2, X_new_d, V_new_d, N, D
    )

    return V_new_d.copy_to_host(), X_new_d.copy_to_host()


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def compute_dependency_matrix(X: np.ndarray, method: str = "correlation") -> np.ndarray:
    D = X.shape[1]
    dep = np.zeros((D, D), dtype=np.float32)
    if method == "correlation":
        corr = np.corrcoef(X.T)
        dep = np.abs(corr).astype(np.float32)
        dep = np.nan_to_num(dep)
        np.fill_diagonal(dep, 0.0)
    return dep


def compute_redundancy(position: np.ndarray, dep_matrix: np.ndarray) -> float:
    selected = np.where(position == 1)[0]
    s = len(selected)
    if s < 2:
        return 0.0
    subset_dep = dep_matrix[np.ix_(selected, selected)]
    dep_sum = float(np.sum(np.triu(subset_dep, k=1)))
    return dep_sum / (s * (s - 1) + 1e-8)


def evaluate_fitness_single(position, X_train, y_train, classifier, dep_matrix, alpha, beta, gamma, cv_folds=5):
    D = len(position)
    selected = np.where(position == 1)[0]
    if len(selected) == 0:
        return 0.0

    X_sub = X_train[:, selected]
    try:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        accs = []
        for tr_idx, val_idx in skf.split(X_sub, y_train):
            clf = clone(classifier)
            clf.fit(X_sub[tr_idx], y_train[tr_idx])
            preds = clf.predict(X_sub[val_idx])
            accs.append(accuracy_score(y_train[val_idx], preds))
        acc = float(np.mean(accs))
    except Exception:
        acc = 0.0

    sparsity = 1.0 - (len(selected) / D)
    redundancy = compute_redundancy(position, dep_matrix)
    fitness = alpha * acc + beta * sparsity - gamma * redundancy
    return float(fitness) if not (np.isnan(fitness) or np.isinf(fitness)) else 0.0


def evaluate_population_fitness(X_pop, X_train, y_train, classifier, dep_matrix, config):
    N = X_pop.shape[0]
    X_pop_cpu = cp.asnumpy(X_pop) if USE_GPU and isinstance(X_pop, cp.ndarray) else np.asarray(X_pop)

    fitness = np.zeros(N, dtype=np.float32)
    for i in range(N):
        fitness[i] = evaluate_fitness_single(
            X_pop_cpu[i], X_train, y_train, classifier, dep_matrix,
            config.alpha, config.beta, config.gamma, config.cv_folds
        )
    return fitness


def reinforced_inertia_update(w, fitness_new, fitness_old, eta, w_min, w_max):
    if np.isnan(fitness_new) or np.isnan(fitness_old):
        return w
    delta = fitness_new - fitness_old
    return float(np.clip(w + eta * delta, w_min, w_max))


# ─── MAIN CLASS ──────────────────────────────────────────────────────────────
class CUDA_AMSR_PSO:
    def __init__(self, config=None, classifier=None):
        if config is None:
            self.config = AMSRPSOConfig()
        else:
            self.config = config
        self.config.n_swarms = max(3, self.config.n_swarms)
        self.classifier = classifier or KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

        self.best_position_ = None
        self.best_fitness_ = -np.inf
        self.best_features_ = None
        self.convergence_ = []
        self.inertia_history_ = []

    def _init_swarm(self, D: int, N: int):
        X = (xp.random.random((N, D)) > 0.5).astype(xp.int8)
        V = xp.random.uniform(-1.0, 1.0, (N, D)).astype(xp.float32)
        return X, V

    def _partition_swarms(self, N: int, M: int) -> List[List[int]]:
        indices = list(range(N))
        size = N // M
        swarms = []
        for m in range(M):
            start = m * size
            end = N if m == M - 1 else start + size
            swarms.append(indices[start:end])
        return swarms

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, dep_matrix=None):
        cfg = self.config
        N, M, T, D = cfg.n_particles, cfg.n_swarms, cfg.n_iter, X_train.shape[1]

        print(f"\n{'='*75}")
        print(f"  CUDA-AMSR-PSO (Numba CUDA) | Features={D} | Particles={N} | Iters={T}")
        print(f"  Device: {'GPU (Numba CUDA Kernel)' if USE_GPU else 'CPU Fallback'}")
        print(f"{'='*75}\n")

        if dep_matrix is None:
            dep_matrix = compute_dependency_matrix(X_train)

        X, V = self._init_swarm(D, N)
        Pbest = X.copy()
        Pbest_fit = -np.ones(N, dtype=np.float32) * np.inf

        fit_arr = evaluate_population_fitness(X, X_train, y_train, self.classifier, dep_matrix, cfg)
        Pbest_fit[:] = fit_arr

        gbest_idx = int(np.argmax(fit_arr))
        Gbest = xp.array(cp.asnumpy(X[gbest_idx]) if USE_GPU else X[gbest_idx], dtype=xp.int8)
        gbest_fit = float(fit_arr[gbest_idx])
        prev_gbest_fit = gbest_fit

        swarm_indices = self._partition_swarms(N, M)
        w = cfg.w_max

        self.convergence_ = []
        self.inertia_history_ = []

        for t in range(1, T + 1):
            for swarm_list in swarm_indices:
                idx = xp.array(swarm_list, dtype=int)
                X_sub = X[idx]
                V_sub = V[idx]
                Pbest_sub = Pbest[idx]

                V_new, X_new = fused_velocity_position_update(
                    V_sub, X_sub, Pbest_sub, Gbest, w, cfg.c1, cfg.c2
                )

                V[idx] = V_new
                X[idx] = X_new

            fit_arr = evaluate_population_fitness(X, X_train, y_train, self.classifier, dep_matrix, cfg)

            improved = fit_arr > Pbest_fit
            for i in np.where(improved)[0]:
                Pbest[i] = X[i].copy()
                Pbest_fit[i] = fit_arr[i]

            best_idx = int(np.argmax(fit_arr))
            if fit_arr[best_idx] > gbest_fit:
                gbest_fit = float(fit_arr[best_idx])
                Gbest = xp.array(cp.asnumpy(X[best_idx]) if USE_GPU else X[best_idx], dtype=xp.int8)

            w = reinforced_inertia_update(w, gbest_fit, prev_gbest_fit, cfg.eta, cfg.w_min, cfg.w_max)
            prev_gbest_fit = gbest_fit

            if t % cfg.exchange_interval == 0:
                for swarm_list in swarm_indices:
                    sub_fits = fit_arr[swarm_list]
                    inject_idx = swarm_list[int(np.argmin(sub_fits))]
                    X[inject_idx] = Gbest.copy()
                    Pbest[inject_idx] = Gbest.copy()

            self.convergence_.append(gbest_fit)
            self.inertia_history_.append(w)

            if t % 10 == 0 or t == 1:
                n_selected = int(xp.sum(Gbest))
                print(f"  Iter {t:3d}/{T} | BestFit={gbest_fit:.4f} | "
                      f"Selected={n_selected}/{D} | w={w:.4f}")

        gbest_cpu = cp.asnumpy(Gbest) if USE_GPU else np.asarray(Gbest)
        self.best_position_ = gbest_cpu
        self.best_fitness_ = gbest_fit
        self.best_features_ = list(np.where(gbest_cpu == 1)[0])

        print(f"\n✓ Optimization finished. Best fitness: {gbest_fit:.4f} | "
              f"Selected features: {len(self.best_features_)}")
        return self

    def transform(self, X):
        if self.best_features_ is None:
            raise RuntimeError("Call fit() first.")
        return X[:, self.best_features_]

    def fit_transform(self, X_train, y_train, dep_matrix=None):
        self.fit(X_train, y_train, dep_matrix)
        return self.transform(X_train)