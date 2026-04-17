# CUDA-AMSR-PSO — Setup & Run Guide
### BMS College of Engineering | Your Omen Laptop (Windows + VSCode)

---

## Files in this package

| File | Purpose |
|---|---|
| `cuda_amsr_pso.py` | Core algorithm (PSO engine, fitness, inertia, multi-swarm) |
| `experiment_runner.py` | Loads datasets, runs all baselines, produces Tables I–III |
| `plot_results.py` | Generates Figures 4–7 (convergence, tradeoff, ablation, runtime) |
| `quick_test.py` | ✅ Run this FIRST to verify your setup works |

---

## Step 1 — Check your GPU (Omen laptop)

Open a terminal in VSCode and run:
```
nvidia-smi
```
You should see your GPU and **CUDA Version** (top-right corner).
Note the CUDA version (e.g., 12.x or 11.x).

---

## Step 2 — Create a Python virtual environment

```bash
# In VSCode terminal (PowerShell or CMD)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
```

---

## Step 3 — Install dependencies

```bash
pip install numpy scikit-learn pandas matplotlib seaborn scipy tqdm
```

Now install CuPy for your CUDA version:
```bash
# CUDA 12.x  (most common on newer Omen laptops)
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x

# Not sure? Install the generic build (slower to install but auto-detects):
pip install cupy
```

Verify CuPy works:
```bash
python -c "import cupy; print(cupy.array([1,2,3]))"
```

---

## Step 4 — Run the smoke test first

```bash
python quick_test.py
```

Expected output (example):
```
✔ CuPy OK | GPU: NVIDIA GeForce RTX 3060
Dataset: Breast Cancer | D=30 features | N_samples=569
...
✔ Smoke test passed! Ready to run experiment_runner.py
```

If CuPy fails, the code falls back to CPU automatically. You'll still get results,
just slower. Set `USE_GPU = False` at the top of `cuda_amsr_pso.py` to silence the warning.

---

## Step 5 — Download the datasets

### Dataset 1: KDD Cup 1999 (auto-downloaded by sklearn)
No manual download needed. `experiment_runner.py` fetches it automatically via:
```python
from sklearn.datasets import fetch_kddcup99
```

### Dataset 2: UNSW-NB15 (manual download required)
1. Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download `UNSW_NB15_training-set.csv`
3. Create a `data/` folder next to your scripts
4. Put the CSV there: `data/UNSW_NB15_training-set.csv`

### Dataset 3: HIGGS (optional, large file)
1. Go to: https://archive.ics.uci.edu/ml/datasets/HIGGS
2. Download `HIGGS.csv.gz`, unzip it
3. Place at: `data/HIGGS.csv`

> **If datasets are missing**, the code generates synthetic stand-ins so you can
> test the pipeline end-to-end. Replace with real data before final experiments.

---

## Step 6 — Run the full experiment

```bash
python experiment_runner.py
```

This will:
- Print Table II (Performance Comparison) to the terminal
- Save CSVs to `results/`
- Save convergence `.npy` arrays to `results/`

Expected runtime per dataset per run (with GPU):
- ~5–15 minutes (depends on N_iter, N_particles, and dataset size)

---

## Step 7 — Generate figures

```bash
python plot_results.py
```

Figures saved to `figures/`:
- `convergence.png`   → Figure 4 (convergence curve)
- `tradeoff.png`      → Figure 5 (accuracy vs feature reduction)
- `ablation.png`      → Figure 6 (ablation bar chart)
- `runtime.png`       → Figure 7 (runtime comparison)

---

## Tuning the hyperparameters

Edit `AMSRPSOConfig` in `experiment_runner.py`:

```python
cfg = AMSRPSOConfig(
    n_particles = 30,    # More = better coverage, slower
    n_swarms    = 3,     # 3-5 works well
    n_iter      = 100,   # 50 for quick test, 100-200 for publication
    alpha       = 0.9,   # Accuracy weight
    beta        = 0.05,  # Sparsity weight (penalises selecting more features)
    gamma       = 0.05,  # Redundancy penalty
    eta         = 0.05,  # Inertia learning rate (Eq. 6)
    cv_folds    = 3,     # 5 for final publication runs
)
```

For a faster test run:
```python
cfg = AMSRPSOConfig(n_particles=10, n_swarms=2, n_iter=20, cv_folds=2)
```

---

## Ablation study

To fill in Figure 6, run `experiment_runner.py` 4 times with these configs and record accuracy:

| Variant | Change |
|---|---|
| Full CUDA-AMSR-PSO | Default config |
| No Multi-swarm | `n_swarms=1` |
| No Reinforced Inertia | `eta=0.0, w_max=0.7, w_min=0.7` |
| No Redundancy Penalty | `gamma=0.0` |
| Standard BPSO | `n_swarms=1, eta=0.0, gamma=0.0` |

Then paste the accuracy values into `plot_results.py → ablation dict`.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: cupy` | Run `pip install cupy-cuda12x` (match your CUDA version) |
| `cupy` import crashes | Set `USE_GPU = False` in `cuda_amsr_pso.py` |
| Memory error | Reduce `n_particles` or `max_rows` in dataset loader |
| KDD99 download fails | Add `data_home="./data"` to `fetch_kddcup99()` |
| Very slow | Use `cv_folds=2`, `n_iter=30`, `n_particles=10` for testing |

---

## VSCode recommended extensions

- Python (Microsoft)
- Pylance
- Jupyter (for interactive exploration)
- GitLens (for version control)
