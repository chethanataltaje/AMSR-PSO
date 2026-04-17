# CUDA-AMSR-PSO: GPU-Accelerated Adaptive Multi-Swarm PSO for Feature Selection

## Overview

This project implements a **CUDA-accelerated Adaptive Multi-Swarm Reinforced Particle Swarm Optimization (AMSR-PSO)** framework for efficient and scalable **feature selection**.

The system leverages **GPU parallelism (Numba CUDA)** to significantly speed up optimization while maintaining high classification performance.

---

## Objectives

* Reduce feature dimensionality without sacrificing accuracy
* Improve scalability using GPU acceleration
* Compare against standard feature selection techniques
* Provide an interactive dashboard for visualization

---

## Key Features

* **CUDA Acceleration** using Numba
* **Adaptive Multi-Swarm PSO**
* **Feature Selection Optimization**
* Benchmarking on multiple datasets:

  * KDD Cup 1999
  * UNSW-NB15
  * HIGGS
* 📈 Performance metrics:

  * Accuracy
  * F1-score
  * Precision / Recall
  * Feature Reduction
  * Runtime
* 🖥️ Interactive dashboards:

  * HTML dashboard (FastAPI backend)
  * Streamlit dashboard

---

## Project Structure

```
AMSR-PSO/
│
├── cuda_amsr_pso.py        # Core PSO implementation (GPU accelerated)
├── experiment_runner.py    # Runs experiments on datasets
├── plot_results.py         # Generates plots (convergence, tradeoff)
├── dashboard.html          # Frontend dashboard
├── app_streamlit.py        # Streamlit dashboard
│
├── results/                # Final results (CSV)
├── figures/                # Generated plots
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

## How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run Experiments

```bash
python experiment_runner.py
```

This will:

* Train models
* Perform feature selection
* Save results in `results/`
* Generate plots in `figures/`

--

```
http://localhost:8501
``

```
index.html
```


## Visualizations

* Convergence curves
* Feature reduction vs accuracy tradeoff
* Method comparison tables

---

## Methodology

The proposed AMSR-PSO introduces:

* Multi-swarm exploration
* Adaptive inertia weight
* Reinforcement-based feature selection
* GPU-parallel fitness evaluation

This enables efficient search in high-dimensional feature spaces.

---

## 🔬 Baseline Comparisons

* Information Gain
* Recursive Feature Elimination (RFE)
* Standard Binary PSO

---

## Key Insights

* Achieves **high dimensionality reduction (~70–95%)**
* Maintains competitive accuracy
* GPU acceleration enables scalability
* Effective on both small and large datasets

---

## Future Work

* Try different classifiers (RF, XGBoost, SVM)
* Improve fitness function
* Add real-time optimization UI
* Extend to deep feature selection

---

## Authors

* Chethana T V


---

## License

This project is for academic and research purposes.

