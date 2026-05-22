# CUDA-AMSR-PSO: GPU-Accelerated Adaptive Multi-Swarm PSO for Scalable Feature Selection

CUDA-AMSR-PSO is a research-oriented feature selection framework that combines **Adaptive Multi-Swarm Particle Swarm Optimization (AMSR-PSO)** with **GPU acceleration using Numba CUDA** for efficient optimization in high-dimensional datasets.

The project focuses on improving feature subset quality while maintaining strong classification performance across large-scale datasets such as **HIGGS**, **KDD Cup 1999**, and **UNSW-NB15**. The implementation supports both **GPU execution** and **CPU fallback mode**, making it practical for experimentation on different hardware configurations.

---

# Overview

Feature selection is a critical step in machine learning, especially for high-dimensional datasets where redundant and noisy features negatively impact performance, interpretability, and computational efficiency.

Traditional wrapper-based methods provide high-quality feature subsets but are computationally expensive because they repeatedly retrain classifiers during optimization. CUDA-AMSR-PSO addresses this challenge using:

- GPU-parallel swarm updates with Numba CUDA
- Adaptive inertia weight adjustment
- Multi-swarm exploration
- Dependency-aware fitness evaluation
- Wrapper-based optimization using KNN evaluation

The proposed system accelerates the optimization process while maintaining strong feature selection quality.

---

# Key Features

## GPU-Accelerated Optimization
- CUDA-enabled swarm updates using Numba CUDA
- Parallel particle processing on GPU
- Fused CUDA kernel for velocity + position updates
- CPU fallback mode for non-GPU systems

## Adaptive Multi-Swarm PSO
- Multiple interacting swarms for improved diversity
- Adaptive inertia mechanism based on fitness improvement
- Reduced premature convergence

## Dependency-Aware Feature Selection
- Penalizes redundant feature subsets
- Encourages compact and informative features
- Improves subset stability and interpretability

## Benchmark Dataset Support
The framework supports:
- HIGGS Dataset
- KDD Cup 1999
- UNSW-NB15

## Visualization and Analysis
Automatically generates:
- Convergence plots
- Runtime comparison plots
- Accuracy vs feature reduction tradeoff plots
- Ablation study visualizations

---

# Project Structure

```text
SCL_MINIPROJECT/
│
├── ablation_results/          # Ablation study outputs
│
├── data/                      # Input datasets
│
├── figures/                   # Generated visualizations
│   ├── convergence.png
│   ├── tradeoff.png
│   ├── runtime.png
│   └── ablation.png
│
├── results/                   # Experimental outputs
│   ├── HIGGS_results.csv
│   ├── KDD_Cup_1999_results.csv
│   ├── UNSW-NB15_results.csv
│   └── *_convergence.npy
│
├── src/
│   ├── cuda_amsr_pso.py       # Core CUDA-AMSR-PSO implementation
│   ├── experiment_runner.py   # Runs experiments on datasets
│   ├── plot_results.py        # Generates plots and analysis
│   ├── app.py                 # Main execution / dashboard backend
│   └── quick_test.py          # Lightweight testing script
│
├── README.md
├── .gitignore
├── requirements.txt
└── venv/
```

---

# Methodology

The CUDA-AMSR-PSO framework combines several optimization strategies:

## 1. Binary Particle Representation

Each particle represents a candidate feature subset using a binary vector:

```math
x = [x_1, x_2, ..., x_D], \quad x_d \in \{0,1\}
```

- `1` → feature selected
- `0` → feature excluded

---

## 2. Adaptive Inertia Update

The inertia weight dynamically changes based on optimization progress:

```math
w(t+1)=clip(w(t)+\eta \Delta F_t, w_{min}, w_{max})
```

This helps balance:
- Exploration of new regions
- Exploitation of promising solutions

---

## 3. Multi-Swarm Coordination

The population is divided into multiple sub-swarms:
- Maintains diversity
- Avoids premature convergence
- Improves robustness in high-dimensional search spaces

---

## 4. Dependency-Aware Fitness Function

The optimization objective balances:
- Classification accuracy
- Feature reduction
- Redundancy suppression

```math
F(x)=\alpha Acc(x)+\beta\left(1-\frac{||x||_1}{D}\right)-\gamma R(x)
```

where:
- `Acc(x)` → classification accuracy
- `||x||₁` → number of selected features
- `R(x)` → redundancy penalty

---

# Experimental Setup

## Datasets

| Dataset | Description |
|---|---|
| HIGGS | High-dimensional binary classification dataset |
| KDD Cup 1999 | Intrusion detection benchmark |
| UNSW-NB15 | Modern cybersecurity intrusion dataset |

## Classifier
- K-Nearest Neighbors (KNN)
- Stratified K-Fold Cross Validation

## Evaluation Metrics
- Accuracy
- F1 Score
- Precision
- Recall
- Feature Reduction
- Runtime
- Number of Selected Features

---

# Results Summary

The framework achieves:
- Significant feature reduction
- Competitive classification accuracy
- Better exploration in large search spaces
- Stable convergence behavior

Key observations from experiments:
- HIGGS: ~70% feature reduction
- KDD Cup 1999: ~94% reduction
- UNSW-NB15: ~96% reduction

---

# Generated Visualizations

The project automatically generates:

## Convergence Curves
Shows optimization progress across iterations.

## Accuracy vs Feature Reduction Tradeoff
Visualizes balance between dimensionality reduction and performance.

## Runtime Analysis
Compares computational cost across methods.

## Ablation Studies
Analyzes contribution of:
- Adaptive inertia
- Multi-swarm coordination
- Redundancy penalty

---

# Dataset Notice

The datasets used in this project are not included in the repository due to their large size.

Please download them manually and place them inside the `data/` directory.

## Required Datasets

### 1. HIGGS Dataset

Download from:
https://archive.ics.uci.edu/dataset/280/higgs

Rename and place as:

```text
data/HIGGS.csv
```

---

### 2. UNSW-NB15 Dataset

Download from:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Place as:

```text
data/UNSW_NB15_training-set.csv
```

---

### 3. KDD Cup 1999 Dataset

This dataset is automatically loaded using scikit-learn:

```python
from sklearn.datasets import fetch_kddcup99
```

No manual download required.

---

# How to Run

## 1. Clone the Repository

```bash
git clone <repository-url>
cd SCL_MINIPROJECT
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run Experiments

```bash
python src/experiment_runner.py
```

This will:
- Perform feature selection
- Run experiments on datasets
- Save outputs to `results/`
- Generate plots in `figures/`

---

## 4. Generate Visualizations

```bash
python src/plot_results.py
```

---

## 5. Run Quick Testing

```bash
python src/quick_test.py
```

---

## 6. Launch Application

```bash
streamlit run src/app.py
```

---

# Baseline Methods

The project compares CUDA-AMSR-PSO against:
- Information Gain
- Recursive Feature Elimination (RFE)
- Standard Binary PSO

---

# Advantages of CUDA-AMSR-PSO

- Better feature subset quality
- Reduced premature convergence
- GPU-parallel optimization
- Strong scalability for large datasets
- Reproducible experimentation pipeline
- Modular and extensible implementation

---

# Limitations

Although GPU acceleration improves swarm updates, wrapper-based fitness evaluation still runs on the CPU and remains the primary runtime bottleneck.

Additional limitations include:
- GPU memory constraints on very large datasets
- CPU-GPU transfer overhead
- Computationally expensive cross-validation

---

# Future Work

Potential improvements include:
- GPU-based classifier evaluation
- Multi-GPU optimization
- Surrogate fitness estimation
- Hybrid deep-learning feature selection
- Hyperparameter optimization
- Real-time optimization dashboards

---

# Authors

- Chethana T V

# License

This project is intended for academic and research purposes only.

