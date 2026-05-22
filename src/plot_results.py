"""
plot_results.py
───────────────
Generates all paper figures (including ablation plots).

Run AFTER experiment_runner.py has completed:
    python plot_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)

COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]

RESULTS_DIR = "results"
ABLATION_DIR = "ablation_results"
FIGURES_DIR = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Load CSVs from both folders
# ─────────────────────────────────────────────
def load_all_csvs():
    csv_files = []

    if os.path.exists(RESULTS_DIR):
        csv_files += [
            os.path.join(RESULTS_DIR, f)
            for f in os.listdir(RESULTS_DIR)
            if f.endswith(".csv")
        ]

    if os.path.exists(ABLATION_DIR):
        csv_files += [
            os.path.join(ABLATION_DIR, f)
            for f in os.listdir(ABLATION_DIR)
            if f.endswith(".csv")
        ]

    return csv_files


# ─────────────────────────────────────────────
# Convergence Plot
# ─────────────────────────────────────────────
def plot_convergence():
    convergence_data = {}

    for folder in [RESULTS_DIR, ABLATION_DIR]:
        if not os.path.exists(folder):
            continue

        for f in os.listdir(folder):
            if f.endswith("_convergence.npy"):
                name = f.replace("_convergence.npy", "").replace("_", " ")
                path = os.path.join(folder, f)
                convergence_data[name] = np.load(path)

    if not convergence_data:
        print("No convergence files found.")
        return

    fig, axes = plt.subplots(
        1,
        len(convergence_data),
        figsize=(6 * len(convergence_data), 5),
    )

    if len(convergence_data) == 1:
        axes = [axes]

    for ax, (name, runs) in zip(axes, convergence_data.items()):
        runs_arr = np.array(runs)
        mean_conv = runs_arr.mean(axis=0)
        std_conv = runs_arr.std(axis=0)

        iters = np.arange(1, len(mean_conv) + 1)

        ax.plot(iters, mean_conv, linewidth=2.5)
        ax.fill_between(
            iters,
            mean_conv - std_conv,
            mean_conv + std_conv,
            alpha=0.2,
        )

        ax.set_title(name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "convergence.png")
    plt.savefig(out, dpi=200)
    print("Saved:", out)
    plt.close()


# ─────────────────────────────────────────────
# Tradeoff Plot (All methods)
# ─────────────────────────────────────────────
def plot_tradeoff(csv_files):
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, path in enumerate(csv_files):
        df = pd.read_csv(path)
        filename = os.path.basename(path).replace(".csv", "")
        label_name = filename.replace("_", " ")

        for _, row in df.iterrows():
            try:
                acc = float(str(row["Accuracy (%)"]).split("±")[0])
                red = float(str(row["Reduction (%)"]).split("±")[0])
            except:
                continue

            ax.scatter(
                red,
                acc,
                s=120,
                color=COLORS[i % len(COLORS)],
                label=label_name,
            )

    ax.set_xlabel("Feature Reduction (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Feature Reduction")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "tradeoff.png")
    plt.savefig(out, dpi=200)
    print("Saved:", out)
    plt.close()


# ─────────────────────────────────────────────
# Ablation-only Plot
# ─────────────────────────────────────────────
def plot_ablation(csv_files):
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, path in enumerate(csv_files):
        df = pd.read_csv(path)
        filename = os.path.basename(path).replace(".csv", "")

        for _, row in df.iterrows():
            if row["Method"] != "CUDA-AMSR-PSO":
                continue

            try:
                acc = float(str(row["Accuracy (%)"]).split("±")[0])
                red = float(str(row["Reduction (%)"]).split("±")[0])
            except:
                continue

            label_name = filename.replace("_", " ")

            ax.scatter(
                red,
                acc,
                s=150,
                color=COLORS[i % len(COLORS)],
                label=label_name,
            )

    ax.set_xlabel("Feature Reduction (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation Study: Accuracy vs Reduction")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=9)

    ax.grid(True)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "ablation.png")
    plt.savefig(out, dpi=200)
    print("Saved:", out)
    plt.close()


# ─────────────────────────────────────────────
# Runtime Plot
# ─────────────────────────────────────────────
def plot_runtime(csv_files):
    runtime_data = {}

    for path in csv_files:
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            method = row["Method"]

            try:
                val = float(str(row["Runtime (s)"]).split("±")[0])
            except:
                continue

            runtime_data.setdefault(method, []).append(val)

    runtime_avg = {k: np.mean(v) for k, v in runtime_data.items()}

    methods = list(runtime_avg.keys())
    runtimes = list(runtime_avg.values())

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        methods,
        runtimes,
        color=[COLORS[i % len(COLORS)] for i in range(len(methods))],
    )

    for bar, val in zip(bars, runtimes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.1f}s",
            ha="center",
        )

    ax.set_yscale("log")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime Comparison")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "runtime.png")
    plt.savefig(out, dpi=200)
    print("Saved:", out)
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating all figures...\n")

    csv_files = load_all_csvs()

    if not csv_files:
        print("No CSV files found. Run experiments first.")
    else:
        plot_tradeoff(csv_files)
        plot_ablation(csv_files)

    plot_convergence()
    plot_runtime(csv_files)
