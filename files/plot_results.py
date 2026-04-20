"""
plot_results.py
───────────────
Generates all paper figures (Figures 4–7) from saved experiment outputs.

Run AFTER experiment_runner.py has completed:
    python plot_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─── Fig 4: Convergence Curve ─────────────────────────────────────────────────

def plot_convergence(convergence_data: dict, out_path: str = None):
    """
    convergence_data: dict of {dataset_name: list_of_runs}
    Each run is a list of best-fitness values over iterations.
    """
    fig, axes = plt.subplots(1, len(convergence_data), figsize=(7 * len(convergence_data), 5))
    if len(convergence_data) == 1:
        axes = [axes]

    for ax, (name, runs) in zip(axes, convergence_data.items()):
        runs_arr = np.array(runs)  # (n_runs, T)
        mean_conv = runs_arr.mean(axis=0)
        std_conv  = runs_arr.std(axis=0)
        iters = np.arange(1, len(mean_conv) + 1)

        ax.plot(iters, mean_conv, color=COLORS[0], linewidth=2.5, label="CUDA-AMSR-PSO")
        ax.fill_between(iters, mean_conv - std_conv, mean_conv + std_conv,
                        alpha=0.2, color=COLORS[0])


        ax.set_xlabel("Iteration", fontsize=13)
        ax.set_ylabel("Best Fitness", fontsize=13)
        ax.set_title(f"Convergence — {name}", fontsize=14)
        ax.legend()
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = out_path or os.path.join(FIGURES_DIR, "convergence.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ─── Fig 5: Accuracy vs Feature Reduction Trade-off ──────────────────────────

def plot_tradeoff(results_csv_paths: list, out_path: str = None):
    """
    Each CSV has columns: Method, Accuracy (%), Reduction (%)
    (Loaded from experiment_runner output)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"CUDA-AMSR-PSO": "o", "Standard BPSO": "s",
               "Information Gain": "^", "RFE": "D"}

    for path in results_csv_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        dataset_name = os.path.basename(path).replace("_results.csv", "").replace("_", " ")

        for i, (_, row) in enumerate(df.iterrows()):
            method = row["Method"]
            # Parse "mean ± std" strings
            try:
                acc = float(str(row["Accuracy (%)"]).split("±")[0].strip())
                red = float(str(row["Reduction (%)"]).split("±")[0].strip())
            except Exception:
                continue
            marker = markers.get(method, "o")
            color  = COLORS[i % len(COLORS)]
            ax.scatter(red, acc, s=120, marker=marker, color=color,
                       label=f"{method} ({dataset_name})", zorder=3)

    ax.set_xlabel("Feature Reduction (%)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs Feature Reduction Trade-off", fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate labels
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h); l2.append(l); seen.add(l)
    ax.legend(h2, l2, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.4)

    out = out_path or os.path.join(FIGURES_DIR, "tradeoff.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()



# ─── Fig 7: Runtime Comparison ───────────────────────────────────────────────

def plot_runtime(results_csv_paths: list, out_path: str = None):
    """
    Builds runtime comparison directly from experiment CSV outputs.
    Each CSV contains "Method" and "Runtime (s)" as "mean ± std".
    """

    import pandas as pd
    import numpy as np

    runtime_data = {}

    # ── Extract runtime from all CSVs ─────────────────────────────
    for path in results_csv_paths:
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        for _, row in df.iterrows():
            method = row["Method"]

            try:
                # Extract mean from "mean ± std"
                val = float(str(row["Runtime (s)"]).split("±")[0].strip())
            except:
                continue

            if method not in runtime_data:
                runtime_data[method] = []

            runtime_data[method].append(val)

    # ── Average across datasets ──────────────────────────────────
    runtime_avg = {k: np.mean(v) for k, v in runtime_data.items()}

    methods  = list(runtime_avg.keys())
    runtimes = list(runtime_avg.values())

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(methods, runtimes,
                  color=[COLORS[i % len(COLORS)] for i in range(len(methods))],
                  edgecolor="white", width=0.5)

    for bar, val in zip(bars, runtimes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(runtimes)*0.02,
                f"{val:.1f}s", ha="center", fontsize=11)

    ax.set_ylabel("Runtime (seconds)", fontsize=13)
    ax.set_yscale("log")
    ax.set_title("Runtime Comparison Across Methods", fontsize=14)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.4)

    out = out_path or os.path.join(FIGURES_DIR, "runtime.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper figures...\n")

    # ── Fig 4: Convergence ────────────────────────────────────────────────
    conv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_convergence.npy")]
    if conv_files:
        convergence_data = {}
        for f in conv_files:
            name = f.replace("_convergence.npy", "").replace("_", " ")
            convergence_data[name] = np.load(os.path.join(RESULTS_DIR, f))
        plot_convergence(convergence_data)
    else:
        print("  No convergence .npy files found — run experiment_runner.py first.")

    # ── Fig 5: Accuracy vs Reduction Trade-off ────────────────────────────
    csv_files = [os.path.join(RESULTS_DIR, f)
                 for f in os.listdir(RESULTS_DIR) if f.endswith("_results.csv")]
    if csv_files:
        plot_tradeoff(csv_files)
    else:
        print("  No results CSV files found — run experiment_runner.py first.")



    # ── Fig 7: Runtime ────────────────────────────────────────────────────
    # Replace with your measured runtimes from experiment_runner output
    csv_files = [os.path.join(RESULTS_DIR, f)
             for f in os.listdir(RESULTS_DIR)
             if f.endswith("_results.csv")]

    if csv_files:
        plot_runtime(csv_files)
    else:
        print("No results CSV files found.")

    print("\nAll figures saved to ./figures/")
