import streamlit as st
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CUDA-AMSR-PSO Dashboard",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main text */
html, body, [class*="css"] {
    font-size: 18px;
}

/* Bigger headers */
h1 {
    font-size: 42px !important;
}

h2 {
    font-size: 32px !important;
}

h3 {
    font-size: 26px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1e1e2f;
}

/* Dataframe text */
[data-testid="stDataFrame"] {
    font-size: 18px;
}

/* Metric values */
[data-testid="stMetricValue"] {
    font-size: 28px;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    font-size: 18px;
}

/* Markdown paragraphs */
p {
    font-size: 18px !important;
    line-height: 1.7;
}

/* Lists */
li {
    font-size: 18px !important;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# PATH HANDLING
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results_dir = os.path.join(BASE_DIR, "results")
figures_dir = os.path.join(BASE_DIR, "figures")
ablation_dir = os.path.join(BASE_DIR, "ablation_results")

# =========================================================
# HEADER
# =========================================================

st.title("⚡ CUDA-AMSR-PSO Dashboard")

st.markdown("""
### GPU Accelerated Adaptive Multi-Swarm PSO for Scalable Feature Selection
""")

# =========================================================
# HELPERS
# =========================================================

def safe_image(filename, caption=""):

    path = os.path.join(figures_dir, filename)

    if os.path.exists(path):

        st.image(
            path,
            caption=caption,
            width=750
        )

    else:
        st.error(f"Image not found: {filename}")


def load_results(name):

    formatted = name.replace(" ", "_")
    path = os.path.join(results_dir, f"{formatted}_results.csv")

    if os.path.exists(path):
        return pd.read_csv(path)

    else:
        st.warning(f"File not found: {path}")
        return pd.DataFrame()


def load_ablation_results():

    files = [
        "HIGGS_no_adaptive.csv",
        "HIGGS_no_dependency.csv",
        "HIGGS_no_multiswarm.csv"
    ]

    dfs = []

    for file in files:

        path = os.path.join(ablation_dir, file)

        if os.path.exists(path):

            df = pd.read_csv(path)
            df["Variant"] = file.replace(".csv", "")
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame()

# =========================================================
# SIDEBAR
# =========================================================

page = st.sidebar.selectbox(
    "Go to",
    [
        "Overview",
        "HIGGS",
        "KDD Cup 1999",
        "UNSW-NB15",
        "Ablation Study",
        "Publication Figures"
    ]
)

# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.header("Project Overview")

    st.markdown("""
CUDA-AMSR-PSO is a GPU accelerated feature selection framework
designed for high-dimensional machine learning and big data analytics.

Traditional wrapper-based feature selection methods often provide
high-quality feature subsets but suffer from extremely high
computational cost due to repeated model evaluation.

This project introduces an Adaptive Multi-Swarm Reinforced
Particle Swarm Optimization framework accelerated using
Numba CUDA to improve scalability and optimization efficiency.
""")

    st.header("Core Contributions")

    st.markdown("""
- ⚡ CUDA accelerated swarm updates using Numba CUDA  
- 🧠 Adaptive inertia mechanism based on fitness improvement  
- 🌐 Multi-swarm coordination for better search diversity  
- 📉 Dependency-aware redundancy reduction  
- 📊 Automated convergence and runtime analysis  
- 🔬 Experimental benchmarking on real-world datasets  
""")

    st.header("Supported Datasets")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
### HIGGS Dataset

High-dimensional binary classification dataset used in particle physics experiments.
""")

    with col2:
        st.info("""
### KDD Cup 1999

Intrusion detection benchmark dataset widely used in cybersecurity research.
""")

    with col3:
        st.info("""
### UNSW-NB15

Modern network intrusion detection dataset with realistic attack traffic.
""")

    st.header("Performance Highlights")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Best Accuracy",
            "99.62%",
            "KDD Cup 1999"
        )

    with c2:
        st.metric(
            "Best Reduction",
            "96.03%",
            "UNSW-NB15"
        )

    with c3:
        st.metric(
            "Selected Features",
            "1.67",
            "UNSW-NB15"
        )

    st.header("Methodology")

    st.markdown("""
The framework combines:

- Binary Particle Swarm Optimization
- Adaptive inertia weight adjustment
- Multi-swarm exploration strategy
- Dependency-aware fitness evaluation
- GPU-parallel particle updates

The optimization objective balances:

- Classification accuracy
- Feature reduction
- Redundancy suppression

Wrapper-based evaluation is performed using
K-Nearest Neighbors with stratified cross-validation.
""")

    st.header("Key Observation")

    st.success("""
CUDA-AMSR-PSO achieves strong feature reduction while
maintaining competitive accuracy across large datasets.

The GPU acceleration primarily improves the optimization
search process, while wrapper evaluation remains the
dominant runtime bottleneck.
""")

# =========================================================
# DATASET PAGES
# =========================================================

elif page in ["HIGGS", "KDD Cup 1999", "UNSW-NB15"]:

    st.header(f"{page} Results")

    df = load_results(page)

    if not df.empty:

        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )

        st.header("Quick Metrics")

        cols = st.columns(len(df))

        for i, (_, row) in enumerate(df.iterrows()):

            with cols[i]:

                try:

                    acc = float(
                        str(row["Accuracy (%)"])
                        .split("±")[0]
                        .strip()
                    )

                    red = float(
                        str(row["Reduction (%)"])
                        .split("±")[0]
                        .strip()
                    )

                    st.metric(
                        label=row["Method"],
                        value=f"{acc:.2f}%",
                        delta=f"{red:.1f}% reduction"
                    )

                except:
                    st.metric(row["Method"], "N/A")

    else:
        st.error("No results found for this dataset.")

# =========================================================
# ABLATION STUDY PAGE
# =========================================================

elif page == "Ablation Study":

    st.header("Ablation Study Analysis")

    st.markdown("""
This section evaluates the contribution of different
components in the CUDA-AMSR-PSO framework.

The study analyzes the impact of:

- Adaptive Inertia
- Dependency Penalty
- Multi-Swarm Coordination
""")

    st.header("Ablation Visualization")

    safe_image(
        "ablation.png",
        "Ablation Study on HIGGS Dataset"
    )

    st.header("Ablation Results")

    ablation_df = load_ablation_results()

    if not ablation_df.empty:

        st.dataframe(
            ablation_df,
            width="stretch",
            hide_index=True
        )

    else:
        st.warning(
            "No ablation CSV files found in ablation_results/"
        )

# =========================================================
# PUBLICATION FIGURES
# =========================================================

elif page == "Publication Figures":

    st.header("Publication Figures")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Runtime",
        "Trade-off",
        "Convergence",
        "Ablation"
    ])

    with tab1:

        safe_image(
            "runtime.png",
            "Runtime Comparison Across Methods"
        )

    with tab2:

        safe_image(
            "tradeoff.png",
            "Accuracy vs Feature Reduction Trade-off"
        )

    with tab3:

        safe_image(
            "convergence.png",
            "Convergence Curves"
        )

    with tab4:

        safe_image(
            "ablation.png",
            "Ablation Study on HIGGS Dataset"
        )