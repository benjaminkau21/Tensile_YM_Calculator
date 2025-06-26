import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- Cached parser ---
@st.cache_data
def parse_sample_tables_from_csv(uploaded_file):
    headers = ["Time", "Displacement", "Force", "Tensile stress", "Tensile strain (Strain 1)"]
    sample_tables = {}
    raw_df = pd.read_csv(uploaded_file, header=None)

    start_idx = raw_df[0][raw_df[0] == "Results Table 2"].index
    if len(start_idx) == 0:
        return {}
    start_row = start_idx[0] + 1

    i = start_row
    sample_index = 1

    while i < len(raw_df):
        row = raw_df.iloc[i, 1:]
        if set(headers).issubset(set(row.values)):
            header_indices = {name: row[row == name].index[0] for name in headers}
            i += 2

            data_rows = []
            while i < len(raw_df):
                next_row = raw_df.iloc[i]
                test_row = next_row[1:]
                if set(headers).issubset(set(test_row.values)) or next_row.isnull().all():
                    break
                values = {key: next_row[idx] for key, idx in header_indices.items()}
                data_rows.append(values)
                i += 1

            if data_rows:
                df_sample = pd.DataFrame(data_rows).apply(pd.to_numeric, errors='coerce')
                sample_tables[f"Sample_{sample_index}"] = df_sample.reset_index(drop=True)
                sample_index += 1
        else:
            i += 1
    return sample_tables

# --- Regression method ---
def compute_regression_modulus(df, strain_range=(2, 5)):
    strain_col = "Tensile strain (Strain 1)"
    stress_col = "Tensile stress"
    df = df[[strain_col, stress_col]].dropna().copy()

    is_percent = df[strain_col].max() > 1
    if is_percent:
        df[strain_col] /= 100.0

    min_strain = strain_range[0] / 100.0
    max_strain = strain_range[1] / 100.0
    df_window = df[(df[strain_col] >= min_strain) & (df[strain_col] <= max_strain)]
    if len(df_window) < 2:
        return None, None, None

    df_grouped = df_window.groupby(strain_col, as_index=False)[stress_col].min()
    x = df_grouped[strain_col].values
    y = df_grouped[stress_col].values

    slope, intercept, r_value, _, _ = linregress(x, y)
    return slope, x, y

# --- Point-to-point method ---
def compute_gradient_modulus(df, strain_range=(2, 5)):
    strain_col = "Tensile strain (Strain 1)"
    stress_col = "Tensile stress"
    df = df[[strain_col, stress_col]].dropna().copy()

    is_percent = df[strain_col].max() > 1
    if is_percent:
        df[strain_col] /= 100.0

    target_strains = [strain_range[0] / 100.0, strain_range[1] / 100.0]
    closest_rows = []

    for target in target_strains:
        idx = (df[strain_col] - target).abs().idxmin()
        closest_rows.append(df.loc[idx])

    if len(closest_rows) < 2:
        return None, None, None

    x0, y0 = closest_rows[0][strain_col], closest_rows[0][stress_col]
    x1, y1 = closest_rows[1][strain_col], closest_rows[1][stress_col]

    if x1 == x0:
        return None, None, None

    slope = (y1 - y0) / (x1 - x0)
    return slope, [x0, x1], [y0, y1]

# --- Streamlit UI ---
st.set_page_config("Tensile Stress Analyzer", layout="centered")
st.title("🔬 Tensile Test Analyzer & Young’s Modulus Calculator")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    sample_tables = parse_sample_tables_from_csv(uploaded_file)

    if not sample_tables:
        st.error("❌ No 'Results Table 2' or valid samples found.")
    else:
        with st.sidebar:
            st.header("⚙️ Settings")
            strain_min = st.slider("Min Range Strain (%)", 0.0, 100.0, 2.0, 0.1)
            strain_max = st.slider("Max Range Strain (%)", 0.0, 100.0, 5.0, 0.1)
            sample_selected = st.selectbox("🧪 Choose Sample", list(sample_tables.keys()))
            method = st.radio("Modulus Calculation Method", ["Linear Regression", "Point-to-Point"])

        # Choose function
        compute_modulus = compute_regression_modulus if method == "Linear Regression" else compute_gradient_modulus

        st.subheader(f"📊 Stress–Strain Curve — {sample_selected}")
        df = sample_tables[sample_selected]
        slope, x_fit, y_fit = compute_modulus(df, (strain_min, strain_max))

        # Plot
        fig, ax = plt.subplots()
        strain_col = "Tensile strain (Strain 1)"
        stress_col = "Tensile stress"
        strain_data = df[strain_col] * (100 if df[strain_col].max() < 1 else 1)
        ax.plot(strain_data, df[stress_col], label="Raw Data", alpha=0.6)

        if slope is not None:
            x_plot = np.array(x_fit) * 100  # convert strain back to %
            ax.plot(x_plot, slope * np.array(x_fit), 'r--', label=f"Fit: E ≈ {slope:.2f} MPa")
            st.success(f"**Estimated Young’s Modulus:** {slope:.2f} MPa")
        else:
            st.warning("⚠️ Not enough valid data points in selected strain range.")

        ax.set_xlabel("Tensile Strain / %")
        ax.set_ylabel("Tensile Stress / MPa")
        ax.legend()
        st.pyplot(fig)

        # Export all moduli
        st.subheader("📤 Export Young’s Moduli for All Samples")
        results = []
        for sample_name, df_sample in sample_tables.items():
            E, _, _ = compute_modulus(df_sample, (strain_min, strain_max))
            results.append({
                "Sample": sample_name,
                "Young’s Modulus (MPa)": round(E, 2) if E is not None else "Not Computed"
            })

        df_moduli = pd.DataFrame(results)
        csv = df_moduli.to_csv(index=False).encode()
        st.download_button("Download Moduli CSV", csv, file_name="youngs_moduli.csv", mime="text/csv")
