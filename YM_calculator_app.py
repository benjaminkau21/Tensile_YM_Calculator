import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- Parse Sample Tables from CSV ---
def parse_sample_tables_from_csv(file):
    headers = ["Time", "Displacement", "Force", "Tensile stress", "Tensile strain (Strain 1)"]
    sample_tables = {}
    raw_df = pd.read_csv(file, header=None)

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

# --- Compute Young's Modulus using linear regression ---
def compute_regression_modulus(df, strain_range=(0.02, 0.05)):
    strain_col = "Tensile strain (Strain 1)"
    stress_col = "Tensile stress"

    df = df[[strain_col, stress_col]].dropna().copy()
    if df[strain_col].max() > 1:
        df[strain_col] = df[strain_col] / 100.0

    df_window = df[(df[strain_col] >= strain_range[0]) & (df[strain_col] <= strain_range[1])]
    if len(df_window) < 2:
        return None, None, None

    df_grouped = df_window.groupby(strain_col, as_index=False)[stress_col].min()
    x = df_grouped[strain_col].values
    y = df_grouped[stress_col].values

    slope, intercept, r_value, _, _ = linregress(x, y)
    return slope, x, y

# --- Streamlit UI ---
st.set_page_config(page_title="Tensile Stress-Strain Analyzer", layout="centered")

st.title("🔬 Tensile Test Analyzer & Young’s Modulus Calculator")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    sample_tables = parse_sample_tables_from_csv(uploaded_file)

    if not sample_tables:
        st.error("❌ No 'Results Table 2' found or no valid samples detected.")
    else:
        st.success(f"✅ Loaded {len(sample_tables)} samples")

        strain_min = st.slider("📉 Minimum Strain (%)", 0.0, 10.0, 2.0, 0.1)
        strain_max = st.slider("📈 Maximum Strain (%)", 0.0, 10.0, 5.0, 0.1)

        sample_selected = st.selectbox("🧪 Choose a Sample", list(sample_tables.keys()))

        df = sample_tables[sample_selected]
        slope, x_fit, y_fit = compute_regression_modulus(df, (strain_min / 100, strain_max / 100))

        st.subheader("📊 Stress–Strain Curve")
        fig, ax = plt.subplots()
        ax.plot(df["Tensile strain (Strain 1)"], df["Tensile stress"], label="Raw Data", alpha=0.6)

        if slope is not None:
            ax.plot(x_fit, slope * x_fit, 'r--', label=f"Fit: E ≈ {slope:.2f}")
            st.markdown(f"**Estimated Young’s Modulus:** {slope:.2f} (units same as stress)")
        else:
            st.warning("⚠️ Not enough valid data points in the selected strain range.")

        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress")
        ax.legend()
        st.pyplot(fig)
