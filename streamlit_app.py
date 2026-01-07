import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# --- Utility Functions ---

@st.cache_data
def parse_sample_tables_from_csv(file):
    headers = ["Time", "Displacement", "Force", "Tensile stress", "Tensile strain (Strain 1)"]
    sample_tables = {}

    raw_df = pd.read_csv(file, header=None)

    # Locate "Results Table 2"
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
            i += 2  # skip header + units

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
                df_sample = pd.DataFrame(data_rows)
                df_sample = df_sample.apply(pd.to_numeric, errors='coerce')
                sample_tables[f"Sample_{sample_index}"] = df_sample.reset_index(drop=True)
                sample_index += 1
        else:
            i += 1

    return sample_tables


def compute_regression_modulus(df, strain_range=(0.02, 0.05)):
    strain_col = "Tensile strain (Strain 1)"
    stress_col = "Tensile stress"

    df = df[[strain_col, stress_col]].dropna().copy()

    # Auto-convert percent to decimal if needed
    if df[strain_col].max() > 1:
        df[strain_col] /= 100.0

    df_window = df[(df[strain_col] >= strain_range[0]) & (df[strain_col] <= strain_range[1])]

    if len(df_window) < 2:
        return None, None, None

    # Optional: minimize over repeated strain values
    df_grouped = df_window.groupby(strain_col, as_index=False)[stress_col].min()

    x = df_grouped[strain_col].values
    y = df_grouped[stress_col].values

    slope, intercept, r_value, _, _ = linregress(x, y)
    return slope, x, y

# --- Streamlit App UI ---

st.title("Young's Modulus Calculator (Tensile Test)")

uploaded_file = st.file_uploader("Upload tensile CSV file", type=["csv"])

if uploaded_file:
    samples = parse_sample_tables_from_csv(uploaded_file)

    if not samples:
        st.error("No valid 'Results Table 2' or data found.")
    else:
        st.success(f"Found {len(samples)} samples!")

        strain_min = st.slider("Select minimum strain (%)", 0.0, 10.0, 2.0, step=0.1)
        strain_max = st.slider("Select maximum strain (%)", 0.0, 10.0, 5.0, step=0.1)

        sample_selected = st.selectbox("Choose a sample to plot", list(samples.keys()))

        if sample_selected:
            df = samples[sample_selected]
            slope, x, y = compute_regression_modulus(df, strain_range=(strain_min / 100, strain_max / 100))

            st.subheader(f"{sample_selected} — Stress-Strain Plot")

            fig, ax = plt.subplots()
            ax.plot(df["Tensile strain (Strain 1)"], df["Tensile stress"], label="Raw Data", alpha=0.7)
            if slope is not None:
                y_fit = slope * x
                ax.plot(x, y_fit, 'r--', label=f"Fit: E = {slope:.2f}")
                st.markdown(f"**Estimated Young's Modulus:** {slope:.2f}")
            else:
                st.warning("Not enough data in the selected strain range to compute modulus.")

            ax.set_xlabel("Strain")
            ax.set_ylabel("Stress")
            ax.legend()
            st.pyplot(fig)
