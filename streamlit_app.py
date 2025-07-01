import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from difflib import get_close_matches

# --- Cached parser ---
@st.cache_data
def parse_sample_tables_from_csv(uploaded_file):
    import pandas as pd

    headers = ["Time", "Displacement", "Force", "Tensile stress", "Tensile strain (Strain 1)"]
    metadata_target_keys = [
        "Tensile stress at Maximum Force",
        "Tensile stress at TENSILE STRESS at breaks"
    ]

    # Normalize target keys (strip + single spaces)
    metadata_target_keys = [key.strip().replace("  ", " ") for key in metadata_target_keys]

    sample_tables = {}
    sample_metadata = {}
    global_metadata = {}
    raw_df = pd.read_csv(uploaded_file, header=None)

    mode = None
    i = 0
    sample_index = 1
    metadata_df = None

    while i < len(raw_df):
        row = raw_df.iloc[i]
        first_cell = str(row[0]) if len(row) > 0 else None

        if first_cell == "Results Table 1":
            mode = "metadata"
            i += 1
            continue

        elif first_cell == "Results Table 2":
            mode = "data"
            i += 1
            continue

        if mode == "metadata":
            row_data = row.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            # Look for presence of any metadata keys in this header row
            matched_keys = [key for key in metadata_target_keys if key in row_data.values]
            if matched_keys:
                header_indices = {key: row_data[row_data == key].index[0] for key in matched_keys}
                i += 2  # Skip units row

                data_rows = []
                while i < len(raw_df):
                    next_row = raw_df.iloc[i]
                    if str(next_row[0]) == "Results Table 2":
                        break
                    values = {key: next_row[idx] for key, idx in header_indices.items() if idx < len(next_row)}
                    data_rows.append(values)
                    i += 1

                if data_rows:
                    metadata_df = pd.DataFrame(data_rows)

                continue

        elif mode == "data":
            row_data = row[1:]
            if set(headers).issubset(set(row_data.values)):
                header_indices = {name: row_data[row_data == name].index[0] for name in headers}
                i += 2  # skip unit row

                data_rows = []
                while i < len(raw_df):
                    next_row = raw_df.iloc[i]
                    test_row = next_row[1:]
                    if set(headers).issubset(set(test_row.values)) or next_row.isnull().all():
                        break
                    values = {key: next_row[idx] for key, idx in header_indices.items() if idx < len(next_row)}
                    data_rows.append(values)
                    i += 1

                if data_rows:
                    df_sample = pd.DataFrame(data_rows).apply(pd.to_numeric, errors='coerce')
                    sample_name = f"Sample_{sample_index}"
                    sample_tables[sample_name] = df_sample.reset_index(drop=True)
                    sample_metadata[sample_name] = {}  # Fill later
                    sample_index += 1
                continue

        i += 1

    # Extract metadata values from metadata_df
    if metadata_df is not None:
        for key in metadata_df.columns:
            series = metadata_df[key].dropna()
            if not series.empty:
                global_metadata[key] = series.iloc[0]

        for name in sample_metadata:
            sample_metadata[name] = global_metadata.copy()

    return sample_tables, sample_metadata, list(global_metadata.keys())

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

uploaded_file = st.file_uploader("📂 Upload CSV File with Tensile Data", type=["csv"])

strain_at_break = {}  # Dict to store all strain at break values

if uploaded_file:
    sample_tables, sample_metadata, metadata_keys = parse_sample_tables_from_csv(uploaded_file)
    if not sample_tables:
        st.error("❌ No 'Results Table 2' or valid samples found.")
    else:
        with st.sidebar:
            st.header("⚙️ Settings")
            strain_min = st.slider("Min Range Strain (%)", 0.0, 100.0, 2.0, 0.1)
            strain_max = st.slider("Max Range Strain (%)", 0.0, 100.0, 5.0, 0.1)
            sample_selected = st.selectbox("🧪 Choose Sample", list(sample_tables.keys()))
            method = st.radio("Modulus Calculation Method", ["Linear Regression", "Point-to-Point"])

            # Optional file upload for strain at break
            strain_break_file = st.file_uploader("📎 Upload CSV: Strain at Break", type=["csv"])
            if strain_break_file:
                try:
                    df_strain_break = pd.read_csv(strain_break_file)
                    if "Sample" in df_strain_break.columns and "Tensile strain at Break / %" in df_strain_break.columns:
                        for i, row in df_strain_break.iterrows():
                            matched = get_close_matches(row["Sample"], sample_tables.keys(), n=1, cutoff=0.8)
                            if matched:
                                strain_at_break[matched[0]] = row["Tensile strain at Break / %"]
                            else:
                                st.warning(f"⚠️ No match found for '{row['Sample']}' in tensile data samples.")
                    else:
                        st.warning("⚠️ CSV must contain columns: 'Sample', 'Tensile strain at Break / %'")
                except Exception as e:
                    st.error(f"Could not parse strain at break file: {e}")

        df = sample_tables[sample_selected]
        meta = sample_metadata.get(sample_selected, {})

        # Allow manual entry of strain at break if not already available
        if sample_selected not in strain_at_break:
            manual_val = st.number_input(
                f"Enter 'Tensile strain at Break / %' for {sample_selected} (if known):",
                min_value=0.0, max_value=100.0, step=0.01
            )
            if manual_val > 0:
                strain_at_break[sample_selected] = manual_val

        # --- Display parameters ---
        # Display table summary
        st.subheader("📋 Summary of All Samples")
        
        summary_rows = []
        for sample_name, df in sample_tables.items():
            slope, _, _ = compute_regression_modulus(df, (strain_min / 100, strain_max / 100))
            E_value = round(slope, 2) if slope is not None else "Not Computed"
            
            meta = sample_metadata.get(sample_name, {})
            row = {
                "Sample": sample_name,
                "Young’s Modulus (MPa)": E_value,
                "Max Stress (MPa)": meta.get("Tensile stress at Maximum Force", "—"),
                "Stress at Break (MPa)": meta.get("Tensile stress at TENSILE STRESS at breaks", "—"),
            }
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        with st.expander("📈 Show Stress–Strain Plot"):
            selected_sample = st.selectbox("Choose Sample to Plot", list(sample_tables.keys()))
            if st.button("Show Plot"):
                st.subheader(f"Stress–Strain Curve — {selected_sample}")
                df = sample_tables[selected_sample]
                slope, x_fit, y_fit = compute_regression_modulus(df, (strain_min, strain_max))
        
                fig, ax = plt.subplots()
                strain_col = "Tensile strain (Strain 1)"
                stress_col = "Tensile stress"
                strain_data = df[strain_col] * (100 if df[strain_col].max() < 1 else 1)
                ax.plot(strain_data, df[stress_col], label="Raw Data", alpha=0.6)
        
                if slope is not None:
                    ax.plot(x_fit * 100, slope * x_fit, 'r--', label=f"Fit: E ≈ {slope:.2f} MPa")
                    st.success(f"Estimated Young’s Modulus: {slope:.2f} MPa")
                else:
                    st.warning("⚠️ Not enough valid data points in selected strain range.")
        
                ax.set_xlabel("Tensile Strain / %")
                ax.set_ylabel("Tensile Stress / MPa")
                ax.legend()
                st.pyplot(fig)
        
        # Optional: download CSV
        csv_data = summary_df.to_csv(index=False).encode()
        st.download_button("📥 Download Summary CSV", csv_data, file_name="tensile_summary.csv", mime="text/csv")
