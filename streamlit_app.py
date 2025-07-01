import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from difflib import get_close_matches

# --- Cached parser ---
@st.cache_data
def parse_sample_tables_from_csv(uploaded_file):
    headers = ["Time", "Displacement", "Force", "Tensile stress", "Tensile strain (Strain 1)"]
    metadata_target_keys = [
        "Tensile stress at Maximum Force",
        "Tensile stress at TENSILE STRESS  at breaks"  # note: double space
    ]

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
        first_cell = row[0] if len(row) > 0 else None

        if first_cell == "Results Table 1":
            mode = "metadata"
            i += 1
            continue

        elif first_cell == "Results Table 2":
            mode = "data"
            i += 1
            continue

        if mode == "metadata":
            row_data = row
            if any(key in row_data.values for key in metadata_target_keys):
                # Found the header row
                header_indices = {name: row_data[row_data == name].index[0] for name in metadata_target_keys if name in row_data.values}
                i += 2  # skip units row

                # Extract rows into metadata_df
                data_rows = []
                while i < len(raw_df):
                    next_row = raw_df.iloc[i]
                    if next_row[0] == "Results Table 2":
                        break
                    values = {key: next_row[idx] for key, idx in header_indices.items()}
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
                    values = {key: next_row[idx] for key, idx in header_indices.items()}
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

    # Extract metadata values (first non-null from each column)
    if metadata_df is not None:
        for key in metadata_target_keys:
            series = metadata_df[key].dropna()
            if not series.empty:
                global_metadata[key] = series.iloc[0]

        # Propagate to all sample metadata
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
        st.subheader("📌 Sample Parameters")
        for key in metadata_keys:
            value = meta.get(key)
            if pd.notna(value):
                st.write(f"**{key}**: {value} MPa")
            else:
                st.write(f"**{key}**: _Not available_")

        if sample_selected in strain_at_break:
            st.write(f"**Tensile strain at Break / %**: {strain_at_break[sample_selected]:.2f} %")

        # --- Compute and Plot ---
        compute_modulus = compute_regression_modulus if method == "Linear Regression" else compute_gradient_modulus
        slope, x_fit, y_fit = compute_modulus(df, (strain_min, strain_max))

        st.subheader(f"📊 Stress–Strain Curve — {sample_selected}")
        fig, ax = plt.subplots()
        strain_col = "Tensile strain (Strain 1)"
        stress_col = "Tensile stress"
        strain_data = df[strain_col] * (100 if df[strain_col].max() < 1 else 1)
        ax.plot(strain_data, df[stress_col], label="Raw Data", alpha=0.6)

        if slope is not None:
            x_plot = np.array(x_fit) * 100
            ax.plot(x_plot, slope * np.array(x_fit), 'r--', label=f"Fit: E ≈ {slope:.2f} MPa")
            st.success(f"**Estimated Young’s Modulus:** {slope:.2f} MPa")
        else:
            st.warning("⚠️ Not enough valid data points in selected strain range.")

        ax.set_xlabel("Tensile Strain / %")
        ax.set_ylabel("Tensile Stress / MPa")
        ax.legend()
        st.pyplot(fig)

        # --- Export all moduli ---
        st.subheader("📤 Export Young’s Moduli for All Samples")
        results = []
        for sample_name, df_sample in sample_tables.items():
            E, _, _ = compute_modulus(df_sample, (strain_min, strain_max))
            row = {
                "Sample": sample_name,
                "Young’s Modulus (MPa)": round(E, 2) if E is not None else "Not Computed"
            }
            row.update(sample_metadata.get(sample_name, {}))
            if strain_at_break.get(sample_name) is not None:
                row["Tensile strain at Break / %"] = strain_at_break[sample_name]
            results.append(row)

        df_moduli = pd.DataFrame(results)
        csv = df_moduli.to_csv(index=False).encode()
        st.download_button("Download Moduli CSV", csv, file_name="youngs_moduli.csv", mime="text/csv")
