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
        "Tensile stress at TENSILE STRESS at breaks"
    ]
    metadata_target_keys = [key.strip().replace("  ", " ") for key in metadata_target_keys]

    sample_tables = {}
    sample_metadata = {}
    raw_df = pd.read_csv(uploaded_file, header=None)

    mode = None
    i = 0
    sample_index = 1
    metadata_rows = []

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
            matched_keys = [key for key in metadata_target_keys if key in row_data.values]
            if matched_keys:
                header_indices = {key: row_data[row_data == key].index[0] for key in matched_keys}
                i += 2  # Skip unit row

                while i < len(raw_df):
                    next_row = raw_df.iloc[i]
                    if str(next_row[0]) == "Results Table 2":
                        break
                    values = {key: next_row[idx] for key, idx in header_indices.items() if idx < len(next_row)}
                    metadata_rows.append(values)
                    i += 1
                continue

        elif mode == "data":
            row_data = row[1:]
            if set(headers).issubset(set(row_data.values)):
                header_indices = {name: row_data[row_data == name].index[0] for name in headers}
                i += 2

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
                    if sample_index - 1 < len(metadata_rows):
                        sample_metadata[sample_name] = metadata_rows[sample_index - 1]
                    else:
                        sample_metadata[sample_name] = {}
                    sample_index += 1
                continue
        i += 1
    return sample_tables, sample_metadata, metadata_target_keys

# --- Regression method ---
def compute_regression_modulus(df, strain_range=(2, 5)):
    strain_col = "Tensile strain (Strain 1)"
    stress_col = "Tensile stress"
    df = df[[strain_col, stress_col]].dropna().copy()
    if df[strain_col].max() > 1:
        df[strain_col] /= 100.0

    min_strain = strain_range[0] / 100.0
    max_strain = strain_range[1] / 100.0
    df_window = df[(df[strain_col] >= min_strain) & (df[strain_col] <= max_strain)]
    if len(df_window) < 2:
        return None, None, None

    df_grouped = df_window.groupby(strain_col, as_index=False)[stress_col].min()
    x = df_grouped[strain_col].values
    y = df_grouped[stress_col].values
    slope, _, _, _, _ = linregress(x, y)
    return slope, x, y

# --- Point-to-point method ---
def compute_gradient_modulus(df, strain_range=(2, 5)):
    strain_col = "Tensile strain (Strain 1)"
    stress_col = "Tensile stress"
    df = df[[strain_col, stress_col]].dropna().copy()
    if df[strain_col].max() > 1:
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
strain_at_break = {}

if uploaded_file:
    sample_tables, sample_metadata, metadata_keys = parse_sample_tables_from_csv(uploaded_file)

    if not sample_tables:
        st.error("❌ No 'Results Table 2' or valid samples found.")
    else:
        with st.sidebar:
            st.header("⚙️ Settings")
            strain_min = st.slider("Min Range Strain (%)", 0.0, 100.0, 2.0, 0.1)
            strain_max = st.slider("Max Range Strain (%)", 0.0, 100.0, 5.0, 0.1)
            sample_selected = st.selectbox("🧪 Choose Sample to Plot", list(sample_tables.keys()))
            method = st.radio("Modulus Calculation Method", ["Linear Regression", "Point-to-Point"])

            strain_break_file = st.file_uploader("📎 Upload CSV: Strain at Break", type=["csv"])
            if strain_break_file:
                try:
                    df_strain_break = pd.read_csv(strain_break_file)
                    if "Sample" in df_strain_break.columns and "Tensile strain at Break / %" in df_strain_break.columns:
                        for _, row in df_strain_break.iterrows():
                            matched = get_close_matches(row["Sample"], sample_tables.keys(), n=1, cutoff=0.8)
                            if matched:
                                strain_at_break[matched[0]] = row["Tensile strain at Break / %"]
                            else:
                                st.warning(f"⚠️ No match found for '{row['Sample']}' in tensile data samples.")
                    else:
                        st.warning("⚠️ CSV must contain columns: 'Sample', 'Tensile strain at Break / %'")
                except Exception as e:
                    st.error(f"Could not parse strain at break file: {e}")

        # Fill strain at break manually if not uploaded
        if sample_selected not in strain_at_break:
            manual_val = st.number_input(
                f"Enter 'Tensile strain at Break / %' for {sample_selected} (if known):",
                min_value=0.0, max_value=100.0, step=0.01
            )
            if manual_val > 0:
                strain_at_break[sample_selected] = manual_val

        # --- Display Summary Table ---
        st.subheader("📋 Summary of All Samples")
        summary_rows = []
        for sample_name, df_sample in sample_tables.items():
            if method == "Linear Regression":
                slope, _, _ = compute_regression_modulus(df_sample, (strain_min, strain_max))
            else:
                slope, _, _ = compute_gradient_modulus(df_sample, (strain_min, strain_max))

            E_value = round(slope, 2) if slope is not None else "Not Computed"
            meta = sample_metadata.get(sample_name, {})
            row = {
                "Sample": sample_name,
                "Youngs Modulus/ MPa": E_value,
                "Tensile Stress at Maximum Force/ MPa": meta.get("Tensile stress at Maximum Force", "—"),
                "Tensile Stress at Break/ MPa": meta.get("Tensile stress at TENSILE STRESS at breaks", "—"),
                "Tensile Strain at Break/ %": strain_at_break.get(sample_name, "—")
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("👥 Grouping and Averages")

        # Step 1: Grouping UI
        group_dict = {}
        all_samples = list(sample_tables.keys())
        existing_groups = []
        
        with st.expander("➕ Define Sample Groups"):
            num_groups = st.number_input("Number of Groups", min_value=1, max_value=10, value=1, step=1)
        
            for i in range(num_groups):
                group_name = st.text_input(f"Group {i+1} Name", value=f"Group_{i+1}")
                selected_samples = st.multiselect(f"Select Samples for {group_name}", all_samples, key=f"group_{i}")
                group_dict[group_name] = selected_samples
                existing_groups.extend(selected_samples)
        
            # Step 2: Display average summary per group
            st.subheader("📈 Group Summary Statistics")
        
            group_summary_rows = []
            for group_name, samples in group_dict.items():
                group_data = []
                for sample in samples:
                    slope = None
                    if method == "Linear Regression":
                        slope, _, _ = compute_regression_modulus(sample_tables[sample], (strain_min, strain_max))
                    else:
                        slope, _, _ = compute_gradient_modulus(sample_tables[sample], (strain_min, strain_max))
        
                    meta = sample_metadata.get(sample, {})
                    strain_break = strain_at_break.get(sample)
        
                    group_data.append({
                        "Young’s Modulus": slope,
                        "Max Stress": meta.get("Tensile stress at Maximum Force"),
                        "Stress at Break": meta.get("Tensile stress at TENSILE STRESS at breaks"),
                        "Strain at Break": strain_break
                    })
        
                # Calculate averages
                df_group = pd.DataFrame(group_data)
                averages = df_group.mean(numeric_only=True)
        
                group_summary_rows.append({
                    "Group": group_name,
                    "Avg Young’s Modulus (MPa)": round(averages["Young’s Modulus"], 2) if not pd.isna(averages["Young’s Modulus"]) else "—",
                    "Avg Max Stress (MPa)": round(averages["Max Stress"], 2) if not pd.isna(averages["Max Stress"]) else "—",
                    "Avg Stress at Break (MPa)": round(averages["Stress at Break"], 2) if not pd.isna(averages["Stress at Break"]) else "—",
                    "Avg Strain at Break (%)": round(averages["Strain at Break"], 2) if not pd.isna(averages["Strain at Break"]) else "—",
                })
        
            df_group_summary = pd.DataFrame(group_summary_rows)
            st.dataframe(df_group_summary)
        
            # Optional CSV export
            csv_group = df_group_summary.to_csv(index=False).encode()
            st.download_button("📥 Download Group Averages CSV", csv_group, file_name="group_averages.csv", mime="text/csv")
        
        
            with st.expander("📈 Show Stress–Strain Plot"):
                if st.button("Show Plot"):
                    st.subheader(f"Stress–Strain Curve — {sample_selected}")
                    df = sample_tables[sample_selected]
                    if method == "Linear Regression":
                        slope, x_fit, y_fit = compute_regression_modulus(df, (strain_min, strain_max))
                    else:
                        slope, x_fit, y_fit = compute_gradient_modulus(df, (strain_min, strain_max))
    
                    fig, ax = plt.subplots()
                    strain_col = "Tensile strain (Strain 1)"
                    stress_col = "Tensile stress"
                    strain_data = df[strain_col] * (100 if df[strain_col].max() < 1 else 1)
                    ax.plot(strain_data, df[stress_col], label="Raw Data", alpha=0.6)
    
                    if slope is not None:
                        ax.plot(np.array(x_fit) * 100, y_fit, 'r--', label=f"Fit: E ≈ {slope:.2f} MPa")
                        st.success(f"Estimated Young’s Modulus: {slope:.2f} MPa")
                    else:
                        st.warning("⚠️ Not enough valid data points in selected strain range.")
    
                    ax.set_xlabel("Tensile Strain / %")
                    ax.set_ylabel("Tensile Stress / MPa")
                    ax.legend()
                    st.pyplot(fig)

        csv_data = summary_df.to_csv(index=False).encode()
        st.download_button("📥 Download Summary CSV", csv_data, file_name="tensile_summary.csv", mime="text/csv")
