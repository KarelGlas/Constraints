import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Excel 2D Plot", layout="wide")
st.title("2D Plot from Excel (thick lines, Plotly)")

# --- Sidebar ---
st.sidebar.header("Excel")
uploaded = st.sidebar.file_uploader(
    "Upload Excel (sheet with S1 + constraint columns; optional sheet 'shadows')",
    type=["xlsx", "xls"]
)

if not uploaded:
    st.info("Upload an Excel file in the sidebar to begin.")
    st.stop()

# List sheets
xlsx = pd.ExcelFile(uploaded)
sheet = st.sidebar.selectbox("Select sheet", xlsx.sheet_names)

@st.cache_data
def load_df(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    if "S1" in df.columns:
        x_col = "S1"
    else:
        x_col = df.columns[0]
    return df, x_col

df, default_x = load_df(uploaded, sheet)

st.write("Preview", df.head())

# --- Axis selections in sidebar ---
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
x_col = st.sidebar.selectbox(
    "X axis", 
    options=numeric_cols + [c for c in df.columns if c not in numeric_cols], 
    index=(numeric_cols.index(default_x) if default_x in numeric_cols else 0)
)
y_candidates = [c for c in numeric_cols if c != x_col]
y_cols = st.sidebar.multiselect("Y series", y_candidates, default=y_candidates)

if not y_cols:
    st.warning("Select at least one Y series in the sidebar.")
    st.stop()

# Melt for Plotly
plot_df = df[[x_col] + y_cols].copy()
long_df = plot_df.melt(id_vars=x_col, value_vars=y_cols, var_name="Series", value_name="Value")

# Plot
fig = px.line(long_df, x=x_col, y="Value", color="Series", markers=False)
fig.update_traces(line=dict(width=4))  # thick lines

# Hard limits: axes start at 0
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    legend_title_text="",
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(range=[0, long_df[x_col].max()]),
    yaxis=dict(range=[0, long_df["Value"].max()])
)

st.plotly_chart(fig, use_container_width=True)