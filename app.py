import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


st.set_page_config(page_title="Constraint Plot", layout="wide")
st.title("Constraint plot")

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

# --- controls in sidebar ---
upper_series = st.sidebar.selectbox("Upper bound series", y_candidates, index=0, key="upper")
lower_series = st.sidebar.selectbox("Lower bound series", [c for c in y_candidates if c != upper_series], index=0, key="lower")
shade_on = st.sidebar.checkbox("Highlight area between bounds", value=True)

# --- long format (kept) ---
plot_df = df[[x_col] + y_cols].copy()
long_df = plot_df.melt(id_vars=x_col, value_vars=y_cols, var_name="Series", value_name="Value")

# --- base lines ---
fig = px.line(long_df, x=x_col, y="Value", color="Series", markers=False)
fig.update_traces(line=dict(width=4))

# --- hard limits start at 0 ---
xmax = max(0, float(long_df[x_col].max()))
ymax = max(0, float(long_df["Value"].max()))
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    legend_title_text="",
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(range=[0, xmax]),
    yaxis=dict(range=[0, ymax])
)

# --- shaded area between two constraints ---
if shade_on and upper_series in plot_df and lower_series in plot_df:
    # align on a shared x-grid (union of x's)
    x = np.sort(plot_df[x_col].unique())
    u = np.interp(x, plot_df[x_col], plot_df[upper_series])
    l = np.interp(x, plot_df[x_col], plot_df[lower_series])

    # ensure correct ordering, clip to axes ≥ 0
    top = np.maximum(u, l)
    bot = np.minimum(u, l)
    top = np.clip(top, 0, None)
    bot = np.clip(bot, 0, None)
    x = np.clip(x, 0, None)

    # add two invisible lines to create a filled band
    fig.add_scatter(x=x, y=top, mode="lines", line=dict(width=0), name=f"Upper: {upper_series}",
                    showlegend=False)
    fig.add_scatter(x=x, y=bot, mode="lines", line=dict(width=0), name=f"Lower: {lower_series}",
                    fill="tonexty", fillcolor="rgba(255,0,0,0.20)", showlegend=False)

st.plotly_chart(fig, use_container_width=True)