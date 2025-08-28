import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Constraint plot", layout="wide")
st.title("Constraint plot")

# ---- Sidebar: data upload ----
st.sidebar.header("Excel")
uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])
if not uploaded:
    st.info("Upload an Excel file in the sidebar to begin.")
    st.stop()

xlsx = pd.ExcelFile(uploaded)
sheet = st.sidebar.selectbox("Select sheet", xlsx.sheet_names)

@st.cache_data
def load_df(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    xcol = "S1" if "S1" in df.columns else df.columns[0]
    return df, xcol

df, xcol = load_df(uploaded, sheet)

# ---- Series selection ----
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
ycols = [c for c in num_cols if c != xcol]
default_sel = ycols if ycols else []
sel = st.sidebar.multiselect("Series to show", ycols, default=default_sel)

if not sel:
    st.warning("⚠ Please select at least one Y series.")
    st.stop()

# ---- Constraint sets (explicit) ----
st.sidebar.markdown("**Constraint sets**")
upper_set = st.sidebar.multiselect("Upper (≤)", sel)
lower_set = st.sidebar.multiselect("Lower (≥)", [c for c in sel if c not in upper_set])

# ---- Prepare data ----
work = df[[xcol] + sel].dropna().copy()
work = work.sort_values(by=xcol).reset_index(drop=True)
X = work[xcol].to_numpy()

# ---- Base figure ----
long_df = work.melt(id_vars=xcol, var_name="Series", value_name="Value")
fig = px.line(long_df, x=xcol, y="Value", color="Series")
fig.update_traces(line=dict(width=4))

xmax = float(long_df[xcol].max())
ymax = float(long_df["Value"].max())
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    xaxis=dict(range=[0, xmax], title=str(xcol)),
    yaxis=dict(range=[0, ymax], title="Value"),
    margin=dict(l=10, r=10, t=40, b=10)
)

# ---- Click interaction ----
if "clicks" not in st.session_state:
    st.session_state.clicks = []
st.caption("Click twice on the chart to select an x-interval. Third click resets.")
events = plotly_events(fig, click_event=True, select_event=False, hover_event=False)
if events:
    x_clicked = events[0]["x"]
    st.session_state.clicks.append(x_clicked)
    if len(st.session_state.clicks) > 2:
        st.session_state.clicks = [x_clicked]  # reset

# ---- Feasible band ----
if len(st.session_state.clicks) == 2 and (upper_set or lower_set):
    x1, x2 = sorted(st.session_state.clicks)
    gx = np.linspace(max(0, X.min(), x1), max(0, x2), 600)

    vals = {c: np.interp(gx, X, work[c].to_numpy()) for c in sel}

    if upper_set:
        up = np.minimum.reduce([vals[c] for c in upper_set])
    else:
        up = np.full_like(gx, np.inf, dtype=float)

    low_parts = [np.zeros_like(gx)]
    if lower_set:
        low_parts += [vals[c] for c in lower_set]
    low = np.maximum.reduce(low_parts)

    mask = up >= low
    gx_f, up_f, low_f = gx[mask], np.clip(up[mask], 0, None), np.clip(low[mask], 0, None)

    if gx_f.size > 0:
        fig.add_scatter(x=gx_f, y=up_f, mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=gx_f, y=low_f, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(255,0,0,0.28)",
                        hoverinfo="skip", name="Feasible area")

# ---- Render ----
st.plotly_chart(fig, use_container_width=True)

if st.button("Reset selection"):
    st.session_state.clicks = []
