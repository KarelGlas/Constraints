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
    if "S1" in df.columns:
        xcol = "S1"
    else:
        xcol = df.columns[0]
    return df, xcol

df, xcol = load_df(uploaded, sheet)

# ---- Column selection ----
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
ycols = st.sidebar.multiselect("Series to show", [c for c in numeric_cols if c != xcol],
                               default=[c for c in numeric_cols if c != xcol])

if not ycols:
    st.warning("Select at least one Y series in the sidebar.")
    st.stop()

# ---- Classify constraint types ----
st.sidebar.markdown("**Constraint types**")
upper_set = st.sidebar.multiselect("Upper (≤)", ycols,
                                   default=[c for c in ycols if df[c].diff().mean() <= 0])
lower_set = st.sidebar.multiselect("Lower (≥)", [c for c in ycols if c not in upper_set],
                                   default=[c for c in ycols if df[c].diff().mean() > 0])

# ---- Base long df & plot ----
plot_df = df[[xcol] + ycols].copy()
long_df = plot_df.melt(id_vars=xcol, var_name="Series", value_name="Value")

fig = px.line(long_df, x=xcol, y="Value", color="Series")
fig.update_traces(line=dict(width=4))  # thick lines

xmax = float(long_df[xcol].max())
ymax = float(long_df["Value"].max())
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    xaxis=dict(range=[0, xmax], title=str(xcol)),
    yaxis=dict(range=[0, ymax], title="Value"),
    margin=dict(l=10, r=10, t=40, b=10)
)

# ---- Interaction: click to pick interval (two clicks) ----
if "clicks" not in st.session_state:
    st.session_state.clicks = []

st.caption("Click twice on the chart to select an x-interval. Third click resets.")
events = plotly_events(fig, click_event=True, select_event=False, hover_event=False)

if events:
    x_clicked = events[0]["x"]
    st.session_state.clicks.append(x_clicked)
    if len(st.session_state.clicks) > 2:
        st.session_state.clicks = [x_clicked]  # reset

# ---- Shade feasible band inside selected interval ----
if len(st.session_state.clicks) == 2:
    x1, x2 = sorted(st.session_state.clicks)

    # common x grid over interval
    X = plot_df[xcol].values
    grid = np.linspace(max(0, X.min(), x1), max(0, x2), 400)

    # interpolate series on grid
    vals = {c: np.interp(grid, X, plot_df[c].values) for c in ycols}

    # envelopes: upper = min of uppers, lower = max of lowers and 0-axis
    upper = np.minimum.reduce([vals[c] for c in upper_set]) if upper_set else np.inf*np.ones_like(grid)
    lowers = [np.zeros_like(grid)]
    if lower_set:
        lowers += [vals[c] for c in lower_set]
    lower = np.maximum.reduce(lowers)

    mask = upper >= lower
    gx, gtop, gbot = grid[mask], upper[mask], lower[mask]

    if len(gx) > 0:
        fig.add_scatter(x=gx, y=gtop, mode="lines", line=dict(width=0),
                        name="Feasible upper", showlegend=False)
        fig.add_scatter(x=gx, y=gbot, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(255,0,0,0.25)",
                        name="Feasible area", showlegend=True)

# ---- Render ----
st.plotly_chart(fig, use_container_width=True)

# Reset button
if st.button("Reset selection"):
    st.session_state.clicks = []

st.subheader("Data preview")
st.dataframe(df.head())
