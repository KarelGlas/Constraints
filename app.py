import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Constraint plot", layout="wide")
st.title("Constraint plot")

# ---------------- Sidebar: Upload & picks ----------------
st.sidebar.header("Excel")
up = st.sidebar.file_uploader("Upload Excel", type=["xlsx","xls"])
if not up:
    st.info("Upload an Excel file in the sidebar to begin.")
    st.stop()

xlsx = pd.ExcelFile(up)
sheet = st.sidebar.selectbox("Select sheet", xlsx.sheet_names)

@st.cache_data
def load_df(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    xcol = "S1" if "S1" in df.columns else df.columns[0]
    return df, xcol
df, xcol = load_df(up, sheet)

num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
series_all = [c for c in num if c != xcol]
sel = st.sidebar.multiselect("Series to show", series_all, default=series_all)
if not sel:
    st.warning("Select at least one Y series.")
    st.stop()

st.sidebar.subheader("Constraint sets")
upper_set = st.sidebar.multiselect("Upper (≤)", sel)   # envelope = min()
lower_set = st.sidebar.multiselect("Lower (≥)", [c for c in sel if c not in upper_set])  # envelope = max()

# ---------------- State for clicks ----------------
if "clicks" not in st.session_state:
    st.session_state.clicks = []

col_reset, = st.columns(1)
if col_reset.button("Reset selection"):
    st.session_state.clicks = []

# ---------------- Build figure from current state ----------------
def build_fig():
    work = df[[xcol] + sel].dropna().sort_values(xcol).reset_index(drop=True)
    X = work[xcol].to_numpy()
    long_df = work.melt(id_vars=xcol, var_name="Series", value_name="Value")

    fig = px.line(long_df, x=xcol, y="Value", color="Series")
    fig.update_traces(line=dict(width=4))

    xmax = float(long_df[xcol].max())
    ymax = float(long_df["Value"].max())
    fig.update_layout(
        template="plotly_white", hovermode="x unified",
        xaxis=dict(range=[0, xmax], title=str(xcol)),
        yaxis=dict(range=[0, ymax], title="Value"),
        margin=dict(l=10, r=10, t=40, b=10),
        dragmode="select",
    )

    # Shade feasible band if we have 2 clicks and any constraint
    if len(st.session_state.clicks) == 2 and (upper_set or lower_set):
        x1, x2 = sorted(st.session_state.clicks)
        gx = np.linspace(max(0, X.min(), x1), max(0, x2), 600)

        vals = {c: np.interp(gx, X, work[c].to_numpy()) for c in sel}
        up = np.minimum.reduce([vals[c] for c in upper_set]) if upper_set else np.full_like(gx, np.inf)
        low_parts = [np.zeros_like(gx)] + ([vals[c] for c in lower_set] if lower_set else [])
        low = np.maximum.reduce(low_parts)

        mask = up >= low
        if mask.any():
            gx_f = gx[mask]; up_f = np.clip(up[mask], 0, None); low_f = np.clip(low[mask], 0, None)
            fig.add_scatter(x=gx_f, y=up_f, mode="lines", line=dict(width=0),
                            hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=gx_f, y=low_f, mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor="rgba(255,0,0,0.28)",
                            hoverinfo="skip", name="Feasible area")
    return fig

fig = build_fig()

# ---------------- Single render with click capture ----------------
st.caption("Click **twice** on the chart to select an x-interval. Third click starts a new selection.")
ev = plotly_events(
    fig,
    click_event=True, select_event=False, hover_event=False,
    override_height=520,  # adjust as desired
    key="constraint_plot",
)

# Record click(s) -> triggers rerun, next build_fig will shade
if ev:
    x_clicked = ev[0].get("x", None)
    if x_clicked is not None:
        st.session_state.clicks.append(float(x_clicked))
        if len(st.session_state.clicks) > 2:
            st.session_state.clicks = [st.session_state.clicks[-1]]  # keep last as first point
