import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Constraint plot", layout="wide")
st.title("Constraint plot")

# --- Sidebar: Excel ---
st.sidebar.header("Excel")
up = st.sidebar.file_uploader("Upload Excel", type=["xlsx","xls"])
if not up:
    st.info("Upload an Excel file in the sidebar to begin.")
    st.stop()
xlsx = pd.ExcelFile(up)
sheet = st.sidebar.selectbox("Select sheet", xlsx.sheet_names)

@st.cache_data
def load_df(f, s):
    df = pd.read_excel(f, sheet_name=s)
    xcol = "S1" if "S1" in df.columns else df.columns[0]
    return df, xcol
df, xcol = load_df(up, sheet)

# --- Series pick ---
num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
ycols = [c for c in num if c != xcol]
sel = st.sidebar.multiselect("Series to show", ycols, default=ycols)
if not sel:
    st.warning("Select at least one Y series.")
    st.stop()

# --- Constraint sets ---
st.sidebar.subheader("Constraint sets")
upper_set = st.sidebar.multiselect("Upper (≤)", sel)                 # envelope = min()
lower_set = st.sidebar.multiselect("Lower (≥)", [c for c in sel if c not in upper_set])  # envelope = max()

# --- Data prep ---
work = df[[xcol] + sel].dropna().sort_values(xcol).reset_index(drop=True)
X = work[xcol].to_numpy()

# --- Base plot ---
long_df = work.melt(id_vars=xcol, var_name="Series", value_name="Value")
fig = px.line(long_df, x=xcol, y="Value", color="Series")
fig.update_traces(line=dict(width=4))
xmax = float(long_df[xcol].max()); ymax = float(long_df["Value"].max())
fig.update_layout(
    template="plotly_white", hovermode="x unified",
    xaxis=dict(range=[0, xmax], title=str(xcol)),
    yaxis=dict(range=[0, ymax], title="Value"),
    margin=dict(l=10,r=10,t=40,b=10),
    dragmode="select",
)

# --- Interaction: two clicks OR slider fallback ---
if "clicks" not in st.session_state: st.session_state.clicks = []
st.caption("Pick an interval: **click twice** on the chart (drag-select also OK).")
events = plotly_events(fig, click_event=True, select_event=False, hover_event=False)

# record clicks
if events:
    st.session_state.clicks.append(events[0]["x"])
    if len(st.session_state.clicks) > 2:
        st.session_state.clicks = [st.session_state.clicks[-1]]  # reset to last click

# fallback slider (always available)
x1s, x2s = float(X.min()), float(X.max())
x1_fallback, x2_fallback = st.sidebar.slider("Interval (fallback)", x1s, x2s, (x1s, x2s))

# determine interval
if len(st.session_state.clicks) == 2:
    x1, x2 = sorted(st.session_state.clicks)
else:
    x1, x2 = x1_fallback, x2_fallback

# --- Feasible band computation ---
has_constraints = bool(upper_set or lower_set)
if has_constraints and x2 > max(0, x1):  # valid interval
    gx = np.linspace(max(0, X.min(), x1), max(0, x2), 600)

    vals = {c: np.interp(gx, X, work[c].to_numpy()) for c in sel}
    up = (np.minimum.reduce([vals[c] for c in upper_set])
          if upper_set else np.full_like(gx, np.inf, dtype=float))
    low_parts = [np.zeros_like(gx)]
    if lower_set: low_parts += [vals[c] for c in lower_set]
    low = np.maximum.reduce(low_parts)

    mask = up >= low
    if mask.any():
        gx_f = gx[mask]; up_f = np.clip(up[mask], 0, None); low_f = np.clip(low[mask], 0, None)
        # add only the filled band
        fig.add_scatter(x=gx_f, y=up_f, mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=gx_f, y=low_f, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(255,0,0,0.28)",
                        hoverinfo="skip", name="Feasible area")
    else:
        st.info("No feasible area within the selected interval (constraints conflict).")
else:
    st.info("Select two clicks or use the slider, and choose at least one Upper or Lower constraint.")

# --- Render + controls ---
st.plotly_chart(fig, use_container_width=True)
if st.button("Reset selection"): st.session_state.clicks = []
