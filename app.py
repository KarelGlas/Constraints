import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Helpers
# -----------------------------
def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        # try parse numeric if possible
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def compute_feasible_polygon(df: pd.DataFrame, x_col: str, y_cols: list,
                             sense: str = "≤", baseline: float = 0.0):
    """
    Build a polygon representing the feasible region given multiple constraints y_i(x).
    Assumptions:
      - '≤' sense: feasible is y between [baseline, min_i y_i(x)]
      - '≥' sense: feasible is y between [max_i y_i(x), baseline]
    Returns list of (x, y) vertices in order (closed polygon not required by caller).
    """
    if len(y_cols) == 0 or x_col not in df.columns:
        return []

    # Sort by x and drop rows with any NaN in required cols
    work = df[[x_col] + y_cols].dropna().sort_values(by=x_col)
    if work.empty:
        return []

    x = work[x_col].values
    Y = work[y_cols].values  # shape (n, k)

    if sense == "≤":
        envelope = np.min(Y, axis=1)
        lower = np.full_like(envelope, float(baseline))
        # polygon path: along x with envelope (top), then back along reversed x at lower (bottom)
        xs = np.concatenate([x, x[::-1]])
        ys = np.concatenate([envelope, lower[::-1]])
    else:  # '≥'
        envelope = np.max(Y, axis=1)
        upper = np.full_like(envelope, float(baseline))
        xs = np.concatenate([x, x[::-1]])
        ys = np.concatenate([upper, envelope[::-1]])

    # Remove possible self-crossing by dropping nan/inf and compressing duplicates
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 3:
        return []
    return list(zip(xs, ys))

def active_bottleneck(df: pd.DataFrame, x_col: str, y_cols: list, sense: str = "≤"):
    """Return a Series with the constraint name that governs (min or max) at each x."""
    work = df[[x_col] + y_cols].dropna().sort_values(by=x_col).reset_index(drop=True)
    if work.empty:
        return pd.Series(dtype=object)
    mat = work[y_cols].to_numpy()
    idx = np.argmin(mat, axis=1) if sense == "≤" else np.argmax(mat, axis=1)
    return pd.Series([y_cols[i] for i in idx], name="bottleneck"), work[x_col].reset_index(drop=True)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Debottlenecking Constraint Visualizer", layout="wide")
st.title("Debottlenecking Constraint Visualizer")

uploaded_file = st.sidebar.file_uploader("Upload Excel with constraints", type=["xlsx"])
if not uploaded_file:
    st.info("Upload an .xlsx with one main sheet and optional 'shadows' sheet.")
    st.stop()

# Load workbook
xlsx = pd.ExcelFile(uploaded_file)
sheet_names = xlsx.sheet_names
main_candidates = [s for s in sheet_names if s.lower() != 'shadows']
if not main_candidates:
    main_candidates = sheet_names

main_sheet = st.sidebar.selectbox("Select main scenario sheet", main_candidates)

raw_df = pd.read_excel(uploaded_file, sheet_name=main_sheet)
df = sanitize_numeric(raw_df)

# Choose x and y columns
default_x = 'S1' if 'S1' in df.columns else df.columns[0]
x_col = st.sidebar.selectbox("X-axis column", options=list(df.columns), index=list(df.columns).index(default_x) if default_x in df.columns else 0)
y_cols = [c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
y_select = st.sidebar.multiselect("Constraint columns (Y)", options=y_cols, default=y_cols)

if len(y_select) == 0:
    st.error("Select at least one constraint column.")
    st.stop()

# Feasibility settings
sense = st.sidebar.radio("Feasibility sense", options=["≤", "≥"], index=0, help="‘≤’: feasible under the tightest limit; ‘≥’: feasible above the loosest limit.")
baseline = st.sidebar.number_input("Baseline (other bound)", value=0.0, step=1.0, help="Lower bound for ‘≤’, upper bound for ‘≥’.")
show_bottleneck = st.sidebar.checkbox("Show active bottleneck labels", value=True)

# Shadow scenario
shadow_df = None
if any(s.lower() == 'shadows' for s in sheet_names):
    shadow_df_raw = pd.read_excel(uploaded_file, sheet_name=[s for s in sheet_names if s.lower() == 'shadows'][0])
    shadow_df = sanitize_numeric(shadow_df_raw)
show_shadow = st.sidebar.checkbox("Show shadow constraints scenario", value=False) if shadow_df is not None else False

# -----------------------------
# Build figure
# -----------------------------
fig = go.Figure()

# Base constraint lines
palette_width = 3
for col in y_select:
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[col],
            name=col,
            mode='lines',
            line=dict(width=palette_width),
            hoverinfo='skip'  # disable default hover on lines
        )
    )

# Feasible region polygon
vertices = compute_feasible_polygon(df, x_col, y_select, sense=sense, baseline=baseline)
if vertices:
    poly_x, poly_y = zip(*vertices)
    # close polygon
    poly_x = list(poly_x) + [poly_x[0]]
    poly_y = list(poly_y) + [poly_y[0]]
    fig.add_trace(
        go.Scatter(
            x=poly_x,
            y=poly_y,
            fill='toself',
            fillcolor='rgba(150,150,250,0.3)',
            line=dict(color='blue', width=1),
            name="Feasible Region",
            hoverinfo='skip'
        )
    )

# Hover tips (transparent markers around midpoints)
n = len(df)
mid_idx = n // 2 if n > 0 else 0
for col in y_select:
    if n == 0 or mid_idx >= n:
        continue
    mid_x = df[x_col].iloc[mid_idx]
    mid_y = df[col].iloc[mid_idx]
    fig.add_trace(
        go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='markers',
            marker=dict(opacity=0, size=12),
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Raise <b>{col}</b> limit to increase output"
        )
    )

# Shadow scenario
if show_shadow and (shadow_df is not None):
    sh_cols = [c for c in shadow_df.columns if c != x_col and pd.api.types.is_numeric_dtype(shadow_df[c])]
    for col in sh_cols:
        fig.add_trace(
            go.Scatter(
                x=shadow_df[x_col],
                y=shadow_df[col],
                name=f"{col} (Shadow)",
                mode='lines',
                line=dict(width=2, dash='dash'),
                hoverinfo='text',
                hovertext=f"{col} increase costs €5,000"
            )
        )

# Layout tweaks
fig.update_layout(
    template="plotly_white",
    hovermode="closest",
    legend_title_text="",
    margin=dict(l=20, r=20, t=40, b=10),
    xaxis_title=str(x_col),
    yaxis_title="Constraint value"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Bottleneck table (optional)
# -----------------------------
if show_bottleneck:
    bneck, x_sorted = active_bottleneck(df, x_col, y_select, sense=sense)
    if not bneck.empty:
        tbl = pd.DataFrame({x_col: x_sorted, "active_bottleneck": bneck})
        st.dataframe(tbl, use_container_width=True, height=260)
    else:
        st.info("Bottleneck table not available (insufficient/invalid data).")

# -----------------------------
# Tips
# -----------------------------
with st.expander("How to structure your Excel"):
    st.markdown(
        """
- **Main sheet:** one column for X (e.g., `S1`) + one column per constraint (numeric).
- **Optional `shadows` sheet:** same structure for alternative scenario (costly upgrades, etc.).
- **Sense:**  
  - `≤` → Feasible region is between *baseline* and the **minimum** of constraints.  
  - `≥` → Feasible region is between the **maximum** of constraints and *baseline*.
- **Baseline:** set to 0 if your lower bound is zero; set to a high number for `≥` cases if needed.
        """
    )
