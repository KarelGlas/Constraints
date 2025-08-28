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
    Feasible region = points (x,y) where all constraints are satisfied AND x+y > baseline.
    """
    if len(y_cols) == 0 or x_col not in df.columns:
        return []

    work = df[[x_col] + y_cols].dropna().sort_values(by=x_col)
    if work.empty:
        return []

    x = work[x_col].values
    Y = work[y_cols].values  # shape (n, k)

    if sense == "≤":
        envelope = np.min(Y, axis=1)
        # baseline curve: lowest allowed y so that x+y > baseline
        lower = baseline - x
        feasible_top = np.minimum(envelope, np.max([lower, envelope], axis=0))
        xs = np.concatenate([x, x[::-1]])
        ys = np.concatenate([feasible_top, lower[::-1]])
    else:  # '≥'
        envelope = np.max(Y, axis=1)
        upper = baseline - x  # now acts as cap
        feasible_bottom = np.maximum(envelope, np.min([upper, envelope], axis=0))
        xs = np.concatenate([x, x[::-1]])
        ys = np.concatenate([upper, feasible_bottom[::-1]])

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

# -----------------------------
# Shadow scenarios: selectable + feasible region
# -----------------------------
shadow_scenarios = {}

# 1) Single 'shadows' sheet (optionally with a 'scenario' column)
if any(s.lower() == 'shadows' for s in sheet_names):
    sh = sanitize_numeric(pd.read_excel(uploaded_file, sheet_name='shadows'))
    if 'scenario' in sh.columns:
        for name, g in sh.groupby('scenario'):
            shadow_scenarios[str(name)] = g.drop(columns=['scenario'])
    else:
        shadow_scenarios['shadows'] = sh

# 2) Any sheet starting with shadow_ or shadow:
for s in sheet_names:
    sl = s.lower()
    if sl.startswith('shadow_') or sl.startswith('shadow:'):
        shadow_scenarios[s] = sanitize_numeric(pd.read_excel(uploaded_file, sheet_name=s))

chosen_shadows = []
show_shadow_region = False
if shadow_scenarios:
    chosen_shadows = st.sidebar.multiselect("Shadow scenarios", list(shadow_scenarios.keys()))
    show_shadow_region = st.sidebar.checkbox("Show shadow feasibility region", value=True)


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

# --- plot selected shadow scenarios ---
for name in chosen_shadows:
    sdf = shadow_scenarios[name]
    if x_col not in sdf.columns:
        st.warning(f"Shadow '{name}' missing x-column '{x_col}'. Skipped.")
        continue

    # use only constraints that exist in both main selection and this shadow df
    sh_cols = [c for c in y_select if c in sdf.columns and pd.api.types.is_numeric_dtype(sdf[c])]
    if not sh_cols:
        st.warning(f"Shadow '{name}' has no matching numeric constraint columns. Skipped.")
        continue

    # lines
    for col in sh_cols:
        fig.add_trace(
            go.Scatter(
                x=sdf[x_col], y=sdf[col],
                name=f"{col} [{name}]",
                mode='lines',
                line=dict(width=2, dash='dash'),
                hoverinfo='text',
                hovertext=f"{col} in {name}"
            )
        )

    # shadow feasible region (x + y > baseline)
    if show_shadow_region:
        v = compute_feasible_polygon(sdf, x_col, sh_cols, sense=sense, baseline=baseline)
        if v:
            px, py = zip(*v)
            px, py = list(px) + [px[0]], list(py) + [py[0]]
            fig.add_trace(
                go.Scatter(
                    x=px, y=py,
                    fill='toself',
                    fillcolor='rgba(120,120,120,0.18)',
                    line=dict(color='rgba(90,90,90,0.8)', width=1, dash='dot'),
                    name=f"Feasible [{name}]",
                    hoverinfo='skip'
                )
            )

    # ----- extend hard-axis bounds (if you use the clamp code) -----
    if 'x_candidates' in locals():
        x_candidates.append(pd.to_numeric(sdf[x_col], errors='coerce').max(skipna=True))
    if 'y_candidates' in locals():
        for c in sh_cols:
            y_candidates.append(pd.to_numeric(sdf[c], errors='coerce').max(skipna=True))
        if show_shadow_region and v:
            _, vy = zip(*v)
            y_candidates.append(np.nanmax(vy))

# Layout tweaks
fig.update_layout(
    template="plotly_white",
    hovermode="closest",
    legend_title_text="",
    margin=dict(l=20, r=20, t=40, b=10),
    xaxis=dict(title=str(x_col), rangemode="tozero", range=[0, None]),
    yaxis=dict(title="Constraint value", rangemode="tozero", range=[0, None])
)

# ---- hard nonnegative axes ----
import numpy as np

# make sure X is numeric
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")

x_candidates = [df[x_col].max(skipna=True)]
y_candidates = [pd.to_numeric(df[c], errors="coerce").max(skipna=True) for c in y_select]

if shadow_df is not None and show_shadow:
    if x_col in shadow_df:
        x_candidates.append(pd.to_numeric(shadow_df[x_col], errors="coerce").max(skipna=True))
    for c in [c for c in shadow_df.columns if c != x_col]:
        y_candidates.append(pd.to_numeric(shadow_df[c], errors="coerce").max(skipna=True))

if 'vertices' in locals() and vertices:
    _, poly_y = zip(*vertices)
    y_candidates.append(np.nanmax(poly_y))

x_max = np.nanmax(x_candidates)
y_max = np.nanmax(y_candidates)

# fallback/padding
x_max = 1.0 if not np.isfinite(x_max) or x_max <= 0 else float(x_max) * 1.05
y_max = 1.0 if not np.isfinite(y_max) or y_max <= 0 else float(y_max) * 1.05

fig.update_xaxes(autorange=False, range=[0, x_max], zeroline=True)
fig.update_yaxes(autorange=False, range=[0, y_max], zeroline=True)

st.plotly_chart(fig, use_container_width=True)


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
