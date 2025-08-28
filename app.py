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
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def compute_feasible_polygon(df: pd.DataFrame, x_col: str, y_cols: list,
                             sense: str = "≤", baseline: float = 0.0):
    """
    Feasible region: y between baseline line (x+y>baseline -> y>baseline-x) and
    the envelope of constraints (min for ≤, max for ≥). Handles NaNs.
    """
    if len(y_cols) == 0 or x_col not in df.columns:
        return []

    work = df[[x_col] + y_cols].sort_values(by=x_col)
    if work.empty:
        return []

    x = pd.to_numeric(work[x_col], errors="coerce").to_numpy()
    # Build Y with smart NaN fill per sense
    Y_cols = []
    for c in y_cols:
        v = pd.to_numeric(work[c], errors="coerce").to_numpy()
        Y_cols.append(v)
    if len(Y_cols) == 0:
        return []
    Y = np.vstack(Y_cols).T  # shape (n, k)

    if sense == "≤":
        # NaN -> +inf so it won't govern the min
        Y_min = np.nanmin(np.where(np.isnan(Y), np.inf, Y), axis=1)
        lower = baseline - x
        # Valid only where both finite and a non-empty band exists
        mask = np.isfinite(x) & np.isfinite(lower) & np.isfinite(Y_min) & (Y_min > lower)
        if not np.any(mask):
            return []
        xf = x[mask]
        top = Y_min[mask]
        bot = lower[mask]
        xs = np.concatenate([xf, xf[::-1]])
        ys = np.concatenate([top, bot[::-1]])
    else:
        # sense '≥'
        # NaN -> -inf so it won't govern the max
        Y_max = np.nanmax(np.where(np.isnan(Y), -np.inf, Y), axis=1)
        upper = baseline - x
        mask = np.isfinite(x) & np.isfinite(upper) & np.isfinite(Y_max) & (Y_max < upper)
        if not np.any(mask):
            return []
        xf = x[mask]
        top = upper[mask]
        bot = Y_max[mask]
        xs = np.concatenate([xf, xf[::-1]])
        ys = np.concatenate([top, bot[::-1]])

    mask2 = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask2], ys[mask2]
    if xs.size < 3:
        return []
    return list(zip(xs.tolist(), ys.tolist()))

def active_bottleneck(df: pd.DataFrame, x_col: str, y_cols: list, sense: str = "≤"):
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
baseline = st.sidebar.number_input("Baseline (other bound)", value=32.0, step=1.0, help="Lower bound for ‘≤’, upper bound for ‘≥’.")
sense = "≤"

# Shadow scenario — column selector
shadow_df = None
shadow_cols_select = []
shadow_interp = None
if any(s.lower() == 'shadows' for s in sheet_names):
    shadow_sheet = [s for s in sheet_names if s.lower() == 'shadows'][0]
    shadow_df_raw = pd.read_excel(uploaded_file, sheet_name=shadow_sheet)
    shadow_df = sanitize_numeric(shadow_df_raw)

    if x_col not in shadow_df.columns:
        st.sidebar.warning(f"'shadows' sheet missing X column '{x_col}'. No shadows plotted.")
    else:
        sh_cols_all = [c for c in shadow_df.columns if c != x_col and pd.api.types.is_numeric_dtype(shadow_df[c])]
        shadow_cols_select = st.sidebar.multiselect(
            "Shadow columns to add",
            options=sh_cols_all,
            default=[],
            help="Pick specific shadow constraints to overlay."
        )
        # Align shadow to main X (key fix)
        base_x = pd.to_numeric(df[x_col], errors="coerce")
        shadow_interp = (
            shadow_df
            .assign(**{x_col: pd.to_numeric(shadow_df[x_col], errors="coerce")})
            .set_index(x_col)
            .reindex(base_x)
            .interpolate(method="values")
            .reset_index(names=x_col)
        )

show_shadow = bool(shadow_df is not None and shadow_interp is not None and len(shadow_cols_select) > 0)

# -----------------------------
# Build figure
# -----------------------------
fig = go.Figure()

# Base constraint lines
for col in y_select:
    combined_output = pd.to_numeric(df[x_col], errors="coerce") + pd.to_numeric(df[col], errors="coerce")
    fig.add_trace(
        go.Scatter(
            x=pd.to_numeric(df[x_col], errors="coerce"),
            y=pd.to_numeric(df[col], errors="coerce"),
            name=col,
            mode='lines',
            line=dict(width=3),
            hovertemplate=(
                "Constraint: <b>%{meta}</b><br>"
                + f"{x_col}=%{{x}}<br>"
                  f"{col}=%{{y}}<br>"
                  "Output (X+Y)=%{customdata}<br>"
                "<extra></extra>"
            ),
            meta=col,
            customdata=combined_output,
        )
    )

# Feasible region polygon (old)
old_vertices = compute_feasible_polygon(df, x_col, y_select, sense=sense, baseline=baseline)
if old_vertices:
    ox, oy = zip(*old_vertices)
    fig.add_trace(
        go.Scatter(
            x=list(ox)+[ox[0]],
            y=list(oy)+[oy[0]],
            fill='toself',
            fillcolor='rgba(150,150,250,0.3)',
            line=dict(color='blue', width=1),
            name="Feasible Region",
            hoverinfo='skip'
        )
    )

# ---- HYBRID DF = base + selected shadows (aligned & interpolated) ----
hybrid = df[[x_col] + y_select].copy()
effective_y = list(y_select)

if show_shadow:
    for s_col in shadow_cols_select:
        series = pd.to_numeric(shadow_interp[s_col], errors="coerce")
        if s_col in hybrid.columns:
            hybrid[s_col] = series
        else:
            hybrid[s_col] = series
            effective_y.append(s_col)
    hybrid = hybrid[[x_col] + effective_y]

# New feasible region (hybrid)
new_vertices = compute_feasible_polygon(hybrid, x_col, effective_y, sense=sense, baseline=baseline)

# Try boolean polygon difference
try:
    from shapely.geometry import Polygon
    have_shapely = True
except Exception:
    have_shapely = False

def _poly_from_vertices(verts):
    if not verts or len(verts) < 3:
        return None
    xs, ys = zip(*verts)
    return Polygon(list(zip(xs, ys)))

if have_shapely and old_vertices and new_vertices:
    poly_old = _poly_from_vertices(old_vertices)
    poly_new = _poly_from_vertices(new_vertices)

    if poly_old and poly_new and (not poly_new.is_empty):
        delta = poly_new.difference(poly_old)

        geoms = list(delta.geoms) if getattr(delta, "geom_type", "") == "MultiPolygon" else [delta]
        for g in geoms:
            if g.is_empty:
                continue
            x_arr, y_arr = g.exterior.xy
            x = np.asarray(x_arr)
            y = np.asarray(y_arr)
            custom_out = (x + y).tolist()
            fig.add_trace(
                go.Scatter(
                    x=x.tolist(),
                    y=y.tolist(),
                    fill='toself',
                    fillcolor='rgba(250,150,150,0.35)',
                    name="Added Feasible Area",
                    customdata=custom_out,
                    hovertemplate=(
                        "Added feasible area<br>"
                        + f"{x_col}=%{{x}}<br>"
                        "Constraint=%{y}<br>"
                        "Output (X+Y)=%{customdata}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        nx, ny = zip(*new_vertices)
        fig.add_trace(
            go.Scatter(
                x=list(nx)+[nx[0]], y=list(ny)+[ny[0]],
                mode='lines',
                line=dict(color='red', width=1),
                name="Feasible (Hybrid) outline",
                hoverinfo='skip'
            )
        )
    else:
        st.warning("Feasible polygon(s) invalid; skipping delta shading.")
else:
    # Fallback: draw old/new
    if new_vertices:
        nx, ny = zip(*new_vertices)
        fig.add_trace(
            go.Scatter(
                x=list(nx)+[nx[0]], y=list(ny)+[ny[0]],
                fill='toself',
                fillcolor='rgba(150,150,250,0.25)',
                line=dict(color='blue', width=1),
                name="Feasible (Hybrid)",
                hoverinfo='skip'
            )
        )
    if old_vertices:
        ox, oy = zip(*old_vertices)
        fig.add_trace(
            go.Scatter(
                x=list(ox)+[ox[0]], y=list(oy)+[oy[0]],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name="Feasible (Old)",
                hoverinfo='skip'
            )
        )
    if not have_shapely:
        st.info("Install shapely to show only the added feasible area: `pip install shapely`.")

# Shadow lines (aligned grid)
if show_shadow:
    for col in shadow_cols_select:
        combined_output_shadow = pd.to_numeric(shadow_interp[x_col], errors="coerce") + pd.to_numeric(shadow_interp[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(shadow_interp[x_col], errors="coerce"),
                y=pd.to_numeric(shadow_interp[col], errors="coerce"),
                name=f"{col} (Shadow)",
                mode='lines',
                line=dict(width=2, dash='dash'),
                customdata=combined_output_shadow,
                hovertemplate=(
                    "Constraint (Shadow): <b>%{meta}</b><br>"
                    + f"{x_col}=%{{x}}<br>"
                      f"{col}=%{{y}}<br>"
                      "Output (X+Y)=%{customdata}<br>"
                    "<extra></extra>"
                ),
                meta=col,
            )
        )

# Layout tweaks
fig.update_layout(
    template="plotly_white",
    hovermode="closest",
    legend_title_text="",
    margin=dict(l=20, r=20, t=40, b=10),
    xaxis=dict(title=str(x_col), rangemode="tozero", range=[0, None]),
    yaxis=dict(title="Constraint value", rangemode="tozero", range=[0, None])
)

# Hard nonnegative axes
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
x_candidates = [df[x_col].max(skipna=True)]
y_candidates = [pd.to_numeric(df[c], errors="coerce").max(skipna=True) for c in y_select]

if show_shadow:
    x_candidates.append(pd.to_numeric(shadow_interp[x_col], errors="coerce").max(skipna=True))
    for c in shadow_cols_select:
        y_candidates.append(pd.to_numeric(shadow_interp[c], errors="coerce").max(skipna=True))

if old_vertices:
    _, poly_y = zip(*old_vertices)
    y_candidates.append(np.nanmax(poly_y))
if new_vertices:
    _, poly_y2 = zip(*new_vertices)
    y_candidates.append(np.nanmax(poly_y2))

x_max = np.nanmax(x_candidates)
y_max = np.nanmax(y_candidates)
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
- **Optional `shadows` sheet:** same structure for alternative scenario.
- **Sense:**  
  - `≤` → Feasible region is between *baseline* and the **minimum** of constraints.  
  - `≥` → Feasible region is between the **maximum** of constraints and *baseline*.
- **Baseline:** set to 0 if your lower bound is zero; set to a high number for `≥` cases if needed.
        """
    )
