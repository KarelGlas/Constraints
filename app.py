import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="2D Capacity Map", layout="wide")

# ---------------- Sidebar: Excel + basic controls ----------------
st.sidebar.header("Excel")
excel_file = st.sidebar.file_uploader(
    "Upload Excel (sheet with S1 + constraint columns; optional sheet 'shadows')",
    type=["xlsx", "xls"]
)

st.sidebar.header("Display / Logic")
click_tol_frac = st.sidebar.slider("Click tolerance (fraction of S2 range)", 0.0, 0.2, 0.02, 0.005)
grid_density = st.sidebar.number_input("Grid density (S1 & S2)", min_value=51, max_value=1001, value=201, step=50)
apply_s2_cap = st.sidebar.checkbox("Apply S2 hard cap from sidebar", value=False)
s2_cap_value = st.sidebar.number_input("S2 hard cap value (if enabled)", value=100.0)
show_heat = st.sidebar.checkbox("Show feasibility heat", value=True)

# ---------------- Helpers ----------------
def norm(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower().replace("-", "_").replace(" ", "_") if ch.isalnum() or ch == "_")

# ---------------- Load Excel ----------------
if not excel_file:
    st.warning("Upload Excel to proceed. See template below.")
    st.stop()

# Try reading main data sheet: pick first sheet
xls = pd.ExcelFile(excel_file)
main_sheet = xls.sheet_names[0]
df = pd.read_excel(xls, sheet_name=main_sheet)

# Normalize column names for matching but keep originals
df.columns = [c if isinstance(c, str) else str(c) for c in df.columns]
cols_norm = {norm(c): c for c in df.columns}

# Find S1 column
s1_col = None
for key in ("s1", "stream1", "x", "stream_1"):
    if key in cols_norm:
        s1_col = cols_norm[key]
        break
if s1_col is None:
    st.error("Missing S1 column. Accepted headers: S1 / Stream1 / X / Stream_1")
    st.stop()

# Coerce and clean S1
df[s1_col] = pd.to_numeric(df[s1_col], errors="coerce")
df = df.dropna(subset=[s1_col]).sort_values(s1_col).drop_duplicates(s1_col)
df = df.rename(columns={s1_col: "S1"})

# Constraint columns = all numeric columns except S1
num_df = df.apply(pd.to_numeric, errors="ignore")
candidate_cols = [c for c in num_df.columns if c != "S1"]
# Keep only columns that are numeric or coercible
constraints = []
for c in candidate_cols:
    col = pd.to_numeric(num_df[c], errors="coerce")
    if col.notna().sum() >= 2:
        constraints.append(c)

if not constraints:
    st.error("No constraint columns detected. Add one or more numeric columns besides S1.")
    st.stop()

# Optional: shadows sheet
shadow_map = {}  # name -> list of dicts [{delta, cost}, ...]
if "shadows" in [s.lower() for s in xls.sheet_names]:
    sheet_name = [s for s in xls.sheet_names if s.lower() == "shadows"][0]
    sh = pd.read_excel(xls, sheet_name=sheet_name)
    expected = {"constraint", "delta", "cost"}
    if expected.issubset({norm(c) for c in sh.columns}):
        # normalize access
        colmap = {norm(c): c for c in sh.columns}
        sh = sh.rename(columns={colmap["constraint"]: "constraint",
                                colmap["delta"]: "delta",
                                colmap["cost"]: "cost"})
        sh["constraint"] = sh["constraint"].astype(str)
        sh["delta"] = pd.to_numeric(sh["delta"], errors="coerce")
        sh["cost"] = pd.to_numeric(sh["cost"], errors="coerce")
        sh = sh.dropna(subset=["constraint", "delta", "cost"])
        # Keep order as given (level 1..n)
        for name, grp in sh.groupby("constraint", sort=False):
            shadow_map[name] = [{"delta": float(r["delta"]), "cost": float(r["cost"])} for _, r in grp.iterrows()]
    else:
        st.warning("Sheet 'shadows' present but missing required columns: constraint, delta, cost. Ignoring.")
else:
    # No shadows sheet -> empty ladders
    shadow_map = {c: [] for c in constraints}

# Ensure every constraint has an entry
for c in constraints:
    shadow_map.setdefault(c, [])

# ---------------- State: current levels ----------------
if "levels" not in st.session_state:
    st.session_state.levels = {c: 0 for c in constraints}
else:
    # sanitize (new uploads)
    for c in constraints:
        st.session_state.levels.setdefault(c, 0)
    # drop levels for removed constraints
    for k in list(st.session_state.levels.keys()):
        if k not in constraints:
            st.session_state.levels.pop(k, None)

# ---------------- Build grids ----------------
S1_raw = df["S1"].values
S1_min, S1_max = float(np.min(S1_raw)), float(np.max(S1_raw))
S1 = np.linspace(S1_min, S1_max, int(grid_density))
# S2 range inferred from data
all_vals = []
for c in constraints:
    all_vals.append(pd.to_numeric(df[c], errors="coerce").values)
all_vals = np.concatenate([v[np.isfinite(v)] for v in all_vals]) if all_vals else np.array([0.0, 1.0])
S2_min, S2_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
if apply_s2_cap:
    s2_cap = float(s2_cap_value)
else:
    s2_cap = None
pad = 0.05 * max(1e-9, S2_max - S2_min)
y_min_viz = S2_min - pad
y_max_viz = S2_max + pad

# Interpolator from Excel
def interp_from_excel(colname: str, delta: float = 0.0):
    x = df["S1"].values
    y = pd.to_numeric(df[colname], errors="coerce").values
    # fallback replace nans with median
    med = np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 0.0
    y = np.nan_to_num(y, nan=med)
    return np.interp(S1, x, y) + delta

# Constraint curve at current level
def constraint_curve(name: str):
    lvl = int(st.session_state.levels.get(name, 0))
    ladder = shadow_map.get(name, [])
    add = sum((ladder[i]["delta"] for i in range(min(lvl, len(ladder)))), start=0.0)
    return interp_from_excel(name, add)

# Evaluate all curves
curves = []
for c in constraints:
    curves.append(constraint_curve(c))
curves = np.vstack(curves)  # shape [n_constraints, len(S1)]

# Envelope (min of curves, optionally cap)
envelope = np.nanmin(curves, axis=0)
if s2_cap is not None:
    envelope = np.minimum(envelope, s2_cap)

# Feasible grid
nS2 = int(grid_density)
S2_vals = np.linspace(S2_min, S2_max, nS2)
S1_mesh, S2_mesh = np.meshgrid(S1, S2_vals, indexing="xy")
envelope_mesh = np.tile(envelope, (nS2, 1))
feasible = S2_mesh <= envelope_mesh

# Binding index per S1
binding_idx = np.nanargmin(curves, axis=0)

# Area (trapz of clipped envelope)
area_feasible = float(np.trapz(np.clip(envelope, S2_min, S2_max), S1))
area_feasible = max(area_feasible, 0.0)

# ---------------- Plot ----------------
colL, colR = st.columns([0.62, 0.38])
with colL:
    st.markdown("### Production Capacity Map (Excel-driven)")

    fig = go.Figure()

    if show_heat:
        z = feasible.astype(int)
        fig.add_trace(go.Heatmap(
            x=S1, y=S2_vals, z=z, showscale=False, opacity=0.25,
            hovertemplate="S1=%{x:.2f}<br>S2=%{y:.2f}<br>Feasible=%{z}<extra></extra>"
        ))

    # Envelope (draw on top, thicker, dashed)
    fig.add_trace(go.Scatter(
        x=S1, y=envelope, mode="lines", name="Feasible envelope",
        line=dict(width=3, dash="dash"), connectgaps=True
    ))

    # Individual constraints (thick, connect gaps)
    for i, c in enumerate(constraints):
        fig.add_trace(go.Scatter(
            x=S1, y=curves[i],
            mode="lines", name=f"{c} (lvl {st.session_state.levels.get(c,0)})",
            line=dict(width=2.5),
            connectgaps=True,
            hovertemplate=f"{c}: S2_max=%{{y:.2f}}"
        ))

    fig.update_layout(
        height=650,
        margin=dict(l=90, r=30, t=30, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        xaxis=dict(title="Stream 1 (S1)", automargin=True, title_standoff=12),
        yaxis=dict(title="Stream 2 (S2)", automargin=True, title_standoff=12, range=[y_min_viz, y_max_viz])
    )

    clicked = plotly_events(
        fig,
        click_event=True, hover_event=False, select_event=False,
        override_height=650, override_width="100%"
    )


# Click → pick constraint
def identify_clicked_constraint(pt):
    if not pt:
        return None
    x = pt[0].get("x"); y = pt[0].get("y")
    if x is None or y is None:
        return None
    idx = int(np.clip(np.searchsorted(S1, x), 1, len(S1) - 1))
    env = envelope[idx]
    tol = click_tol_frac * (S2_max - S2_min)
    if y > env + tol:
        bid = binding_idx[idx]
    else:
        diffs = np.abs(curves[:, idx] - y)
        diffs = np.where(np.isnan(diffs), np.inf, diffs)
        bid = int(np.nanargmin(diffs))
    return constraints[bid]

chosen = identify_clicked_constraint(clicked)

with colR:
    st.markdown("### Controls")
    st.write(f"**Feasible area (approx.):** {area_feasible:,.1f} (S1·S2)")

    if chosen:
        st.success(f"Selected ≈ **{chosen}**")

    # Next shadow info + Δarea estimate
    def next_shadow_info(name):
        ladder = shadow_map.get(name, [])
        lvl = int(st.session_state.levels.get(name, 0))
        if lvl >= len(ladder):
            return {"status": "maxed"}
        nxt = ladder[lvl]
        return {"status": "ok", "next_delta": nxt["delta"], "next_cost": nxt["cost"], "next_level": lvl + 1}

    if chosen:
        info = next_shadow_info(chosen)
        if info and info["status"] == "ok":
            # Temporarily increment chosen level to estimate Δarea
            lvl0 = st.session_state.levels[chosen]
            st.session_state.levels[chosen] = lvl0 + 1
            # recompute curves & envelope
            curves_new = []
            for c in constraints:
                curves_new.append(constraint_curve(c))
            curves_new = np.vstack(curves_new)
            env_new = np.nanmin(curves_new, axis=0)
            if s2_cap is not None:
                env_new = np.minimum(env_new, s2_cap)
            dA = float(np.trapz(np.clip(env_new, S2_min, S2_max) - np.clip(envelope, S2_min, S2_max), S1))
            dA = max(dA, 0.0)
            # revert
            st.session_state.levels[chosen] = lvl0

            st.info(f"Next relax **{chosen}** → ΔS2={info['next_delta']} @ cost={info['next_cost']}.  "
                    f"Estimated **Δarea ≈ {dA:,.1f}**")

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Apply relax", use_container_width=True, type="primary", disabled=(chosen is None)):
            if chosen:
                ladder = shadow_map.get(chosen, [])
                lvl = int(st.session_state.levels.get(chosen, 0))
                if lvl < len(ladder):
                    st.session_state.levels[chosen] = lvl + 1
                    st.experimental_rerun()
    with colB:
        if st.button("Reset levels", use_container_width=True):
            for k in list(st.session_state.levels.keys()):
                st.session_state.levels[k] = 0
            st.experimental_rerun()
    with colC:
        if st.button("Recompute", use_container_width=True):
            st.experimental_rerun()

    # Levels & spent cost
    rows = []
    total_cost = 0.0
    for c in constraints:
        lvl = int(st.session_state.levels.get(c, 0))
        ladder = shadow_map.get(c, [])
        spent = sum((ladder[i]["cost"] for i in range(min(lvl, len(ladder)))), start=0.0)
        total_cost += spent
        rows.append({"constraint": c, "level": lvl, "cost_spent": spent})
    st.dataframe(pd.DataFrame(rows))
    st.write(f"**Total cost spent:** {total_cost:,.0f}")

# ---------------- Template hint ----------------
with st.expander("Excel template (structure)"):
    st.markdown("""
**Main sheet (first sheet, any name):**
- Column **S1** (monotonic or sortable numeric).
- One column per constraint: **C_LIN**, **C_QUAD**, **...** (numeric S2 maxima vs S1).

**Optional `shadows` sheet:**
- Columns: **constraint, delta, cost**.
- One row per level, ordered top-to-bottom = level 1..n.
- `constraint` must exactly match a column name from main sheet.

**Notes:**
- S2 range inferred from data; optional hard cap via sidebar.
- No JSON needed; constraints = Excel columns.
""")
with st.expander("Debug"):
    st.write("Constraints:", constraints)
    st.write("S1 range:", float(S1.min()), float(S1.max()))
    st.write("Y range (viz):", y_min_viz, y_max_viz)
    st.write("First curve sample:", {c: float(curves[i][len(S1)//2]) for i, c in enumerate(constraints)})
