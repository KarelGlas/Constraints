import json
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="2D Capacity Map", layout="wide")

# ---------- Defaults ----------
DEFAULT_JSON = {
    "title": "Production Capacity Map",
    "s1_min": 0, "s1_max": 100, "s1_step": 1,
    "s2_min": 0, "s2_max": 100,
    "bounds": {"s1_max": 100, "s2_max": 100},           # hard caps (optional)
    "constraints": [
        {
            "name": "C_LIN",
            "type": "excel_column",                      # or "linear" / "quadratic"
            "expr": None,                                # for formula types
            "shadows": [                                 # incremental relax levels (ΔS2 allowed) with cost
                {"delta": 3.0, "cost": 1000},
                {"delta": 4.0, "cost": 1500},
                {"delta": 6.0, "cost": 2500}
            ]
        },
        {
            "name": "C_QUAD",
            "type": "excel_column",
            "expr": None,
            "shadows": [
                {"delta": 2.0, "cost": 900},
                {"delta": 3.0, "cost": 1400},
                {"delta": 5.0, "cost": 2200}
            ]
        }
    ],
    "click_tolerance": 0.02,                             # fraction of S2 range to treat as “binding”
    "grid_density": 101
}

# ---------- Sidebar inputs ----------
st.sidebar.header("Config (JSON)")
config_text = st.sidebar.text_area(
    "Edit JSON config",
    value=json.dumps(DEFAULT_JSON, indent=2),
    height=420
)

# Upload Excel data
st.sidebar.header("Excel constraints")
excel_file = st.sidebar.file_uploader("Upload Excel (S1 + constraint columns)", type=["xlsx", "xls"])

# Session state: shadow levels per constraint
if "levels" not in st.session_state:
    st.session_state.levels = {}  # {name: level_index}

# Parse config
def parse_config(text):
    try:
        cfg = json.loads(text)
        return cfg, None
    except Exception as e:
        return None, f"JSON error: {e}"

cfg, cfg_err = parse_config(config_text)
if cfg_err:
    st.error(cfg_err)
    st.stop()

# Ensure levels keys exist
for c in cfg["constraints"]:
    st.session_state.levels.setdefault(c["name"], 0)

colL, colR = st.columns([0.62, 0.38])


# Validate Excel format
# ---------- Load & validate Excel with robust name matching ----------
def norm(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower().replace("-", "_").replace(" ", "_") if ch.isalnum() or ch=="_")

excel_df = None
excel_map = {}   # maps JSON constraint name -> actual Excel column

if excel_file:
    excel_df = pd.read_excel(excel_file)
    # normalize column index
    excel_df.columns = [c if isinstance(c, str) else str(c) for c in excel_df.columns]
    excel_cols_norm = {norm(c): c for c in excel_df.columns}
    # require S1 (any case/spacing)
    s1_key = None
    for k,v in excel_cols_norm.items():
        if k in ("s1","stream1","x","stream_1"):
            s1_key = v; break
    if not s1_key:
        st.error("Excel must contain column for S1 (accepted headers: S1 / Stream1 / X / Stream_1).")
        st.stop()
    # coerce S1
    excel_df[s1_key] = pd.to_numeric(excel_df[s1_key], errors="coerce")
    excel_df = excel_df.dropna(subset=[s1_key]).sort_values(s1_key).drop_duplicates(s1_key)
    excel_df = excel_df.rename(columns={s1_key: "S1"})

    # build mapping for constraints of type 'excel_column'
    missing = []
    for c in cfg["constraints"]:
        if c.get("type","excel_column") != "excel_column":
            continue
        want = c["name"]
        want_norm = norm(want)
        if want_norm in excel_cols_norm:
            excel_map[want] = excel_cols_norm[want_norm]
        else:
            # heuristic suggestions: contains or startswith
            candidates = [col for nk,col in excel_cols_norm.items() if want_norm in nk or nk in want_norm or nk.startswith(want_norm[:4])]
            if candidates:
                excel_map[want] = candidates[0]  # best guess
                st.warning(f"Mapping '{want}' → '{candidates[0]}' (auto). Verify header names.")
            else:
                excel_map[want] = None
                missing.append(want)

    if missing:
        st.error(f"Missing constraint columns in Excel: {missing}. "
                 "Rename headers to match JSON names or adjust JSON.")
else:
    # fallback synthetic to keep app usable
    excel_s1 = np.arange(cfg["s1_min"], cfg["s1_max"] + cfg["s1_step"], cfg["s1_step"])
    excel_df = pd.DataFrame({"S1": excel_s1})
    for c in cfg["constraints"]:
        if c.get("type","excel_column")=="excel_column":
            excel_df[c["name"]] = np.maximum(cfg["s2_max"] - 0.5*(excel_s1 - cfg["s1_min"]), 0.0)
            excel_map[c["name"]] = c["name"]

# helper: interpolator using mapped column
def interp_from_excel(colname, delta=0.0):
    mapped = excel_map.get(colname)
    if mapped is None or mapped not in excel_df.columns:
        return np.full_like(S1, np.nan)
    x = excel_df["S1"].values
    y = pd.to_numeric(excel_df[mapped], errors="coerce").values
    y = np.nan_to_num(y, nan=np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 0.0)
    return np.interp(S1, x, y) + delta

# Build S1 grid for evaluating constraints and area
S1 = np.linspace(cfg["s1_min"], cfg["s1_max"], cfg.get("grid_density", 101))
S2_min, S2_max = cfg["s2_min"], cfg["s2_max"]

# Interpolator for piecewise columns
def interp_from_excel(colname, delta=0.0):
    y = excel_df[colname].values if colname in excel_df.columns else np.full_like(excel_s1, np.nan, dtype=float)
    return np.interp(S1, excel_s1, np.nan_to_num(y, nan=0.0)) + delta

# Evaluate constraint envelopes at current shadow levels
def constraint_curve(c):
    name = c["name"]
    lvl = st.session_state.levels.get(name, 0)
    add = 0.0
    if lvl > 0 and "shadows" in c and len(c["shadows"]) >= lvl:
        add = sum(sh["delta"] for sh in c["shadows"][:lvl])

    t = c.get("type", "excel_column")
    if t == "excel_column":
        return interp_from_excel(name, add)
    elif t == "linear":
        # expr example: "a + b*S1"  with {"a": 90, "b": -0.5}
        p = c.get("expr", {"a": 90.0, "b": -0.5})
        return p["a"] + p["b"]*S1 + add
    elif t == "quadratic":
        # expr example: "a + b*(S1-c)^2" with {"a": 90, "b": -0.02, "c": 30}
        p = c.get("expr", {"a": 90.0, "b": -0.02, "c": 30.0})
        return p["a"] + p["b"]*(S1 - p["c"])**2 + add
    else:
        return np.full_like(S1, np.nan)

# Compute active envelope and binding index
curves = []
names = []
for c in cfg["constraints"]:
    if c.get("type","excel_column")=="excel_column" and (excel_df is None or c["name"] not in excel_df.columns):
        continue  # skip absent column
    curves.append(constraint_curve(c))
    names.append(c["name"])
if len(curves)==0:
    st.error("No valid constraints available. Check Excel column names and JSON.")
    st.stop()
curves = np.vstack(curves)

# Apply hard bounds on S2 if given
s2_cap = cfg.get("bounds", {}).get("s2_max", S2_max)
envelope = np.minimum(np.nanmin(curves, axis=0), s2_cap)

# Feasible mask on grid (dense)
nS1 = len(S1)
nS2 = int(cfg.get("grid_density", 101))
S2_vals = np.linspace(S2_min, S2_max, nS2)
S1_mesh, S2_mesh = np.meshgrid(S1, S2_vals, indexing="xy")
envelope_mesh = np.tile(envelope, (nS2, 1))
feasible = S2_mesh <= envelope_mesh

# Binding constraint at each S1 (index into names)
binding_idx = np.nanargmin(curves, axis=0)

# Area estimation (Riemann sum)
dx = (S1.max() - S1.min()) / (nS1 - 1) if nS1 > 1 else 0.0
area_feasible = np.trapz(np.clip(envelope, S2_min, S2_max), S1)
area_feasible = float(max(area_feasible, 0.0))

# Plot
with colL:
    st.markdown(f"### {cfg.get('title','Capacity Map')}")
    # Color grid by feasibility & binding constraint (simple 2-color: feasible/infeasible)
    z = feasible.astype(int)  # 1 feasible, 0 infeasible

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=S1, y=S2_vals, z=z, showscale=False, opacity=0.25,
        hovertemplate="S1=%{x:.2f}<br>S2=%{y:.2f}<br>Feasible=%{z}<extra></extra>"
    ))

    # Envelope line
    fig.add_trace(go.Scatter(
        x=S1, y=np.clip(envelope, S2_min, S2_max), mode="lines",
        name="Feasible envelope"
    ))

    # Individual constraints
    for i, c in enumerate(cfg["constraints"]):
        fig.add_trace(go.Scatter(
            x=S1, y=np.clip(curves[i], S2_min, S2_max),
            mode="lines", name=f"{c['name']} (lvl {st.session_state.levels.get(c['name'],0)})",
            hovertemplate=f"{c['name']}: S2_max=%{{y:.2f}}"
        ))

    fig.update_layout(
        xaxis_title="Stream 1 (S1)",
        yaxis_title="Stream 2 (S2)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=650
    )

    clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=650, override_width="100%")

# Determine clicked patch → which constraint to relax?
def identify_clicked_constraint(pt):
    if not pt:
        return None
    x = pt[0].get("x")
    y = pt[0].get("y")
    if x is None or y is None:
        return None
    # Find nearest S1 index
    idx = int(np.clip(np.searchsorted(S1, x), 1, len(S1)-1))
    # If infeasible point: find first violating (lowest envelope) constraint
    # Else: if close to envelope, take binding constraint; otherwise pick binding at that S1
    env = envelope[idx]
    tol = cfg.get("click_tolerance", 0.02) * (S2_max - S2_min)
    if y > env + tol:
        # infeasible → bottleneck is the constraint that defines envelope
        bid = binding_idx[idx]
    else:
        # near/under envelope → choose the binding constraint (closest)
        diffs = curves[:, idx] - y
        diffs = np.where(np.isnan(diffs), np.inf, np.abs(diffs))
        bid = int(np.nanargmin(diffs))
    return names[bid]

chosen = identify_clicked_constraint(clicked)

with colR:
    st.markdown("### Controls")
    st.write(f"**Feasible area (approx.):** {area_feasible:,.1f} (S1·S2 units)")

    if chosen:
        st.success(f"Selected region ≈ constraint: **{chosen}**")

    # Show shadow ladder + next cost
    def next_shadow_info(name):
        c = next((x for x in cfg["constraints"] if x["name"] == name), None)
        if not c:
            return None
        lvl = st.session_state.levels.get(name, 0)
        ladder = c.get("shadows", [])
        if lvl >= len(ladder):
            return {"status": "maxed"}
        nxt = ladder[lvl]
        return {"status": "ok", "next_delta": nxt["delta"], "next_cost": nxt["cost"], "level_after": lvl + 1}

    if chosen:
        info = next_shadow_info(chosen)
        if info and info["status"] == "ok":
            # Rough Δarea estimate: integrate min(new envelope, cap) - old envelope (clipped >=0)
            # Recompute chosen constraint with +delta
            cdef = next(c for c in cfg["constraints"] if c["name"] == chosen)
            lvl_tmp = st.session_state.levels[chosen]
            st.session_state.levels[chosen] = lvl_tmp + 1
            # recompute
            curves_new = []
            for c in cfg["constraints"]:
                curves_new.append(constraint_curve(c))
            curves_new = np.vstack(curves_new)
            env_new = np.minimum(np.nanmin(curves_new, axis=0), cfg.get("bounds", {}).get("s2_max", S2_max))
            dA = np.trapz(np.clip(env_new, S2_min, S2_max) - np.clip(envelope, S2_min, S2_max), S1)
            dA = float(max(dA, 0.0))
            # revert
            st.session_state.levels[chosen] = lvl_tmp

            st.info(f"Next relax for **{chosen}** → +ΔS2={info['next_delta']} at cost={info['next_cost']}.  \nEstimated **Δarea ≈ {dA:,.1f}**")

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Apply relax", use_container_width=True, type="primary", disabled=(chosen is None)):
            if chosen:
                c = next((x for x in cfg["constraints"] if x["name"] == chosen), None)
                lvl = st.session_state.levels.get(chosen, 0)
                if c and lvl < len(c.get("shadows", [])):
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

    # Show current levels table & costs sunk
    rows = []
    total_cost = 0.0
    for c in cfg["constraints"]:
        name = c["name"]
        lvl = st.session_state.levels.get(name, 0)
        cost = 0.0
        for i in range(lvl):
            cost += c.get("shadows", [])[i]["cost"]
        total_cost += cost
        rows.append({"constraint": name, "level": lvl, "cost_spent": cost})
    st.dataframe(pd.DataFrame(rows))
    st.write(f"**Total cost spent:** {total_cost:,.0f}")
