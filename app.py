"""
Capacity Envelope Unlock Simulator — Streamlit
----------------------------------------------
Single-file Streamlit app to map 2D capacity space (x=Stream1, y=Stream2),
apply constraints (linear / quadratic / piecewise), and visualize the
"unlockable" areas when relaxing constraints to their next shadow level with costs.

Run locally
-----------
1) pip install streamlit plotly shapely numpy streamlit-plotly-events
2) streamlit run app.py

Deploy (Streamlit Community Cloud)
----------------------------------
- Put this file as app.py in a Git repo + requirements.txt with the packages above.
- Deploy from share.streamlit.io.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.ops import unary_union
from streamlit_plotly_events import plotly_events

# ------------------------------
# Geometry helpers
# ------------------------------

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def polygon_under_function(f, x_min, x_max, y_min, y_max, sense: str = "le", n:int=400) -> Polygon:
    xs = np.linspace(x_min, x_max, n)
    ys = np.array([f(x) for x in xs])
    ys = np.clip(ys, y_min, y_max)
    if sense == "le":
        bottom = [(x_min, y_min), (x_max, y_min)]
        curve = list(zip(xs[::-1], ys[::-1]))
        coords = bottom + curve
    else:
        top = [(x_min, y_max), (x_max, y_max)]
        curve = list(zip(xs, ys))
        coords = curve + top
    return Polygon(coords).intersection(box(x_min, x_max, y_min, y_max))


def polygon_left_of_vertical(x_cut, x_min, x_max, y_min, y_max, sense: str = "le") -> Polygon:
    x0 = clamp(x_cut, x_min, x_max)
    if sense == "le":
        coords = [(x_min, y_min), (x0, y_min), (x0, y_max), (x_min, y_max)]
    else:
        coords = [(x0, y_min), (x_max, y_min), (x_max, y_max), (x0, y_max)]
    return Polygon(coords)


def shapely_to_traces(geom: Polygon | MultiPolygon, name: str, hovertext: str, fillalpha=0.40):
    traces = []
    if geom.is_empty:
        return traces
    geoms = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    for g in geoms:
        x, y = g.exterior.xy
        tr = go.Scatter(x=list(x), y=list(y), mode="lines", fill="toself",
                        name=name, hoverinfo="text", text=hovertext, opacity=fillalpha)
        traces.append(tr)
    return traces


# ------------------------------
# Constraint model
# ------------------------------

@dataclass
class ShadowStep:
    delta: float
    cost: float

@dataclass
class Constraint:
    id: str
    kind: str  # 'linear' | 'quadratic' | 'piecewise'
    sense: str # 'le' or 'ge'
    params: Dict[str, Any]
    shadows: List[ShadowStep] = field(default_factory=list)

    def polygon(self, bounds: Tuple[float,float,float,float], level:int=0, n:int=400) -> Polygon:
        x_min, x_max, y_min, y_max = bounds
        delta = 0.0
        if level > 0 and self.shadows:
            level = min(level, len(self.shadows))
            delta = sum(s.delta for s in self.shadows[:level])
        if self.kind == 'linear':
            a,b,c = self.params['a'], self.params['b'], self.params['c']
            c_shift = c + delta
            if abs(b) < 1e-9:  # vertical
                x_cut = c_shift / a
                sense = self.sense
                if a < 0:
                    sense = 'ge' if self.sense=='le' else 'le'
                return polygon_left_of_vertical(x_cut, x_min, x_max, y_min, y_max, sense)
            def f(x):
                return (c_shift - a*x)/b
            sense = self.sense if b>0 else ('ge' if self.sense=='le' else 'le')
            return polygon_under_function(f, x_min, x_max, y_min, y_max, sense, n)
        elif self.kind == 'quadratic':
            a,b,c = self.params['a'], self.params['b'], self.params['c']
            c_shift = c + delta
            def f(x):
                return a*(x**2) + b*x + c_shift
            return polygon_under_function(f, x_min, x_max, y_min, y_max, self.sense, n)
        elif self.kind == 'piecewise':
            pts = self.params['points']
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts]) + delta
            def f(x):
                return np.interp(x, xs, ys, left=ys[0], right=ys[-1])
            x0, x1 = max(x_min, xs.min()), min(x_max, xs.max())
            return polygon_under_function(f, x0, x1, y_min, y_max, self.sense, n)
        else:
            raise ValueError(f"Unknown constraint kind: {self.kind}")


def intersect_all(polys: List[Polygon]) -> Polygon:
    if not polys:
        return Polygon()
    region = polys[0]
    for p in polys[1:]:
        region = region.intersection(p)
        if region.is_empty:
            return region
    return region


def compute_regions(config: Dict[str,Any], levels: Dict[str,int]):
    bounds = tuple(config['bounds'])
    cons = [Constraint(
        id=c['id'], kind=c['kind'], sense=c['sense'], params=c['params'],
        shadows=[ShadowStep(**s) for s in c.get('shadows',[])]
    ) for c in config['constraints']]

    base_polys = [c.polygon(bounds, levels.get(c.id, 0)) for c in cons]
    base_region = intersect_all(base_polys)

    unlock_traces = []
    unlock_rows = []
    for c in cons:
        cur = levels.get(c.id, 0)
        if len(c.shadows) <= cur:
            continue
        alt_levels = dict(levels)
        alt_levels[c.id] = cur + 1
        alt_polys = [cc.polygon(bounds, alt_levels.get(cc.id,0)) for cc in cons]
        new_region = intersect_all(alt_polys)
        diff = new_region.difference(base_region)
        area_gain = diff.area if not diff.is_empty else 0.0
        step = c.shadows[cur]
        hover = (f"Constraint: {c.id}<br>Next Δ={step.delta}"
                 f"<br>ΔArea={area_gain:.2f}<br>Cost=€{step.cost:,.0f}"
                 f"<br>Area/€={area_gain/step.cost if step.cost else np.nan:.6f}")
        traces = shapely_to_traces(diff, name=f"unlock:{c.id}", hovertext=hover, fillalpha=0.50)
        for t in traces:
            t.customdata = [[c.id, cur, step.delta, step.cost]] * len(t.x)
        unlock_traces.extend(traces)
        unlock_rows.append({
            "constraint": c.id,
            "kind": c.kind,
            "level": cur,
            "delta": step.delta,
            "cost": step.cost,
            "area_gain": area_gain,
            "area_per_cost": (area_gain/step.cost if step.cost else np.nan)
        })
    return base_region, unlock_traces, unlock_rows


def default_config():
    return {
        "bounds": [0, 120, 0, 120],
        "constraints": [
            {"id":"machine_capacity","kind":"linear","sense":"le","params":{"a":1.0,"b":1.0,"c":100.0},
             "shadows":[{"delta":10.0,"cost":15000},{"delta":10.0,"cost":25000}]},
            {"id":"s1_max","kind":"linear","sense":"le","params":{"a":1.0,"b":0.0,"c":90.0},
             "shadows":[{"delta":5.0,"cost":8000}]},
            {"id":"s2_max","kind":"linear","sense":"le","params":{"a":0.0,"b":1.0,"c":80.0},
             "shadows":[{"delta":10.0,"cost":9000}]},
            {"id":"quad_mix","kind":"quadratic","sense":"le","params":{"a":0.01,"b":0.2,"c":10.0},
             "shadows":[{"delta":5.0,"cost":12000}]},
            {"id":"pw_downstream","kind":"piecewise","sense":"le","params":{"points":[[0,60],[40,50],[80,30],[120,25]]},
             "shadows":[{"delta":7.0,"cost":11000},{"delta":5.0,"cost":18000}]}
        ]
    }


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Capacity Envelope — Streamlit", layout="wide")
st.title("Capacity Envelope Unlock Simulator")

colL, colR = st.columns([2.2, 1.3], gap="large")

with colR:
    st.subheader("Config (JSON)")
    cfg_text = st.text_area("", value=json.dumps(default_config(), indent=2), height=300)
    cfg = json.loads(cfg_text)

    # Init levels
    if "levels" not in st.session_state or set(st.session_state["levels"].keys()) != {c['id'] for c in cfg['constraints']}:
        st.session_state["levels"] = {c['id']: 0 for c in cfg['constraints']}

    st.markdown("**Current relax levels**")
    st.code("\n".join([f"- {k}: {v}" for k,v in st.session_state["levels"].items()]))

with colL:
    st.subheader("Capacity Map")
    base_region, unlock_traces, rows = compute_regions(cfg, st.session_state["levels"])

    # Build figure
    x_min, x_max, y_min, y_max = cfg['bounds']
    fig = go.Figure()
    base_traces = shapely_to_traces(base_region, name='feasible_now', hovertext='Feasible region (current)', fillalpha=0.35)
    for t in base_traces:
        t.fillcolor = 'rgba(33,150,243,0.35)'
        fig.add_trace(t)
    palette = [
        'rgba(244,67,54,0.45)','rgba(76,175,80,0.45)','rgba(255,193,7,0.45)',
        'rgba(156,39,176,0.45)','rgba(0,150,136,0.45)','rgba(121,85,72,0.45)'
    ]
    for i, t in enumerate(unlock_traces):
        t.fillcolor = palette[i % len(palette)]
        fig.add_trace(t)
    fig.update_layout(
        xaxis_title='Stream 1 (x)', yaxis_title='Stream 2 (y)',
        xaxis=dict(range=[x_min, x_max], zeroline=False, scaleratio=1),
        yaxis=dict(range=[y_min, y_max], scaleanchor='x', zeroline=False),
        margin=dict(l=10,r=10,t=10,b=10), showlegend=False, dragmode='pan')

    selected = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="plt")

    clicked_info = None
    if selected:
        p = selected[0]
        cd = p.get("customdata")
        if cd and isinstance(cd, list) and len(cd) >= 4:
            cid, lev, delta, cost = cd
            clicked_info = {"constraint": cid, "level": int(lev), "delta": float(delta), "cost": float(cost)}

    if clicked_info:
        st.info(f"Selected: {clicked_info['constraint']} | next Δ={clicked_info['delta']} | cost=€{int(clicked_info['cost']):,}")
    else:
        st.caption("Click a colored patch to inspect an unlock option.")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Apply selected relax", disabled=(clicked_info is None)):
            cid = clicked_info['constraint']
            # Guard against exceeding available shadows
            cons_map = {c['id']: c for c in cfg['constraints']}
            max_level = len(cons_map[cid].get('shadows',[]))
            cur = st.session_state['levels'].get(cid, 0)
            if cur < max_level:
                st.session_state['levels'][cid] = cur + 1
                st.experimental_rerun()
    with c2:
        if st.button("Reset levels"):
            st.session_state['levels'] = {c['id']: 0 for c in cfg['constraints']}
            st.experimental_rerun()

with colR:
    st.subheader("Unlock options (next step)")
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("area_per_cost", ascending=False)
        st.dataframe(df, use_container_width=True, height=260)
    else:
        st.write("No further relax steps available.")

st.caption("Areas are in squared stream-units. Piecewise/quadratic are polyline-approximated; adjust sampling in code if needed.")
