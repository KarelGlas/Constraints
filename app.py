import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Debottlenecking Constraint Visualizer")
uploaded_file = st.sidebar.file_uploader("Upload Excel with constraints", type=["xlsx"])
if not uploaded_file:
    st.stop()
xlsx = pd.ExcelFile(uploaded_file)
main_sheet = st.sidebar.selectbox("Select main scenario sheet", [s for s in xlsx.sheet_names if s != 'shadows'])
df = pd.read_excel(uploaded_file, sheet_name=main_sheet)
x_col = 'S1' if 'S1' in df.columns else df.columns[0]  # x-axis column
y_cols = [c for c in df.columns if c != x_col]

# Load shadow scenario if available
shadow_df = None
if 'shadows' in xlsx.sheet_names:
    shadow_df = pd.read_excel(uploaded_file, sheet_name='shadows')

# Build the Plotly figure
fig = go.Figure()
# Plot base constraint lines
for col in y_cols:
    fig.add_trace(go.Scatter(x=df[x_col], y=df[col], name=col, mode='lines',
                             line=dict(width=4)))  # base lines thick solid
# Highlight feasible region
vertices = compute_feasible_polygon(df, x_col, y_cols)  # user-defined function for intersections
if vertices:
    poly_x, poly_y = zip(*vertices)
    poly_x += (poly_x[0],); poly_y += (poly_y[0],)  # close polygon
    fig.add_trace(go.Scatter(x=poly_x, y=poly_y, fill='toself',
                             fillcolor='rgba(150,150,250,0.3)', line=dict(color='blue', width=1),
                             name="Feasible Region", hoverinfo='skip'))

# Add tooltips on lines
for col in y_cols:
    mid_x = df[x_col].iloc[len(df)//2]
    mid_y = df[col].iloc[len(df)//2]
    fig.add_trace(go.Scatter(x=[mid_x], y=[mid_y], mode='markers', 
                             marker=dict(opacity=0), showlegend=False,
                             hoverinfo='text', hovertext=f"Raise {col} limit to increase output"))
    # Disable default hover on the line itself:
    fig.update_traces(hoverinfo='skip', selector=dict(name=col))

# Plot shadow constraints if toggled
if shadow_df is not None:
    if st.sidebar.checkbox("Show shadow constraints scenario"):
        for col in [c for c in shadow_df.columns if c != x_col]:
            fig.add_trace(go.Scatter(x=shadow_df[x_col], y=shadow_df[col],
                                     name=f"{col} (Shadow)", mode='lines',
                                     line=dict(width=3, dash='dash'),
                                     hoverinfo='text', hovertext=f"{col} increase costs €5,000"))

# Final figure adjustments
fig.update_layout(template="plotly_white", hovermode="closest",
                  legend_title_text="", margin=dict(l=20, r=20, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)