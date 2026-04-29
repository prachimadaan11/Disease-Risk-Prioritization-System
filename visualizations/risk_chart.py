# ============================================================
# Risk Assessment Chart Visualization
# Part of: Disease Risk Prioritization System
# ============================================================

import plotly.graph_objects as go
import pandas as pd

# Data
data = [
    {
        "Patient_Type": "High Risk",
        "ML_High_Risk_Prob": 0.958,
        "Fuzzy_Score": 0.8,
        "Age_Factor": 0.8,
        "Clinical_Score": 0.51,
        "Lifestyle_Score": 0.83,
        "Overall_Risk_Estimate": 0.821
    },
    {
        "Patient_Type": "Medium Risk", 
        "ML_High_Risk_Prob": 0.333,
        "Fuzzy_Score": 0.5,
        "Age_Factor": 0.425,
        "Clinical_Score": 0.155,
        "Lifestyle_Score": 0.15,
        "Overall_Risk_Estimate": 0.365
    },
    {
        "Patient_Type": "Low Risk",
        "ML_High_Risk_Prob": 0.000,
        "Fuzzy_Score": 0.2,
        "Age_Factor": 0.2,
        "Clinical_Score": 0.0,
        "Lifestyle_Score": 0.06,
        "Overall_Risk_Estimate": 0.106
    }
]

df = pd.DataFrame(data)

# Create the figure
fig = go.Figure()

# Define colors using more distinct brand colors
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#D2BA4C', '#964325', '#944454']

# Risk components and their display names (keeping under 15 characters)
risk_components = [
    ('ML_High_Risk_Prob', 'ML Risk Prob'),
    ('Fuzzy_Score', 'Fuzzy Score'),
    ('Age_Factor', 'Age Factor'),
    ('Clinical_Score', 'Clinical Score'),
    ('Lifestyle_Score', 'Lifestyle Score'),
    ('Overall_Risk_Estimate', 'Overall Risk')
]

# Add bars for each risk component
for i, (col, name) in enumerate(risk_components):
    fig.add_trace(go.Bar(
        name=name,
        x=df['Patient_Type'],
        y=df[col],
        marker_color=colors[i],
        text=[f'{val:.2f}' if val > 0 else '' for val in df[col]],
        textposition='outside',
        textfont=dict(size=10)
    ))

# Update traces with cliponaxis=False for bar charts
fig.update_traces(cliponaxis=False)

# Update layout with better spacing and formatting
fig.update_layout(
    title='Risk Assessment by Patient Profile',
    xaxis_title='Patient Type',
    yaxis_title='Risk Score',
    barmode='group',
    bargap=0.2,
    bargroupgap=0.1,
    yaxis=dict(
        range=[0, 1.1],
        tickmode='linear',
        tick0=0,
        dtick=0.2,
        tickformat='.1f'
    )
)

# Center legend under title since we have 6 items (more than 5, so keep default position)
# Keep default legend position for 6 items

# Save as PNG and SVG
fig.write_image('assets/risk_assessment_chart.png')
fig.write_image('assets/risk_assessment_chart.svg', format='svg')