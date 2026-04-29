# ============================================================
# Disease Risk System Flowchart Visualization
# Part of: Disease Risk Prioritization System
# ============================================================

import plotly.graph_objects as go

# Create a flowchart using Plotly with boxes and arrows
fig = go.Figure()

# Define positions for each component (x, y coordinates)
positions = {
    # Input layer
    'Patient Data': (0.5, 9),
    
    # Processing layer
    'ML Models': (0.2, 7),
    'Fuzzy Logic': (0.5, 7), 
    'Risk Factors': (0.8, 7),
    
    # Sub-components
    'RF/GB/LR/SVM': (0.2, 5.5),
    'Age/BMI/BP Sets': (0.5, 5.5),
    'Age/Clinical/Life': (0.8, 5.5),
    
    # TODIM Algorithm
    'TODIM Algorithm': (0.5, 4),
    'Decision Matrix': (0.2, 2.5),
    'Weight Calc': (0.4, 2.5),
    'Dominance Calc': (0.6, 2.5),
    'Prospect Theory': (0.8, 2.5),
    
    # Output layer
    'Risk Priority': (0.2, 1),
    'Recommendations': (0.5, 1),
    'Trust Scores': (0.8, 1)
}

# Define colors for different layers
colors = {
    'input': '#B3E5FC',      # Light cyan
    'processing': '#FFCDD2',  # Light red
    'algorithm': '#A5D6A7',   # Light green
    'output': '#FFF59D'       # Light yellow
}

# Add boxes for each component - BIGGER BOXES
box_width = 0.15
box_height = 0.6

for name, (x, y) in positions.items():
    # Determine color based on component type
    if name == 'Patient Data':
        color = colors['input']
    elif name in ['ML Models', 'Fuzzy Logic', 'Risk Factors', 'RF/GB/LR/SVM', 'Age/BMI/BP Sets', 'Age/Clinical/Life']:
        color = colors['processing']
    elif name in ['TODIM Algorithm', 'Decision Matrix', 'Weight Calc', 'Dominance Calc', 'Prospect Theory']:
        color = colors['algorithm']
    else:
        color = colors['output']
    
    # Add rectangle
    fig.add_shape(
        type="rect",
        x0=x-box_width/2, y0=y-box_height/2,
        x1=x+box_width/2, y1=y+box_height/2,
        fillcolor=color,
        line=dict(color="#333333", width=2)
    )
    
    # Add text
    fig.add_annotation(
        x=x, y=y,
        text=name.replace('/', '<br>'),
        showarrow=False,
        font=dict(size=11, color="#000000", family="Arial Black"),
        align="center"
    )

# Define connections with proper arrow positioning
def add_arrow(fig, start_name, end_name, positions, box_height):
    """Add arrow from bottom of start box to top of end box"""
    start_pos = positions[start_name]
    end_pos = positions[end_name]
    
    # Start from bottom of start box
    start_y = start_pos[1] - box_height/2
    # End at top of end box
    end_y = end_pos[1] + box_height/2
    
    fig.add_annotation(
        x=end_pos[0], 
        y=end_y,
        ax=start_pos[0], 
        ay=start_y,
        xref="x", yref="y",
        axref="x", ayref="y",
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2.5,
        arrowcolor="#333333",
        showarrow=True
    )

# Add all connections
connections = [
    ('Patient Data', 'ML Models'),
    ('Patient Data', 'Fuzzy Logic'),
    ('Patient Data', 'Risk Factors'),
    
    ('ML Models', 'RF/GB/LR/SVM'),
    ('Fuzzy Logic', 'Age/BMI/BP Sets'),
    ('Risk Factors', 'Age/Clinical/Life'),
    
    ('RF/GB/LR/SVM', 'TODIM Algorithm'),
    ('Age/BMI/BP Sets', 'TODIM Algorithm'),
    ('Age/Clinical/Life', 'TODIM Algorithm'),
    
    ('TODIM Algorithm', 'Decision Matrix'),
    ('TODIM Algorithm', 'Weight Calc'),
    ('TODIM Algorithm', 'Dominance Calc'),
    ('TODIM Algorithm', 'Prospect Theory'),
    
    ('Decision Matrix', 'Risk Priority'),
    ('Prospect Theory', 'Recommendations'),
    ('Prospect Theory', 'Trust Scores')
]

for start, end in connections:
    add_arrow(fig, start, end, positions, box_height)

# Configure layout
fig.update_layout(
    title={
        'text': "Disease Risk Prioritization System",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24, 'color': '#333333', 'family': 'Arial Black'}
    },
    showlegend=False,
    width=1200,
    height=1000,
    xaxis=dict(
        range=[-0.1, 1.1],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    yaxis=dict(
        range=[0, 10],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=50, r=50, t=80, b=50)
)

# Show the figure
fig.show()

# Optionally save (requires kaleido: pip install kaleido)
try:
    fig.write_image('assets/disease_risk_flowchart.png', width=1200, height=1000, scale=2)
    fig.write_image('assets/disease_risk_flowchart.svg', format='svg')
    print("Files saved successfully in assets/ folder!")
except Exception as e:
    print(f"Could not save files (install kaleido if needed): {e}")