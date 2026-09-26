# ----------------------
# NEW: MONKEY-PATCH FOR PYTHON 3.14+
# This is a hack to fix the 'pkgutil.find_loader' error.
# We are manually adding the old function back and pointing it
# to the new function that Dash should be using.
import importlib.util
import pkgutil

if not hasattr(pkgutil, 'find_loader'):
    pkgutil.find_loader = importlib.util.find_spec
# END OF PATCH
# ----------------------

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import random

# ----------------------
# Load and Process Real Data
# ----------------------
try:
    # 1. Load the real CSV file
    # Make sure you have renamed 'master.csv' to 'suicide_data.csv'
    df_raw = pd.read_csv("suicide_data.csv")

    # 2. Rename columns to match our old code and be more standard
    df_raw.rename(columns={
        'suicides_no': 'Suicides',
        'population': 'Population',
        'country': 'Country',
        'year': 'Year',
        'suicides/100k pop': 'Rate_per_100k'
    }, inplace=True)

    # 3. Process the data
    # The raw file has data broken down by 'sex' and 'age'.
    # We need to group it to get ONE row per country/year.
    df_processed = df_raw.groupby(['Country', 'Year'])[['Suicides', 'Population']].sum().reset_index()

    # 4. Recalculate the rate based on the grouped data
    df_processed['Rate_per_100k'] = (df_processed['Suicides'] / df_processed['Population']) * 100_000
    
    # 5. Clean up any bad data (where population might be 0, creating 'inf' values)
    df_processed.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df_processed.dropna(inplace=True)

    # 6. Get the list of years for our dropdown
    all_years = sorted(df_processed['Year'].unique())

except FileNotFoundError:
    print("="*50)
    print("ERROR: 'suicide_data.csv' not found.")
    print("Please download it from Kaggle, unzip it, rename 'master.csv' to 'suicide_data.csv',")
    print("and place it in the same folder as this app.py file.")
    print("Link: https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016")
    print("="*50)
    exit() # Stop the app if the file isn't found
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()


# ----------------------
# Dashboard App
# ----------------------
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([

    html.H1("🌍 Real-Time Suicide Rate Dashboard (1985-2016)", style={"textAlign": "center"}),

    # Add a dropdown to select the year
    html.Div([
        html.Label("Select Year:", style={'fontSize': '20px', 'margin-right': '10px'}),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': year, 'value': year} for year in all_years],
            value=all_years[-1],  # Default to the latest year in the dataset
            clearable=False,
            style={'width': '200px', 'display': 'inline-block'}
        )
    ], style={'textAlign': 'center', 'padding': '20px'}),


    # KPI Metrics container (will be filled by the callback)
    html.Div(
        id='kpi-container',
        style={"display": "flex", "justifyContent": "space-around", "flexWrap": "wrap"}
    ),

    # Graphs (will be filled by thecallback)
    dcc.Graph(id="world-map"),
    dcc.Graph(id="bar-chart"),
    dcc.Graph(id="scatter"),

    # Interval timer to trigger "live" updates
    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,  # 5000 milliseconds = 5 seconds
        n_intervals=0
    )
])


# ----------------------
# The Callback to update everything
# ----------------------
@app.callback(
    [Output('kpi-container', 'children'),
     Output('world-map', 'figure'),
     Output('bar-chart', 'figure'),
     Output('scatter', 'figure')],
    # It takes inputs from both the dropdown and the timer
    [Input('year-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(selected_year, n):
    
    # 1. Get the real data for the selected year
    # We must use .copy() to avoid changing the original dataframe
    dff = df_processed[df_processed['Year'] == selected_year].copy()

    # 2. Apply the "live" simulation
    # We add a small random number to simulate new data "coming in"
    # This loop goes through every country and adds a small random number
    for i in dff.index:
        dff.at[i, 'Suicides'] += random.randint(0, 5)

    # 3. Re-calculate rate after simulation
    dff["Rate_per_100k"] = (dff["Suicides"] / dff["Population"]) * 100_000

    # 4. Create the new KPI components
    kpis = [
        html.Div(f"Total Suicides ({selected_year}): {dff['Suicides'].sum():,}", style={"fontSize": "20px", "margin": "10px"}),
        html.Div(f"Highest Country: {dff.loc[dff['Suicides'].idxmax(), 'Country']}", style={"fontSize": "20px", "margin": "10px"}),
        html.Div(f"Highest Rate: {dff.loc[dff['Rate_per_100k'].idxmax(), 'Country']} ({dff['Rate_per_100k'].max():.2f} per 100k)", style={"fontSize": "20px", "margin": "10px"}),
    ]

    # 5. Re-create all the figures
    fig_map = px.choropleth(
        dff,
        locations="Country",
        locationmode="country names",
        color="Rate_per_100k",
        hover_name="Country",
        hover_data={"Suicides": True, "Population": True, "Rate_per_100k": ':.2f'},
        title=f"Suicide Rate per 100k Population ({selected_year})",
        color_continuous_scale="Reds"
    )
    
    # Bar chart for the top 15 countries by suicide count
    fig_bar = px.bar(
        dff.nlargest(15, 'Suicides').sort_values('Suicides', ascending=False),
        x="Country",
        y="Suicides",
        title=f"Total Suicide Count by Country (Top 15 for {selected_year})"
    )
    
    # Scatter plot
    fig_scatter = px.scatter(
        dff,
        x="Population",
        y="Rate_per_100k",
        size="Suicides",
        hover_name="Country",
        title=f"Suicide Rate vs Population ({selected_year})",
        log_x=True, # Use a log scale for population, as it varies widely
        size_max=60 # Control the max bubble size
    )

    # 6. Return all the new components to the layout
    return kpis, fig_map, fig_bar, fig_scatter


# Run the app
if __name__ == "__main__":
    app.run(debug=True)

