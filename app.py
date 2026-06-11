# IMPORTS
# Dash builds the web app ,pandas reads our data file ,plotly makes the chart
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px

# Read the CSV file into a table we can work with
df = pd.read_csv('yield.csv')

# CREATE APP
app = Dash()

app.layout = html.Div([

    dcc.Dropdown(id='country', options=sorted(df['Area'].unique()), value=''),
    dcc.RangeSlider(id='year', min=1961, max=2016, value=[1961, 2010]),
    dcc.Graph(id='graph')
])
# This runs every time the user changes the dropdown or slider
@app.callback(Output('graph', 'figure'), Input('country', 'value'), Input('year', 'value'))
def update(country, year):

    # Keep only rows that match the selected country and year range
    f = df[(df['Area'] == country) & (df['Year'].between(year[0], year[1]))]

    # Draw the line chart
    return px.line(f, x='Year', y='Value', color='Item')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)