#Task 2 
from dash import Dash, html, dcc, Input, Output
import pandas as pd, plotly.express as px

df = pd.read_csv('merged_dataset.csv')
df['Rainfall_mm'] = pd.to_numeric(df['Rainfall_mm'], errors='coerce')
df = df.dropna()
app = Dash()

app.layout = html.Div([
    dcc.Dropdown(id='cont', options=sorted(df['Continent'].unique()), value='Europe'),
    dcc.Dropdown(id='attr', options=['Yield_hg_ha','Rainfall_mm','Temp','Pesticides_ton_ha'], value='Rainfall_mm'),
    dcc.Dropdown(id='stat', options=['mean','median','var'], value='mean'),
    dcc.Graph(id='line'), dcc.Graph(id='box')
])

@app.callback(Output('line','figure'), Output('box','figure'),
              Input('cont','value'), Input('attr','value'), Input('stat','value'))
def update(cont, attr, stat):
    f = df[df['Continent']==cont]
    color = 'Item' if attr=='Yield_hg_ha' else None
    grp = f.groupby(['Year'] + (['Item'] if color else []))[attr].agg(stat).reset_index()
    return (px.line(grp, x='Year', y=attr, color=color, title=f'{stat} of {attr} in {cont}'),
            px.box(f, x='Area', y=attr, color=color, title=f'Distribution of {attr} in {cont}'))

if __name__ == '__main__':
    app.run(debug=True)