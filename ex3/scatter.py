import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

"""
This code shows farming data from around the world as a scatter plot.
It lets you filter by year, crop, and continent, and change how
the data looks using color, size, and shape controls on the left side.
"""

# Load data
df = pd.read_csv("cleaned_dataset.csv")
# Removes rows where Yield, Rainfall, or Temp values are missing.
df = df.dropna(subset=["Yield_hg_ha", "Rainfall_mm", "Temp"])
df["Year"] = df["Year"].astype(int)

# ATTRS: the four attributes that can be visually encoded by the user.
# Rainfall (x-axis) and Yield (y-axis) are fixed and not included here.

ATTRS     = ["Temp", "Pesticides_ton_ha", "Item", "Continent"]
ENCODINGS = ["color", "size", "symbol", "none"]
DEFAULTS  = {"Temp": "size", "Pesticides_ton_ha": "none", "Item": "symbol", "Continent": "color"}
YEARS     = sorted(df["Year"].unique())

app = Dash(__name__)

# The sidebar holds all the controls for filtering and encoding.
sidebar = html.Div([
    html.H5("Encodings"),
    # One row of radio buttons per attribute, letting the user
    # assign it a visual role: color, size, symbol, or none.
    *[html.Div([
        html.B(a + ": "),
        dcc.RadioItems(id="r_"+a, options=ENCODINGS, value=DEFAULTS[a], inline=True)
    ]) for a in ATTRS],
    html.Hr(),

    # Filters the data to only show rows from the selected year.
    html.B("Year"),
    dcc.Slider(id="year", min=YEARS[0], max=YEARS[-1], step=1, value=YEARS[-1],
               marks={int(y): str(y) for y in YEARS[::10]},
               updatemode="drag"),

    # Filters the data to only show the selected crop types.
    html.B("Crop"),
    dcc.Dropdown(id="item", options=sorted(df["Item"].unique()), value=sorted(df["Item"].unique()), multi=True),

    # Filters the data to only show the selected continents.
    html.B("Continent"),
    dcc.Dropdown(id="cont", options=sorted(df["Continent"].unique()), value=sorted(df["Continent"].unique()), multi=True),

    # One range slider per numeric column. Filters the data to only include
    # rows whose values fall within the selected min/max range.
    *[html.Div([html.B(c), dcc.RangeSlider(id="rs_"+c, min=float(df[c].min()), max=float(df[c].max()),
        value=[float(df[c].min()), float(df[c].max())], marks={}, tooltip={"always_visible": False})])
      for c in ["Yield_hg_ha", "Rainfall_mm", "Temp", "Pesticides_ton_ha"]],
], style={"width": "280px", "padding": "10px", "overflowY": "auto",
          "height": "100vh", "backgroundColor": "#f8f9fa", "borderRight": "1px solid #ddd"})

app.layout = html.Div([
    html.H4("Agricultural Data Explorer", style={"padding": "10px", "borderBottom": "1px solid #ddd"}),
    html.Div([sidebar, dcc.Graph(id="plot", style={"flex": 1, "height": "92vh"})],
             style={"display": "flex"}),
])

@app.callback(
    Output("plot", "figure"),
    [Input("r_"+a, "value") for a in ATTRS] + [
        Input("year", "value"), Input("item", "value"), Input("cont", "value"),
        Input("rs_Yield_hg_ha", "value"), Input("rs_Rainfall_mm", "value"),
        Input("rs_Temp", "value"), Input("rs_Pesticides_ton_ha", "value"),
    ]
)
def update(*args):
    """
This function is called whenever any of the controls change. 
It receives the current values of all controls as arguments, applies the necessary filters to the dataset,
and returns a new scatter plot figure based on the filtered data and selected encodings.
    """

    encs = args[:4]
    year, items, conts = args[4], args[5], args[6]
    yield_r, rain_r, temp_r, pest_r = args[7], args[8], args[9], args[10]

    # Resolves encoding conflicts: if two attributes are assigned the same
    # visual role, the second one is set to "none".
    used, final = set(), {}
    for attr, enc in zip(ATTRS, encs):
        final[attr] = enc if enc != "none" and enc not in used else "none"
        if enc != "none":
            used.add(enc)

    # Applies all active filters to the dataset.
    d = df[(df["Year"] == year) & df["Item"].isin(items) & df["Continent"].isin(conts)]
    d = d[(d["Yield_hg_ha"].between(*yield_r)) & (d["Rainfall_mm"].between(*rain_r)) &
          (d["Temp"].between(*temp_r)) & (d["Pesticides_ton_ha"].between(*pest_r))]

    # Randomly samples 2000 rows if the filtered dataset is too large.
    if len(d) > 2000:
        d = d.sample(2000, random_state=42)

    # Renders the scatter plot with fixed axes and user-defined encodings.
    return px.scatter(d,
        x="Rainfall_mm", y="Yield_hg_ha",
        opacity=0.75, template="plotly_white",
        color  = next((a for a, e in final.items() if e == "color"),  None),
        size   = next((a for a, e in final.items() if e == "size"),   None),
        symbol = next((a for a, e in final.items() if e == "symbol"), None),
        hover_name="Item", title=f"Year {year}")

if __name__ == "__main__":
    # Starts the local development server.
    app.run(debug=True)