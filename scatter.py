"""
Scatter Plot & Optional t-SNE Tasks for Crop Data Dashboard
Author: [Solomon Haile Dereje]
"""
import pandas as pd, numpy as np, pycountry
import plotly.express as px, plotly.graph_objects as go
from sklearn.manifold import TSNE; from sklearn.preprocessing import StandardScaler
from dash import Dash, dcc, html, Input, Output, State, ctx

df = pd.read_csv("cleaned_dataset.csv").dropna()

MANUAL_ISO = {
    "Russia":"RUS","Turkey":"TUR","Ivory Coast":"CIV","Czechia":"CZE",
    "North Korea":"PRK","South Korea":"KOR","Iran":"IRN","Syria":"SYR",
    "Taiwan":"TWN","Vietnam":"VNM","Bolivia":"BOL","Tanzania":"TZA",
    "Laos":"LAO","Democratic Republic of the Congo":"COD","Republic of the Congo":"COG",
    "Réunion":"REU","Hong Kong":"HKG","Micronesia":"FSM","Brunei":"BRN",
    "Moldova":"MDA","Eswatini":"SWZ","North Macedonia":"MKD","Cabo Verde":"CPV",
    "French Guiana":"GUF","Wallis and Futuna Islands":"WLF","Serbia and Montenegro":"SCG",
}

def to_iso3(name):
    if name in MANUAL_ISO: return MANUAL_ISO[name]
    try: return pycountry.countries.lookup(name).alpha_3
    except: return None
df["ISO3"] = df["Area"].map(to_iso3)
df = df.dropna(subset=["ISO3"])

ATTR_INFO = [("Yield_hg_ha","Yield (hg/ha)"),("Rainfall_mm","Rainfall (mm)"),("Temp","Temp (°C)"),("Pesticides_ton_ha","Pesticides (ton/ha)")]
ATTRS, LABELS = zip(*ATTR_INFO); ATTRS, LABELS = list(ATTRS), list(LABELS)
CONT_COLORS = dict(zip(["Africa","Asia","Europe","North America","Oceania","South America"],
                       ["#e6194B","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4"]))

app = Dash(__name__)
app.layout = html.Div([
    html.H2("🌾 Crop Data Dashboard"),
    # Task 1 — Filtering controls: Year slider (1b), Item dropdown (1c), Attribute dropdown (1d)
    html.Div([
        dcc.Slider(id="yr", min=df.Year.min(), max=df.Year.max(), step=1, value=2000,
                   marks={int(y): str(y) for y in df.Year.unique() if y % 10 == 0},
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Dropdown(id="itm", options=sorted(df.Item.unique()), value=df.Item.iloc[0], clearable=False),
        dcc.Dropdown(id="atr", options=[{"label": l, "value": a} for a, l in zip(ATTRS, LABELS)],
                     value=ATTRS[0], clearable=False),
    ], style={"display": "flex", "gap": "20px", "alignItems": "center", "marginBottom": "12px"}),
    html.Div([
        html.Div([dcc.Graph(id="map"), dcc.Store(id="sel", data=[])], style={"flex": 6}),
        html.Div(dcc.Graph(id="sct"), style={"flex": 4}),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),
    html.Div([
        dcc.Graph(id="par", style={"flex": 1}),
        dcc.Graph(id="tsn", config={"modeBarButtonsToAdd": ["lasso2d"]}, style={"flex": 1}),
    ], style={"display": "flex", "gap": "12px"}),
])

def subset(yr, itm):
    return df[(df.Year == yr) & (df.Item == itm)].dropna(subset=ATTRS)

def highlight_colors(areas, sel, full, dim="rgba(180,180,180,0.2)"):
    if not sel: return [full] * len(areas)
    return [full if a in sel else dim for a in areas]

@app.callback(Output("sel", "data"),
              Input("map", "clickData"), Input("tsn", "selectedData"),
              State("sel", "data"), State("yr", "value"), State("itm", "value"),
              prevent_initial_call=True)
def update_selection(map_click, tsne_sel, sel, yr, itm):
    """
    Task 2c-d / Optional Task: Manages shared selection state.
    Map clicks toggle individual countries (Task 2c).
    t-SNE lasso selections are merged with existing map selections (Optional Task).
    Ends: returns updated list of selected area names stored in dcc.Store.
    """
    sel = sel or []
    if ctx.triggered_id == "map" and map_click:
        iso = map_click["points"][0].get("location")
        rows = df[df.ISO3 == iso]["Area"]
        if rows.empty: return sel
        a = rows.iloc[0]
        return [x for x in sel if x != a] if a in sel else sel + [a]
    if ctx.triggered_id == "tsn" and tsne_sel:
        sub = subset(yr, itm).reset_index(drop=True)
        new = [p["text"] for p in tsne_sel["points"] if "text" in p]

        return list(set(sel + new))
    return sel

@app.callback(Output("map", "figure"),
              Input("yr", "value"), Input("itm", "value"),
              Input("atr", "value"), Input("sel", "data"))
def update_map(yr, itm, atr, sel):
    """
    Task 2: Choropleth map filtered by Year (Task 1b), Item (Task 1c), Attribute (Task 1d).
    Task 2a: Area names converted to ISO-3 via to_iso3() using pycountry + MANUAL_ISO.
    Task 2b: Countries coloured by selected attribute with a colorbar legend.
    Task 2d: When areas are selected, unselected countries are dimmed (opacity=0.2);
             a second highlighted trace with a red border is drawn on top for selected ones.
    Ends: returns the choropleth figure.
    """
    d = subset(yr, itm); sel = sel or []
    opacity = [1.0 if (not sel or a in sel) else 0.2 for a in d["Area"]]
    fig = go.Figure(go.Choropleth(
        locations=d.ISO3, z=d[atr], colorscale="Viridis",
        marker_opacity=opacity, marker_line_width=0.5,
        colorbar_title=LABELS[ATTRS.index(atr)],
        customdata=d[["Area"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{z:.2f}<extra></extra>",
    ))
    if sel:
        ds = d[d.Area.isin(sel)]
        fig.add_trace(go.Choropleth(locations=ds.ISO3, z=ds[atr], colorscale="Viridis",
                                    zmin=d[atr].min(), zmax=d[atr].max(), showscale=False,
                                    marker_line_color="#e94560", marker_line_width=2,
                                    customdata=ds[["Area"]],
                                    hovertemplate="<b>%{customdata[0]}</b> ✓<br>%{z:.2f}<extra></extra>"))
    fig.update_layout(geo=dict(showframe=False, projection_type="natural earth"),
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig

@app.callback(Output("sct", "figure"),
              Input("itm", "value"), Input("atr", "value"), Input("sel", "data"))
def update_scatter(itm, atr, sel):
    """
    Task 3: Comparison scatter plot — x=Year, y=selected attribute (Task 3a).
    Task 3b: If no area selected, shows all areas across all years.
    Task 3c: If areas are selected, filters to only those areas across all years.
    Task 3d: Points coloured by Area with a legend (capped at 20 areas for readability).
    Ends: returns the scatter figure.
    """
    sel = sel or []
    d = df[df.Item == itm]
    if sel: d = d[d.Area.isin(sel)]
    d = d[d.Area.isin(d.Area.unique()[:20])]
    fig = px.scatter(d, x="Year", y=atr, color="Area",
                     labels={atr: LABELS[ATTRS.index(atr)]},
                     title=f"{itm} — {'selected areas' if sel else 'all areas'}")
    return fig

@app.callback(Output("par", "figure"),
              Input("yr", "value"), Input("itm", "value"), Input("sel", "data"))
def update_parallel(yr, itm, sel):
    """
    Task 4: Parallel coordinates plot filtered by Year and Item (Task 4a).
    Task 4b: Four parallel axes — Yield, Rainfall, Temp, Pesticides (Task 4b).
    Task 4c: Selected areas drawn in red (#e94560); unselected in semi-transparent blue (Task 4c).
    Ends: returns the parallel coordinates figure.
    """
    sel = sel or []
    d = subset(yr, itm).copy()
    d["_color"] = d.Area.isin(sel).astype(int)
    dims = [dict(range=[d[a].min(), d[a].max()], label=LABELS[i], values=d[a])
            for i, a in enumerate(ATTRS)]
    fig = go.Figure(go.Parcoords(
        line=dict(color=d["_color"], colorscale=[[0, "rgba(100,149,237,0.3)"], [1, "#e94560"]],
                  cmin=0, cmax=1, showscale=False),
        dimensions=dims,
    ))
    fig.update_layout(margin=dict(l=90, r=60, t=40, b=20), title=f"{itm} · {yr}")
    return fig

@app.callback(Output("tsn", "figure"),
              Input("yr", "value"), Input("itm", "value"), Input("sel", "data"))
def update_tsne(yr, itm, sel):
    """
    Optional Task: t-SNE dimensionality reduction scatter plot.
    Optional a: Data filtered by Year and Item.
    Optional b: scikit-learn TSNE reduces the 4 numerical attributes to 2 components.
    Optional c: Points coloured by Continent with a legend.
    Optional d: Selected areas shown at full opacity with a red outline; others dimmed.
                Lasso selection feeds back into the shared selection store via update_selection.
    Ends: returns the t-SNE scatter figure.
    """
    sel = sel or []
    d = subset(yr, itm).copy().reset_index(drop=True)
    if len(d) < 5: return go.Figure()
    try:
        coords = TSNE(2, perplexity=min(30, len(d)-1), random_state=42, max_iter=500).fit_transform(
            StandardScaler().fit_transform(d[ATTRS]))
    except TypeError:
        coords = TSNE(2, perplexity=min(30, len(d)-1), random_state=42, n_iter=500).fit_transform(
            StandardScaler().fit_transform(d[ATTRS]))
    d["tx"], d["ty"] = coords[:, 0], coords[:, 1]
    fig = go.Figure()
    for cont, g in d.groupby("Continent"):
        colors = highlight_colors(g.Area.tolist(), sel, full=CONT_COLORS.get(cont, "#aaa"),
                                  dim="rgba(180,180,180,0.2)")
        fig.add_trace(go.Scatter(x=g.tx, y=g.ty, mode="markers", name=cont, text=g.Area,
                                 marker=dict(color=colors, size=9,
                                             line=dict(color=["#e94560" if a in sel else "rgba(0,0,0,0)"
                                                               for a in g.Area], width=2)),
                                 hovertemplate="<b>%{text}</b><extra></extra>"))
    fig.update_layout(dragmode="lasso", xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
                      title=f"t-SNE · {itm} · {yr}", margin=dict(l=40, r=20, t=40, b=30))
    return fig

if __name__ == "__main__": app.run(debug=False, host="0.0.0.0", port=8050)