import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# TASK 1 : Dimensionality Reduction

# Task 1a: Load dataset
df = pd.read_csv("cleaned_dataset.csv")
F  = ["Yield_hg_ha", "Rainfall_mm", "Temp", "Pesticides_ton_ha"]

# Task 1b: sklearn PCA, reduce to 2 components
def task1b_pca(X):
    return PCA(2).fit_transform(X)

# TASK 2 — Clustering (sklearn K-means)

# Task 2a: K-means on raw 4D features
def task2a_kmeans_4d(X):
    return KMeans(3, n_init=10, random_state=0).fit_predict(X)

# Task 2b: K-means on the 2 PCA components
def task2b_kmeans_pca(Xp):
    return KMeans(3, n_init=10, random_state=0).fit_predict(Xp)


# TASK 3 : Manual Implementation (PCA + K-means)

# Manual PCA: centre, covariance via matmul, eigendecompose, sort, project
def task3_manual_pca(X):
    X   = X - X.mean(axis=0)
    cov = (X.T @ X) / (len(X) - 1)
    vals, vecs = np.linalg.eig(cov)
    return (X @ vecs[:, np.argsort(vals)[::-1]][:, :2]).real

# Manual K-means: random init, assign by norm, guard empty clusters
def task3_manual_kmeans(X, k=3, seed=42):
    rng     = np.random.default_rng(seed)
    centres = X[rng.choice(len(X), k, replace=False)].copy()
    labels  = np.zeros(len(X), dtype=int)
    for _ in range(300):
        dists      = np.stack([np.linalg.norm(X - c, axis=1) for c in centres])
        new_labels = np.argmin(dists, axis=0)
        if np.all(new_labels == labels): break
        labels  = new_labels
        centres = [X[labels==i].mean(axis=0) if (labels==i).any() else centres[i] for i in range(k)]
    return labels


# TASK 4 : Visualisation (scatter plot setup)

SYMBOLS    = ["circle", "square", "diamond", "cross", "x", "triangle-up"]
CONTINENTS = sorted(df.Continent.unique())
SYM_MAP    = {c: s for c, s in zip(CONTINENTS, SYMBOLS)}

def task4_make_figure(d, pc, labels):
    ax = float(np.abs(pc).max() * 1.05)
    d["Cluster"] = labels.astype(str)
    fig = px.scatter(d, x=pc[:, 0], y=pc[:, 1], color="Cluster", symbol="Continent",
                     symbol_map=SYM_MAP, template="plotly_white")
    fig.update_layout(
        xaxis=dict(title="PC1", range=[-ax, ax], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="PC2", range=[-ax, ax]),
    )
    return fig


# TASK 5 : Filtering (Year slider + Item dropdown, applied before all compute)

app = Dash(__name__)
app.layout = html.Div([
    html.H3("PCA & K-means — Crop Yield"),
    dcc.Slider(id="yr", min=1961, max=2016, step=1, value=2010,
               marks={y: str(y) for y in range(1961, 2017, 10)},
               tooltip={"always_visible": True}),
    dcc.Dropdown(id="itm", options=["all"] + sorted(df.Item.unique()), value="all", clearable=False),
    dcc.Tabs(id="tab", value="t1", children=[
        dcc.Tab(label="Plot 1 — sklearn KMeans on raw 4D features", value="t1"),
        dcc.Tab(label="Plot 2 — sklearn KMeans on PCA components",  value="t2"),
        dcc.Tab(label="Plot 3 — Manual PCA + Manual KMeans",        value="t3"),
    ]),
    dcc.Graph(id="g"),
])

@app.callback(Output("g", "figure"), Input("yr", "value"), Input("itm", "value"), Input("tab", "value"))
def update(yr, itm, tab):

    # Task 5: Filter by Year and Item — applied before all computations, including NaN dropping
    d = df[df.Year == yr] if itm == "all" else df[(df.Year == yr) & (df.Item == itm)]

    # Task 1a: Drop NaNs on the four feature columns
    d = d.dropna(subset=F).reset_index(drop=True)
    if d.empty:
        return go.Figure().add_annotation(
            text="No data available for this selection.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14)
        )

    # Task 1b: Standardise then reduce to 2 PCA components
    X  = StandardScaler().fit_transform(d[F])
    Xp = task1b_pca(X)

    if tab == "t1":
        # Task 2a: K-means on raw 4D features, plot in PCA space
        labels = task2a_kmeans_4d(X)
        pc     = Xp

    elif tab == "t2":
        # Task 2b: K-means on 2 PCA components
        labels = task2b_kmeans_pca(Xp)
        pc     = Xp

    else:
        # Task 3: Manual PCA + Manual K-means 
        pc     = task3_manual_pca(X)
        labels = task3_manual_kmeans(pc)

    # Task 4: Build and return scatter plot
    return task4_make_figure(d, pc, labels)


if __name__ == "__main__":
    app.run(debug=True)