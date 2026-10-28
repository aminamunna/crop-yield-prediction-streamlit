"""
Crop Yield Explorer - Streamlit dashboard
Dataset: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset (yield_df.csv)
Run:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Crop Yield Explorer", page_icon="🌾", layout="wide")

NUM_COLS = ["Year", "rainfall_mm", "pesticides_t", "avg_temp"]
CAT_COLS = ["Area", "Item"]


# ----------------------------------------------------------------------------
# 1. DATA PREPARATION
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparing data...")
def prepare(raw: pd.DataFrame):
    log = {}
    df = raw.copy()
    log["Raw rows"] = len(df)

    # drop index artefact
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # tidy names
    df = df.rename(columns={
        "hg/ha_yield": "yield_hg_ha",
        "average_rain_fall_mm_per_year": "rainfall_mm",
        "pesticides_tonnes": "pesticides_t",
    })
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")  # a few files store it as text

    log["Missing values removed"] = int(df.isna().any(axis=1).sum())
    df = df.dropna()

    log["Duplicate rows removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # feature engineering
    df["yield_t_ha"] = df["yield_hg_ha"] / 10_000          # hectograms -> tonnes per hectare
    df["decade"] = (df["Year"] // 10 * 10).astype(str) + "s"
    df["temp_band"] = pd.qcut(df["avg_temp"], 4, labels=["Cool", "Mild", "Warm", "Hot"])
    df["rain_band"] = pd.qcut(df["rainfall_mm"], 4, labels=["Dry", "Moderate", "Wet", "Very wet"])
    df["pest_band"] = pd.qcut(df["pesticides_t"].rank(method="first"), 4,
                              labels=["Low", "Medium", "High", "Very high"])
    log["Clean rows"] = len(df)
    return df, log


@st.cache_data
def load_default():
    for p in ["yield_df.csv", "data/yield_df.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


# ----------------------------------------------------------------------------
# 2. MODEL
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models...")
def train_models(df: pd.DataFrame):
    X = df[CAT_COLS + NUM_COLS]
    y = df["yield_t_ha"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])
    models = {
        "Linear Regression (baseline)": LinearRegression(),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42),
    }
    results, fitted = [], {}
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("model", m)]).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results.append({"Model": name, "R²": r2_score(y_te, pred),
                        "MAE (t/ha)": mean_absolute_error(y_te, pred)})
        fitted[name] = pipe

    # permutation importance on the boosted model (grouped by original column)
    from sklearn.inspection import permutation_importance
    best = fitted["Gradient Boosting"]
    imp = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=42)
    imp_df = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
              .sort_values("importance"))
    return pd.DataFrame(results), fitted, imp_df, (y_te, fitted["Gradient Boosting"].predict(X_te))


# ----------------------------------------------------------------------------
# 3. LOAD DATA
# ----------------------------------------------------------------------------
st.title("🌾 Crop Yield Explorer")
st.caption("FAO + World Bank data · 10 major crops · 1990-2013")

raw = load_default()
if raw is None:
    up = st.file_uploader("Upload yield_df.csv (from the Kaggle dataset)", type="csv")
    if up is None:
        st.info("Place `yield_df.csv` next to `app.py`, or upload it above.")
        st.stop()
    raw = pd.read_csv(up)

df_all, prep_log = prepare(raw)

# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")
crops = st.sidebar.multiselect("Crop", sorted(df_all["Item"].unique()),
                               default=sorted(df_all["Item"].unique()))
yr_min, yr_max = int(df_all["Year"].min()), int(df_all["Year"].max())
years = st.sidebar.slider("Years", yr_min, yr_max, (yr_min, yr_max))
countries = st.sidebar.multiselect("Country (blank = all)", sorted(df_all["Area"].unique()))

df = df_all[df_all["Item"].isin(crops) & df_all["Year"].between(*years)]
if countries:
    df = df[df["Area"].isin(countries)]
if df.empty:
    st.warning("No data for these filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(df):,}")
k2.metric("Countries", df["Area"].nunique())
k3.metric("Mean yield (t/ha)", f"{df['yield_t_ha'].mean():.1f}")
k4.metric("Median yield (t/ha)", f"{df['yield_t_ha'].median():.1f}")

tab_over, tab_rel, tab_geo, tab_cat, tab_model, tab_ins, tab_data = st.tabs(
    ["Overview", "Relationships", "Geography", "Categorical", "Prediction model", "Insights", "Data prep"])

# ----------------------------------------------------------------------------
# OVERVIEW  (Viz 1-3)
# ----------------------------------------------------------------------------
with tab_over:
    c1, c2 = st.columns(2)
    with c1:  # Viz 1
        log_x = st.checkbox("Log scale (x)", value=True)
        hdf = df.assign(log10_yield=np.log10(df["yield_t_ha"].clip(lower=1e-3)))
        xcol = "log10_yield" if log_x else "yield_t_ha"
        fig = px.histogram(hdf, x=xcol, color="Item", nbins=60, barmode="overlay", opacity=0.6,
                           title="1 · Yield distribution by crop",
                           labels={"log10_yield": "log10 yield (0 = 1 t/ha, 1 = 10 t/ha)",
                                   "yield_t_ha": "Yield (t/ha)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:  # Viz 2
        fig = px.box(df, x="Item", y="yield_t_ha", color="Item", points=False,
                     title="2 · Yield spread by crop", labels={"yield_t_ha": "Yield (t/ha)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Viz 3 - treemap
    top = (df.groupby(["Item", "Area"], as_index=False)["yield_t_ha"].mean()
             .sort_values("yield_t_ha", ascending=False).groupby("Item").head(8))
    fig = px.treemap(top, path=[px.Constant("World"), "Item", "Area"], values="yield_t_ha",
                     color="yield_t_ha", color_continuous_scale="YlGn",
                     title="3 · Treemap: top 8 countries per crop (mean yield, t/ha)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# RELATIONSHIPS  (Viz 4-8)
# ----------------------------------------------------------------------------
with tab_rel:
    sample = df.sample(min(len(df), 6000), random_state=1)
    c1, c2, c3 = st.columns(3)
    with c1:  # Viz 4
        st.plotly_chart(px.scatter(sample, x="avg_temp", y="yield_t_ha", color="Item", opacity=.5,
                                   title="4 · Temperature vs yield",
                                   labels={"avg_temp": "Avg temp (°C)", "yield_t_ha": "Yield (t/ha)"}),
                        use_container_width=True)
    with c2:  # Viz 5
        st.plotly_chart(px.scatter(sample, x="rainfall_mm", y="yield_t_ha", color="Item", opacity=.5,
                                   title="5 · Rainfall vs yield",
                                   labels={"rainfall_mm": "Rainfall (mm/yr)", "yield_t_ha": "Yield (t/ha)"}),
                        use_container_width=True)
    with c3:  # Viz 6
        st.plotly_chart(px.scatter(sample, x="pesticides_t", y="yield_t_ha", color="Item", opacity=.5,
                                   log_x=True, title="6 · Pesticides (log) vs yield",
                                   labels={"pesticides_t": "Pesticides (t, log)", "yield_t_ha": "Yield (t/ha)"}),
                        use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:  # Viz 7 - correlogram
        corr = df[["yield_t_ha"] + NUM_COLS].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="7 · Correlogram (Pearson)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:  # Viz 8 - trend
        trend = df.groupby(["Year", "Item"], as_index=False)["yield_t_ha"].mean()
        st.plotly_chart(px.line(trend, x="Year", y="yield_t_ha", color="Item",
                                title="8 · Mean yield over time", labels={"yield_t_ha": "Yield (t/ha)"}),
                        use_container_width=True)

    # Viz 9 - scatter matrix
    st.plotly_chart(px.scatter_matrix(sample, dimensions=["yield_t_ha", "avg_temp", "rainfall_mm", "pesticides_t"],
                                      color="Item", opacity=.4, height=650,
                                      title="9 · Scatter-plot matrix"),
                    use_container_width=True)

# ----------------------------------------------------------------------------
# GEOGRAPHY  (Viz 10-11)
# ----------------------------------------------------------------------------
with tab_geo:
    geo = df.groupby("Area", as_index=False).agg(yield_t_ha=("yield_t_ha", "mean"),
                                                 pesticides_t=("pesticides_t", "mean"),
                                                 rainfall_mm=("rainfall_mm", "mean"),
                                                 avg_temp=("avg_temp", "mean"))
    fig = px.choropleth(geo, locations="Area", locationmode="country names", color="yield_t_ha",
                        color_continuous_scale="YlGn", title="10 · Mean yield by country (t/ha)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Countries whose names don't match Plotly's list (e.g. 'Russian Federation') may appear blank.")

    fig = px.scatter(geo, x="pesticides_t", y="yield_t_ha", size="rainfall_mm", color="avg_temp",
                     hover_name="Area", log_x=True, color_continuous_scale="Turbo",
                     title="11 · Bubble chart: pesticides vs yield (size = rainfall, colour = temp)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# CATEGORICAL  (Viz 12-14)
# ----------------------------------------------------------------------------
with tab_cat:
    c1, c2 = st.columns(2)
    with c1:  # Viz 12 - mosaic
        ct = df.assign(temp_band=df["temp_band"].astype(str), Item=df["Item"].astype(str))
        bands = [b for b in ["Cool", "Mild", "Warm", "Hot"] if b in ct["temp_band"].values]
        band_share = ct["temp_band"].value_counts(normalize=True)
        crop_share = pd.crosstab(ct["temp_band"], ct["Item"], normalize="index")
        widths = [float(band_share[b]) for b in bands]
        centers = [float(l + w / 2) for l, w in zip(np.cumsum([0] + widths[:-1]), widths)]
        palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
        fig_m = go.Figure()
        for i, crop in enumerate(sorted(ct["Item"].unique())):
            ys = [float(crop_share.loc[b, crop]) for b in bands]
            fig_m.add_bar(
                x=centers, y=ys, width=widths, name=crop,
                marker=dict(color=palette[i % len(palette)], line=dict(color="white", width=1.5)),
                text=[crop if y > 0.08 else "" for y in ys], textposition="inside",
                insidetextanchor="middle", textfont=dict(color="black", size=11),
                customdata=[[b, w] for b, w in zip(bands, widths)],
                hovertemplate=f"<b>{crop}</b><br>Temp band: %{{customdata[0]}}"
                              "<br>Share of band: %{y:.1%}<br>Band share of records: %{customdata[1]:.1%}<extra></extra>",
            )
        fig_m.update_layout(
            title="12 · Mosaic plot: temperature band × crop",
            barmode="stack", bargap=0,
            xaxis=dict(tickmode="array", tickvals=centers, ticktext=bands,
                       title="Temperature band (column width = share of records)", range=[0, 1]),
            yaxis=dict(tickformat=".0%", title="Crop share within band", range=[0, 1]),
            legend=dict(title="Crop"), height=520,
        )
        st.plotly_chart(fig_m, use_container_width=True)
    with c2:  # Viz 13 - heatmap
        pivot = df.pivot_table(index="Item", columns="decade", values="yield_t_ha", aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, text_auto=".1f", color_continuous_scale="YlGn", aspect="auto",
                                  title="13 · Mean yield: crop × decade (t/ha)"),
                        use_container_width=True)

    # Viz 14 - violin by temp band
    st.plotly_chart(px.violin(df, x="temp_band", y="yield_t_ha", color="temp_band", box=True,
                              category_orders={"temp_band": ["Cool", "Mild", "Warm", "Hot"]},
                              title="14 · Yield distribution by temperature band",
                              labels={"yield_t_ha": "Yield (t/ha)", "temp_band": "Temperature band"}),
                    use_container_width=True)

    # Viz 15 - sunburst
    sb = df.groupby(["Item", "rain_band"], as_index=False, observed=True)["yield_t_ha"].mean()
    st.plotly_chart(px.sunburst(sb, path=["Item", "rain_band"], values="yield_t_ha",
                                color="yield_t_ha", color_continuous_scale="YlGn",
                                title="15 · Sunburst: mean yield by crop and rainfall band"),
                    use_container_width=True)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
with tab_model:
    results, fitted, imp_df, (y_te, y_hat) = train_models(df_all)
    st.subheader("Model comparison (80/20 split, whole dataset)")
    st.dataframe(results.style.format({"R²": "{:.3f}", "MAE (t/ha)": "{:.2f}"}), hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h",
                               title="Permutation importance (drop in R²)"), use_container_width=True)
    with c2:
        samp = pd.DataFrame({"Actual": y_te, "Predicted": y_hat}).sample(min(3000, len(y_te)), random_state=1)
        fig = px.scatter(samp, x="Actual", y="Predicted", opacity=.4, title="Actual vs predicted (t/ha)")
        mx = float(max(samp.max()))
        fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("What-if yield predictor")
    a, b, c = st.columns(3)
    area = a.selectbox("Country", sorted(df_all["Area"].unique()))
    item = a.selectbox("Crop", sorted(df_all["Item"].unique()))
    year = b.slider("Year", yr_min, yr_max + 10, yr_max)
    rain = b.slider("Rainfall (mm/yr)", int(df_all.rainfall_mm.min()), int(df_all.rainfall_mm.max()),
                    int(df_all.rainfall_mm.median()))
    pest = c.slider("Pesticides (tonnes)", 0.0, float(df_all.pesticides_t.quantile(.99)),
                    float(df_all.pesticides_t.median()))
    temp = c.slider("Avg temp (°C)", float(df_all.avg_temp.min()), float(df_all.avg_temp.max()),
                    float(df_all.avg_temp.median()))
    row = pd.DataFrame([{"Area": area, "Item": item, "Year": year, "rainfall_mm": rain,
                         "pesticides_t": pest, "avg_temp": temp}])
    st.success(f"Predicted yield: **{fitted['Gradient Boosting'].predict(row)[0]:.1f} t/ha**")
    st.caption("Random split → neighbouring years of the same country appear in train and test, "
               "so scores are optimistic. Discuss this limitation in your report.")

# ----------------------------------------------------------------------------
# INSIGHTS (computed from the filtered data)
# ----------------------------------------------------------------------------
with tab_ins:
    st.subheader("Auto-generated insights (update with filters)")
    by_crop = df.groupby("Item")["yield_t_ha"].mean().sort_values(ascending=False)
    by_area = df.groupby("Area")["yield_t_ha"].mean().sort_values(ascending=False)
    corr = df[["yield_t_ha"] + NUM_COLS].corr()["yield_t_ha"]
    first, last = df[df.Year == df.Year.min()], df[df.Year == df.Year.max()]
    growth = (last["yield_t_ha"].mean() / first["yield_t_ha"].mean() - 1) * 100 if len(first) and len(last) else np.nan

    st.markdown(f"""
- **Crop type dominates.** `{by_crop.index[0]}` has the highest mean yield ({by_crop.iloc[0]:.1f} t/ha) and
  `{by_crop.index[-1]}` the lowest ({by_crop.iloc[-1]:.1f} t/ha), a {by_crop.iloc[0] / by_crop.iloc[-1]:.1f}× gap.
- **Country matters.** Highest-yielding country: `{by_area.index[0]}` ({by_area.iloc[0]:.1f} t/ha);
  lowest: `{by_area.index[-1]}` ({by_area.iloc[-1]:.1f} t/ha).
- **Climate correlations with yield:** temperature r = {corr['avg_temp']:.2f}, rainfall r = {corr['rainfall_mm']:.2f},
  pesticides r = {corr['pesticides_t']:.2f}, year r = {corr['Year']:.2f}.
  Remember: correlation across *mixed crops* is confounded by crop type; check per-crop views too.
- **Change over the period** ({df.Year.min()}→{df.Year.max()}): mean yield changed by {growth:+.1f}%.
""")
    st.markdown("**Per-crop correlation with yield** (removes the crop-mix effect)")
    pc = (df.groupby("Item")[["yield_t_ha"] + NUM_COLS].corr()["yield_t_ha"]
            .unstack()[["avg_temp", "rainfall_mm", "pesticides_t", "Year"]])
    st.dataframe(pc.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"))

# ----------------------------------------------------------------------------
# DATA PREP LOG
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Cleaning log")
    st.table(pd.DataFrame(prep_log.items(), columns=["Step", "Count"]))
    st.subheader("Summary statistics")
    st.dataframe(df_all[["yield_t_ha"] + NUM_COLS].describe().T)
    st.subheader("Sample rows")
    st.dataframe(df.head(50))
    st.download_button("Download filtered data", df.to_csv(index=False), "filtered_crop_data.csv")
