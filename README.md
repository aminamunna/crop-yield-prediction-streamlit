# Crop Yield Explorer

## Run it
```bash
pip install streamlit pandas numpy plotly matplotlib scikit-learn statsmodels
# download yield_df.csv from Kaggle (patelris/crop-yield-prediction-dataset) into this folder
streamlit run app.py
```

## Visualization checklist (15 in the dashboard)
| # | Chart | Tab | Question it answers |
|---|-------|-----|---------------------|
| 1 | Histogram (log) | Overview | How skewed is yield? |
| 2 | Box plot by crop | Overview | Which crops differ most? |
| 3 | Treemap | Overview | Which countries lead each crop? |
| 4-6 | Scatter plots (temp, rain, pesticides) | Relationships | Do inputs/climate relate to yield? |
| 7 | Correlogram | Relationships | Which variables move together? |
| 8 | Line chart | Relationships | Is yield improving over time? |
| 9 | Scatter-plot matrix | Relationships | Pairwise structure at a glance |
| 10 | Choropleth | Geography | Where are yields high/low? |
| 11 | Bubble chart | Geography | Do high-input countries get higher yield? |
| 12 | Mosaic plot | Categorical | Which crops sit in which temperature bands? |
| 13 | Heatmap crop × decade | Categorical | Which crops improved most? |
| 14 | Violin by temp band | Categorical | Does hot climate shift the yield distribution? |
| 15 | Sunburst | Categorical | Crop × rainfall interaction |

## Report outline (fill in with numbers from your dashboard)
1. **Introduction**: food security context; objective; dataset source (FAO, World Bank), 10 crops, 1990-2013.
2. **Data preparation**: use the "Data prep" tab log (rows, duplicates, missing); unit conversion hg/ha → t/ha; engineered bands and decades; note that climate/pesticide values are per country-year and repeat across crops.
3. **Visualization & interpretation**: for each chart, 2-3 sentences: *what it shows → what you see → why it matters*. Use the "Insights" tab numbers.
4. **Model development**: baseline linear regression vs gradient boosting; R², MAE; feature importance; limitation of the random split (leakage across years within a country) and what a country- or time-based split would change.
5. **Actionable insights** (see prompts below).
6. **Limitations & future work**: no soil, irrigation, fertilizer, or area data; country-level averages hide local variation; correlation ≠ causation.
7. **Conclusion** and references.

## Prompts for actionable insights (confirm each against your own charts before writing it)
- *Crop selection:* which crops give the most tonnes/ha in hot vs cool and dry vs wet bands? (Mosaic, violin, sunburst)
- *Input efficiency:* does extra pesticide use keep paying off, or flatten out? Check the log-scale scatter and the per-crop correlation table.
- *Climate risk:* which crops show strongest negative association with high temperature? (per-crop table in Insights)
- *Closing yield gaps:* countries well below the crop's median yet in a similar climate band are the targets for knowledge transfer. (Treemap, choropleth)
- *Trend:* which crops/countries stagnated over the decades? (Line chart, decade heatmap)

## Presentation outline (8-10 slides)
1. Title & question · 2. Dataset & cleaning · 3. Overview charts (1-3) · 4. Climate & input relationships (4-9) · 5. Geography (10-11) · 6. Categorical views (12-15) · 7. Model & performance · 8. Key insights · 9. Recommendations · 10. Limitations & next steps
