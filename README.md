# 🌾 Crop Yield Explorer

An interactive Streamlit dashboard that analyses global crop yield using 15 visualizations, a predictive model, and auto-generated insights.

**Author:** Amina Munna · **Course:** Predictive Analytics and Data Visualization · **Assessment:** Data Visualization – Dashboard, Report and Presentation
**Live dashboard:** [paste Streamlit link here, if deployed]

---

## 1. Dataset
- **Source:** [Crop Yield Prediction Dataset (Kaggle, patelris)](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
- **Original data:** FAO (fao.org) and World Bank (data.worldbank.org)
- **File used:** `yield_df.csv`, a merge of `yield.csv`, `rainfall.csv`, `temp.csv` and `pesticides.csv`
- **Coverage:** 1990-2013 · 101 countries · 10 crops (cassava, maize, plantains and others, potatoes, rice, sorghum, soybeans, sweet potatoes, wheat, yams)
- **Variables:** Area, Item (crop), Year, yield (hg/ha), average rainfall (mm/year), pesticides (tonnes), average temperature (°C)

## 2. Data Preparation
| Step | Result |
|---|---|
| Raw rows | 28,242 |
| Missing values removed | 0 |
| Duplicate rows removed | 2,310 |
| **Clean rows** | **25,932** |

Other steps: dropped the index column, renamed columns, converted yield from hg/ha to **t/ha**, and created **decade** and **quartile bands** (temperature, rainfall, pesticides).

## 3. Dashboard Features
- Sidebar filters: crop, year range, country
- KPI cards: records, countries, mean and median yield
- 7 tabs: Overview · Relationships · Geography · Categorical · Prediction model · Insights · Data prep
- What-if yield predictor
- Download button for filtered data

### The 15 visualizations
| # | Chart | Tab |
|---|-------|-----|
| 1 | Histogram (log scale) | Overview |
| 2 | Box plot by crop | Overview |
| 3 | Treemap (top countries per crop) | Overview |
| 4 | Scatter: temperature vs yield | Relationships |
| 5 | Scatter: rainfall vs yield | Relationships |
| 6 | Scatter: pesticides vs yield | Relationships |
| 7 | Correlogram | Relationships |
| 8 | Line chart: yield over time | Relationships |
| 9 | Scatter-plot matrix | Relationships |
| 10 | Choropleth map | Geography |
| 11 | Bubble chart | Geography |
| 12 | Mosaic plot (temperature band × crop) | Categorical |
| 13 | Heatmap (crop × decade) | Categorical |
| 14 | Violin plot by temperature band | Categorical |
| 15 | Sunburst (crop × rainfall band) | Categorical |

## 4. Model
80/20 train-test split, with one-hot encoding for crop and country and scaling for numeric features.

| Model | R² | MAE (t/ha) |
|---|---|---|
| Linear Regression (baseline) | 0.749 | 2.99 |
| Gradient Boosting | **0.966** | **0.85** |

Most important features: **crop > country > pesticides, rainfall, temperature > year**.

> **Note:** the random split places neighbouring years of the same country in both train and test sets, so the scores are probably optimistic.

## 5. Key Findings
1. **Crop type matters most.** Potatoes average 19.9 t/ha vs 1.7 t/ha for soybeans, an 11.7× gap.
2. **Large country gaps.** United Kingdom ≈ 24.1 t/ha vs Botswana ≈ 0.7 t/ha.
3. **Heat risk.** Nine of ten crops show lower yield in hotter regions (strongest: maize −0.55, yams −0.50). Cassava is the exception (+0.35).
4. **Pooled correlations hide this.** Across all crops, yield vs temperature is only −0.11, so crops must be analysed separately.
5. **Pesticides:** modest positive link (soybeans 0.35, potatoes 0.21), near zero for wheat, plantains and yams.
6. **Progress over time:** mean yield rose 35.4% (1990-2013), but plantains were flat (10.6 → 10.7 t/ha).

## 6. Run Locally
```bash
git clone [your-repo-url]
cd [repo-folder]
pip install -r requirements.txt
streamlit run app.py
```
Keep `yield_df.csv` in the same folder as `app.py`. If it is missing, the app asks you to upload it.

## 7. Limitations
- Country-level averages hide local variation.
- No soil, irrigation, fertilizer or planted-area data.
- Correlation does not imply causation.
- A few country names may not appear on the choropleth map.

## 8. Tech Stack
Python · Pandas · NumPy · Plotly · Scikit-learn · Streamlit

## 9. Files
```
├── app.py             # Streamlit dashboard
├── yield_df.csv       # Dataset
├── requirements.txt   # Dependencies
└── README.md
```
