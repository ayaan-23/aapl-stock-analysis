# AAPL Stock Analysis & Prediction

End-to-end analysis of AAPL stock data — exploratory data analysis, visualizations, and next-day price prediction using Linear Regression & Random Forest, benchmarked against a naive baseline.

## 📊 Overview

This project analyzes daily OHLCV (Open, High, Low, Close, Volume) data for Apple Inc. (AAPL) from **Feb 2015 to Feb 2017** (506 trading days), covering:

- Exploratory data analysis (price trends, returns distribution, volume, correlations)
- Feature engineering (moving averages, lagged prices, volatility, momentum)
- Predictive modeling for next-day closing price
- Honest benchmarking against a naive "no-change" baseline

## 🔑 Key Finding

Neither Linear Regression nor Random Forest meaningfully outperformed a naive baseline that simply predicts tomorrow's close equals today's close. This reflects the **random-walk nature of daily stock prices** — short-horizon price levels are extremely hard to out-predict with standard tabular features, a well-known result in quantitative finance.

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Linear Regression | 0.893 | 1.336 | 0.9614 |
| Random Forest | 1.669 | 2.391 | 0.8763 |
| **Naive Baseline (t = t-1)** | **0.792** | **1.247** | **0.9664** |

## 📁 Repo Contents

| File | Description |
|---|---|
| `AAPL_Stock_Analysis_Report.html` | Full report with embedded charts, findings, and conclusions — open in any browser |
| `analysis.py` | Python pipeline: data loading, feature engineering, model training, chart generation |
| `aapl_processed.csv` | Cleaned dataset with engineered features (moving averages, lags, volatility, returns) |

## 🛠️ Tools & Libraries

- `pandas`, `numpy` — data wrangling
- `matplotlib` — visualization
- `scikit-learn` — Linear Regression, Random Forest, evaluation metrics

## 📈 What's Inside the Report

1. Data overview & summary statistics
2. Price trend with 10-day and 30-day moving averages
3. Daily returns distribution
4. Trading volume patterns
5. Feature correlation heatmap
6. Model comparison (Linear Regression vs Random Forest vs naive baseline)
7. Feature importance analysis
8. Conclusions & next steps

## 🚀 Running It Yourself

```bash
pip install pandas numpy matplotlib scikit-learn
python analysis.py
```

This regenerates `aapl_processed.csv`, `charts.json`, and `results.json`. Run `build_report.py` afterward to rebuild the HTML report.

## 💡 Next Steps / Ideas

- Predict price **direction** (up/down) instead of exact price — often more tractable than regression
- Add technical indicators (RSI, MACD, Bollinger Bands)
- Model volatility directly (e.g., GARCH)
- Extend to longer forecast horizons where fundamentals matter more than daily noise
- Compare across multiple tickers or sectors

## 📄 Data Source

Public AAPL OHLCV dataset (Feb 2015–Feb 2017), sourced from a public GitHub dataset repository.

---

*Built as a data science portfolio project applying end-to-end analysis and prediction to real-world financial data.*
