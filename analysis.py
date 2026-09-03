"""
End-to-End Stock Price Data Analysis & Prediction
Dataset: Apple Inc. (AAPL) daily OHLCV, Feb 2015 - Feb 2017
Source: plotly/datasets (public GitHub dataset)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64
import io
import json

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("apple_stock.csv", parse_dates=["Date"])
df = df.rename(columns={
    "AAPL.Open": "Open", "AAPL.High": "High", "AAPL.Low": "Low",
    "AAPL.Close": "Close", "AAPL.Volume": "Volume", "AAPL.Adjusted": "Adj_Close"
})
df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Adj_Close"]].sort_values("Date").reset_index(drop=True)

summary_stats = df.describe().to_dict()

# ---------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["Daily_Return"] = df["Close"].pct_change() * 100
df["MA10"] = df["Close"].rolling(10).mean()
df["MA30"] = df["Close"].rolling(30).mean()
df["Volatility_10"] = df["Daily_Return"].rolling(10).std()
df["Lag1"] = df["Close"].shift(1)
df["Lag2"] = df["Close"].shift(2)
df["Lag3"] = df["Close"].shift(3)
df["Return_Lag1"] = df["Daily_Return"].shift(1)
df["Volume_Change"] = df["Volume"].pct_change() * 100
df["Target_NextClose"] = df["Close"].shift(-1)  # predict next day's close

model_df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (time-based, no shuffling)
# ---------------------------------------------------------------
features = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA30",
            "Volatility_10", "Lag1", "Lag2", "Lag3", "Return_Lag1", "Volume_Change"]
X = model_df[features]
y = model_df["Target_NextClose"]

split_idx = int(len(model_df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = model_df["Date"].iloc[split_idx:]

results = {}

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
results["Linear Regression"] = {
    "MAE": mean_absolute_error(y_test, pred_lr),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred_lr)),
    "R2": r2_score(y_test, pred_lr),
    "preds": pred_lr
}

rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
results["Random Forest"] = {
    "MAE": mean_absolute_error(y_test, pred_rf),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred_rf)),
    "R2": r2_score(y_test, pred_rf),
    "preds": pred_rf
}

# naive baseline: predict tomorrow's close = today's close
pred_naive = X_test["Close"].values
results["Naive Baseline (t=t-1)"] = {
    "MAE": mean_absolute_error(y_test, pred_naive),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred_naive)),
    "R2": r2_score(y_test, pred_naive),
    "preds": pred_naive
}

best_model_name = min(results, key=lambda k: results[k]["RMSE"])

feat_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# ---------------------------------------------------------------
# 4. VISUALIZATIONS -> base64 PNGs for embedding in HTML report
# ---------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

charts = {}

# Chart 1: Price trend with moving averages
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Date"], df["Close"], label="Close Price", color="#2563eb", linewidth=1.3)
ax.plot(df["Date"], df["MA10"], label="10-Day MA", color="#f59e0b", linewidth=1)
ax.plot(df["Date"], df["MA30"], label="30-Day MA", color="#dc2626", linewidth=1)
ax.set_title("AAPL Closing Price with Moving Averages")
ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
ax.legend()
fig.autofmt_xdate()
charts["price_trend"] = fig_to_base64(fig)

# Chart 2: Daily returns distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["Daily_Return"].dropna(), bins=40, color="#0ea5e9", edgecolor="white")
ax.axvline(df["Daily_Return"].mean(), color="#dc2626", linestyle="--", label=f"Mean = {df['Daily_Return'].mean():.2f}%")
ax.set_title("Distribution of Daily Returns")
ax.set_xlabel("Daily Return (%)"); ax.set_ylabel("Frequency")
ax.legend()
charts["returns_dist"] = fig_to_base64(fig)

# Chart 3: Volume over time
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df["Date"], df["Volume"], color="#64748b", width=1.5)
ax.set_title("Daily Trading Volume")
ax.set_xlabel("Date"); ax.set_ylabel("Volume")
fig.autofmt_xdate()
charts["volume"] = fig_to_base64(fig)

# Chart 4: Correlation heatmap
corr_cols = ["Open", "High", "Low", "Close", "Volume", "Daily_Return", "MA10", "MA30", "Volatility_10"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=45, ha="right")
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                 color="white" if abs(corr.iloc[i,j])>0.5 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Feature Correlation Heatmap")
charts["correlation"] = fig_to_base64(fig)

# Chart 5: Actual vs Predicted (test set, best model)
best_preds = results[best_model_name]["preds"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates_test, y_test.values, label="Actual Close", color="#16a34a", linewidth=1.5)
ax.plot(dates_test, best_preds, label=f"Predicted ({best_model_name})", color="#dc2626", linewidth=1.5, linestyle="--")
ax.set_title(f"Actual vs Predicted Next-Day Close — {best_model_name} (Test Set)")
ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
ax.legend()
fig.autofmt_xdate()
charts["actual_vs_pred"] = fig_to_base64(fig)

# Chart 6: Feature importance (Random Forest)
fig, ax = plt.subplots(figsize=(8, 5))
feat_importance.sort_values().plot(kind="barh", ax=ax, color="#7c3aed")
ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance")
charts["feat_importance"] = fig_to_base64(fig)

# ---------------------------------------------------------------
# 5. SAVE OUTPUTS FOR REPORT BUILDING
# ---------------------------------------------------------------
with open("charts.json", "w") as f:
    json.dump(charts, f)

with open("results.json", "w") as f:
    json.dump({
        "n_rows": len(df),
        "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "price_start": float(df["Close"].iloc[0]),
        "price_end": float(df["Close"].iloc[-1]),
        "total_return_pct": float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100),
        "avg_daily_return": float(df["Daily_Return"].mean()),
        "std_daily_return": float(df["Daily_Return"].std()),
        "max_close": float(df["Close"].max()),
        "min_close": float(df["Close"].min()),
        "best_model": best_model_name,
        "model_results": {k: {kk: (vv if kk!="preds" else None) for kk, vv in v.items()} for k, v in results.items()},
        "feat_importance": feat_importance.to_dict(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }, f, indent=2, default=str)

df.to_csv("aapl_processed.csv", index=False)

print("DONE")
for name, r in results.items():
    print(f"{name}: MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}  R2={r['R2']:.4f}")
print("Best model:", best_model_name)
