import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Power Supply Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Power Supply Demand Forecasting")
st.markdown("**Model:** TBATS (Best Performer — RMSE: 763) | **Dataset:** PJM Hourly Energy Consumption")
st.markdown("---")

# ── Load Data & Model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('saved_models.pkl', 'rb') as f:
        models = pickle.load(f)
    return models['tbats']

@st.cache_data
def load_data():
    df = pd.read_excel('PJMW_MW_Hourly.xlsx')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    df.set_index('Datetime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.dropna(inplace=True)
    return df

@st.cache_data
def load_daily():
    return pd.read_csv('daily_data.csv', index_col='Datetime', parse_dates=True)

# Load everything
df     = load_data()
daily  = load_daily()
model  = load_model()

# Train/test split (same as notebook)
train_size = int(len(daily) * 0.8)
train = daily.iloc[:train_size]
test  = daily.iloc[train_size:]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)", min_value=7, max_value=60, value=30, step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Why TBATS?")
st.sidebar.markdown("""
TBATS handles **multiple seasonalities** which makes it ideal for energy data:
- Daily cycles (hour of day)
- Weekly cycles (weekday vs weekend)  
- Yearly cycles (summer/winter peaks)

| Model | MAE | RMSE |
|-------|-----|------|
| ARIMA | 601 | 786 |
| SARIMAX | 619 | 819 |
| **TBATS** | **622** | **763** ✅ |
| PROPHET | 814 | 1074 |
| Holt-Winters | 2859 | 3366 |
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Project Info")
st.sidebar.markdown("""
- **Dataset:** PJMW_MW_Hourly  
- **Records:** 143,206  
- **Period:** 2002 – 2018  
- **Train/Test:** 80% / 20%  
""")

# ── Helper: metrics ───────────────────────────────────────────────────────────
def compute_metrics(true, pred):
    pred = np.array(pred[:len(true)])
    true = np.array(true)
    mae  = round(np.mean(np.abs(true - pred)), 2)
    rmse = round(np.sqrt(np.mean((true - pred) ** 2)), 2)
    mape = round(np.mean(np.abs((true - pred) / true)) * 100, 2)
    return mae, rmse, mape

# ── EDA Section ───────────────────────────────────────────────────────────────
st.header("📈 Exploratory Data Analysis")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",    f"{len(df):,}")
col2.metric("Date Range",       f"{df.index.min().date()} → {df.index.max().date()}")
col3.metric("Mean Demand (MW)", f"{df['PJMW_MW'].mean():,.0f}")
col4.metric("Peak Demand (MW)", f"{df['PJMW_MW'].max():,.0f}")

tab1, tab2, tab3 = st.tabs(["📅 Historical Trend", "📦 Distributions", "🔢 Raw Data"])

with tab1:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    df['PJMW_MW'].resample('D').mean().plot(ax=axes[0], color='steelblue')
    axes[0].set_title('Daily Average Energy Demand')
    axes[0].set_ylabel('MW')

    df['PJMW_MW'].resample('ME').mean().plot(ax=axes[1], color='darkorange')
    axes[1].set_title('Monthly Average Energy Demand')
    axes[1].set_ylabel('MW')

    df['PJMW_MW'].resample('YE').mean().plot(ax=axes[2], color='green', marker='o')
    axes[2].set_title('Yearly Average Energy Demand')
    axes[2].set_ylabel('MW')

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hour_avg = df.groupby(df.index.hour)['PJMW_MW'].mean()
    axes[0].bar(hour_avg.index, hour_avg.values, color='steelblue')
    axes[0].set_title('Average Demand by Hour of Day')
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('MW')

    month_avg = df.groupby(df.index.month)['PJMW_MW'].mean()
    axes[1].bar(month_avg.index, month_avg.values, color='darkorange')
    axes[1].set_title('Average Demand by Month')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('MW')

    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.dataframe(df.tail(100), use_container_width=True)

# ── Forecasting Section ───────────────────────────────────────────────────────
st.markdown("---")
st.header(f"🔮 TBATS Forecast — Next {forecast_days} Days")

with st.spinner("Generating forecast... ⏳"):
    # Test predictions
    test_pred = model.forecast(steps=len(test))
    test_pred_series = pd.Series(test_pred, index=test.index)

    # Future predictions
    future_idx = pd.date_range(start=daily.index[-1], periods=forecast_days + 1, freq='D')[1:]
    future_pred = model.forecast(steps=forecast_days)
    future_series = pd.Series(future_pred, index=future_idx)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.subheader("📊 TBATS Model Performance")
mae, rmse, mape = compute_metrics(test.values, test_pred)

col1, col2, col3 = st.columns(3)
col1.metric("MAE",     f"{mae:,.2f} MW")
col2.metric("RMSE",    f"{rmse:,.2f} MW",  delta="Best among all models", delta_color="off")
col3.metric("MAPE",    f"{mape:.2f}%")

# ── Actual vs Predicted ───────────────────────────────────────────────────────
st.subheader("🧪 Test Period: Actual vs TBATS Predicted")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test.index, test.values,      label='Actual',         color='steelblue',  linewidth=1.5)
ax.plot(test.index, test_pred[:len(test)], label='TBATS Predicted', color='darkorange', linewidth=1.2, linestyle='--')
ax.set_title('Actual vs TBATS Predicted — Test Period (2015–2018)')
ax.set_xlabel('Date')
ax.set_ylabel('MW')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
st.pyplot(fig)

# ── Future Forecast ───────────────────────────────────────────────────────────
st.subheader(f"📅 Next {forecast_days} Days Forecast")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily.index[-120:], daily.values[-120:],
        label='Historical', color='steelblue', linewidth=1.5)
ax.plot(future_series.index, future_series.values,
        label=f'TBATS Forecast ({forecast_days} days)', color='darkorange', linewidth=2)
ax.axvline(x=daily.index[-1], color='gray', linestyle=':', linewidth=1.5, label='Forecast Start')
ax.fill_between(future_series.index, future_series.values * 0.95,
                future_series.values * 1.05, alpha=0.2, color='darkorange', label='±5% Confidence Band')
ax.set_title(f'TBATS: Next {forecast_days} Days Energy Demand Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('MW')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig)

# ── Forecast Table + Download ─────────────────────────────────────────────────
st.subheader("🗂️ Forecast Data Table")
forecast_df = pd.DataFrame({
    'Date': future_series.index.date,
    'TBATS_Forecast_MW': future_series.values.round(2)
})
st.dataframe(forecast_df, use_container_width=True)

csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Forecast CSV", csv, "tbats_forecast.csv", "text/csv")
