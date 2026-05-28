# ⚡ Forecasting Power Supply & Demand

A machine learning project for forecasting hourly electrical power supply and demand using real-world grid data from the PJM Interconnection.

---

## 📋 Overview

Accurate power demand forecasting is critical for grid operators to balance generation against consumption, reduce waste, and ensure reliable energy delivery. This project applies time-series analysis and machine learning techniques to predict megawatt-hour (MW) load at an hourly granularity.

---

## 📁 Repository Structure

```
Forecasting-Power-Supply-Demand/
│
├── Forecasting Power Supply_Demand.ipynb   # Main analysis and modeling notebook
├── PJMW_MW_Hourly.xlsx                     # Hourly MW load dataset (PJM West region)
├── aappp.py                                # Python application/utility script
└── README.md                               # Project documentation
```

---

## 📊 Dataset

**File:** `PJMW_MW_Hourly.xlsx`

- **Source:** PJM Interconnection — one of the largest electricity transmission organizations in the US
- **Region:** PJM West (PJMW)
- **Granularity:** Hourly
- **Features:** Timestamp, Megawatt (MW) load

PJM operates a competitive wholesale electricity market and manages the high-voltage electric power system serving 65 million people across 13 states and the District of Columbia.

---

## 🔍 Project Workflow

1. **Data Loading & Exploration** — Load the hourly MW dataset and inspect distributions, trends, and seasonality
2. **Preprocessing** — Handle missing values, parse timestamps, engineer time-based features (hour, day of week, month, season)
3. **Exploratory Data Analysis (EDA)** — Visualize demand patterns across hours, days, and seasons
4. **Feature Engineering** — Create lag features and rolling statistics for time-series modeling
5. **Model Training** — Train forecasting models (e.g., XGBoost, LSTM, ARIMA, or similar)
6. **Evaluation** — Assess performance using metrics such as MAE, RMSE, and MAPE
7. **Visualization** — Plot predicted vs. actual demand over time

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| Pandas | Data manipulation and time-series handling |
| NumPy | Numerical computations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine learning models and metrics |
| XGBoost / LSTM | Forecasting models |
| Jupyter Notebook | Interactive analysis environment |
| OpenPyXL | Reading Excel (.xlsx) data files |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl jupyter
```

### Run the Notebook

```bash
git clone https://github.com/shwetabp14/Forecasting-Power-Supply-Demand.git
cd Forecasting-Power-Supply-Demand
jupyter notebook "Forecasting Power Supply_Demand.ipynb"
```

### Run the Python Script

```bash
python aappp.py
```

---

## 📈 Results

The model forecasts hourly MW power demand and is evaluated on held-out test data. Key metrics reported:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error

---

## 🌐 Use Cases

- Grid load balancing and dispatch planning
- Renewable energy integration (scheduling solar/wind around demand peaks)
- Cost optimization for energy procurement
- Anomaly detection in consumption patterns

---

## 👤 Author

**Shweta Patil** ([@shwetabp14](https://github.com/shwetabp14))
