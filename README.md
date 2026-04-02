# 📈 Market Insight Dashboard

A real-time stock analysis dashboard built with Python, Pandas, Streamlit, and the Claude AI API.

**Live demo:** market-dashboard-c5ytwygv2kgyrqcxappmfkc

---

## What It Does

- Fetches daily OHLCV stock data from Alpha Vantage (free API)
- Computes returns, volatility, RSI, and max drawdown using **Pandas**
- Renders interactive charts with **Plotly**
- Generates a plain-English market summary using **Claude AI**
- Deployed as a web app via **Streamlit Cloud** — no server needed

---

## Quick Start (Run Locally)

### 1. Clone the repo
```bash
git clone https://github.com/PedroVBAmorim/market-dashboard.git
cd market-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get free API keys
- **Alpha Vantage** (stock data): https://www.alphavantage.co/support/#api-key
- **Anthropic** (AI summaries): https://console.anthropic.com

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter your API keys in the sidebar and click **Analyze**.

---

## Deploy to Streamlit Cloud (Free Hosting)

1. Push your code to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set main file as `app.py`
4. Click **Deploy** — done. You get a public URL like `yourapp.streamlit.app`

> ⚠️ Don't hardcode your API keys. Enter them in the sidebar at runtime — they never touch your code.

---

## Project Structure

```
market-dashboard/
├── app.py              # Main Streamlit app (all logic lives here)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🐼 Pandas Crash Course — Everything Used In This Project

This section teaches you every Pandas concept actually used in the code. Read this and you'll understand the entire `compute_metrics()` function.

---

### What is a DataFrame?

A DataFrame is a table — rows and columns, like a spreadsheet — but with superpowers.

```python
import pandas as pd

# Create a DataFrame manually
df = pd.DataFrame({
    "date":  ["2024-01-01", "2024-01-02", "2024-01-03"],
    "close": [150.0,        153.5,         151.2],
    "volume":[10_000_000,   12_500_000,    9_800_000]
})

print(df)
#          date   close      volume
# 0  2024-01-01  150.0  10000000
# 1  2024-01-02  153.5  12500000
# 2  2024-01-03  151.2   9800000
```

Each column is a **Series** — a single column of data with an index.

---

### The Index

The index is the "row label." By default it's 0, 1, 2... but you can set it to dates, strings, anything.

```python
# Set the date column as the index
df = df.set_index("date")

# Now rows are labeled by date
print(df.loc["2024-01-02"])   # get row by date label
# close      153.5
# volume    12500000

# In the project, we do this when loading from Alpha Vantage:
df.index = pd.to_datetime(df.index)   # convert string dates to real datetime objects
df = df.sort_index()                   # sort oldest → newest
```

---

### Selecting Columns

```python
# Single column → returns a Series
closes = df["close"]

# Multiple columns → returns a DataFrame
subset = df[["close", "volume"]]
```

---

### pct_change() — Daily Returns

This is the most important function in the whole project.

```python
# pct_change() computes (current - previous) / previous for every row
df["daily_return"] = df["close"].pct_change()

# Example:
# Day 1: 150.0  →  NaN  (no previous day)
# Day 2: 153.5  →  (153.5 - 150.0) / 150.0 = 0.0233  (+2.33%)
# Day 3: 151.2  →  (151.2 - 153.5) / 153.5 = -0.0150  (-1.50%)
```

---

### Cumulative Return

How much did a $1 investment grow over the whole period?

```python
# (1 + r1) * (1 + r2) * (1 + r3) ... - 1
cumulative = (1 + df["daily_return"].dropna()).prod() - 1

# .dropna() removes the NaN in the first row before multiplying
# .prod() multiplies all values together
```

---

### rolling() — Moving Averages

A rolling window looks at the last N rows at each point.

```python
# 20-day moving average of closing price
df["ma20"] = df["close"].rolling(window=20).mean()

# First 19 rows will be NaN because there aren't 20 days of history yet
# Row 20 onward will have the average of the last 20 closes
```

Think of it like a sliding window that moves one day at a time.

---

### std() — Volatility

Standard deviation measures how much daily returns vary. More variation = more risk.

```python
import numpy as np

# Annualized volatility
# Multiply daily std by sqrt(252) because there are ~252 trading days/year
volatility = df["daily_return"].std() * np.sqrt(252)

# A volatility of 0.30 means 30% annualized — that's fairly high (like a growth stock)
# S&P 500 averages around 15-20% historically
```

---

### cummax() — Max Drawdown

Drawdown measures how far a price has fallen from its peak.

```python
# cummax() returns the running maximum up to each point in time
rolling_max = df["close"].cummax()

# At each day: how far below the peak are we?
drawdown = (df["close"] - rolling_max) / rolling_max

# The worst single drop over the period
max_drawdown = drawdown.min()   # will be a negative number like -0.15 = -15%
```

---

### diff() and clip() — RSI

RSI (Relative Strength Index) measures momentum. Between 0 and 100.

```python
# diff() = today's close minus yesterday's close
delta = df["close"].diff()

# clip() cuts values at a boundary
gain = delta.clip(lower=0)    # keep only positive moves, replace negatives with 0
loss = -delta.clip(upper=0)   # keep only negative moves (flip sign to make positive)

# 14-period rolling average of gains and losses
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))

# RSI > 70: stock may be overbought (due for a pullback)
# RSI < 30: stock may be oversold (potential buying opportunity)
```

---

### tail() — Slicing Recent Data

```python
# Get the last 30 rows (most recent 30 trading days)
recent = df.tail(30)

# You can also slice by date
recent = df.loc["2024-01-01":"2024-03-01"]
```

---

### apply() — Transform Every Column At Once

```python
# Convert all columns from string to float in one line
df = df.apply(pd.to_numeric)

# Equivalent to doing this for every column individually:
# df["open"]   = pd.to_numeric(df["open"])
# df["close"]  = pd.to_numeric(df["close"])
# ... etc
```

---

### from_dict() — Building a DataFrame from a Nested Dict

Alpha Vantage returns JSON that looks like this:

```json
{
  "2024-01-03": {"1. open": "185.0", "4. close": "187.2", ...},
  "2024-01-02": {"1. open": "183.5", "4. close": "185.0", ...}
}
```

```python
# orient="index" means the outer keys (dates) become row labels
df = pd.DataFrame.from_dict(raw_json, orient="index")

# Result:
#            1. open  2. high  3. low  4. close  5. volume
# 2024-01-03   185.0    188.1   184.2     187.2   45000000
# 2024-01-02   183.5    186.0   182.9     185.0   38000000
```

---

### iloc vs loc

```python
# .iloc — position-based (like array indexing)
df.iloc[0]     # first row
df.iloc[-1]    # last row
df.iloc[0:5]   # first 5 rows

# .loc — label-based (use the actual index value)
df.loc["2024-01-03"]              # row with this date
df.loc["2024-01-01":"2024-02-01"] # date range (inclusive on both ends)
```

---

### The Cheat Sheet

| Operation | Code | What it does |
|---|---|---|
| Load CSV | `pd.read_csv("file.csv")` | Create DataFrame from file |
| Select column | `df["close"]` | Get one column as Series |
| Filter rows | `df[df["close"] > 150]` | Keep rows where condition is true |
| Daily returns | `df["close"].pct_change()` | % change row-to-row |
| Moving average | `df["close"].rolling(20).mean()` | 20-day average |
| Cumulative product | `series.prod()` | Multiply all values |
| Running max | `df["close"].cummax()` | Highest value seen so far |
| Std deviation | `df["close"].std()` | Spread of values |
| Remove NaN | `df.dropna()` | Drop rows with missing values |
| Last N rows | `df.tail(30)` | Most recent 30 rows |
| First N rows | `df.head(5)` | First 5 rows |
| Convert type | `pd.to_numeric(series)` | String → number |
| Group & aggregate | `df.groupby("sector").mean()` | Average by group |
| Sort | `df.sort_index()` or `df.sort_values("close")` | Sort rows |

---

## Tech Stack

| Tool | Role | Why |
|---|---|---|
| **Pandas** | Data processing | Industry standard for tabular data in Python |
| **NumPy** | Math operations | Fast array math (annualizing volatility etc.) |
| **Streamlit** | Web UI | Build web apps in pure Python, no HTML/CSS needed |
| **Plotly** | Charts | Interactive, zoomable charts in the browser |
| **Anthropic SDK** | AI summaries | Claude API for plain-English market analysis |
| **Alpha Vantage** | Market data | Free stock data API, no credit card required |

---

## Skills Demonstrated

- **Python** — primary language throughout
- **Pandas** — data ingestion, transformation, financial metric computation
- **REST API integration** — Alpha Vantage fetch with error handling and caching
- **AI API integration** — Anthropic Claude for data-driven text generation
- **Data visualization** — Plotly interactive charts (scatter, line, RSI)
- **Deployment** — Streamlit Cloud (live public URL, zero infrastructure)

---

*Built by Pedro Villas Boas Amorim*
