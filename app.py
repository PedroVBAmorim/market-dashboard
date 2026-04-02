import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import anthropic
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Insight Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Market Insight Dashboard")
st.caption("Real-time stock analysis powered by Python + AI")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Alpha Vantage API Key", type="password",
                            help="Free key at alphavantage.co")
    claude_key = st.text_input("Anthropic API Key", type="password",
                               help="Get one at console.anthropic.com")
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, NVDA")
    days = st.slider("Days of history", min_value=7, max_value=90, value=30)
    run = st.button("🚀 Analyze", use_container_width=True)

# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)   # cache for 5 minutes so we don't hammer the free API
def fetch_daily(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Fetches daily OHLCV data from Alpha Vantage.
    Returns a DataFrame indexed by date with columns:
    open, high, low, close, volume
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={ticker}"
        f"&outputsize=compact&apikey={api_key}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    if "Time Series (Daily)" not in data:
        st.warning(f"Could not fetch {ticker}: {data.get('Note') or data.get('Information') or 'Unknown error'}")
        return pd.DataFrame()

    raw = data["Time Series (Daily)"]

    # --- THIS IS THE CORE PANDAS PATTERN YOU NEED TO UNDERSTAND ---
    # pd.DataFrame.from_dict turns a nested dict into a DataFrame
    # orient="index" means the outer keys (dates) become the row index
    df = pd.DataFrame.from_dict(raw, orient="index")

    # Rename columns — Alpha Vantage prefixes them with "1. open" etc.
    df.columns = ["open", "high", "low", "close", "volume"]

    # Convert every column to numeric (they come in as strings from JSON)
    df = df.apply(pd.to_numeric)

    # Convert the index from strings to real datetime objects
    df.index = pd.to_datetime(df.index)

    # Sort oldest → newest (API returns newest first)
    df = df.sort_index()

    return df


def compute_metrics(df: pd.DataFrame, days: int) -> dict:
    """
    Given a full history DataFrame, slice to `days` and compute key metrics.
    This is where Pandas really shines — one-liners for complex math.
    """
    # Slice to the last N trading days
    df = df.tail(days).copy()

    # pct_change() computes (today - yesterday) / yesterday for every row
    # This gives us daily returns as decimals e.g. 0.012 = +1.2%
    df["daily_return"] = df["close"].pct_change()

    # Cumulative return: how much did $1 grow over the period?
    # (1 + r1) * (1 + r2) * ... - 1
    cumulative_return = (1 + df["daily_return"].dropna()).prod() - 1

    # Rolling 20-day average — smooths out noise
    df["ma20"] = df["close"].rolling(window=20).mean()

    # Volatility = standard deviation of daily returns, annualized
    # Multiply by sqrt(252) because there are ~252 trading days in a year
    volatility = df["daily_return"].std() * np.sqrt(252)

    # Max drawdown: biggest peak-to-trough drop in the period
    rolling_max = df["close"].cummax()           # running highest price so far
    drawdown = (df["close"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Simple RSI (14-period Relative Strength Index)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)                   # keep only positive moves
    loss = -delta.clip(upper=0)                  # keep only negative moves (flip sign)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return {
        "df": df,
        "cumulative_return": cumulative_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "latest_rsi": df["rsi"].iloc[-1],
        "latest_close": df["close"].iloc[-1],
        "avg_volume": df["volume"].mean(),
        "start_price": df["close"].iloc[0],
    }


def generate_ai_summary(metrics_by_ticker: dict, days: int, claude_key: str) -> str:
    """
    Sends a structured data summary to Claude and asks for a plain-English analysis.
    """
    client = anthropic.Anthropic(api_key=claude_key)

    # Build a compact text summary of the metrics to send as context
    lines = [f"Stock performance summary over the last {days} trading days:\n"]
    for ticker, m in metrics_by_ticker.items():
        lines.append(
            f"{ticker}: Price ${m['latest_close']:.2f} | "
            f"Return {m['cumulative_return']*100:.1f}% | "
            f"Volatility {m['volatility']*100:.1f}% annualized | "
            f"Max Drawdown {m['max_drawdown']*100:.1f}% | "
            f"RSI {m['latest_rsi']:.1f}"
        )

    prompt = "\n".join(lines) + (
        "\n\nWrite a concise 3-paragraph market analysis for an investor. "
        "Paragraph 1: overall performance summary. "
        "Paragraph 2: standout winners and losers with reasons why. "
        "Paragraph 3: key risks to watch based on the volatility and RSI data. "
        "Use plain English. No bullet points. No markdown headers."
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# ── Main dashboard ────────────────────────────────────────────────────────────
if not run:
    st.info("👈 Enter your API keys in the sidebar and click **Analyze** to get started.")
    st.markdown("""
    **What this dashboard does:**
    - Fetches real stock data from Alpha Vantage (free API)
    - Computes returns, volatility, RSI, and drawdown using Pandas
    - Visualizes everything with interactive Plotly charts
    - Generates a plain-English AI market summary via Claude

    **Free API keys:**
    - Alpha Vantage: [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
    - Anthropic: [console.anthropic.com](https://console.anthropic.com)
    """)
    st.stop()

if not api_key or not claude_key:
    st.error("Please enter both API keys in the sidebar.")
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# ── Fetch and process ──────────────────────────────────────────────────────────
all_metrics = {}
with st.spinner("Fetching market data..."):
    for ticker in tickers:
        df_full = fetch_daily(ticker, api_key)
        if not df_full.empty:
            all_metrics[ticker] = compute_metrics(df_full, days)

if not all_metrics:
    st.error("No data returned. Check your API key and tickers.")
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.subheader("📊 Key Metrics")
cols = st.columns(len(all_metrics))
for col, (ticker, m) in zip(cols, all_metrics.items()):
    ret = m["cumulative_return"] * 100
    col.metric(
        label=ticker,
        value=f"${m['latest_close']:.2f}",
        delta=f"{ret:+.1f}% ({days}d)"
    )

st.divider()

# ── Price chart ────────────────────────────────────────────────────────────────
st.subheader("📉 Normalized Price Performance")
st.caption("All prices indexed to 100 at start of period so you can compare stocks fairly")

fig_price = go.Figure()
for ticker, m in all_metrics.items():
    df = m["df"]
    # Index to 100: divide every close by the starting close, multiply by 100
    normalized = (df["close"] / m["start_price"]) * 100
    fig_price.add_trace(go.Scatter(
        x=df.index, y=normalized,
        name=ticker, mode="lines", line=dict(width=2)
    ))

fig_price.update_layout(
    yaxis_title="Indexed Price (start = 100)",
    hovermode="x unified",
    height=380,
    margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig_price, use_container_width=True)

# ── Returns vs Volatility scatter ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Return vs. Risk")
    scatter_data = pd.DataFrame({
        "Ticker": list(all_metrics.keys()),
        "Return (%)": [m["cumulative_return"] * 100 for m in all_metrics.values()],
        "Volatility (%)": [m["volatility"] * 100 for m in all_metrics.values()],
        "RSI": [m["latest_rsi"] for m in all_metrics.values()],
    })
    fig_scatter = px.scatter(
        scatter_data, x="Volatility (%)", y="Return (%)",
        text="Ticker", size="RSI", color="Return (%)",
        color_continuous_scale="RdYlGn",
        height=320
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("📋 Full Metrics Table")
    table_data = pd.DataFrame({
        "Ticker": list(all_metrics.keys()),
        "Price": [f"${m['latest_close']:.2f}" for m in all_metrics.values()],
        "Return": [f"{m['cumulative_return']*100:+.1f}%" for m in all_metrics.values()],
        "Volatility": [f"{m['volatility']*100:.1f}%" for m in all_metrics.values()],
        "Max DD": [f"{m['max_drawdown']*100:.1f}%" for m in all_metrics.values()],
        "RSI": [f"{m['latest_rsi']:.0f}" for m in all_metrics.values()],
    })
    st.dataframe(table_data, hide_index=True, use_container_width=True)

# ── RSI chart ──────────────────────────────────────────────────────────────────
st.subheader("📡 RSI (Relative Strength Index)")
st.caption("Above 70 = overbought (potential sell signal) · Below 30 = oversold (potential buy signal)")

fig_rsi = go.Figure()
for ticker, m in all_metrics.items():
    df = m["df"]
    fig_rsi.add_trace(go.Scatter(
        x=df.index, y=df["rsi"],
        name=ticker, mode="lines", line=dict(width=1.5)
    ))

fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
fig_rsi.update_layout(yaxis=dict(range=[0, 100]), height=280, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_rsi, use_container_width=True)

# ── AI Summary ────────────────────────────────────────────────────────────────
st.subheader("🤖 AI Market Analysis")
with st.spinner("Generating insights with Claude..."):
    try:
        summary = generate_ai_summary(all_metrics, days, claude_key)
        st.info(summary)
    except Exception as e:
        st.error(f"AI summary failed: {e}")

st.caption(f"Data sourced from Alpha Vantage · Analysis generated {datetime.now().strftime('%B %d, %Y %H:%M')}")
