import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quotex Remix AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #080d18; color: #eef2ff; }
      [data-testid="stSidebar"] { background: #101827; }
      .hero { padding: 8px 0 14px 0; }
      .hero h1 { font-size: clamp(2rem, 5vw, 3.25rem); margin: 0; color: #f5f7ff; }
      .hero p { color: #aab5c8; margin-top: 6px; }
      .signal-card { border: 1px solid #273449; border-radius: 16px; padding: 18px;
                     background: #111b2b; margin: 8px 0 14px 0; }
      .signal-main { font-size: 1.8rem; font-weight: 800; letter-spacing: .03em; }
      .muted { color: #9aa8bd; }
      .engine-card { border: 1px solid #273449; border-radius: 14px; padding: 14px;
                     background: #101827; min-height: 118px; margin-bottom: 10px; }
      .engine-name { color: #edf2ff; font-size: 1rem; font-weight: 750; }
      .engine-state { font-size: 1.05rem; font-weight: 800; margin: 7px 0; }
      .engine-detail { color: #aab5c8; font-size: .88rem; }
      div[data-testid="stMetric"] { background: #101827; border: 1px solid #273449;
          padding: 12px 14px; border-radius: 13px; }
      div[data-testid="stMetricLabel"] { color: #aab5c8; }
      .footnote { font-size: .86rem; line-height: 1.5; color: #9aa8bd; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data access
# -----------------------------------------------------------------------------
def get_api_key() -> str | None:
    """Read the Twelve Data key from Hugging Face/Streamlit secrets or env."""
    try:
        key = st.secrets.get("TWELVE_DATA_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.getenv("TWELVE_DATA_API_KEY", "").strip() or None


@st.cache_data(ttl=20, show_spinner=False)
def fetch_candles(symbol: str, interval: str, api_key: str) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 150,
        "format": "JSON",
        "apikey": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Twelve Data: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Twelve Data returned a response that was not valid JSON.") from exc

    if payload.get("status") == "error" or "values" not in payload:
        message = payload.get("message") or payload.get("code") or "Unknown API response"
        raise RuntimeError(f"Twelve Data error: {message}")

    df = pd.DataFrame(payload["values"])
    required = {"datetime", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise RuntimeError("The market-data response is missing OHLC fields.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for column in ("open", "high", "low", "close"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "open", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )
    if len(df) < 60:
        raise RuntimeError(
            f"Only {len(df)} candles arrived. At least 60 are needed for the six-engine analysis."
        )
    return df


# -----------------------------------------------------------------------------
# Six rule-based analysis engines
# -----------------------------------------------------------------------------
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data["close"]

    data["ema9"] = close.ewm(span=9, adjust=False).mean()
    data["ema21"] = close.ewm(span=21, adjust=False).mean()
    data["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))
    data.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100
    data.loc[(avg_loss == 0) & (avg_gain == 0), "rsi"] = 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    data["bb_mid"] = close.rolling(20).mean()
    data["bb_std"] = close.rolling(20).std(ddof=0)
    data["bb_upper"] = data["bb_mid"] + 2 * data["bb_std"]
    data["bb_lower"] = data["bb_mid"] - 2 * data["bb_std"]
    return data


def run_engines(data: pd.DataFrame) -> tuple[list[dict], str, int, int]:
    last = data.iloc[-1]
    prev = data.iloc[-2]
    older3 = data.iloc[-4]
    older8 = data.iloc[-9]
    engines: list[dict] = []

    # 1. EMA trend alignment
    if last["ema9"] > last["ema21"] > last["ema50"] and last["close"] > last["ema9"]:
        state, detail = "BULLISH", "EMA 9 > 21 > 50; price above EMA 9"
    elif last["ema9"] < last["ema21"] < last["ema50"] and last["close"] < last["ema9"]:
        state, detail = "BEARISH", "EMA 9 < 21 < 50; price below EMA 9"
    else:
        state, detail = "NEUTRAL", "EMA trend lines are not fully aligned"
    engines.append({"name": "TITAN X", "state": state, "detail": detail})

    # 2. RSI momentum regime (not an overbought/oversold reversal promise)
    rsi = float(last["rsi"])
    if rsi >= 55:
        state, detail = "BULLISH", f"RSI {rsi:.1f} — positive momentum zone"
    elif rsi <= 45:
        state, detail = "BEARISH", f"RSI {rsi:.1f} — negative momentum zone"
    else:
        state, detail = "NEUTRAL", f"RSI {rsi:.1f} — middle zone"
    engines.append({"name": "QUANTUM STRIKE", "state": state, "detail": detail})

    # 3. MACD momentum
    if last["macd_hist"] > 0 and last["macd"] > last["macd_signal"]:
        state, detail = "BULLISH", "MACD is above its signal line"
    elif last["macd_hist"] < 0 and last["macd"] < last["macd_signal"]:
        state, detail = "BEARISH", "MACD is below its signal line"
    else:
        state, detail = "NEUTRAL", "MACD momentum is mixed"
    engines.append({"name": "NEXUS FUSION", "state": state, "detail": detail})

    # 4. 3-candle + 8-candle price momentum
    if last["close"] > older3["close"] and last["close"] > older8["close"]:
        state, detail = "BULLISH", "Price is above both 3- and 8-candle references"
    elif last["close"] < older3["close"] and last["close"] < older8["close"]:
        state, detail = "BEARISH", "Price is below both 3- and 8-candle references"
    else:
        state, detail = "NEUTRAL", "Short- and medium-momentum do not agree"
    engines.append({"name": "OMEGA QX", "state": state, "detail": detail})

    # 5. Bollinger middle-line direction and slope
    if last["close"] > last["bb_mid"] and last["bb_mid"] > data.iloc[-4]["bb_mid"]:
        state, detail = "BULLISH", "Price above rising Bollinger middle line"
    elif last["close"] < last["bb_mid"] and last["bb_mid"] < data.iloc[-4]["bb_mid"]:
        state, detail = "BEARISH", "Price below falling Bollinger middle line"
    else:
        state, detail = "NEUTRAL", "Bollinger direction is not clear"
    engines.append({"name": "APEX VISION", "state": state, "detail": detail})

    # 6. Latest candle structure and wick rejection
    candle_range = max(float(last["high"] - last["low"]), 1e-12)
    body = abs(float(last["close"] - last["open"]))
    upper_wick = float(last["high"] - max(last["open"], last["close"]))
    lower_wick = float(min(last["open"], last["close"]) - last["low"])
    if lower_wick / candle_range >= 0.45 and last["close"] >= last["open"]:
        state, detail = "BULLISH", "Long lower wick with a non-bearish close"
    elif upper_wick / candle_range >= 0.45 and last["close"] <= last["open"]:
        state, detail = "BEARISH", "Long upper wick with a non-bullish close"
    elif body / candle_range >= 0.45 and last["close"] > last["open"] and last["close"] >= prev["close"]:
        state, detail = "BULLISH", "Strong bullish candle body and higher close"
    elif body / candle_range >= 0.45 and last["close"] < last["open"] and last["close"] <= prev["close"]:
        state, detail = "BEARISH", "Strong bearish candle body and lower close"
    else:
        state, detail = "NEUTRAL", "Latest candle has no clear rejection pattern"
    engines.append({"name": "PHOENIX STRUCTURE", "state": state, "detail": detail})

    bullish = sum(item["state"] == "BULLISH" for item in engines)
    bearish = sum(item["state"] == "BEARISH" for item in engines)

    # A signal is displayed only when at least 5 of 6 engines agree and none oppose.
    if bullish >= 5 and bearish == 0:
        final_signal = "UP BIAS"
    elif bearish >= 5 and bullish == 0:
        final_signal = "DOWN BIAS"
    else:
        final_signal = "WAIT"
    return engines, final_signal, bullish, bearish


def make_candle_chart(data: pd.DataFrame) -> go.Figure:
    view = data.tail(90)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=view["datetime"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="Price",
            increasing_line_color="#27c58b",
            decreasing_line_color="#ff5b67",
        )
    )
    fig.add_trace(go.Scatter(x=view["datetime"], y=view["ema9"], name="EMA 9", line=dict(width=1.4, color="#f5c451")))
    fig.add_trace(go.Scatter(x=view["datetime"], y=view["ema21"], name="EMA 21", line=dict(width=1.4, color="#6aa9ff")))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080d18",
        plot_bgcolor="#080d18",
        height=430,
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.08, x=0),
        font=dict(color="#dce5f5"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#243044", side="right")
    return fig


# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Dashboard settings")
symbol = st.sidebar.selectbox(
    "Market",
    ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "EUR/JPY", "GBP/JPY", "XAU/USD", "BTC/USD"],
    index=0,
)
interval = st.sidebar.selectbox("Timeframe", ["1min", "5min", "15min"], index=0)
auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
refresh_seconds = st.sidebar.select_slider("Refresh interval", options=[30, 60, 90, 120], value=60)
if st.sidebar.button("↻ Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.markdown(
    """
    <div class="hero">
      <h1>⚡ QUOTEX REMIX AI</h1>
      <p>Live market-data dashboard · Six independent rule-based analysis engines</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = get_api_key()
if not api_key:
    st.error("Twelve Data API key not found. Add TWELVE_DATA_API_KEY in your Hugging Face Space → Settings → Secrets, then restart the Space.")
    st.stop()

run_every = refresh_seconds if auto_refresh else None


@st.fragment(run_every=run_every)
def live_dashboard():
    try:
        raw = fetch_candles(symbol, interval, api_key)
        data = build_indicators(raw)
        engines, final_signal, bullish, bearish = run_engines(data)
    except Exception as exc:
        st.error(str(exc))
        st.info("Check the Twelve Data key, API quota, selected symbol, and Space logs.")
        return

    latest = data.iloc[-1]
    signal_color = "#27c58b" if final_signal == "UP BIAS" else "#ff5b67" if final_signal == "DOWN BIAS" else "#f3c64e"
    signal_help = (
        "5 or more engines agree; this is still not a guaranteed next-candle result."
        if final_signal != "WAIT"
        else "The six engines do not have the required agreement. Avoid forcing a direction."
    )

    st.subheader("Live market snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market", symbol)
    c2.metric("Timeframe", interval.upper())
    c3.metric("Latest close", f"{latest['close']:.5f}" if latest["close"] < 100 else f"{latest['close']:.2f}")
    c4.metric("Engine agreement", f"{max(bullish, bearish)}/6")

    st.subheader("Live candlestick chart")
    st.plotly_chart(make_candle_chart(data), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    st.subheader("Final AI analysis")
    st.markdown(
        f"""
        <div class="signal-card">
          <div class="muted">CURRENT RULE-BASED READING</div>
          <div class="signal-main" style="color:{signal_color};">{final_signal}</div>
          <div class="muted">{bullish} bullish · {bearish} bearish · {6 - bullish - bearish} neutral</div>
          <div class="muted" style="margin-top:8px;">{signal_help}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Six analysis engines")
    cols = st.columns(2)
    for index, engine in enumerate(engines):
        state_color = "#27c58b" if engine["state"] == "BULLISH" else "#ff5b67" if engine["state"] == "BEARISH" else "#aab5c8"
        with cols[index % 2]:
            st.markdown(
                f"""
                <div class="engine-card">
                  <div class="engine-name">{engine['name']}</div>
                  <div class="engine-state" style="color:{state_color};">{engine['state']}</div>
                  <div class="engine-detail">{engine['detail']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(f"Last candle timestamp from data provider: {latest['datetime']} · Updated on screen: {datetime.now().strftime('%H:%M:%S')}")


live_dashboard()

st.markdown("---")
st.markdown(
    "<div class='footnote'><b>Important:</b> This is an educational technical-analysis tool, not financial advice. Engine agreement is not a win probability. It cannot guarantee the next candle. Twelve Data prices may differ from Quotex prices, especially OTC instruments. Verify the same symbol and feed on Quotex before making any decision. No trades are placed by this app.</div>",
    unsafe_allow_html=True,
)
