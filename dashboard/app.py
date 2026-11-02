# run with: streamlit run dashboard/app.py
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("NEPSE_API_URL", "http://127.0.0.1:8000")

NAMES = {
    "NABIL": "Nabil Bank",
    "NICA": "NIC Asia Bank",
    "GBIME": "Global IME Bank",
    "SCB": "Standard Chartered Bank Nepal",
    "SHIVM": "Shivam Cements",
}

st.set_page_config(page_title="NEPSE Stock Price Viewer", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, timeout: int = 20, **params):
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def rs(value: float) -> str:
    return f"Rs {value:,.2f}"


def describe_move(change_pct: float) -> str:
    if abs(change_pct) < 0.5:
        return "almost no change"
    word = "rise" if change_pct > 0 else "fall"
    return f"a {abs(change_pct):.1f}% {word}"


def next_trading_day(dates: pd.Series) -> pd.Timestamp:
    # weekdays seen in the recent data count as trading days (holidays ignored)
    counts = dates.tail(60).dt.day_of_week.value_counts()
    open_days = set(counts[counts >= 3].index)
    day = dates.iloc[-1] + pd.Timedelta(days=1)
    for _ in range(7):
        if day.weekday() in open_days:
            return day
        day += pd.Timedelta(days=1)
    return dates.iloc[-1] + pd.Timedelta(days=1)


def build_chart(hist, detailed, pred, metrics, record_df, next_date):
    fig = go.Figure()
    if detailed:
        fig.add_trace(
            go.Candlestick(
                x=hist["date"],
                open=hist["open"],
                high=hist["high"],
                low=hist["low"],
                close=hist["close"],
                name="Daily prices",
            )
        )
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["ma_5"], name="5-day average"))
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["ma_20"], name="20-day average"))
    else:
        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["close"],
                mode="lines+markers",
                name="Real closing price",
                line={"width": 2},
                marker={"size": 5},
            )
        )
    if record_df is not None:
        fig.add_trace(
            go.Scatter(
                x=record_df["date"],
                y=record_df["predicted_close"],
                mode="lines",
                name="Model's guess for each past day (made the day before)",
                line={"dash": "dot", "color": "#e8590c", "width": 2},
            )
        )
    if pred:
        fig.add_trace(
            go.Scatter(
                x=[hist["date"].iloc[-1], next_date],
                y=[pred["current_close"], pred["predicted_close"]],
                mode="lines",
                name="One step ahead",
                line={"dash": "dash", "color": "#868e96", "width": 1},
                showlegend=False,
            )
        )
        guess = {
            "x": [next_date],
            "y": [pred["predicted_close"]],
            "mode": "markers",
            "name": "Guess for the next trading day (with typical miss)",
            "marker": {"size": 14, "symbol": "diamond", "color": "#e8590c"},
        }
        if metrics:
            guess["error_y"] = {
                "type": "constant",
                "value": metrics["model_rmse"],
                "visible": True,
            }
        fig.add_trace(go.Scatter(**guess))
        fig.add_annotation(
            x=next_date,
            y=pred["predicted_close"],
            text=f"Next trading day<br>guess: {rs(pred['predicted_close'])}",
            showarrow=True,
            arrowhead=2,
            ax=-70,
            ay=-60,
            font={"color": "#e8590c"},
        )
    fig.update_layout(
        height=450,
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title="Price (Rs)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.15},
    )
    return fig


st.title("NEPSE Stock Price Viewer")
st.caption(
    "Recent prices and a guess for the next trading day from a simple model. "
    "Educational project. Not financial advice."
)

try:
    stocks = api_get("/stocks")
except requests.RequestException as exc:
    st.error(f"Cannot reach the data service at {API_URL}. Start the API first. ({exc})")
    st.stop()

if not stocks:
    st.warning("No price data loaded yet. Run: python -m app.loader data/")
    st.stop()

symbol = st.sidebar.selectbox(
    "Choose a stock",
    [s["symbol"] for s in stocks],
    format_func=lambda s: f"{s} - {NAMES.get(s, s)}",
)
days = st.sidebar.slider(
    "How many past days to show",
    10,
    120,
    20,
    help="Only the past is shown. The model never looks further ahead than one trading day.",
)
detailed = st.sidebar.checkbox(
    "Show candlesticks and averages",
    value=False,
    help="Advanced chart used by traders. Leave off for a simple price line.",
)
st.sidebar.info("The model guesses **one trading day ahead only**.")

hist_all = pd.DataFrame(api_get(f"/stocks/{symbol}/history", limit=max(days + 25, 60)))
hist_all["date"] = pd.to_datetime(hist_all["date"])
hist_all["ma_5"] = hist_all["close"].rolling(5).mean()
hist_all["ma_20"] = hist_all["close"].rolling(20).mean()
next_date = next_trading_day(hist_all["date"])
hist = hist_all.tail(days).reset_index(drop=True)

pred, pred_problem = None, None
try:
    with st.spinner("Loading the model's guess (the first request can be slow)..."):
        pred = api_get(f"/stocks/{symbol}/predict", timeout=90)
except requests.HTTPError as exc:
    status = exc.response.status_code if exc.response is not None else "?"
    if status == 503:
        pred_problem = "No trained model for this stock yet."
    else:
        pred_problem = f"The guess could not be loaded (error {status})."
except requests.RequestException as exc:
    pred_problem = f"The guess could not be loaded: {exc}"

try:
    metrics = api_get(f"/stocks/{symbol}/metrics")
except requests.RequestException:
    metrics = None

try:
    record = api_get(f"/stocks/{symbol}/track-record", timeout=60, days=days)
except requests.RequestException:
    record = None
record_df = None
if record:
    record_df = pd.DataFrame(record["rows"])
    record_df["date"] = pd.to_datetime(record_df["date"])

name = NAMES.get(symbol, symbol)
next_label = next_date.strftime("%A, %d %b %Y")

if pred:
    st.markdown(
        f"### {name} closed at **{rs(pred['current_close'])}** on {pred['as_of']}.\n"
        f"The model's guess for the **next trading day ({next_label})** is "
        f"**{rs(pred['predicted_close'])}**, which means {describe_move(pred['change_pct'])}."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Last closing price", rs(pred["current_close"]))
    c2.metric(
        f"Guess for {next_date.strftime('%d %b')}",
        rs(pred["predicted_close"]),
        f"{pred['change_pct']:+.2f}%",
        delta_color="off",
    )
    if metrics:
        c3.metric(
            "Typical daily miss of the model",
            rs(metrics["model_rmse"]),
            help="Average size of the model's mistakes in past tests. Lower is better.",
        )
else:
    st.subheader(name)
    st.info(pred_problem or "No guess available.")

st.subheader(f"Last {days} trading days and the guess for {next_date.strftime('%d %b')}")
st.plotly_chart(
    build_chart(hist, detailed, pred, metrics, record_df, next_date),
    use_container_width=True,
)
st.caption(
    "Left of the orange diamond is the past. The solid line is the real price and the "
    "dotted line is what the model would have guessed for each day, one day ahead. "
    "The diamond is the only forecast: the next trading day. The line through it shows "
    "the model's typical miss. Holidays are not taken into account."
)

if record_df is not None:
    st.subheader("Real price vs the model's guess")
    last = record["rows"][-1]
    miss_model = abs(last["actual_close"] - last["predicted_close"])
    miss_rule = abs(last["actual_close"] - last["previous_close"])
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(f"Real close on {last['date']}", rs(last["actual_close"]))
    r2.metric("Model's guess for that day", rs(last["predicted_close"]))
    r3.metric("Model's miss", rs(miss_model))
    r4.metric(
        "Miss of 'same as yesterday'",
        rs(miss_rule),
        help="What you get by guessing the price will not change.",
    )
    st.write(
        f"Over the last {record['days']} trading days the model's guess was closer "
        f"to the real price than 'same as yesterday' on "
        f"**{record['model_closer_days']} of {record['days']} days**."
    )
    with st.expander("Day-by-day table: real price vs guess"):
        table = pd.DataFrame(
            {
                "Date": record_df["date"].dt.date,
                "Real close": record_df["actual_close"],
                "Model's guess": record_df["predicted_close"],
                "Model's miss": (record_df["actual_close"] - record_df["predicted_close"])
                .abs()
                .round(2),
                "'Same as yesterday' guess": record_df["previous_close"],
                "Its miss": (record_df["actual_close"] - record_df["previous_close"])
                .abs()
                .round(2),
            }
        )
        st.dataframe(table.iloc[::-1], hide_index=True, use_container_width=True)
        st.caption(record["note"])

if pred:
    with st.expander("Technical details (for people who know trading terms)"):
        st.write(f"Rule-based hint: **{pred['signal']}**")
        for reason in pred["reasons"]:
            st.write(f"- {reason}")
        st.caption(
            "The hint comes from simple rules on trend and RSI. "
            "It is not a recommendation to buy or sell."
        )


# ---- direction model and strategy reality check ----
st.divider()
st.subheader("Will it go up tomorrow? And would trading on that have worked?")
try:
    dirn = api_get(f"/stocks/{symbol}/direction", timeout=60)
    strat = api_get(f"/stocks/{symbol}/strategy", timeout=60)
except requests.RequestException:
    dirn = strat = None
    st.info("Direction model not available for this stock.")

if dirn and strat:
    ev = dirn["evidence"]
    d1, d2, d3 = st.columns(3)
    d1.metric("Chance of going up tomorrow", f"{dirn['prob_up'] * 100:.0f}%")
    d2.metric(
        "Past hit rate",
        f"{ev['accuracy'] * 100:.1f}%",
        f"{ev['edge_points']:+.1f} pts vs always guessing the common direction",
        delta_color="off",
    )
    d3.metric("Could it be luck?", f"p = {ev['p_value']:.2f}", help="Below 0.05 is the usual bar.")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Trading on it (after fees and tax)", f"{strat['strategy_net_pct']:+.1f}%")
    t2.metric("Same, before fees", f"{strat['strategy_gross_pct']:+.1f}%")
    t3.metric("Just buying and holding", f"{strat['buy_hold_net_pct']:+.1f}%")
    t4.metric("Number of trades", strat["trades"])
