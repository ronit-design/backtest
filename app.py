import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Backtester", layout="wide")

NUMERIC_OPS = [">", "<", ">=", "<=", "=="]
CROSS_OPS = ["crosses above", "crosses below"]
ALL_OPS = NUMERIC_OPS + CROSS_OPS

FREQ_MAP = {"Daily": 252, "Weekly": 52, "Monthly": 12, "Quarterly": 4}

# ── core logic ────────────────────────────────────────────────────────────────

def evaluate_condition(df, col, op, val):
    s = df[col]
    if op == ">":            return s > val
    if op == "<":            return s < val
    if op == ">=":           return s >= val
    if op == "<=":           return s <= val
    if op == "==":           return s == val
    if op == "crosses above": return (s > val) & (s.shift(1) <= val)
    if op == "crosses below": return (s < val) & (s.shift(1) >= val)
    return pd.Series(False, index=df.index)


def build_signal(df, conditions):
    if not conditions:
        return pd.Series(False, index=df.index)
    sig = pd.Series(True, index=df.index)
    for c in conditions:
        sig &= evaluate_condition(df, c["col"], c["op"], c["val"])
    return sig


def run_backtest(df, buy_conds, sell_conds, initial_capital=10_000):
    buy_sig  = build_signal(df, buy_conds)
    sell_sig = build_signal(df, sell_conds)

    position = [0] * len(df)
    in_pos = False
    trades = []  # (entry_idx, exit_idx)
    entry_i = None

    for i in range(len(df)):
        if not in_pos and buy_sig.iloc[i]:
            in_pos = True
            entry_i = i
        elif in_pos and sell_sig.iloc[i]:
            in_pos = False
            trades.append((entry_i, i))
            entry_i = None
        position[i] = 1 if in_pos else 0

    if in_pos and entry_i is not None:
        trades.append((entry_i, len(df) - 1))

    pos_series = pd.Series(position, index=df.index)
    price_ret  = df["close"].pct_change().fillna(0)

    # shift(1): signal seen at bar close, fills next bar
    strat_ret  = pos_series.shift(1).fillna(0) * price_ret

    strat_eq   = initial_capital * (1 + strat_ret).cumprod()
    bh_eq      = initial_capital * (1 + price_ret).cumprod()

    return {
        "strat_ret":  strat_ret,
        "bh_ret":     price_ret,
        "strat_eq":   strat_eq,
        "bh_eq":      bh_eq,
        "position":   pos_series,
        "trades":     trades,
        "buy_sig":    buy_sig,
        "sell_sig":   sell_sig,
    }


def calc_metrics(ret, equity, ppy):
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n = len(ret)
    ann_ret  = (1 + total_ret) ** (ppy / n) - 1
    ann_vol  = ret.std() * np.sqrt(ppy)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    roll_max = equity.cummax()
    max_dd   = ((equity - roll_max) / roll_max).min()
    return {
        "Total Return":     total_ret,
        "Ann. Return":      ann_ret,
        "Ann. Volatility":  ann_vol,
        "Sharpe Ratio":     sharpe,
        "Max Drawdown":     max_dd,
    }


def forward_return_stats(df, signal: pd.Series, periods: int) -> dict | None:
    """Cumulative forward return over `periods` bars after each signal fire."""
    close = df["close"]
    signal_indices = np.where(signal.values)[0]
    fwd_rets = []
    for i in signal_indices:
        end = i + periods
        if end >= len(close):
            end = len(close) - 1
        if end > i:
            fwd_rets.append(close.iloc[end] / close.iloc[i] - 1)
    if not fwd_rets:
        return None
    arr = np.array(fwd_rets)
    # mode: bucket to nearest 1%
    rounded = np.round(arr * 100).astype(int)
    counts  = pd.Series(rounded).value_counts()
    mode_val = counts.index[0] / 100
    return {
        "n_signals": len(arr),
        "mean":      arr.mean(),
        "median":    np.median(arr),
        "mode":      mode_val,
        "min":       arr.min(),
    }


# ── UI helpers ────────────────────────────────────────────────────────────────

def condition_row(key_prefix, numeric_cols, label):
    cols = st.columns([2, 1.5, 1.5])
    col  = cols[0].selectbox("Column",   numeric_cols,  key=f"{key_prefix}_col",   label_visibility="collapsed")
    op   = cols[1].selectbox("Operator", ALL_OPS,       key=f"{key_prefix}_op",    label_visibility="collapsed")
    val  = cols[2].number_input("Value", value=0.0,     key=f"{key_prefix}_val",   label_visibility="collapsed",
                                format="%.4f", step=0.01)
    return {"col": col, "op": op, "val": val}


def format_pct(v):  return f"{v:.2%}"
def format_f2(v):   return f"{v:.2f}" if not np.isnan(v) else "—"


# ── main app ──────────────────────────────────────────────────────────────────

st.title("Strategy Backtester")
st.caption("Upload your OHLCV + indicator CSV, define buy/sell rules, compare vs buy & hold.")

# ── Step 1: upload ────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV", type="csv")
if not uploaded:
    st.info("Upload a CSV with columns: time, open, high, low, close, Volume, ROC, 2nd Derivative (Acceleration)")
    st.stop()

df_raw = pd.read_csv(uploaded)

# normalise column names
df_raw.columns = df_raw.columns.str.strip()

required = {"time", "open", "high", "low", "close"}
missing  = required - set(df_raw.columns.str.lower())
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df_raw.columns = [c if c.lower() not in {"time","open","high","low","close","volume"}
                  else c.lower() for c in df_raw.columns]

df_raw["time"] = pd.to_datetime(df_raw["time"])
df_raw = df_raw.sort_values("time").reset_index(drop=True)
df_raw = df_raw.set_index("time")

if "volume" in df_raw.columns:
    roll_avg = df_raw["volume"].rolling(12, min_periods=1).mean()
    df_raw["Vol / 12M Avg"] = df_raw["volume"] / roll_avg

numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

with st.expander("Data preview", expanded=False):
    st.dataframe(df_raw.tail(20), use_container_width=True)

st.divider()

# ── Step 2: strategy builder ─────────────────────────────────────────────────
st.subheader("Strategy Rules")

freq_choice = st.selectbox("Data frequency (for annualisation)", list(FREQ_MAP.keys()), index=2)
ppy = FREQ_MAP[freq_choice]
capital = st.number_input("Starting capital ($)", value=10_000, step=1_000)

left, right = st.columns(2)

with left:
    st.markdown("**Buy conditions** (all must be true)")
    n_buy = st.number_input("# buy conditions", 1, 5, 1, key="n_buy")
    buy_conds = []
    for i in range(int(n_buy)):
        st.caption(f"Buy condition {i+1}")
        buy_conds.append(condition_row(f"buy_{i}", numeric_cols, f"Buy {i+1}"))

with right:
    st.markdown("**Sell conditions** (all must be true)")
    n_sell = st.number_input("# sell conditions", 1, 5, 1, key="n_sell")
    sell_conds = []
    for i in range(int(n_sell)):
        st.caption(f"Sell condition {i+1}")
        sell_conds.append(condition_row(f"sell_{i}", numeric_cols, f"Sell {i+1}"))

run = st.button("Run Backtest", type="primary", use_container_width=True)
if not run:
    st.stop()

# ── Step 3: run & display ─────────────────────────────────────────────────────
with st.spinner("Running backtest…"):
    res = run_backtest(df_raw, buy_conds, sell_conds, initial_capital=capital)

strat_metrics = calc_metrics(res["strat_ret"],  res["strat_eq"],  ppy)
bh_metrics    = calc_metrics(res["bh_ret"],     res["bh_eq"],     ppy)
n_trades      = len(res["trades"])
time_in_mkt   = res["position"].mean()

# ── metrics table ─────────────────────────────────────────────────────────────
st.subheader("Performance Summary")

metric_cols = st.columns(5)
labels = ["Total Return", "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown"]
for col_ui, label in zip(metric_cols, labels):
    sv = strat_metrics[label]
    bv = bh_metrics[label]
    if label == "Sharpe Ratio":
        sv_str, bv_str = format_f2(sv), format_f2(bv)
        delta = f"{sv - bv:+.2f} vs B&H" if not np.isnan(sv) and not np.isnan(bv) else ""
    else:
        sv_str, bv_str = format_pct(sv), format_pct(bv)
        delta = f"{sv - bv:+.2%} vs B&H"
    col_ui.metric(label, sv_str, delta, help=f"Buy & Hold: {bv_str}")

extra = st.columns(2)
extra[0].metric("# Trades", n_trades)
extra[1].metric("Time in Market", format_pct(time_in_mkt))

st.divider()

# ── equity curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Equity", "Position (1=invested, 0=cash)"))

fig.add_trace(go.Scatter(x=res["strat_eq"].index, y=res["strat_eq"],
                         name="Strategy", line=dict(color="#2563eb", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=res["bh_eq"].index,    y=res["bh_eq"],
                         name="Buy & Hold", line=dict(color="#9ca3af", width=1.5, dash="dot")), row=1, col=1)

fig.add_trace(go.Scatter(x=res["position"].index, y=res["position"],
                         fill="tozeroy", name="In market",
                         line=dict(color="#10b981", width=1),
                         fillcolor="rgba(16,185,129,0.15)"), row=2, col=1)

fig.update_layout(height=550, legend=dict(orientation="h", y=1.02),
                  margin=dict(l=0, r=0, t=30, b=0))
fig.update_yaxes(title_text="$", row=1, col=1)
st.plotly_chart(fig, use_container_width=True)

# ── drawdown chart ────────────────────────────────────────────────────────────
st.subheader("Drawdown")

strat_dd = (res["strat_eq"] - res["strat_eq"].cummax()) / res["strat_eq"].cummax()
bh_dd    = (res["bh_eq"]   - res["bh_eq"].cummax())   / res["bh_eq"].cummax()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd,
                          fill="tozeroy", name="Strategy",
                          line=dict(color="#ef4444"), fillcolor="rgba(239,68,68,0.2)"))
fig2.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd,
                          name="Buy & Hold", line=dict(color="#9ca3af", dash="dot")))
fig2.update_layout(height=280, yaxis_tickformat=".0%",
                   legend=dict(orientation="h"), margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig2, use_container_width=True)

# ── forward return analysis ───────────────────────────────────────────────────
st.subheader("Forward Return Analysis (12 periods after signal)")
st.caption("For every bar where the buy or sell condition fired, what did the asset return over the next 12 periods?")

fwd_buy  = forward_return_stats(df_raw, res["buy_sig"],  ppy)
fwd_sell = forward_return_stats(df_raw, res["sell_sig"], ppy)

fwd_left, fwd_right = st.columns(2)

def render_fwd_table(stats, signal_series, label, color, container):
    with container:
        st.markdown(f"**After {label} signal**")
        if stats is None:
            st.info("No signals fired.")
            return
        rows = [
            ("Signals fired",  stats["n_signals"]),
            ("Mean return",    format_pct(stats["mean"])),
            ("Median return",  format_pct(stats["median"])),
            ("Mode return",    format_pct(stats["mode"])),
            ("Lowest return",  format_pct(stats["min"])),
        ]
        st.table(pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric"))

        close = df_raw["close"]
        sig_idx = np.where(signal_series.values)[0]
        rets = []
        for i in sig_idx:
            end = min(i + ppy, len(close) - 1)
            if end > i:
                rets.append(close.iloc[end] / close.iloc[i] - 1)
        if rets:
            fig_h = go.Figure(go.Histogram(
                x=[r * 100 for r in rets],
                nbinsx=20,
                marker_color=color,
                opacity=0.75,
            ))
            fig_h.update_layout(
                height=220, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="12-period forward return (%)", yaxis_title="Count",
            )
            st.plotly_chart(fig_h, use_container_width=True)

render_fwd_table(fwd_buy,  res["buy_sig"],  "Buy",  "#2563eb", fwd_left)
render_fwd_table(fwd_sell, res["sell_sig"], "Sell", "#ef4444", fwd_right)

st.divider()

# ── trades log ────────────────────────────────────────────────────────────────
if res["trades"] and st.checkbox("Show trade log"):
    rows = []
    idx = df_raw.index
    for entry_i, exit_i in res["trades"]:
        entry_price = df_raw["close"].iloc[entry_i]
        exit_price  = df_raw["close"].iloc[exit_i]
        ret = exit_price / entry_price - 1
        rows.append({
            "Entry date":  idx[entry_i].date(),
            "Exit date":   idx[exit_i].date(),
            "Entry price": round(entry_price, 4),
            "Exit price":  round(exit_price, 4),
            "Return":      f"{ret:.2%}",
        })
    trade_df = pd.DataFrame(rows)
    wins = sum(1 for r in rows if float(r["Return"].strip("%")) > 0)
    st.dataframe(trade_df, use_container_width=True)
    st.caption(f"Win rate: {wins}/{len(rows)} = {wins/len(rows):.0%}")
