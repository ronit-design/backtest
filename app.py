import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(page_title="Backtester", layout="wide")

NUMERIC_OPS = [">", "<", ">=", "<=", "=="]
CROSS_OPS   = ["crosses above", "crosses below"]
ALL_OPS     = NUMERIC_OPS + CROSS_OPS

FREQ_MAP = {
    "15-Min":    6_552,   # 26 bars/day × 252 trading days
    "Daily":       252,
    "Weekly":       52,
    "Monthly":      12,
    "Quarterly":     4,
}

# ── core logic ────────────────────────────────────────────────────────────────

def evaluate_condition(df, col, op, val):
    s = df[col]
    if op == ">":             return s > val
    if op == "<":             return s < val
    if op == ">=":            return s >= val
    if op == "<=":            return s <= val
    if op == "==":            return s == val
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
    in_pos   = False
    trades   = []   # (entry_bar_idx, exit_bar_idx)
    entry_i  = None

    for i in range(len(df)):
        if not in_pos and buy_sig.iloc[i]:
            in_pos  = True
            entry_i = i
        elif in_pos and sell_sig.iloc[i]:
            in_pos  = False
            trades.append((entry_i, i))
            entry_i = None
        position[i] = 1 if in_pos else 0

    if in_pos and entry_i is not None:          # open trade at end of data
        trades.append((entry_i, len(df) - 1))

    pos_series = pd.Series(position, index=df.index)
    price_ret  = df["close"].pct_change().fillna(0)

    # signal seen at bar close → fills at next bar's open, approximated as next close
    strat_ret = pos_series.shift(1).fillna(0) * price_ret
    strat_eq  = initial_capital * (1 + strat_ret).cumprod()
    bh_eq     = initial_capital * (1 + price_ret).cumprod()

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


def calc_metrics(ret: pd.Series, equity: pd.Series, ppy: int, n_active: int | None = None) -> dict:
    """
    ret       – return series (pass only active/invested bars for the strategy)
    equity    – full equity curve (for total return and drawdown)
    ppy       – periods per year for the chosen frequency
    n_active  – bars actually invested; if None uses len(ret)
    """
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n         = n_active if (n_active and n_active > 0) else len(ret)
    if n == 0:
        return {"Total Return": total_ret, "Ann. Return": np.nan,
                "Ann. Volatility": np.nan, "Sharpe Ratio": np.nan, "Max Drawdown": np.nan}
    ann_ret   = (1 + total_ret) ** (ppy / n) - 1
    ann_vol   = ret.std() * np.sqrt(ppy)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else np.nan
    roll_max  = equity.cummax()
    max_dd    = ((equity - roll_max) / roll_max).min()
    return {
        "Total Return":    total_ret,
        "Ann. Return":     ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio":    sharpe,
        "Max Drawdown":    max_dd,
    }


def build_trade_records(df, trades, ppy) -> list[dict]:
    idx   = df.index
    close = df["close"]
    low   = df["low"]
    rows  = []
    for entry_i, exit_i in trades:
        ep       = close.iloc[entry_i]
        xp       = close.iloc[exit_i]
        ret      = xp / ep - 1
        bars     = exit_i - entry_i
        ann      = (1 + ret) ** (ppy / bars) - 1 if bars > 0 else np.nan
        # max drawdown: entry close → lowest low during the holding period
        lowest   = low.iloc[entry_i : exit_i + 1].min()
        trade_dd = (lowest - ep) / ep
        rows.append({
            "Entry date":  idx[entry_i].date(),
            "Exit date":   idx[exit_i].date(),
            "Bars held":   bars,
            "Entry price": round(ep, 4),
            "Exit price":  round(xp, 4),
            "Return":      ret,
            "Ann. return": ann,
            "Max DD":      trade_dd,
            "Win":         ret > 0,
        })
    return rows


def trade_stats(records: list[dict]) -> dict | None:
    if not records:
        return None
    rets  = np.array([r["Return"] for r in records])
    rounded = np.round(rets * 100).astype(int)
    mode_val = pd.Series(rounded).value_counts().index[0] / 100
    return {
        "n":       len(rets),
        "wins":    int(sum(r["Win"] for r in records)),
        "mean":    rets.mean(),
        "median":  float(np.median(rets)),
        "mode":    mode_val,
        "min":     rets.min(),
        "max":     rets.max(),
    }


def forward_return_stats(df, signal: pd.Series, periods: int) -> dict | None:
    close     = df["close"]
    sig_idx   = np.where(signal.values)[0]
    fwd_rets  = []
    for i in sig_idx:
        end = min(i + periods, len(close) - 1)
        if end > i:
            fwd_rets.append(close.iloc[end] / close.iloc[i] - 1)
    if not fwd_rets:
        return None
    arr = np.array(fwd_rets)
    return {
        "n_signals": len(arr),
        "mean":      arr.mean(),
        "median":    float(np.median(arr)),
        "max":       arr.max(),
        "min":       arr.min(),
    }


# ── statistical significance ──────────────────────────────────────────────────

def sig_verdict(p: float) -> tuple[str, str]:
    """Return (label, hex_colour) for a p-value."""
    if p < 0.05:   return "✅  Significant (p < 0.05)",   "#16a34a"
    if p < 0.10:   return "⚠️  Borderline (p < 0.10)",    "#d97706"
    return             "❌  Not significant (p ≥ 0.10)", "#dc2626"


def run_significance_tests(
    trade_rets:       np.ndarray,   # per-trade returns
    strat_ret_active: pd.Series,    # bar returns while invested
    bh_ret_invested:  pd.Series,    # B&H returns over same bars
    n_invested:       int,
    ppy:              int,
    sharpe:           float,
) -> list[dict]:
    """
    Returns a list of result dicts, each with keys:
      name, stat_label, stat_val, p, verdict, colour
    Bootstrap CI result has ci_low/ci_high instead of p/verdict.
    """
    n = len(trade_rets)
    results = []

    # ── Test 1: Binomial test on win rate ──────────────────────────────────────
    if n >= 1:
        wins = int(np.sum(trade_rets > 0))
        res1 = stats.binomtest(wins, n, p=0.5, alternative="greater")
        label, colour = sig_verdict(res1.pvalue)
        results.append({
            "test":       "1 · Win Rate vs 50%",
            "method":     "Binomial test  (H₀: win rate = 50%)",
            "stat_label": "Win rate",
            "stat_val":   f"{wins}/{n} = {wins/n:.1%}",
            "p":          res1.pvalue,
            "verdict":    label,
            "colour":     colour,
        })

    # ── Test 2: One-sample t-test — mean trade return > 10% ──────────────────
    MEAN_RET_THRESHOLD = 0.10
    if n >= 2:
        t2, p2 = stats.ttest_1samp(trade_rets, popmean=MEAN_RET_THRESHOLD, alternative="greater")
        label, colour = sig_verdict(p2)
        results.append({
            "test":       "2 · Mean Trade Return > 10%",
            "method":     f"One-sample t-test  (H₀: mean trade return = 10%)",
            "stat_label": "t-statistic",
            "stat_val":   f"{t2:.3f}  (df = {n - 1},  mean = {trade_rets.mean():.2%})",
            "p":          p2,
            "verdict":    label,
            "colour":     colour,
        })

    # ── Test 3: Sharpe significance — Lo (2002), H₀: SR = 0.6 ────────────────
    SR_THRESHOLD = 0.6
    if not np.isnan(sharpe) and n_invested >= ppy:
        t_years  = n_invested / ppy
        # Lo (2002): (SR_hat - threshold) / sqrt((1 + SR_hat²/2) / T)
        t3       = (sharpe - SR_THRESHOLD) * np.sqrt(t_years) / np.sqrt(1 + sharpe ** 2 / 2)
        p3       = 1 - stats.t.cdf(t3, df=max(n_invested - 1, 1))
        label, colour = sig_verdict(p3)
        hlz_flag = "  ·  ⚡ clears HLZ 3.0 threshold" if abs(t3) >= 3.0 else ""
        results.append({
            "test":       "3 · Sharpe Ratio > 0.6",
            "method":     f"Lo (2002) adjusted t-test  (H₀: SR = 0.6,  {t_years:.1f} years invested){hlz_flag}",
            "stat_label": "t-statistic",
            "stat_val":   f"{t3:.3f}  (SR = {sharpe:.3f})",
            "p":          p3,
            "verdict":    label,
            "colour":     colour,
        })

    # ── Test 4: Bootstrap 95% CI on mean trade return (10 000 resamples) ──────
    if n >= 2:
        rng        = np.random.default_rng(42)
        boot_means = rng.choice(trade_rets, size=(10_000, n), replace=True).mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot_means, [5, 95])
        significant  = ci_lo > 0
        label  = "✅  Significant — CI excludes 0" if significant else "❌  Not significant — CI includes 0"
        colour = "#16a34a" if significant else "#dc2626"
        results.append({
            "test":       "4 · Mean Return Bootstrap CI",
            "method":     "Bootstrap resampling  10 000 iterations, 90% CI (5% each tail)",
            "stat_label": "90% CI",
            "stat_val":   f"[{ci_lo:.2%},  {ci_hi:.2%}]",
            "p":          None,
            "verdict":    label,
            "colour":     colour,
            "ci_lo":      ci_lo,
            "ci_hi":      ci_hi,
        })

    # ── Test 5: Strategy vs B&H — Welch's t-test ──────────────────────────────
    if len(strat_ret_active) >= 2 and len(bh_ret_invested) >= 2:
        t5, p5 = stats.ttest_ind(
            strat_ret_active.values,
            bh_ret_invested.values,
            equal_var=False,
            alternative="greater",
        )
        label, colour = sig_verdict(p5)
        results.append({
            "test":       "5 · Strategy Returns > Buy & Hold",
            "method":     "Welch's t-test  (unequal variances, invested bars only)",
            "stat_label": "t-statistic",
            "stat_val":   f"{t5:.3f}",
            "p":          p5,
            "verdict":    label,
            "colour":     colour,
        })

    return results


# ── UI helpers ────────────────────────────────────────────────────────────────

def condition_row(key_prefix, numeric_cols):
    cols = st.columns([2, 1.5, 1.5])
    col  = cols[0].selectbox("Column",   numeric_cols, key=f"{key_prefix}_col", label_visibility="collapsed")
    op   = cols[1].selectbox("Operator", ALL_OPS,      key=f"{key_prefix}_op",  label_visibility="collapsed")
    val  = cols[2].number_input("Value", value=0.0,    key=f"{key_prefix}_val", label_visibility="collapsed",
                                format="%.4f", step=0.01)
    return {"col": col, "op": op, "val": val}


def fmt_pct(v):  return f"{v:.2%}"
def fmt_f2(v):   return f"{v:.2f}" if not np.isnan(v) else "—"


# ── app ───────────────────────────────────────────────────────────────────────

st.title("Strategy Backtester")
st.caption("Upload your OHLCV + indicator CSV, define buy/sell rules, compare vs buy & hold.")

# ── upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV", type="csv")
if not uploaded:
    st.info("Expected columns: time, open, high, low, close, Volume, ROC, 2nd Derivative (Acceleration)")
    st.stop()

df_raw = pd.read_csv(uploaded)
df_raw.columns = df_raw.columns.str.strip()

required = {"time", "open", "high", "low", "close"}
missing  = required - set(df_raw.columns.str.lower())
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df_raw.columns = [c.lower() if c.lower() in {"time","open","high","low","close","volume"} else c
                  for c in df_raw.columns]

def parse_datetime_col(series: pd.Series) -> pd.Series:
    """Parse datetime column robustly, handling mixed formats and mixed tz-aware/naive rows."""
    for kwargs in [
        {"format": "mixed", "dayfirst": False},
        {"format": "mixed", "dayfirst": False, "utc": True},
        {"infer_datetime_format": True},
        {"errors": "coerce"},
    ]:
        try:
            parsed = pd.to_datetime(series, **kwargs)
            # strip timezone so index is always tz-naive
            if hasattr(parsed, "dt") and parsed.dt.tz is not None:
                parsed = parsed.dt.tz_localize(None)
            elif hasattr(parsed, "tz") and parsed.tz is not None:
                parsed = parsed.tz_localize(None)
            return parsed
        except Exception:
            continue
    st.error("Could not parse the 'time' column. Please ensure it contains valid dates or timestamps.")
    st.stop()

df_raw["time"] = parse_datetime_col(df_raw["time"])
df_raw = df_raw.sort_values("time").reset_index(drop=True).set_index("time")

with st.expander("Data preview", expanded=False):
    st.dataframe(df_raw.tail(20), use_container_width=True)

st.divider()

# ── strategy builder ──────────────────────────────────────────────────────────
st.subheader("Strategy Rules")

c1, c2, c3 = st.columns(3)
freq_choice = c1.selectbox("Data frequency", list(FREQ_MAP.keys()), index=3)
ppy         = FREQ_MAP[freq_choice]
capital     = c2.number_input("Starting capital ($)", value=10_000, step=1_000)
fwd_periods = c3.number_input(
    "Forward-return window (bars)",
    min_value=1, value=ppy,
    help=f"Periods to look ahead in signal analysis. Default = 1 year ({ppy} bars for {freq_choice})."
)

# vol ratio: rolling 2-year window in bars (computed here so ppy is known)
vol_window  = 2 * ppy
vol_col     = f"Vol / 2Y Avg"
if "volume" in df_raw.columns:
    roll_avg = df_raw["volume"].rolling(vol_window, min_periods=vol_window).mean()
    df_raw[vol_col] = df_raw["volume"] / roll_avg

numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

left, right = st.columns(2)
with left:
    st.markdown("**Buy conditions** (all must be true)")
    n_buy = st.number_input("# buy conditions", 1, 5, 1, key="n_buy")
    buy_conds = []
    for i in range(int(n_buy)):
        st.caption(f"Buy condition {i+1}")
        buy_conds.append(condition_row(f"buy_{i}", numeric_cols))

with right:
    st.markdown("**Sell conditions** (all must be true)")
    n_sell = st.number_input("# sell conditions", 1, 5, 1, key="n_sell")
    sell_conds = []
    for i in range(int(n_sell)):
        st.caption(f"Sell condition {i+1}")
        sell_conds.append(condition_row(f"sell_{i}", numeric_cols))

run = st.button("Run Backtest", type="primary", use_container_width=True)
if not run:
    st.stop()

# ── run ───────────────────────────────────────────────────────────────────────
with st.spinner("Running…"):
    res = run_backtest(df_raw, buy_conds, sell_conds, initial_capital=capital)

# invested bars (shift(1) matches how strat_ret is built)
invested_mask = res["position"].shift(1).fillna(0).astype(bool)
n_invested    = int(invested_mask.sum())
active_ret    = res["strat_ret"][invested_mask]   # returns only while in position

strat_metrics = calc_metrics(active_ret, res["strat_eq"], ppy, n_active=n_invested)
bh_metrics    = calc_metrics(res["bh_ret"], res["bh_eq"], ppy)

records     = build_trade_records(df_raw, res["trades"], ppy)
tstats      = trade_stats(records)
time_in_mkt = res["position"].mean()

# override strategy max DD: worst (entry → lowest low) across all trades
if records:
    strat_metrics["Max Drawdown"] = min(r["Max DD"] for r in records)

# ── performance summary ───────────────────────────────────────────────────────
st.subheader("Performance Summary")
st.caption("Strategy: annualised over time invested; max DD = entry price → lowest low per trade. Buy & Hold: full-period annualisation; max DD = equity curve peak-to-trough.")

metric_cols = st.columns(5)
labels = ["Total Return", "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown"]
for col_ui, label in zip(metric_cols, labels):
    sv = strat_metrics[label]
    bv = bh_metrics[label]
    if label == "Sharpe Ratio":
        sv_str, bv_str = fmt_f2(sv), fmt_f2(bv)
        delta = f"{sv - bv:+.2f} vs B&H" if not np.isnan(sv) and not np.isnan(bv) else ""
    else:
        sv_str = fmt_pct(sv) if not np.isnan(sv) else "—"
        bv_str = fmt_pct(bv) if not np.isnan(bv) else "—"
        delta  = f"{sv - bv:+.2%} vs B&H" if not np.isnan(sv) and not np.isnan(bv) else ""
    col_ui.metric(label, sv_str, delta, help=f"Buy & Hold: {bv_str}")

extra = st.columns(3)
extra[0].metric("# Trades", len(records))
extra[1].metric("Time in Market", fmt_pct(time_in_mkt))
extra[2].metric("Bars Invested", f"{n_invested} / {len(df_raw)}")

st.divider()

# ── equity curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                    subplot_titles=("Equity ($)", "Invested (1 = in position)"))
fig.add_trace(go.Scatter(x=res["strat_eq"].index, y=res["strat_eq"],
                         name="Strategy", line=dict(color="#2563eb", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=res["bh_eq"].index, y=res["bh_eq"],
                         name="Buy & Hold", line=dict(color="#9ca3af", width=1.5, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=res["position"].index, y=res["position"],
                         fill="tozeroy", name="In market",
                         line=dict(color="#10b981", width=1),
                         fillcolor="rgba(16,185,129,0.15)"), row=2, col=1)
fig.update_layout(height=540, legend=dict(orientation="h", y=1.02),
                  margin=dict(l=0, r=0, t=30, b=0))
fig.update_yaxes(title_text="$", row=1, col=1)
st.plotly_chart(fig, use_container_width=True)

# ── drawdown ──────────────────────────────────────────────────────────────────
st.subheader("Drawdown")
strat_dd = (res["strat_eq"] - res["strat_eq"].cummax()) / res["strat_eq"].cummax()
bh_dd    = (res["bh_eq"]   - res["bh_eq"].cummax())   / res["bh_eq"].cummax()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd, fill="tozeroy", name="Strategy",
                          line=dict(color="#ef4444"), fillcolor="rgba(239,68,68,0.2)"))
fig2.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd, name="Buy & Hold",
                          line=dict(color="#9ca3af", dash="dot")))
fig2.update_layout(height=260, yaxis_tickformat=".0%",
                   legend=dict(orientation="h"), margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── per-trade statistics ──────────────────────────────────────────────────────
st.subheader("Trade Statistics")

if tstats is None:
    st.info("No completed trades.")
else:
    win_rate = tstats["wins"] / tstats["n"]
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    tc1.metric("Trades",    tstats["n"])
    tc2.metric("Win rate",  fmt_pct(win_rate))
    tc3.metric("Avg return", fmt_pct(tstats["mean"]))
    tc4.metric("Median return", fmt_pct(tstats["median"]))
    tc5.metric("Worst trade",  fmt_pct(tstats["min"]))

    ts_left, ts_right = st.columns(2)
    with ts_left:
        stat_rows = [
            ("Mean return",    fmt_pct(tstats["mean"])),
            ("Median return",  fmt_pct(tstats["median"])),
            ("Mode return",    fmt_pct(tstats["mode"])),
            ("Best trade",     fmt_pct(tstats["max"])),
            ("Worst trade",    fmt_pct(tstats["min"])),
        ]
        st.table(pd.DataFrame(stat_rows, columns=["Metric", "Value"]).set_index("Metric"))

    with ts_right:
        trade_rets = [r["Return"] * 100 for r in records]
        fig_h = go.Figure(go.Histogram(x=trade_rets, nbinsx=20,
                                       marker_color="#2563eb", opacity=0.75))
        fig_h.add_vline(x=tstats["mean"] * 100, line_dash="dash", line_color="#ef4444",
                        annotation_text="mean", annotation_position="top right")
        fig_h.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_title="Return per trade (%)", yaxis_title="Count")
        st.plotly_chart(fig_h, use_container_width=True)

st.divider()

# ── statistical significance ──────────────────────────────────────────────────
st.subheader("Statistical Significance")

_n_trades = len(records)
if _n_trades < 10:
    st.error(f"⚠️  Only {_n_trades} trades — too few for meaningful significance testing. Results below are illustrative only.")
elif _n_trades < 30:
    st.warning(f"⚠️  {_n_trades} trades — tests have low statistical power. Treat results as indicative, not conclusive.")

if _n_trades >= 2:
    trade_ret_arr   = np.array([r["Return"] for r in records])
    bh_ret_invested = res["bh_ret"][invested_mask]
    sig_results     = run_significance_tests(
        trade_ret_arr, active_ret, bh_ret_invested,
        n_invested, ppy, strat_metrics["Sharpe Ratio"],
    )

    for sr in sig_results:
        with st.container(border=True):
            row_l, row_r = st.columns([3, 1])
            with row_l:
                st.markdown(f"**{sr['test']}**")
                st.caption(sr["method"])
                st.markdown(f"`{sr['stat_label']}:` **{sr['stat_val']}**")
                if sr["p"] is not None:
                    st.markdown(f"`p-value:` **{sr['p']:.4f}**")
            with row_r:
                st.markdown(
                    f"<div style='text-align:center; padding:14px 0; "
                    f"color:{sr['colour']}; font-weight:700; font-size:0.9rem;'>"
                    f"{sr['verdict']}</div>",
                    unsafe_allow_html=True,
                )

    # ── Multiple testing warning (Harvey, Liu & Zhu 2016) ────────────────────
    with st.expander("⚡  About multiple testing & the HLZ threshold"):
        st.markdown("""
**If you have run this backtester across multiple strategy configurations** (different
thresholds, different indicators, different tickers), the standard p < 0.05 threshold
understates your probability of finding a false positive.

Harvey, Liu & Zhu (2016) showed that given the number of strategies tested across the
finance industry, a result should clear a **t-statistic of ≥ 3.0** (p ≈ 0.003) before
being considered credible — not the standard 1.96.

Test 3 flags automatically when the Sharpe t-statistic clears this bar. A strategy
that passes at 95% confidence is *promising*. One that also clears t = 3.0 is *credible*.
        """)
else:
    st.info("Need at least 2 completed trades to run significance tests.")

st.divider()

# ── trade log ─────────────────────────────────────────────────────────────────
st.subheader("Trade Log")
if not records:
    st.info("No trades executed.")
else:
    display_rows = []
    for r in records:
        display_rows.append({
            "Entry date":    r["Entry date"],
            "Exit date":     r["Exit date"],
            "Bars held":     r["Bars held"],
            "Entry price":   r["Entry price"],
            "Exit price":    r["Exit price"],
            "Return":        fmt_pct(r["Return"]),
            "Ann. return":   fmt_pct(r["Ann. return"]) if not np.isnan(r["Ann. return"]) else "—",
            "Max DD":        fmt_pct(r["Max DD"]),
            "Win":           "✓" if r["Win"] else "✗",
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

st.divider()

# ── forward return analysis ───────────────────────────────────────────────────
st.subheader(f"Forward Return Analysis ({fwd_periods} bars after signal)")
st.caption(f"Every time the buy or sell condition fires, what did the asset return over the next {fwd_periods} bars ({freq_choice} data)?")

fwd_buy  = forward_return_stats(df_raw, res["buy_sig"],  fwd_periods)
fwd_sell = forward_return_stats(df_raw, res["sell_sig"], fwd_periods)

fwd_left, fwd_right = st.columns(2)

def render_fwd(stats, signal_series, label, color, container):
    with container:
        st.markdown(f"**After {label} signal**")
        if stats is None:
            st.info("No signals fired.")
            return
        rows = [
            ("Signals fired",   stats["n_signals"]),
            ("Mean return",     fmt_pct(stats["mean"])),
            ("Median return",   fmt_pct(stats["median"])),
            ("Highest return",  fmt_pct(stats["max"])),
            ("Lowest return",   fmt_pct(stats["min"])),
        ]
        st.table(pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric"))
        close   = df_raw["close"]
        sig_idx = np.where(signal_series.values)[0]
        rets    = []
        for i in sig_idx:
            end = min(i + ppy, len(close) - 1)
            if end > i:
                rets.append(close.iloc[end] / close.iloc[i] - 1)
        if rets:
            fig_h = go.Figure(go.Histogram(x=[r * 100 for r in rets], nbinsx=20,
                                           marker_color=color, opacity=0.75))
            fig_h.update_layout(height=210, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title=f"{fwd_periods}-bar forward return (%)", yaxis_title="Count")
            st.plotly_chart(fig_h, use_container_width=True)

render_fwd(fwd_buy,  res["buy_sig"],  "Buy",  "#2563eb", fwd_left)
render_fwd(fwd_sell, res["sell_sig"], "Sell", "#ef4444", fwd_right)
