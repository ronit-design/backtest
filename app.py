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


def build_signal(df, conditions, logic="AND"):
    """
    logic = "AND"  → all conditions must be true simultaneously
    logic = "OR"   → any single condition being true is enough
    """
    if not conditions:
        return pd.Series(False, index=df.index)
    if logic == "OR":
        sig = pd.Series(False, index=df.index)
        for c in conditions:
            sig |= evaluate_condition(df, c["col"], c["op"], c["val"])
    else:
        sig = pd.Series(True, index=df.index)
        for c in conditions:
            sig &= evaluate_condition(df, c["col"], c["op"], c["val"])
    return sig


def run_backtest(df, buy_conds, sell_conds, initial_capital=10_000, take_profit=None,
                 buy_logic="AND", sell_logic="AND"):
    """
    take_profit : float | None
        If set (e.g. 0.20 for 20%), exit as soon as the return from
        entry close reaches this threshold, regardless of sell signal.
    buy_logic / sell_logic : "AND" | "OR"
    trades stored as (entry_bar_idx, exit_bar_idx, exit_reason)
    """
    buy_sig   = build_signal(df, buy_conds,  logic=buy_logic)
    sell_sig  = build_signal(df, sell_conds, logic=sell_logic)
    close_arr = df["close"].values

    position  = [0] * len(df)
    in_pos    = False
    trades    = []   # (entry_bar_idx, exit_bar_idx, exit_reason)
    entry_i   = None
    entry_px  = None

    for i in range(len(df)):
        if not in_pos and buy_sig.iloc[i]:
            in_pos   = True
            entry_i  = i
            entry_px = close_arr[i]
        elif in_pos:
            tp_hit   = (take_profit is not None and
                        (close_arr[i] / entry_px - 1) >= take_profit)
            sig_hit  = sell_sig.iloc[i]
            if tp_hit or sig_hit:
                reason  = "Take Profit" if tp_hit else "Signal"
                in_pos  = False
                trades.append((entry_i, i, reason))
                entry_i = entry_px = None
        position[i] = 1 if in_pos else 0

    # open trade: do NOT add to trades list (excluded from all trade statistics)
    open_trade = entry_i if (in_pos and entry_i is not None) else None

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
        "open_trade": open_trade,   # entry bar index if position still open, else None
        "buy_sig":    buy_sig,
        "sell_sig":   sell_sig,
    }


def calc_metrics(ret: pd.Series, equity: pd.Series, ppy: int,
                 n_active: int | None = None, rf: float = 0.055) -> dict:
    """
    ret       – return series (pass only active/invested bars for the strategy)
    equity    – full equity curve (for total return and drawdown)
    ppy       – periods per year for the chosen frequency
    n_active  – bars actually invested; if None uses len(ret)
    rf        – annual risk-free rate (default 5.5%)
    """
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n         = n_active if (n_active and n_active > 0) else len(ret)
    if n == 0:
        return {"Total Return": total_ret, "Ann. Return": np.nan,
                "Ann. Volatility": np.nan, "Sharpe Ratio": np.nan, "Max Drawdown": np.nan}
    ann_ret   = (1 + total_ret) ** (ppy / n) - 1
    ann_vol   = ret.std() * np.sqrt(ppy)
    sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
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
    for trade in trades:
        entry_i, exit_i, exit_reason = (*trade, "Signal")[:3] if len(trade) == 2 else trade

        ep       = close.iloc[entry_i]
        xp       = close.iloc[exit_i]
        ret      = xp / ep - 1
        bars     = exit_i - entry_i
        ann      = (1 + ret) ** (ppy / bars) - 1 if bars > 0 else np.nan
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
            "Exit":        exit_reason,
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


# ── monte carlo ───────────────────────────────────────────────────────────────

def run_monte_carlo(pos_series: pd.Series, price_ret: pd.Series,
                    ppy: int, n_sims: int, seed: int = 42) -> dict | None:
    """
    Permutation test: randomly select which n_invested bars to be invested
    (preserving the same count as the real strategy), compute Sharpe for each
    permutation, compare real Sharpe against the resulting null distribution.
    """
    positions = pos_series.shift(1).fillna(0).values.astype(float)
    rets      = price_ret.values.astype(float)
    n_bars    = len(rets)
    n_inv     = int((positions > 0).sum())

    if n_inv < 2:
        return None

    active_real = rets[positions > 0]
    std_real    = active_real.std(ddof=1)
    real_sr     = active_real.mean() / std_real * np.sqrt(ppy) if std_real > 0 else 0.0

    rng = np.random.default_rng(seed)

    if n_bars * n_sims <= 20_000_000:
        # Fully vectorised: argsort of random floats = random permutation
        rand_idx     = rng.random((n_sims, n_bars)).argsort(axis=1)[:, :n_inv]
        sim_rets_all = rets[rand_idx]                          # (n_sims, n_inv)
        sim_means    = sim_rets_all.mean(axis=1)
        sim_stds     = sim_rets_all.std(axis=1, ddof=1)
        valid        = sim_stds > 0
        sim_sharpes  = np.where(valid, sim_means / sim_stds * np.sqrt(ppy), 0.0)
    else:
        # Loop fallback for large intraday datasets
        sim_sharpes = np.empty(n_sims)
        for i in range(n_sims):
            idx = rng.choice(n_bars, size=n_inv, replace=False)
            s   = rets[idx]
            std = s.std(ddof=1)
            sim_sharpes[i] = s.mean() / std * np.sqrt(ppy) if std > 0 else 0.0

    percentile = float((sim_sharpes < real_sr).mean())
    return {
        "real_sr":    real_sr,
        "sim_sharpes": sim_sharpes,
        "percentile": percentile,
        "p_value":    1.0 - percentile,
        "n_sims":     n_sims,
        "n_invested": n_inv,
    }


# ── walk-forward ──────────────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, buy_conds: list, sell_conds: list,
                     n_folds: int, ppy: int, initial_capital: float,
                     take_profit: float | None = None,
                     buy_logic: str = "AND", sell_logic: str = "AND",
                     rf: float = 0.055) -> dict:
    n         = len(df)
    fold_size = n // n_folds
    fold_results, eq_pieces, bh_pieces = [], [], []

    for k in range(n_folds):
        s       = k * fold_size
        e       = (k + 1) * fold_size if k < n_folds - 1 else n
        fold_df = df.iloc[s:e].copy()
        if len(fold_df) < 3:
            continue

        fres    = run_backtest(fold_df, buy_conds, sell_conds,
                              initial_capital=initial_capital, take_profit=take_profit,
                              buy_logic=buy_logic, sell_logic=sell_logic)
        inv_m   = fres["position"].shift(1).fillna(0).astype(bool)
        n_inv   = int(inv_m.sum())
        act_r   = fres["strat_ret"][inv_m]
        fm      = calc_metrics(act_r, fres["strat_eq"], ppy, n_active=n_inv, rf=rf)
        frec    = build_trade_records(fold_df, fres["trades"], ppy)

        if frec:
            fm["Max Drawdown"] = min(r["Max DD"] for r in frec)

        win_r = sum(r["Win"] for r in frec) / len(frec) if frec else np.nan

        fold_results.append({
            "Fold":         k + 1,
            "Period":       f"{fold_df.index[0].date()} → {fold_df.index[-1].date()}",
            "Bars":         len(fold_df),
            "Trades":       len(frec),
            "Total Return": fm["Total Return"],
            "Ann. Return":  fm["Ann. Return"],
            "Sharpe":       fm["Sharpe Ratio"],
            "Max DD":       fm["Max Drawdown"],
            "Win Rate":     win_r,
        })
        eq_pieces.append(fres["strat_eq"])
        bh_pieces.append(fres["bh_eq"])

    def stitch(pieces):
        if not pieces:
            return None
        segs, running = [], initial_capital
        for p in pieces:
            scaled = p * (running / p.iloc[0])
            segs.append(scaled)
            running = scaled.iloc[-1]
        return pd.concat(segs)

    sharpes   = [r["Sharpe"] for r in fold_results
                 if not np.isnan(r.get("Sharpe") or np.nan)]
    pos_folds = sum(1 for s in sharpes if s > 0)

    return {
        "fold_results":   fold_results,
        "stitched_eq":    stitch(eq_pieces),
        "stitched_bh":    stitch(bh_pieces),
        "pct_pos_folds":  pos_folds / len(fold_results) if fold_results else 0.0,
        "mean_fold_sr":   float(np.mean(sharpes)) if sharpes else np.nan,
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

# clear cached results whenever a different file is uploaded
_file_id = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("_file_id") != _file_id:
    for _k in ("res", "wf_result", "mc_result", "buy_conds", "sell_conds",
               "ppy", "capital", "take_profit"):
        st.session_state.pop(_k, None)
    st.session_state["_file_id"] = _file_id

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

# ── ATH / ATL distance columns (expanding window — no look-ahead) ─────────────
# ATH Dist % : always ≤ 0  →  -50 means price is 50% below running ATH
# ATL Dist % : always ≥ 0  →   30 means price is 30% above running ATL
df_raw["ATH Dist %"] = (df_raw["close"] / df_raw["close"].expanding().max() - 1) * 100
df_raw["ATL Dist %"] = (df_raw["close"] / df_raw["close"].expanding().min() - 1) * 100

with st.expander("Data preview", expanded=False):
    st.dataframe(df_raw.tail(20), use_container_width=True)

# ── ATH / ATL reference panel ─────────────────────────────────────────────────
with st.expander("📌  ATH / ATL Reference", expanded=False):
    _ath_price   = df_raw["close"].max()
    _atl_price   = df_raw["close"].min()
    _cur_price   = df_raw["close"].iloc[-1]
    _cur_ath_pct = (_cur_price / _ath_price - 1) * 100
    _cur_atl_pct = (_cur_price / _atl_price - 1) * 100
    _ath_date    = df_raw["close"].idxmax().date()
    _atl_date    = df_raw["close"].idxmin().date()

    ref_c1, ref_c2, ref_c3, ref_c4 = st.columns(4)
    ref_c1.metric("All-Time High",  f"{_ath_price:.4f}", f"{_ath_date}")
    ref_c2.metric("Current vs ATH", f"{_cur_ath_pct:.1f}%",
                  help="ATH Dist % value at the last bar")
    ref_c3.metric("All-Time Low",   f"{_atl_price:.4f}", f"{_atl_date}")
    ref_c4.metric("Current vs ATL", f"{_cur_atl_pct:.1f}%",
                  help="ATL Dist % value at the last bar")

    st.caption(
        "**`ATH Dist %`** is always ≤ 0 — enter `-50` to mean '50% or more below ATH'.  "
        "**`ATL Dist %`** is always ≥ 0 — enter `30` to mean '30% or more above ATL'.  "
        "Both use an expanding window (bar only sees past data — no look-ahead bias)."
    )

st.divider()

# ── strategy builder ──────────────────────────────────────────────────────────
st.subheader("Strategy Rules")

c1, c2, c3, c4 = st.columns(4)
freq_choice = c1.selectbox("Data frequency", list(FREQ_MAP.keys()), index=3)
ppy         = FREQ_MAP[freq_choice]
capital     = c2.number_input("Starting capital ($)", value=10_000, step=1_000)
rf          = c3.number_input("Risk-free rate (%)", value=5.5, min_value=0.0,
                               max_value=20.0, step=0.25,
                               help="Annual risk-free rate used in Sharpe calculation (e.g. T-bill rate).") / 100
fwd_periods = c4.number_input(
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
    logic_col, label_col = st.columns([1, 2])
    buy_logic = logic_col.radio(
        "Buy logic", ["AND", "OR"], horizontal=True, key="buy_logic",
        help="AND = all conditions must be true · OR = any one condition is enough"
    )
    label_col.markdown(
        f"**Buy conditions** ({'all' if buy_logic == 'AND' else 'any'} must be true)"
    )
    n_buy = st.number_input("# buy conditions", 1, 5, 1, key="n_buy")
    buy_conds = []
    for i in range(int(n_buy)):
        st.caption(f"Buy condition {i+1}")
        buy_conds.append(condition_row(f"buy_{i}", numeric_cols))

with right:
    logic_col2, label_col2 = st.columns([1, 2])
    sell_logic = logic_col2.radio(
        "Sell logic", ["AND", "OR"], horizontal=True, key="sell_logic",
        help="AND = all conditions must be true · OR = any one condition is enough"
    )
    label_col2.markdown(
        f"**Sell conditions** ({'all' if sell_logic == 'AND' else 'any'} must be true)"
    )
    n_sell = st.number_input("# sell conditions", 1, 5, 1, key="n_sell")
    sell_conds = []
    for i in range(int(n_sell)):
        st.caption(f"Sell condition {i+1}")
        sell_conds.append(condition_row(f"sell_{i}", numeric_cols))

tp_col1, tp_col2 = st.columns([1, 3])
tp_enabled   = tp_col1.toggle("Take Profit exit", value=False)
take_profit  = (tp_col1.number_input(
                    "Take profit threshold (%)", min_value=0.1, max_value=1000.0,
                    value=20.0, step=0.5,
                    help="Exit the trade when return from entry reaches this level."
                ) / 100.0) if tp_enabled else None
if tp_enabled:
    tp_col2.info(f"Strategy will exit any trade once return from entry reaches **{take_profit:.0%}**, "
                 f"whichever comes first — take profit or sell signal.")

run = st.button("Run Backtest", type="primary", use_container_width=True)

if run:
    with st.spinner("Running…"):
        res = run_backtest(df_raw, buy_conds, sell_conds,
                           initial_capital=capital, take_profit=take_profit,
                           buy_logic=buy_logic, sell_logic=sell_logic)
        st.session_state["res"]          = res
        st.session_state["buy_conds"]    = buy_conds
        st.session_state["sell_conds"]   = sell_conds
        # buy_logic / sell_logic are widget-bound keys — Streamlit tracks them automatically
        st.session_state["ppy"]          = ppy
        st.session_state["capital"]      = capital
        st.session_state["rf"]           = rf
        st.session_state["take_profit"]  = take_profit
        # clear advanced results so stale data doesn't show for a new strategy
        st.session_state.pop("wf_result", None)
        st.session_state.pop("mc_result", None)

if "res" not in st.session_state:
    st.stop()

res = st.session_state["res"]

# invested bars (shift(1) matches how strat_ret is built)
invested_mask = res["position"].shift(1).fillna(0).astype(bool)
n_invested    = int(invested_mask.sum())
active_ret    = res["strat_ret"][invested_mask]   # returns only while in position

rf            = st.session_state.get("rf", 0.055)
strat_metrics = calc_metrics(active_ret, res["strat_eq"], ppy, n_active=n_invested, rf=rf)
bh_metrics    = calc_metrics(res["bh_ret"], res["bh_eq"], ppy, rf=rf)

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

# ── live position ─────────────────────────────────────────────────────────────
if res["open_trade"] is not None:
    oi = res["open_trade"]
    entry_px  = df_raw["close"].iloc[oi]
    current_px = df_raw["close"].iloc[-1]
    unreal_ret = current_px / entry_px - 1
    bars_open  = len(df_raw) - 1 - oi
    st.warning(
        f"⚠️  **Live position detected** — entered {df_raw.index[oi].date()} "
        f"at {entry_px:.4f}, currently at {current_px:.4f} "
        f"({unreal_ret:+.2%} unrealised, {bars_open} bars open). "
        f"This trade is **excluded** from all statistics below."
    )

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
            "Exit":          r["Exit"],
            "Win":           "✓" if r["Win"] else "✗",
        })
    # append live trade row if a position is still open (reference only)
    if res["open_trade"] is not None:
        oi         = res["open_trade"]
        entry_px   = df_raw["close"].iloc[oi]
        current_px = df_raw["close"].iloc[-1]
        unreal_ret = current_px / entry_px - 1
        bars_open  = len(df_raw) - 1 - oi
        display_rows.append({
            "Entry date":  df_raw.index[oi].date(),
            "Exit date":   "OPEN",
            "Bars held":   bars_open,
            "Entry price": round(entry_px, 4),
            "Exit price":  round(current_px, 4),
            "Return":      fmt_pct(unreal_ret) + " *",
            "Ann. return": "—",
            "Max DD":      "—",
            "Win":         "~",
        })

    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
    if res["open_trade"] is not None:
        st.caption("* Live position — unrealised P&L shown for reference only, excluded from all statistics.")

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

st.divider()

# ── walk-forward validation ───────────────────────────────────────────────────
st.subheader("Walk-Forward Validation")
st.caption("Splits the dataset into N equal temporal folds and runs the strategy on each independently. Tests consistency — does the strategy work across different time periods, or is it driven by one lucky window?")

wf_c1, wf_c2 = st.columns([1, 3])
n_folds = wf_c1.number_input("Number of folds", min_value=3, max_value=10, value=5, step=1)

if st.button("Run Walk-Forward", use_container_width=True):
    with st.spinner("Running walk-forward validation…"):
        st.session_state["wf_result"] = run_walk_forward(
            df_raw,
            st.session_state["buy_conds"],
            st.session_state["sell_conds"],
            int(n_folds),
            st.session_state["ppy"],
            st.session_state["capital"],
            take_profit=st.session_state.get("take_profit"),
            buy_logic=st.session_state.get("buy_logic", "AND"),
            sell_logic=st.session_state.get("sell_logic", "AND"),
            rf=st.session_state.get("rf", 0.055),
        )

if "wf_result" in st.session_state:
    wf = st.session_state["wf_result"]

    # ── consistency summary ───────────────────────────────────────────────────
    cs1, cs2, cs3 = st.columns(3)
    cs1.metric("Folds with positive Sharpe",
               f"{int(wf['pct_pos_folds'] * len(wf['fold_results']))} / {len(wf['fold_results'])}")
    cs2.metric("Mean fold Sharpe", fmt_f2(wf["mean_fold_sr"]))
    cs3.metric("Consistency",
               f"{wf['pct_pos_folds']:.0%}",
               help="% of folds where Sharpe > 0")

    # ── per-fold table ────────────────────────────────────────────────────────
    fold_display = []
    for r in wf["fold_results"]:
        fold_display.append({
            "Fold":         r["Fold"],
            "Period":       r["Period"],
            "Bars":         r["Bars"],
            "Trades":       r["Trades"],
            "Total Return": fmt_pct(r["Total Return"]) if not np.isnan(r["Total Return"]) else "—",
            "Ann. Return":  fmt_pct(r["Ann. Return"])  if not np.isnan(r["Ann. Return"])  else "—",
            "Sharpe":       fmt_f2(r["Sharpe"]),
            "Max DD":       fmt_pct(r["Max DD"])        if not np.isnan(r["Max DD"])        else "—",
            "Win Rate":     fmt_pct(r["Win Rate"])      if not np.isnan(r["Win Rate"])      else "—",
        })
    st.dataframe(pd.DataFrame(fold_display), use_container_width=True, hide_index=True)

    # ── stitched equity curve ─────────────────────────────────────────────────
    if wf["stitched_eq"] is not None:
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(
            x=wf["stitched_eq"].index, y=wf["stitched_eq"],
            name="Strategy (OOS stitched)", line=dict(color="#2563eb", width=2)))
        if wf["stitched_bh"] is not None:
            fig_wf.add_trace(go.Scatter(
                x=wf["stitched_bh"].index, y=wf["stitched_bh"],
                name="Buy & Hold", line=dict(color="#9ca3af", width=1.5, dash="dot")))
        fig_wf.update_layout(
            height=380, yaxis_title="$",
            legend=dict(orientation="h"), margin=dict(l=0, r=0, t=10, b=0),
            title="Stitched out-of-sample equity curve")
        st.plotly_chart(fig_wf, use_container_width=True)

st.divider()

# ── monte carlo permutation ───────────────────────────────────────────────────
st.subheader("Monte Carlo Permutation Test")
st.caption("Randomly shuffles which bars the strategy is invested in (preserving the same number of invested periods), repeats thousands of times, and asks: how often does a random timing strategy match or beat your Sharpe? A low p-value means your signal timing is genuinely adding value.")

_default_sims = {6_552: 1_000, 252: 5_000, 52: 10_000, 12: 10_000, 4: 10_000}
_default_n    = _default_sims.get(ppy, 5_000)

mc_c1, mc_c2 = st.columns([1, 3])
n_sims_input  = mc_c1.number_input("Simulations", min_value=500, max_value=50_000,
                                    value=_default_n, step=500)

if st.button("Run Monte Carlo", use_container_width=True):
    with st.spinner(f"Running {n_sims_input:,} permutations…"):
        st.session_state["mc_result"] = run_monte_carlo(
            res["position"], res["bh_ret"],
            st.session_state["ppy"], int(n_sims_input)
        )

if "mc_result" in st.session_state:
    mc = st.session_state["mc_result"]

    mc_p      = mc["p_value"]
    mc_label, mc_colour = sig_verdict(mc_p)

    # ── headline metrics ──────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Your Sharpe",          fmt_f2(mc["real_sr"]))
    mc2.metric("Median random Sharpe", fmt_f2(float(np.median(mc["sim_sharpes"]))))
    mc3.metric("Percentile rank",      f"{mc['percentile']:.1%}")
    mc4.metric("p-value",              f"{mc_p:.4f}")

    st.markdown(
        f"<div style='text-align:center; padding:10px; border-radius:6px; "
        f"color:{mc_colour}; font-weight:700; font-size:1rem; "
        f"border: 1px solid {mc_colour};'>{mc_label}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Your Sharpe of {mc['real_sr']:.3f} beats {mc['percentile']:.1%} of {mc['n_sims']:,} randomly-timed strategies with the same number of invested bars ({mc['n_invested']:,}).")

    # ── distribution histogram ────────────────────────────────────────────────
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x=mc["sim_sharpes"], nbinsx=60,
        name="Random strategies",
        marker_color="#9ca3af", opacity=0.7,
    ))
    fig_mc.add_vline(
        x=mc["real_sr"], line_color="#2563eb", line_width=2.5,
        annotation_text=f"  Your SR: {mc['real_sr']:.3f}",
        annotation_font_color="#2563eb",
        annotation_position="top right",
    )
    fig_mc.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Sharpe ratio (random permutations)",
        yaxis_title="Count",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_mc, use_container_width=True)
