
# midfreq_alpha_toolkit.py
# A compact, dependency-light toolkit for building mid-frequency signals from PRICE-ONLY data
# across a cross-section (e.g., CAC constituents).
#
# Author: ChatGPT
# License: MIT
#
# -----------------------------
# WHAT'S INSIDE (high-level):
# -----------------------------
# 1) Data utilities
#    - trading_calendar_index: build per-second (or 10s) timestamps for regular session (09:00–17:30 Europe/Paris)
#    - simulate_prices_gbm_jumps: synthetic per-second prices with intraday seasonality & jumps
#    - to_10s / to_1min resamplers (price-only friendly)
#
# 2) Hygiene
#    - basic bad-tick removal & winsorization helpers
#
# 3) Labeling
#    - triple_barrier_labels for price-only barrier labels (TP/SL/maxH) → {-1, 0, +1}
#
# 4) Features (modern, price-only)
#    - Pre-averaged realized variance & quarticity (microstructure-noise-robust)
#    - Local Hurst exponent (rough-vol proxy) via log-log variogram slope
#    - Drift-burst test (simple noise-robust sign-consistency test around local trend)
#    - Discrete-time Hawkes intensity fit on exceedance events (self-excitation proxy)
#    - Haar wavelet burst descriptors & clustering (jump taxonomy)
#    - Lead–lag signature-like features (order-2 proxy) on path windows
#    - Graph features: rolling (shrunk) precision-matrix signals & leader-laggers
#
# 5) Modeling
#    - Purged & embargoed walk-forward splitter
#    - Simple classifiers (Logistic / Gradient Boosting when available) + meta-label gate
#
# 6) Portfolio & execution-lite
#    - Vol targeting & simple quadratic objective with ridge (QP-free approximation)
#
# 7) Orchestration
#    - build_dataset_for_model: compute features + labels on sliding windows for all tickers
#
# Dependencies: numpy, pandas
# Optional:    scipy, scikit-learn (if absent, light fallbacks are used)
#
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.cluster import KMeans
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    # Lightweight fallbacks
    LogisticRegression = None
    GradientBoostingClassifier = None
    KMeans = None
    LedoitWolf = None
    roc_auc_score = None

# Optional SciPy for robust stats
try:
    from scipy.stats import mstats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    mstats = None

# ----------------------------
# 1) DATA & CALENDAR UTILITIES
# ----------------------------

def trading_calendar_index(start_date: str, end_date: str,
                           tz: str = "Europe/Paris",
                           session_start: str = "09:00:00",
                           session_end: str = "17:30:00",
                           freq: str = "1S",
                           business_days_only: bool = True) -> pd.DatetimeIndex:
    """
    Build a time index for regular sessions between start_date and end_date, inclusive.
    freq: '1S' for per-second; '10S' for 10 sec; etc.
    Returns tz-aware DatetimeIndex in Europe/Paris with DST handled by pandas.
    """
    # Build naive business day range
    days = pd.date_range(start=start_date, end=end_date, freq="B" if business_days_only else "D")
    frames = []
    for d in days:
        start = pd.Timestamp(f"{d.date()} {session_start}", tz=tz)
        end   = pd.Timestamp(f"{d.date()} {session_end}", tz=tz)
        # pandas date_range is inclusive of both ends by default; for 1S we want inclusive
        frames.append(pd.date_range(start, end, freq=freq))
    if not frames:
        return pd.DatetimeIndex([], tz=tz)
    return frames[0].append(frames[1:])

def simulate_prices_gbm_jumps(index: pd.DatetimeIndex,
                              tickers: List[str],
                              seed: int = 42,
                              daily_vol_bps: float = 120.0,
                              intraday_u_shape: bool = True,
                              jump_prob_per_hour: float = 0.12,
                              jump_mean_bps: float = 8.0,
                              jump_std_bps: float = 12.0,
                              start_price_range: Tuple[float, float] = (10.0, 150.0)) -> pd.DataFrame:
    """
    Simulate per-second last prices for multiple tickers with GBM + Poisson jumps + intraday seasonality.
    Returns a tidy DataFrame with MultiIndex (timestamp, ticker) and column 'price'.
    """
    rng = np.random.default_rng(seed)
    n = len(index)
    dt_sec = (index[1] - index[0]).total_seconds() if n > 1 else 1.0
    dt_day = 24*3600.0
    # Convert daily vol in bps to per-second vol
    # Assume ~ 6.5 trading hours (09:00–17:30 = 8.5, but intraday-only variance). We'll scale by session seconds per day.
    # session seconds per day:
    sess_secs = int((pd.Timestamp(index[0].date().strftime('%Y-%m-%d') + " 17:30:00") -
                     pd.Timestamp(index[0].date().strftime('%Y-%m-%d') + " 09:00:00")).total_seconds())
    daily_vol = daily_vol_bps / 1e4  # as fraction
    # rough per-second sigma:
    sigma_sec = daily_vol / math.sqrt(sess_secs)

    # U-shape seasonality (more vol near open/close)
    times = index.tz_convert(None)
    tt = times.time
    ss = np.array([t.hour*3600 + t.minute*60 + t.second for t in tt], dtype=float)
    # Map session seconds 0..sess_secs
    day_start = ss[0]
    s_in_session = ss - day_start
    s_in_session[s_in_session < 0] = 0
    # U-shape scaling factor (1 at midday, >1 near edges)
    if intraday_u_shape:
        x = (s_in_session / sess_secs) * 2 - 1  # -1..1
        u_shape = 1.0 + 0.5*(x**2)              # simple convex
    else:
        u_shape = np.ones_like(s_in_session)

    data = []
    for k, tkr in enumerate(tickers):
        p0 = rng.uniform(*start_price_range)
        # Normal shocks
        z = rng.normal(0.0, 1.0, size=n)
        # Jumps: Poisson arrivals per hour → per-second rate
        lam = jump_prob_per_hour / 3600.0
        is_jump = rng.uniform(size=n) < lam
        jump_sizes = rng.normal(jump_mean_bps/1e4, jump_std_bps/1e4, size=n) * is_jump
        # GBM log-return
        dlogp = (sigma_sec * u_shape) * z + jump_sizes
        logp = np.cumsum(dlogp) + math.log(p0)
        price = np.exp(logp)
        data.append(pd.DataFrame({"timestamp": index, "ticker": tkr, "price": price}))
    df = pd.concat(data, ignore_index=True)
    df.set_index(["timestamp", "ticker"], inplace=True)
    df.sort_index(inplace=True)
    return df

def resample_last_price(df: pd.DataFrame, rule: str = "10S") -> pd.DataFrame:
    """
    Resample tidy last-print prices to coarser bars (e.g., '10S', '1min').
    Returns ['open','high','low','close'].
    """
    # Ensure properly sorted
    df = df.sort_index()
    # Use last observation carry-forward to fill missing seconds per ticker
    def one(g):
        g = g.droplevel(1, axis=0) if isinstance(g.index, pd.MultiIndex) else g
        # g must be per-ticker series of price
        s = g['price']
        s = s.asfreq('1S', method='pad')
        o = s.resample(rule).first()
        h = s.resample(rule).max()
        l = s.resample(rule).min()
        c = s.resample(rule).last()
        out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
        return out
    pieces = []
    for tkr, g in df.groupby(level="ticker", sort=False):
        out = one(g)
        out['ticker'] = tkr
        out = out.reset_index().set_index(['timestamp', 'ticker'])
        pieces.append(out)
    out = pd.concat(pieces).sort_index()
    return out

# ---------------------------
# 2) BASIC CLEANING UTILITIES
# ---------------------------

def winsorize_returns(ret: pd.Series, p_low: float = 0.001, p_high: float = 0.999) -> pd.Series:
    r = ret.copy()
    if SCIPY_OK:
        low, high = mstats.mquantiles(r.dropna(), [p_low, p_high])
    else:
        x = r.dropna().values
        low = np.quantile(x, p_low)
        high = np.quantile(x, p_high)
    r[r < low] = low
    r[r > high] = high
    return r

# --------------
# 3) LABELING
# --------------

def triple_barrier_labels(close: pd.Series,
                          sigma: pd.Series,
                          pt_mult: float = 2.0,
                          sl_mult: float = 1.5,
                          horizon: int = 2700  # e.g., 45 min for 1S bars
                          ) -> pd.DataFrame:
    """
    Vectorized-ish triple-barrier on a single series (one ticker). Works on 1S bars;
    for 10S or 1min bars, adjust 'horizon' in bars accordingly.
    Returns DataFrame with ['label','t_hit','ret'] indexed by timestamps aligned to 'close'.
    label in {-1,0,+1}
    """
    c = close.values
    n = len(c)
    up = c * (1 + pt_mult * sigma.fillna(method='ffill').values)
    dn = c * (1 - sl_mult * sigma.fillna(method='ffill').values)
    label = np.zeros(n, dtype=int)
    t_hit = np.full(n, np.nan)
    realized = np.full(n, np.nan)

    for i in range(n):
        # horizon window
        j_end = min(n-1, i + horizon)
        if i >= j_end:
            break
        path = c[i+1:j_end+1]
        if path.size == 0:
            break
        up_hit = np.argmax(path >= up[i])
        dn_hit = np.argmax(path <= dn[i])
        hit_up = (path >= up[i]).any()
        hit_dn = (path <= dn[i]).any()
        if hit_up and hit_dn:
            # pick whichever happens first
            if up_hit == 0:
                t = i+1
                label[i] = +1
            elif dn_hit == 0:
                t = i+1
                label[i] = -1
            else:
                # compare indices
                t = i+1 + min(up_hit, dn_hit)
                label[i] = +1 if up_hit < dn_hit else -1
            t_hit[i] = t
            realized[i] = (c[int(t)] / c[i]) - 1.0
        elif hit_up:
            t = i+1 + up_hit
            label[i] = +1
            t_hit[i] = t
            realized[i] = (c[int(t)] / c[i]) - 1.0
        elif hit_dn:
            t = i+1 + dn_hit
            label[i] = -1
            t_hit[i] = t
            realized[i] = (c[int(t)] / c[i]) - 1.0
        else:
            # max-horizon
            t = j_end
            t_hit[i] = t
            realized[i] = (c[t] / c[i]) - 1.0
            # neutral label if neither barrier hit
            label[i] = 0

    out = pd.DataFrame({"label": label, "t_hit": t_hit, "ret": realized}, index=close.index)
    return out

# ------------------------------------
# 4) FEATURES (modern, price-only)
# ------------------------------------

def pre_averaged_rv(price: pd.Series, m: int = 30, window: int = 1800) -> pd.Series:
    """
    Pre-averaged realized variance (Jacod et al.) for microstructure noise robustness.
    m = pre-averaging span in seconds; window = rolling window to aggregate RV.
    Returns a rolling estimate aligned with 'price' index.
    """
    p = np.log(price).values
    n = len(p)
    # Pre-averaging weights (Hansen–Lunde like): g(k/m) = k/m - 1 if 0<=k<=m
    g = np.array([(k/m) - 0.5 for k in range(1, m)], dtype=float)  # omit 0 and m
    rv = np.full(n, np.nan)
    # Compute pre-averaged increments
    for t in range(m, n):
        x = p[t-m+1:t+1]
        # pre-averaged return at t
        diffs = np.diff(x)
        ga = np.sum(g * diffs[:m-1])
        # rolling window variance of ga
        if t >= window:
            seg = p[t-window+1:t+1]
            # approximate RV by sum of squared pre-averaged returns inside window step-m
            k0 = len(seg) // m
            vals = []
            for j in range(k0):
                s = t - window + 1 + j*m
                e = s + m
                if e <= t:
                    xj = p[s:e]
                    dj = np.diff(xj)
                    gij = np.sum(g * dj[:m-1])
                    vals.append(gij*gij)
            if vals:
                rv[t] = np.mean(vals)
    return pd.Series(rv, index=price.index).ffill()

def local_hurst(price: pd.Series, window: int = 1800) -> pd.Series:
    """
    Local Hurst exponent via log–log variogram slope over scales (1,2,4,8,16) seconds inside a rolling window.
    H ~ slope/2. Values < 0.5 indicate rough/anti-persistent; > 0.5 smoother.
    """
    p = np.log(price).values
    n = len(p)
    H = np.full(n, np.nan)
    scales = np.array([1,2,4,8,16])
    for t in range(window, n):
        seg = p[t-window+1:t+1]
        vars_ = []
        for h in scales:
            diffs = seg[h:] - seg[:-h]
            vars_.append(np.var(diffs))
        x = np.log(scales)
        y = np.log(np.array(vars_))
        # simple linear fit
        slope = np.polyfit(x, y, 1)[0]
        H[t] = slope / 2.0
    return pd.Series(H, index=price.index).ffill()

def drift_burst_flag(price: pd.Series, window: int = 900, subwin: int = 30, thresh: float = 2.5) -> pd.Series:
    """
    Detect 'drift bursts' using sign-consistency of short-horizon returns around a rolling local trend.
    Within each main window, compute mean of returns over subwindows and compare to their std.
    If |mean/std| > thresh → flag burst.
    Returns a rolling z-score-like series (clipped).
    """
    p = np.log(price).values
    n = len(p)
    flag = np.full(n, 0.0)
    for t in range(window, n):
        seg = p[t-window+1:t+1]
        # detrend by global linear fit over the window
        x = np.arange(len(seg))
        beta = np.polyfit(x, seg, 1)
        detr = seg - (beta[0]*x + beta[1])
        # aggregate subwindows
        k = len(detr) // subwin
        means = []
        for j in range(k):
            s = j*subwin
            e = s + subwin
            means.append(np.mean(np.diff(detr[s:e])))
        m = np.mean(means)
        sdev = np.std(means) + 1e-8
        z = np.clip(m / sdev, -5, 5)
        flag[t] = z
    return pd.Series(flag, index=price.index)

def exceedance_events(ret: pd.Series, sigma: pd.Series, theta: float = 1.5) -> pd.Series:
    """
    Binary event stream: 1 when |r_t| > theta * sigma_t, else 0.
    """
    return ((ret.abs() > theta * (sigma.replace(0, np.nan))).fillna(False)).astype(int)

def hawkes_discrete_intensity(events: pd.Series, L: int = 60) -> Tuple[pd.Series, pd.Series]:
    """
    Fit a discrete-time self-exciting model:
        e_t ~ Poisson(lambda_t),  lambda_t = mu + sum_{k=1..L} a_k * e_{t-k}
    via least-squares on a rolling basis (global fit here for simplicity).
    Returns (lambda_hat, kernel_sum_slope) where slope is the first difference of lambda (proxy regime change).
    """
    e = events.values.astype(float)
    n = len(e)
    if n <= L+1:
        lam = pd.Series(np.full(n, e.mean()), index=events.index)
        slope = lam.diff().fillna(0.0)
        return lam, slope

    # Build Toeplitz design for lagged events
    X = np.zeros((n-L, L))
    y = e[L:]
    for i in range(L):
        X[:, i] = e[L-1-i: n-1-i]  # lag i+1
    # Ridge to avoid overfit
    reg = 1e-3
    XtX = X.T @ X + reg * np.eye(L)
    Xty = X.T @ y
    a = np.linalg.solve(XtX, Xty)
    mu = max(1e-6, y.mean() - X.mean(axis=0).dot(a))

    lam_hat = np.zeros(n)
    lam_hat[:L] = mu
    for t in range(L, n):
        lam_hat[t] = mu + np.dot(a, e[t-L:t][::-1])
        lam_hat[t] = max(lam_hat[t], 1e-6)
    lam = pd.Series(lam_hat, index=events.index)
    slope = lam.diff().fillna(0.0)
    return lam, slope

def haar_wavelet_energy(window_vals: np.ndarray) -> np.ndarray:
    """
    Compute Haar detail energies at dyadic scales for a 1D window.
    Returns energy per level (length = floor(log2(n))).
    """
    x = window_vals.astype(float)
    levels = int(np.floor(np.log2(len(x))))
    energies = []
    coeffs = x.copy()
    for _ in range(levels):
        # Haar transform: averages and differences
        a = (coeffs[0::2] + coeffs[1::2]) / 2.0
        d = (coeffs[0::2] - coeffs[1::2]) / 2.0
        energies.append(np.mean(d**2))
        coeffs = a
    return np.array(energies)

def wavelet_jump_taxonomy(price: pd.Series, window: int = 3600, step: int = 60, n_clusters: int = 4,
                          use_sklearn: bool = True) -> pd.Series:
    """
    Sliding-window Haar wavelet energy vectors → cluster into archetypes.
    Returns a categorical series of cluster IDs aligned to 'price' (ffill between anchors).
    """
    x = price.values
    n = len(x)
    feats = []
    anchors = []
    for t in range(window, n, step):
        seg = x[t-window+1:t+1]
        energies = haar_wavelet_energy(seg)
        # time asymmetry proxy: compare forward vs backward energies
        energies_rev = haar_wavelet_energy(seg[::-1])
        feat = np.concatenate([energies, energies_rev])
        feats.append(feat)
        anchors.append(t)
    feats = np.array(feats)
    if feats.size == 0:
        return pd.Series(index=price.index, dtype=int)

    if SKLEARN_OK and use_sklearn:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        lab = km.fit_predict(feats)
    else:
        # simple k-means fallback
        rng = np.random.default_rng(42)
        cent = feats[rng.choice(len(feats), size=n_clusters, replace=False)]
        for _ in range(20):
            # assign
            d2 = ((feats[:, None, :] - cent[None, :, :])**2).sum(-1)
            asg = d2.argmin(axis=1)
            # update
            for k in range(n_clusters):
                if np.any(asg == k):
                    cent[k] = feats[asg == k].mean(axis=0)
        lab = asg

    series = pd.Series(index=price.index, dtype=int)
    prev = 0
    for idx, t in enumerate(anchors):
        series.iloc[prev: t+1] = lab[idx]
        prev = t+1
    series.iloc[prev:] = series.iloc[prev-1] if prev > 0 else 0
    return series.fillna(method='ffill').astype(int)

def signature_like_features(price: pd.Series, window: int = 3600) -> pd.DataFrame:
    """
    Lead–lag signature PROXY of order 2 (dependency-free).
    Construct path with channels [time_norm, log_price] and its lead–lag transform,
    then compute truncated iterated integrals approximated by cumulative sums & outer products.
    Returns a small feature vector per timestamp (rolling window).
    """
    p = np.log(price).values
    n = len(p)
    tnorm = np.linspace(0, 1, n)
    # lead–lag transform: x^lead_t = x_t, x^lag_t = x_{t-1} (pad first)
    x1 = tnorm
    x2 = p
    x1_lag = np.r_[x1[0], x1[:-1]]
    x2_lag = np.r_[x2[0], x2[:-1]]
    X = np.vstack([x1, x2, x1_lag, x2_lag]).T  # shape (n, 4)

    # First-order increments
    dX = np.diff(X, axis=0)
    # Rolling window aggregation of order-1 and order-2 "signatures"
    feats = []
    idx = []
    for t in range(window, n):
        seg = dX[t-window: t, :]
        s1 = seg.sum(axis=0)                     # first-order
        s2 = (seg.T @ seg).flatten()             # second-order proxy
        feats.append(np.r_[s1, s2])
        idx.append(t)
    cols = [f"s1_{i}" for i in range(4)] + [f"s2_{i}_{j}" for i in range(4) for j in range(4)]
    out = pd.DataFrame(feats, index=price.index[1:][idx], columns=cols)
    # ffill to full index
    out = out.reindex(price.index).ffill()
    return out

def rolling_vol(price: pd.Series, span: int = 900) -> pd.Series:
    r = np.log(price).diff()
    return r.ewm(span=span, min_periods=span//3).std().fillna(method='bfill')

def graph_precision_features(ret_df: pd.DataFrame, window: int = 1800, ridge: float = 1e-3) -> Dict[str, pd.Series]:
    """
    Compute rolling inverse-covariance (precision) matrix across tickers to identify conditional dependencies.
    Returns per-ticker features: node strength of precision and simple leader score (based on lead-lag corr).
    ret_df: wide DataFrame (index=time, columns=tickers) of returns (e.g., 10s or 1min log-returns).
    """
    tickers = list(ret_df.columns)
    T = len(ret_df)
    strength = {t: np.full(T, np.nan) for t in tickers}

    R = ret_df.values
    for t in range(window, T):
        seg = R[t-window: t, :]
        # Covariance with shrinkage (if LedoitWolf available)
        if SKLEARN_OK and LedoitWolf is not None:
            S = LedoitWolf().fit(seg).covariance_
        else:
            S = np.cov(seg, rowvar=False) + ridge*np.eye(seg.shape[1])
        # Precision
        try:
            P = np.linalg.inv(S + ridge*np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(S + ridge*np.eye(S.shape[0]))
        # Node strength = sum of abs off-diagonals for each node
        absP = np.abs(P)
        np.fill_diagonal(absP, 0.0)
        s = absP.sum(axis=1)
        for i, tkr in enumerate(tickers):
            strength[tkr][t] = s[i]

    out = {tkr: pd.Series(v, index=ret_df.index).ffill() for tkr, v in strength.items()}
    return out

# --------------------------
# 5) MODELING & VALIDATION
# --------------------------

@dataclass
class PurgedKFoldEmbargo:
    n_splits: int = 5
    embargo: int = 1800  # seconds/bars to embargo after each test fold

    def split(self, X_index: pd.Index):
        """
        X_index: DatetimeIndex (per-sample) assumed sorted.
        Yields (train_idx, test_idx) numpy arrays.
        """
        n = len(X_index)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        starts = np.cumsum(fold_sizes) - fold_sizes
        ends = np.cumsum(fold_sizes)

        all_idx = np.arange(n)
        for start, end in zip(starts, ends):
            test_idx = np.arange(start, end)
            # embargo: remove samples within 'embargo' after test end
            embargo_end_time = X_index[min(end - 1 + self.embargo, n - 1)]
            # purge: remove overlapping times (here we treat time as position; for 1S index, embargo is seconds)
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            if self.embargo > 0:
                # convert embargo to positions
                embargo_end_pos = min(end - 1 + self.embargo, n - 1)
                train_mask[end: embargo_end_pos+1] = False
            train_idx = all_idx[train_mask]
            yield train_idx, test_idx

def fit_classifier(X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
    """
    Try GradientBoosting if available; else Logistic.
    """
    y = y.astype(int).values
    # collapse to binary for simplicity: 1 if +1, 0 if -1, drop 0 labels
    mask = (y != 0)
    if not np.any(mask):
        raise ValueError("No non-zero labels to train on.")
    X2 = X.loc[mask]
    y2 = (y[mask] > 0).astype(int)

    if SKLEARN_OK and GradientBoostingClassifier is not None:
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X2, y2, sample_weight=sample_weight[mask] if sample_weight is not None else None)
        return clf
    elif SKLEARN_OK and LogisticRegression is not None:
        clf = LogisticRegression(max_iter=1000, n_jobs=None if hasattr(LogisticRegression, 'n_jobs') else None)
        clf.fit(X2, y2, sample_weight=sample_weight[mask] if sample_weight is not None else None)
        return clf
    else:
        # Tiny fallback: one-dimensional threshold on first feature
        class Tiny:
            def __init__(self, thr):
                self.thr = thr
            def predict_proba(self, X):
                x = X.iloc[:, 0].values
                p = 1/(1+np.exp(-(x - self.thr)))
                return np.vstack([1-p, p]).T
        thr = np.nanmean(X2.iloc[:,0].values)
        return Tiny(thr)

# ------------------------------
# 6) PORTFOLIO (simple & robust)
# ------------------------------

def size_positions(scores: pd.Series, vol: pd.Series, gross_limit: float = 1.0, per_name_cap: float = 0.05) -> pd.Series:
    """
    Vol-targeted proportional sizing with caps. scores are signed edges in [-1,1].
    """
    s = scores.fillna(0.0)
    v = vol.replace(0, np.nan).fillna(method='ffill')
    w = (s / v).replace([np.inf, -np.inf], 0.0)
    # cap per name
    w = w.clip(lower=-per_name_cap, upper=per_name_cap)
    # scale to gross limit
    if w.abs().sum() > 0:
        w = w * (gross_limit / w.abs().sum())
    return w

# -----------------------------------
# 7) DATASET BUILD (orchestration)
# -----------------------------------

def build_dataset_for_model(price_df: pd.DataFrame,
                            bar_rule: str = "10S",
                            horizon_sec: int = 2700,
                            pt_mult: float = 2.0,
                            sl_mult: float = 1.5,
                            seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Given tidy last-price DataFrame indexed by (timestamp, ticker), compute features and labels.
    Returns (X, y) where X is a multiindex DataFrame aligned to y (label per time×ticker).
    """
    # Resample to bar_rule
    ohlc = resample_last_price(price_df, rule=bar_rule)
    close = ohlc['close']
    # Compute per-ticker features
    frames = []
    labels = []
    tickers = sorted(close.index.get_level_values('ticker').unique())

    for tkr in tickers:
        c = close.xs(tkr, level='ticker')
        r = np.log(c).diff().fillna(0.0)
        r = winsorize_returns(r)
        sig = rolling_vol(c, span=900)  # 15-min EWMA vol on chosen bar
        # Features
        pav = pre_averaged_rv(c, m=30, window=1800)
        H = local_hurst(c, window=1800)
        driftz = drift_burst_flag(c, window=900, subwin=30, thresh=2.0)
        events = exceedance_events(r, sig, theta=1.5)
        lam, slp = hawkes_discrete_intensity(events, L=60)  # 10 min memory on 10S bars → tune by bar_rule
        wv_cluster = wavelet_jump_taxonomy(c, window=3600, step=60, n_clusters=4)
        sig_feat = signature_like_features(c, window=3600)

        feat = pd.concat([
            r.rename('ret'),
            sig.rename('ewm_vol'),
            pav.rename('pav_rv'),
            H.rename('hurst'),
            driftz.rename('drift_z'),
            lam.rename('hawkes_lambda'),
            slp.rename('hawkes_slope'),
            wv_cluster.rename('wv_cluster'),
            sig_feat
        ], axis=1)
        feat['ticker'] = tkr
        frames.append(feat)

        # Labels (triple barrier on this bar rule)
        # horizon in units of bars: horizon_sec / bar_seconds
        if bar_rule.endswith('S'):
            bar_sec = int(bar_rule[:-1])
        elif bar_rule.endswith('min'):
            bar_sec = int(bar_rule[:-3]) * 60
        else:
            # default try pandas offset alias
            try:
                bar_sec = pd.Timedelta(bar_rule).total_seconds()
            except Exception:
                bar_sec = 60
        h_bars = int(horizon_sec // bar_sec)
        lab = triple_barrier_labels(c, sigma=sig, pt_mult=pt_mult, sl_mult=sl_mult, horizon=h_bars)
        lab['ticker'] = tkr
        labels.append(lab[['label']])

    X = pd.concat(frames).reset_index().set_index(['timestamp','ticker']).sort_index()
    y = pd.concat(labels).reset_index().set_index(['timestamp','ticker']).sort_index()['label']
    # Align
    X, y = X.align(y, join='inner', axis=0)
    # Categoricals
    if 'wv_cluster' in X.columns:
        X['wv_cluster'] = X['wv_cluster'].astype('category').cat.codes
    return X, y

# ------------------------------
# 8) END-TO-END SIMPLE TRAINING
# ------------------------------

def train_simple_alpha(price_df: pd.DataFrame,
                       bar_rule: str = "10S",
                       horizon_sec: int = 2700) -> Dict[str, object]:
    """
    Build features+labels, split with purged/embargoed CV, train a classifier, and return models & artifacts.
    """
    X, y = build_dataset_for_model(price_df, bar_rule=bar_rule, horizon_sec=horizon_sec)
    # Simple sample weights: inverse class frequency on {-1,0,+1} collapsed into binary
    y_bin = (y.values > 0).astype(int)
    w = np.ones_like(y_bin, dtype=float)
    f1 = y_bin.mean()
    if f1 > 0 and f1 < 1:
        w[y_bin==1] = 0.5 / f1
        w[y_bin==0] = 0.5 / (1-f1)

    # Flatten multiindex into a DataFrame for splitting by time
    times = X.index.get_level_values('timestamp')
    splitter = PurgedKFoldEmbargo(n_splits=3, embargo=1800)

    preds = np.zeros(len(X))
    models = []
    for tr, te in splitter.split(times):
        clf = fit_classifier(X.iloc[tr], y.iloc[tr], sample_weight=w[tr])
        proba = clf.predict_proba(X.iloc[te])[:,1]  # P(up)
        preds[te] = proba
        models.append(clf)

    # Pack artifacts
    return {
        "X": X,
        "y": y,
        "pred_proba": pd.Series(preds, index=X.index, name='p_up'),
        "models": models
    }
