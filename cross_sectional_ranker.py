
"""
cross_sectional_ranker.py

Pipeline complet de ranking cross-sectionnel pour un univers d'actions observé
toutes les 10 secondes avec données bid / ask et tailles bid / ask optionnelles.

Objectif
--------
À chaque timestamp de ranking, classer les actions selon leur rendement futur
sur un horizon configurable, par exemple les 10 prochaines minutes.

Modèle
------
LightGBM LambdaRank / LambdaMART.

Format large attendu
--------------------
timestamp,ask_SAP,bid_SAP,ask_BMW,bid_BMW,...,
ask_size_SAP,bid_size_SAP,ask_size_BMW,bid_size_BMW,...

Les tailles sont optionnelles. Les préfixes sont configurables en ligne de
commande.

Principales features
--------------------
- spread absolu et relatif;
- mid-price et microprice;
- déséquilibre bid / ask;
- rendements 10 s, 1 min, 5 min, 10 min et 30 min;
- rendement depuis l'ouverture;
- overnight: clôture précédente -> ouverture;
- open-to-close des trois journées précédentes;
- volatilité réalisée;
- features relatives à l'indice equal-weight;
- normalisation cross-sectionnelle;
- signaux de pairs pondérés par corrélations positives et négatives.

Causalité
---------
Pour le jour D, la matrice de corrélation est calculée uniquement avec les
journées antérieures à D. Aucune donnée future n'est utilisée pour les features.

Installation
------------
pip install numpy pandas scikit-learn lightgbm pyarrow

Exécution
---------
python cross_sectional_ranker.py \
    --input quotes.parquet \
    --timestamp-col timestamp \
    --ask-prefix ask_ \
    --bid-prefix bid_ \
    --ask-size-prefix ask_size_ \
    --bid-size-prefix bid_size_ \
    --base-frequency 10s \
    --ranking-frequency 1min \
    --horizon 10min \
    --output-dir outputs_ranker

Test synthétique
----------------
python cross_sectional_ranker.py --demo --n-estimators 200
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


EPS = 1e-12


@dataclass
class Config:
    timestamp_col: str = "timestamp"

    ask_prefix: str = "ask_"
    bid_prefix: str = "bid_"
    ask_size_prefix: str = "ask_size_"
    bid_size_prefix: str = "bid_size_"

    base_frequency: str = "10s"
    ranking_frequency: str = "1min"
    horizon: str = "10min"

    market_open: str = "09:00:00"
    market_close: str = "17:30:00"

    corr_lookback_days: int = 20
    corr_min_days: int = 5
    corr_top_k: int = 5
    corr_threshold: float = 0.10
    corr_shrinkage: float = 0.10

    relevance_bins: int = 5
    target_mode: str = "raw"  # raw ou volatility_adjusted
    top_k: int = 5

    train_ratio: float = 0.70
    validation_ratio: float = 0.15

    n_estimators: int = 1500
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 100
    feature_fraction: float = 0.80
    bagging_fraction: float = 0.80
    bagging_freq: int = 1
    reg_alpha: float = 1.0
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 100

    seed: int = 42
    n_jobs: int = -1


# ---------------------------------------------------------------------------
# Lecture, détection des colonnes et validation
# ---------------------------------------------------------------------------

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz", ".zip"}:
        return pd.read_csv(path)

    raise ValueError(
        f"Format non supporté: {suffix}. Utiliser CSV, CSV.GZ ou Parquet."
    )


def infer_tickers(
    columns: Iterable[str],
    ask_prefix: str,
    bid_prefix: str,
    ask_size_prefix: Optional[str],
    bid_size_prefix: Optional[str],
) -> List[str]:
    columns = list(map(str, columns))

    def quote_columns(prefix: str, excluded_prefix: Optional[str]) -> set[str]:
        result = set()
        for column in columns:
            if not column.startswith(prefix):
                continue
            if excluded_prefix and column.startswith(excluded_prefix):
                continue
            ticker = column[len(prefix):]
            if ticker:
                result.add(ticker)
        return result

    ask_tickers = quote_columns(ask_prefix, ask_size_prefix)
    bid_tickers = quote_columns(bid_prefix, bid_size_prefix)

    missing_bid = sorted(ask_tickers - bid_tickers)
    missing_ask = sorted(bid_tickers - ask_tickers)

    if missing_bid or missing_ask:
        raise ValueError(
            "Colonnes bid/ask non appariées. "
            f"Sans bid: {missing_bid[:10]}; sans ask: {missing_ask[:10]}"
        )

    tickers = sorted(ask_tickers & bid_tickers)
    if len(tickers) < 2:
        raise ValueError("Au moins deux actions sont nécessaires pour le ranking.")

    return tickers


def required_quote_columns(tickers: Sequence[str], cfg: Config) -> List[str]:
    columns = []
    for ticker in tickers:
        columns.extend(
            [
                f"{cfg.ask_prefix}{ticker}",
                f"{cfg.bid_prefix}{ticker}",
            ]
        )

        ask_size = f"{cfg.ask_size_prefix}{ticker}" if cfg.ask_size_prefix else None
        bid_size = f"{cfg.bid_size_prefix}{ticker}" if cfg.bid_size_prefix else None

        if ask_size:
            columns.append(ask_size)
        if bid_size:
            columns.append(bid_size)

    return columns


def timedelta_to_bars(duration: str, base_frequency: str) -> int:
    duration_td = pd.Timedelta(duration)
    base_td = pd.Timedelta(base_frequency)

    if duration_td < base_td:
        raise ValueError(
            f"{duration} est inférieur à la fréquence de base {base_frequency}."
        )

    ratio = duration_td / base_td
    rounded = round(float(ratio))

    if not np.isclose(ratio, rounded):
        raise ValueError(
            f"{duration} doit être un multiple entier de {base_frequency}."
        )

    return int(rounded)


def resample_flat_quotes(
    raw: pd.DataFrame,
    tickers: Sequence[str],
    cfg: Config,
) -> pd.DataFrame:
    if cfg.timestamp_col not in raw.columns:
        raise ValueError(f"Colonne absente: {cfg.timestamp_col}")

    work = raw.copy()
    work[cfg.timestamp_col] = pd.to_datetime(
        work[cfg.timestamp_col],
        errors="coerce",
    )
    work = work.dropna(subset=[cfg.timestamp_col])
    work = work.sort_values(cfg.timestamp_col)
    work = work.drop_duplicates(subset=[cfg.timestamp_col], keep="last")
    work = work.set_index(cfg.timestamp_col)

    quote_columns = []
    for ticker in tickers:
        quote_columns.extend(
            [f"{cfg.ask_prefix}{ticker}", f"{cfg.bid_prefix}{ticker}"]
        )

    missing_quotes = [column for column in quote_columns if column not in work.columns]
    if missing_quotes:
        raise ValueError(f"Colonnes de prix absentes: {missing_quotes[:10]}")

    size_columns = []
    if cfg.ask_size_prefix and cfg.bid_size_prefix:
        for ticker in tickers:
            ask_size = f"{cfg.ask_size_prefix}{ticker}"
            bid_size = f"{cfg.bid_size_prefix}{ticker}"
            if ask_size in work.columns:
                size_columns.append(ask_size)
            if bid_size in work.columns:
                size_columns.append(bid_size)

    used_columns = quote_columns + size_columns
    work = work[used_columns].apply(pd.to_numeric, errors="coerce")

    daily_parts = []

    for _, day in work.groupby(work.index.normalize(), sort=True):
        day = day.between_time(
            cfg.market_open,
            cfg.market_close,
            inclusive="both",
        )
        if day.empty:
            continue

        # Les données sont des snapshots. Le dernier état connu est porté en avant
        # à l'intérieur de la journée uniquement.
        day = day.resample(cfg.base_frequency).last().ffill()
        daily_parts.append(day)

    if not daily_parts:
        raise ValueError("Aucune donnée disponible dans les horaires de marché.")

    result = pd.concat(daily_parts).sort_index()
    return result


def extract_panels(
    quotes: pd.DataFrame,
    tickers: Sequence[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    ask = pd.DataFrame(
        {
            ticker: quotes[f"{cfg.ask_prefix}{ticker}"]
            for ticker in tickers
        },
        index=quotes.index,
    ).astype("float64")

    bid = pd.DataFrame(
        {
            ticker: quotes[f"{cfg.bid_prefix}{ticker}"]
            for ticker in tickers
        },
        index=quotes.index,
    ).astype("float64")

    valid = (
        ask.gt(0)
        & bid.gt(0)
        & ask.ge(bid)
    )
    ask = ask.where(valid)
    bid = bid.where(valid)

    ask_size = None
    bid_size = None

    if cfg.ask_size_prefix and cfg.bid_size_prefix:
        ask_data = {}
        bid_data = {}

        for ticker in tickers:
            ask_column = f"{cfg.ask_size_prefix}{ticker}"
            bid_column = f"{cfg.bid_size_prefix}{ticker}"

            if ask_column in quotes.columns:
                ask_data[ticker] = quotes[ask_column]
            else:
                ask_data[ticker] = pd.Series(np.nan, index=quotes.index)

            if bid_column in quotes.columns:
                bid_data[ticker] = quotes[bid_column]
            else:
                bid_data[ticker] = pd.Series(np.nan, index=quotes.index)

        ask_size = pd.DataFrame(ask_data, index=quotes.index).astype("float64")
        bid_size = pd.DataFrame(bid_data, index=quotes.index).astype("float64")

        ask_size = ask_size.where(ask_size >= 0)
        bid_size = bid_size.where(bid_size >= 0)

        if ask_size.notna().sum().sum() == 0 or bid_size.notna().sum().sum() == 0:
            ask_size = None
            bid_size = None

    return ask, bid, ask_size, bid_size


# ---------------------------------------------------------------------------
# Transformations temporelles sans traverser la nuit
# ---------------------------------------------------------------------------

def apply_by_day(frame: pd.DataFrame, operation) -> pd.DataFrame:
    parts = []
    for _, day in frame.groupby(frame.index.normalize(), sort=True):
        parts.append(operation(day))
    return pd.concat(parts).sort_index()


def log_return(frame: pd.DataFrame, bars: int) -> pd.DataFrame:
    logged = np.log(frame)
    return apply_by_day(logged, lambda day: day.diff(bars))


def future_log_return(frame: pd.DataFrame, bars: int) -> pd.DataFrame:
    logged = np.log(frame)
    return apply_by_day(
        logged,
        lambda day: day.shift(-bars) - day,
    )


def since_open_return(frame: pd.DataFrame) -> pd.DataFrame:
    logged = np.log(frame)
    return apply_by_day(
        logged,
        lambda day: day - day.iloc[0],
    )


def rolling_std_by_day(
    frame: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    min_periods = min_periods or max(2, window // 3)
    return apply_by_day(
        frame,
        lambda day: day.rolling(
            window=window,
            min_periods=min_periods,
        ).std(),
    )


def rolling_mean_by_day(
    frame: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    min_periods = min_periods or max(2, window // 3)
    return apply_by_day(
        frame,
        lambda day: day.rolling(
            window=window,
            min_periods=min_periods,
        ).mean(),
    )


def daily_repeated_features(
    mid: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    log_mid = np.log(mid)
    normalized_dates = log_mid.index.normalize()
    unique_dates = pd.DatetimeIndex(normalized_dates.unique()).sort_values()

    daily_open = log_mid.groupby(normalized_dates).first().reindex(unique_dates)
    daily_close = log_mid.groupby(normalized_dates).last().reindex(unique_dates)

    overnight_daily = daily_open - daily_close.shift(1)
    open_close_daily = daily_close - daily_open

    output: Dict[str, pd.DataFrame] = {}

    def repeat_daily(daily_values: pd.DataFrame) -> pd.DataFrame:
        values = daily_values.reindex(normalized_dates)
        values.index = log_mid.index
        values.columns = log_mid.columns
        return values

    output["overnight_return"] = repeat_daily(overnight_daily)

    for lag in (1, 2, 3):
        output[f"open_close_day_minus_{lag}"] = repeat_daily(
            open_close_daily.shift(lag)
        )

    return output


def select_ranking_timestamps(
    index: pd.DatetimeIndex,
    base_frequency: str,
    ranking_frequency: str,
) -> pd.DatetimeIndex:
    step = timedelta_to_bars(ranking_frequency, base_frequency)
    selected = []

    for _, day_index in pd.Series(index=index, data=1).groupby(index.normalize()):
        selected.extend(day_index.index[::step])

    return pd.DatetimeIndex(selected)


# ---------------------------------------------------------------------------
# Corrélations causales
# ---------------------------------------------------------------------------

def historical_one_minute_returns(
    mid: pd.DataFrame,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    output = {}

    for day_date, day in mid.groupby(mid.index.normalize(), sort=True):
        minute_mid = day.resample("1min").last().ffill()
        minute_returns = np.log(minute_mid).diff().dropna(how="all")
        output[pd.Timestamp(day_date)] = minute_returns

    return output


def build_weight_matrices(
    correlation: pd.DataFrame,
    top_k: int,
    threshold: float,
    shrinkage: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    corr = correlation.to_numpy(dtype=np.float64)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    n = corr.shape[0]
    corr = corr * (1.0 - shrinkage)
    np.fill_diagonal(corr, 0.0)

    positive = np.zeros_like(corr)
    negative_abs = np.zeros_like(corr)
    signed = np.zeros_like(corr)

    for i in range(n):
        row = corr[i]

        positive_candidates = np.where(row > threshold)[0]
        if len(positive_candidates):
            selected = positive_candidates[
                np.argsort(row[positive_candidates])[-top_k:]
            ]
            positive[i, selected] = row[selected]

        negative_candidates = np.where(row < -threshold)[0]
        if len(negative_candidates):
            selected = negative_candidates[
                np.argsort(np.abs(row[negative_candidates]))[-top_k:]
            ]
            negative_abs[i, selected] = np.abs(row[selected])

        absolute_candidates = np.where(np.abs(row) > threshold)[0]
        if len(absolute_candidates):
            selected = absolute_candidates[
                np.argsort(np.abs(row[absolute_candidates]))[-top_k:]
            ]
            signed[i, selected] = row[selected]

    positive_sum = positive.sum(axis=1, keepdims=True)
    negative_sum = negative_abs.sum(axis=1, keepdims=True)
    signed_sum = np.abs(signed).sum(axis=1, keepdims=True)

    positive = np.divide(
        positive,
        positive_sum,
        out=np.zeros_like(positive),
        where=positive_sum > EPS,
    )
    negative_abs = np.divide(
        negative_abs,
        negative_sum,
        out=np.zeros_like(negative_abs),
        where=negative_sum > EPS,
    )
    signed = np.divide(
        signed,
        signed_sum,
        out=np.zeros_like(signed),
        where=signed_sum > EPS,
    )

    stats = {
        "mean_abs_corr": np.mean(np.abs(corr), axis=1),
        "max_positive_corr": np.max(corr, axis=1),
        "min_negative_corr": np.min(corr, axis=1),
        "positive_peer_count": np.sum(corr > threshold, axis=1).astype(float),
        "negative_peer_count": np.sum(corr < -threshold, axis=1).astype(float),
    }

    return positive, negative_abs, signed, stats


def causal_correlation_features(
    mid: pd.DataFrame,
    selected_index: pd.DatetimeIndex,
    signal_frames: Mapping[str, pd.DataFrame],
    optional_imbalance: Optional[pd.DataFrame],
    cfg: Config,
) -> Dict[str, pd.DataFrame]:
    tickers = list(mid.columns)
    dates = sorted(pd.DatetimeIndex(mid.index.normalize().unique()))
    historical_returns = historical_one_minute_returns(mid)

    selected_set = set(selected_index)
    output_parts: Dict[str, List[pd.DataFrame]] = {}

    signal_sources = dict(signal_frames)
    if optional_imbalance is not None:
        signal_sources["order_imbalance"] = optional_imbalance

    for day_position, day_date in enumerate(dates):
        current_day_index = pd.DatetimeIndex(
            [
                timestamp
                for timestamp in mid.loc[mid.index.normalize() == day_date].index
                if timestamp in selected_set
            ]
        )

        if len(current_day_index) == 0:
            continue

        prior_dates = dates[
            max(0, day_position - cfg.corr_lookback_days):day_position
        ]

        if len(prior_dates) < cfg.corr_min_days:
            continue

        history = [
            historical_returns[date]
            for date in prior_dates
            if date in historical_returns
        ]
        if not history:
            continue

        history_frame = pd.concat(history).dropna(how="all")

        min_periods = max(30, len(history_frame) // 10)
        correlation = history_frame.corr(min_periods=min_periods)
        correlation = correlation.reindex(index=tickers, columns=tickers)

        positive, negative_abs, signed, stats = build_weight_matrices(
            correlation=correlation,
            top_k=cfg.corr_top_k,
            threshold=cfg.corr_threshold,
            shrinkage=cfg.corr_shrinkage,
        )

        for source_name, source_frame in signal_sources.items():
            values = (
                source_frame
                .reindex(index=current_day_index, columns=tickers)
                .to_numpy(dtype=np.float64)
            )

            positive_signal = values @ positive.T
            negative_peer_movement = values @ negative_abs.T
            signed_signal = values @ signed.T

            day_outputs = {
                f"corr_positive_peer_{source_name}": positive_signal,
                f"corr_negative_peer_{source_name}": negative_peer_movement,
                f"corr_inverse_negative_peer_{source_name}": -negative_peer_movement,
                f"corr_signed_peer_{source_name}": signed_signal,
            }

            for feature_name, feature_values in day_outputs.items():
                output_parts.setdefault(feature_name, []).append(
                    pd.DataFrame(
                        feature_values,
                        index=current_day_index,
                        columns=tickers,
                    )
                )

        for feature_name, feature_values in stats.items():
            repeated = np.repeat(
                feature_values.reshape(1, -1),
                repeats=len(current_day_index),
                axis=0,
            )
            output_parts.setdefault(feature_name, []).append(
                pd.DataFrame(
                    repeated,
                    index=current_day_index,
                    columns=tickers,
                )
            )

    output = {}
    for feature_name, parts in output_parts.items():
        output[feature_name] = pd.concat(parts).sort_index().astype("float32")

    return output


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_feature_frames(
    ask: pd.DataFrame,
    bid: pd.DataFrame,
    ask_size: Optional[pd.DataFrame],
    bid_size: Optional[pd.DataFrame],
    cfg: Config,
) -> Tuple[
    Dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DatetimeIndex,
]:
    tickers = list(ask.columns)
    mid = (ask + bid) / 2.0
    spread = ask - bid
    relative_spread = spread / mid

    selected_index = select_ranking_timestamps(
        mid.index,
        cfg.base_frequency,
        cfg.ranking_frequency,
    )

    window_bars = {
        "10s": timedelta_to_bars("10s", cfg.base_frequency),
        "1m": timedelta_to_bars("1min", cfg.base_frequency),
        "5m": timedelta_to_bars("5min", cfg.base_frequency),
        "10m": timedelta_to_bars("10min", cfg.base_frequency),
        "30m": timedelta_to_bars("30min", cfg.base_frequency),
    }

    returns = {
        name: log_return(mid, bars)
        for name, bars in window_bars.items()
    }

    ret_10s = returns["10s"]

    features: Dict[str, pd.DataFrame] = {
        "spread": spread,
        "spread_bps": relative_spread * 10_000.0,
        "return_10s": returns["10s"],
        "return_1m": returns["1m"],
        "return_5m": returns["5m"],
        "return_10m": returns["10m"],
        "return_30m": returns["30m"],
        "return_since_open": since_open_return(mid),
        "realized_vol_1m": rolling_std_by_day(
            ret_10s,
            window_bars["1m"],
        ),
        "realized_vol_5m": rolling_std_by_day(
            ret_10s,
            window_bars["5m"],
        ),
        "realized_vol_10m": rolling_std_by_day(
            ret_10s,
            window_bars["10m"],
        ),
        "realized_vol_30m": rolling_std_by_day(
            ret_10s,
            window_bars["30m"],
        ),
        "spread_bps_mean_5m": rolling_mean_by_day(
            relative_spread * 10_000.0,
            window_bars["5m"],
        ),
        "spread_bps_mean_30m": rolling_mean_by_day(
            relative_spread * 10_000.0,
            window_bars["30m"],
        ),
    }

    features.update(daily_repeated_features(mid))

    imbalance = None

    if ask_size is not None and bid_size is not None:
        total_depth = ask_size + bid_size
        imbalance = (bid_size - ask_size) / total_depth.replace(0.0, np.nan)

        # Microprice au niveau 1:
        # davantage proche de l'ask quand la taille bid est forte, et inversement.
        microprice = (
            ask * bid_size + bid * ask_size
        ) / total_depth.replace(0.0, np.nan)

        log_bid_size = np.log1p(bid_size.clip(lower=0))
        log_ask_size = np.log1p(ask_size.clip(lower=0))

        features.update(
            {
                "log_bid_size": log_bid_size,
                "log_ask_size": log_ask_size,
                "log_total_depth": np.log1p(total_depth.clip(lower=0)),
                "order_imbalance": imbalance,
                "microprice_deviation_bps": (
                    (microprice - mid) / mid * 10_000.0
                ),
                "log_bid_ask_size_ratio": (
                    np.log1p(bid_size.clip(lower=0))
                    - np.log1p(ask_size.clip(lower=0))
                ),
                "bid_size_change_1m": apply_by_day(
                    log_bid_size,
                    lambda day: day.diff(window_bars["1m"]),
                ),
                "ask_size_change_1m": apply_by_day(
                    log_ask_size,
                    lambda day: day.diff(window_bars["1m"]),
                ),
                "imbalance_mean_1m": rolling_mean_by_day(
                    imbalance,
                    window_bars["1m"],
                ),
                "imbalance_mean_5m": rolling_mean_by_day(
                    imbalance,
                    window_bars["5m"],
                ),
            }
        )

    # Indice equal-weight implicite: moyenne des rendements des constituants.
    for horizon_name in ("1m", "5m", "10m", "30m"):
        stock_return = returns[horizon_name]
        index_return = stock_return.mean(axis=1)

        repeated_index = pd.DataFrame(
            np.repeat(
                index_return.to_numpy().reshape(-1, 1),
                repeats=len(tickers),
                axis=1,
            ),
            index=stock_return.index,
            columns=tickers,
        )

        features[f"index_return_{horizon_name}"] = repeated_index
        features[f"relative_to_index_{horizon_name}"] = (
            stock_return - repeated_index
        )

    # Corrélations estimées sur les jours précédents uniquement.
    correlation_features = causal_correlation_features(
        mid=mid,
        selected_index=selected_index,
        signal_frames={
            "return_1m": returns["1m"],
            "return_10m": returns["10m"],
            "return_30m": returns["30m"],
        },
        optional_imbalance=imbalance,
        cfg=cfg,
    )
    features.update(correlation_features)

    horizon_bars = timedelta_to_bars(cfg.horizon, cfg.base_frequency)
    target_future_return = future_log_return(mid, horizon_bars)

    return features, target_future_return, mid, selected_index


def frame_to_long_series(
    frame: pd.DataFrame,
    selected_index: pd.DatetimeIndex,
    tickers: Sequence[str],
    name: str,
) -> pd.Series:
    selected = frame.reindex(index=selected_index, columns=tickers)
    try:
        series = selected.stack(future_stack=True)
    except TypeError:
        # Compatibilité avec les anciennes versions de pandas.
        series = selected.stack(dropna=False)
    series.index = series.index.set_names(["timestamp", "ticker"])
    series.name = name
    return series


def cross_sectional_zscore(
    values: pd.Series,
) -> pd.Series:
    grouped = values.groupby(level="timestamp")
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return (values - mean) / std


def build_learning_table(
    feature_frames: Mapping[str, pd.DataFrame],
    target_future_return: pd.DataFrame,
    selected_index: pd.DatetimeIndex,
    tickers: Sequence[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, List[str]]:
    base_index = pd.MultiIndex.from_product(
        [selected_index, tickers],
        names=["timestamp", "ticker"],
    )
    table = pd.DataFrame(index=base_index)

    for feature_name, feature_frame in feature_frames.items():
        table[feature_name] = frame_to_long_series(
            feature_frame,
            selected_index,
            tickers,
            feature_name,
        ).reindex(base_index).astype("float32")

    table["future_return"] = frame_to_long_series(
        target_future_return,
        selected_index,
        tickers,
        "future_return",
    ).reindex(base_index).astype("float32")

    timestamp_values = table.index.get_level_values("timestamp")
    ticker_values = table.index.get_level_values("ticker")

    market_open_delta = pd.to_timedelta(cfg.market_open)
    seconds_from_midnight = (
        timestamp_values.hour * 3600
        + timestamp_values.minute * 60
        + timestamp_values.second
    )
    open_seconds = int(market_open_delta.total_seconds())
    close_seconds = int(pd.to_timedelta(cfg.market_close).total_seconds())
    session_seconds = max(1, close_seconds - open_seconds)

    progress = np.clip(
        (seconds_from_midnight - open_seconds) / session_seconds,
        0.0,
        1.0,
    )

    table["session_progress"] = progress.astype("float32")
    table["session_sin"] = np.sin(2.0 * np.pi * progress).astype("float32")
    table["session_cos"] = np.cos(2.0 * np.pi * progress).astype("float32")

    ticker_dtype = pd.CategoricalDtype(categories=list(tickers), ordered=False)
    table["ticker_id"] = pd.Series(
        ticker_values,
        index=table.index,
        dtype=ticker_dtype,
    )

    # Quelques normalisations cross-sectionnelles utiles au ranking.
    zscore_sources = [
        "return_1m",
        "return_5m",
        "return_10m",
        "return_30m",
        "return_since_open",
        "spread_bps",
        "realized_vol_10m",
        "relative_to_index_10m",
    ]
    if "order_imbalance" in table.columns:
        zscore_sources.append("order_imbalance")
    if "microprice_deviation_bps" in table.columns:
        zscore_sources.append("microprice_deviation_bps")

    for feature_name in zscore_sources:
        if feature_name in table.columns:
            table[f"{feature_name}_xsec_z"] = cross_sectional_zscore(
                table[feature_name]
            ).astype("float32")

    if cfg.target_mode == "volatility_adjusted":
        volatility = table["realized_vol_30m"].abs().clip(lower=1e-8)
        table["ranking_target"] = table["future_return"] / volatility
    elif cfg.target_mode == "raw":
        table["ranking_target"] = table["future_return"]
    else:
        raise ValueError(
            "--target-mode doit être raw ou volatility_adjusted."
        )

    # Le label de pertinence est créé à l'intérieur de chaque timestamp.
    percentile_rank = (
        table.groupby(level="timestamp")["ranking_target"]
        .rank(method="first", pct=True)
    )
    labels = np.floor(percentile_rank * cfg.relevance_bins)
    labels = labels.clip(upper=cfg.relevance_bins - 1)

    table["label"] = labels.astype("float32")

    # Un groupe de ranking doit contenir tout l'univers.
    valid_target = table["future_return"].notna()
    valid_counts = valid_target.groupby(level="timestamp").sum()
    complete_timestamps = valid_counts[
        valid_counts == len(tickers)
    ].index

    table = table.loc[
        table.index.get_level_values("timestamp").isin(complete_timestamps)
    ].copy()
    table = table[table["future_return"].notna()]
    table["label"] = table["label"].astype("int8")

    non_features = {
        "future_return",
        "ranking_target",
        "label",
    }
    feature_columns = [
        column
        for column in table.columns
        if column not in non_features
    ]

    # Supprime les features entièrement manquantes.
    all_nan_features = [
        column
        for column in feature_columns
        if column != "ticker_id" and table[column].notna().sum() == 0
    ]
    if all_nan_features:
        table = table.drop(columns=all_nan_features)
        feature_columns = [
            column for column in feature_columns
            if column not in all_nan_features
        ]

    return table, feature_columns


# ---------------------------------------------------------------------------
# Split, entraînement et évaluation
# ---------------------------------------------------------------------------

def split_by_trading_days(
    table: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamps = table.index.get_level_values("timestamp")
    days = sorted(pd.DatetimeIndex(timestamps.normalize().unique()))

    if len(days) < 3:
        raise ValueError("Il faut au minimum trois journées exploitables.")

    n_days = len(days)
    n_train = max(1, int(n_days * train_ratio))
    n_validation = max(1, int(n_days * validation_ratio))

    if n_train + n_validation >= n_days:
        n_train = n_days - 2
        n_validation = 1

    train_days = set(days[:n_train])
    validation_days = set(days[n_train:n_train + n_validation])
    test_days = set(days[n_train + n_validation:])

    normalized = timestamps.normalize()

    train = table.loc[normalized.isin(train_days)].copy()
    validation = table.loc[normalized.isin(validation_days)].copy()
    test = table.loc[normalized.isin(test_days)].copy()

    return train, validation, test


def sort_ranking_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.reset_index()
        .sort_values(["timestamp", "ticker"])
        .set_index(["timestamp", "ticker"])
    )


def group_sizes(frame: pd.DataFrame) -> List[int]:
    return (
        frame.groupby(level="timestamp", sort=False)
        .size()
        .astype(int)
        .tolist()
    )


def prepare_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    matrix = frame[list(feature_columns)].copy()

    if "ticker_id" in matrix.columns:
        # Conserve le même dtype catégoriel dans tous les splits.
        if not isinstance(matrix["ticker_id"].dtype, pd.CategoricalDtype):
            matrix["ticker_id"] = matrix["ticker_id"].astype("category")

    return matrix


def train_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: Sequence[str],
    cfg: Config,
) -> lgb.LGBMRanker:
    train = sort_ranking_frame(train)
    validation = sort_ranking_frame(validation)

    x_train = prepare_feature_matrix(train, feature_columns)
    x_validation = prepare_feature_matrix(validation, feature_columns)

    y_train = train["label"].astype(int)
    y_validation = validation["label"].astype(int)

    train_groups = group_sizes(train)
    validation_groups = group_sizes(validation)

    truncation_level = min(max(cfg.top_k * 2, 10), max(train_groups))

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        feature_fraction=cfg.feature_fraction,
        bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        lambdarank_truncation_level=truncation_level,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbosity=-1,
    )

    categorical_features = (
        ["ticker_id"] if "ticker_id" in feature_columns else "auto"
    )

    eval_points = sorted(
        {
            min(cfg.top_k, max(train_groups)),
            min(cfg.top_k * 2, max(train_groups)),
        }
    )

    model.fit(
        x_train,
        y_train,
        group=train_groups,
        eval_set=[(x_validation, y_validation)],
        eval_group=[validation_groups],
        eval_at=eval_points,
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=cfg.early_stopping_rounds,
                verbose=True,
            ),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


def mean_ndcg(
    prediction_frame: pd.DataFrame,
    k: int,
) -> float:
    values = []

    for _, group in prediction_frame.groupby("timestamp", sort=False):
        if len(group) < 2:
            continue

        true_relevance = group["label"].to_numpy(dtype=float).reshape(1, -1)
        predicted_scores = group["score"].to_numpy(dtype=float).reshape(1, -1)

        values.append(
            ndcg_score(
                true_relevance,
                predicted_scores,
                k=min(k, len(group)),
            )
        )

    return float(np.mean(values)) if values else float("nan")


def evaluate_predictions(
    prediction_frame: pd.DataFrame,
    top_k: int,
) -> Dict[str, float]:
    group_metrics = []

    for timestamp, group in prediction_frame.groupby("timestamp", sort=False):
        group = group.sort_values("score", ascending=False)
        k = min(top_k, len(group) // 2)

        if k < 1:
            continue

        spearman_ic = group["score"].corr(
            group["future_return"],
            method="spearman",
        )

        top_return = group.head(k)["future_return"].mean()
        bottom_return = group.tail(k)["future_return"].mean()
        spread = top_return - bottom_return

        group_metrics.append(
            {
                "timestamp": timestamp,
                "spearman_ic": spearman_ic,
                "top_return": top_return,
                "bottom_return": bottom_return,
                "long_short_spread": spread,
            }
        )

    metrics_frame = pd.DataFrame(group_metrics)

    if metrics_frame.empty:
        return {}

    spread = metrics_frame["long_short_spread"]
    spread_std = spread.std(ddof=1)

    return {
        "mean_spearman_ic": float(metrics_frame["spearman_ic"].mean()),
        "median_spearman_ic": float(metrics_frame["spearman_ic"].median()),
        f"ndcg_at_{top_k}": mean_ndcg(prediction_frame, top_k),
        f"ndcg_at_{top_k * 2}": mean_ndcg(prediction_frame, top_k * 2),
        f"mean_top_{top_k}_future_return": float(
            metrics_frame["top_return"].mean()
        ),
        f"mean_bottom_{top_k}_future_return": float(
            metrics_frame["bottom_return"].mean()
        ),
        "mean_long_short_spread": float(spread.mean()),
        "long_short_hit_rate": float((spread > 0).mean()),
        "spread_information_ratio_per_prediction": (
            float(spread.mean() / spread_std)
            if spread_std and np.isfinite(spread_std)
            else float("nan")
        ),
        "n_ranking_timestamps": int(len(metrics_frame)),
    }


def predict_ranker(
    model: lgb.LGBMRanker,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    frame = sort_ranking_frame(frame)
    x = prepare_feature_matrix(frame, feature_columns)

    result = frame.reset_index()[
        [
            "timestamp",
            "ticker",
            "future_return",
            "ranking_target",
            "label",
        ]
    ].copy()

    result["score"] = model.predict(
        x,
        num_iteration=model.best_iteration_,
    )

    result["predicted_rank"] = (
        result.groupby("timestamp")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    result["realized_rank"] = (
        result.groupby("timestamp")["future_return"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return result.sort_values(["timestamp", "predicted_rank"])


# ---------------------------------------------------------------------------
# Données synthétiques
# ---------------------------------------------------------------------------

def generate_demo_quotes(
    n_days: int = 30,
    n_stocks: int = 40,
    bars_per_day: int = 360,
    base_frequency: str = "10s",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"S{i:02d}" for i in range(n_stocks)]

    rows = []
    first_day = pd.Timestamp("2025-01-02")

    # Une partie des actions a un beta négatif pour tester les corrélations signées.
    betas = rng.uniform(0.3, 1.2, size=n_stocks)
    betas[-max(2, n_stocks // 8):] *= -0.6

    previous_close = rng.uniform(50.0, 200.0, size=n_stocks)

    for day_offset in range(n_days):
        day = first_day + pd.offsets.BDay(day_offset)
        timestamps = pd.date_range(
            day + pd.Timedelta(hours=9),
            periods=bars_per_day,
            freq=base_frequency,
        )

        overnight_shock = rng.normal(0.0, 0.003, size=n_stocks)
        opening_price = previous_close * np.exp(overnight_shock)

        market = rng.normal(0.0, 0.00018, size=bars_per_day)
        common_reversal = np.zeros(bars_per_day)
        common_reversal[1:] = -0.15 * market[:-1]

        idiosyncratic = rng.normal(
            0.0,
            0.00025,
            size=(bars_per_day, n_stocks),
        )

        returns = (
            market.reshape(-1, 1) * betas.reshape(1, -1)
            + common_reversal.reshape(-1, 1)
            + idiosyncratic
        )

        log_price = np.log(opening_price) + np.cumsum(returns, axis=0)
        mid = np.exp(log_price)

        spread_bps = rng.uniform(0.5, 4.0, size=(bars_per_day, n_stocks))
        half_spread = mid * spread_bps / 20_000.0

        bid = mid - half_spread
        ask = mid + half_spread

        bid_size = rng.lognormal(
            mean=5.0,
            sigma=0.5,
            size=(bars_per_day, n_stocks),
        )
        ask_size = rng.lognormal(
            mean=5.0,
            sigma=0.5,
            size=(bars_per_day, n_stocks),
        )

        # Ajoute un petit lien entre imbalance et prochain mouvement.
        imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        if bars_per_day > 1:
            returns[1:] += 0.00002 * imbalance[:-1]

        day_data = {"timestamp": timestamps}
        for index, ticker in enumerate(tickers):
            day_data[f"ask_{ticker}"] = ask[:, index]
            day_data[f"bid_{ticker}"] = bid[:, index]
            day_data[f"ask_size_{ticker}"] = ask_size[:, index]
            day_data[f"bid_size_{ticker}"] = bid_size[:, index]

        rows.append(pd.DataFrame(day_data))
        previous_close = mid[-1]

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    raw: pd.DataFrame,
    cfg: Config,
    output_dir: str | Path,
    save_features: bool = False,
) -> Dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = infer_tickers(
        columns=raw.columns,
        ask_prefix=cfg.ask_prefix,
        bid_prefix=cfg.bid_prefix,
        ask_size_prefix=cfg.ask_size_prefix,
        bid_size_prefix=cfg.bid_size_prefix,
    )

    print(f"Actions détectées: {len(tickers)}")
    print(", ".join(tickers[:10]) + (" ..." if len(tickers) > 10 else ""))

    quotes = resample_flat_quotes(raw, tickers, cfg)
    ask, bid, ask_size, bid_size = extract_panels(
        quotes,
        tickers,
        cfg,
    )

    if ask_size is None or bid_size is None:
        print("Tailles bid/ask absentes ou inutilisables: features de profondeur désactivées.")
    else:
        print("Tailles bid/ask détectées: features de profondeur activées.")

    feature_frames, target, _, selected_index = create_feature_frames(
        ask=ask,
        bid=bid,
        ask_size=ask_size,
        bid_size=bid_size,
        cfg=cfg,
    )

    table, feature_columns = build_learning_table(
        feature_frames=feature_frames,
        target_future_return=target,
        selected_index=selected_index,
        tickers=tickers,
        cfg=cfg,
    )

    if table.empty:
        raise ValueError(
            "Aucune observation exploitable. Vérifier l'horizon, les horaires "
            "et le nombre de journées."
        )

    train, validation, test = split_by_trading_days(
        table,
        cfg.train_ratio,
        cfg.validation_ratio,
    )

    print(
        f"Lignes: train={len(train):,}, validation={len(validation):,}, "
        f"test={len(test):,}"
    )
    print(
        f"Groupes: train={train.index.get_level_values('timestamp').nunique():,}, "
        f"validation={validation.index.get_level_values('timestamp').nunique():,}, "
        f"test={test.index.get_level_values('timestamp').nunique():,}"
    )
    print(f"Features utilisées: {len(feature_columns)}")

    model = train_ranker(
        train=train,
        validation=validation,
        feature_columns=feature_columns,
        cfg=cfg,
    )

    predictions = predict_ranker(
        model=model,
        frame=test,
        feature_columns=feature_columns,
    )

    metrics = evaluate_predictions(
        prediction_frame=predictions,
        top_k=cfg.top_k,
    )

    print("\nMétriques test")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.8f}")
        else:
            print(f"{key}: {value}")

    model.booster_.save_model(str(output_dir / "lambdarank_model.txt"))

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain_importance": model.booster_.feature_importance(
                importance_type="gain"
            ),
            "split_importance": model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("gain_importance", ascending=False)

    importance.to_csv(
        output_dir / "feature_importance.csv",
        index=False,
    )
    predictions.to_csv(
        output_dir / "test_rankings.csv.gz",
        index=False,
        compression="gzip",
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(asdict(cfg), file, indent=2)

    with open(output_dir / "tickers.json", "w", encoding="utf-8") as file:
        json.dump(tickers, file, indent=2)

    if save_features:
        try:
            table.reset_index().to_parquet(
                output_dir / "learning_table.parquet",
                index=False,
            )
        except ImportError:
            warnings.warn(
                "pyarrow n'est pas installé; learning_table.parquet non sauvegardé."
            )

    print(f"\nRésultats sauvegardés dans: {output_dir.resolve()}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ranking cross-sectionnel bid/ask avec LambdaRank."
    )

    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs_ranker")
    parser.add_argument("--timestamp-col", type=str, default="timestamp")

    parser.add_argument("--ask-prefix", type=str, default="ask_")
    parser.add_argument("--bid-prefix", type=str, default="bid_")
    parser.add_argument("--ask-size-prefix", type=str, default="ask_size_")
    parser.add_argument("--bid-size-prefix", type=str, default="bid_size_")

    parser.add_argument("--base-frequency", type=str, default="10s")
    parser.add_argument("--ranking-frequency", type=str, default="1min")
    parser.add_argument("--horizon", type=str, default="10min")

    parser.add_argument("--market-open", type=str, default="09:00:00")
    parser.add_argument("--market-close", type=str, default="17:30:00")

    parser.add_argument("--corr-lookback-days", type=int, default=20)
    parser.add_argument("--corr-min-days", type=int, default=5)
    parser.add_argument("--corr-top-k", type=int, default=5)
    parser.add_argument("--corr-threshold", type=float, default=0.10)
    parser.add_argument("--corr-shrinkage", type=float, default=0.10)

    parser.add_argument("--relevance-bins", type=int, default=5)
    parser.add_argument(
        "--target-mode",
        choices=["raw", "volatility_adjusted"],
        default="raw",
    )
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)

    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--feature-fraction", type=float, default=0.80)
    parser.add_argument("--bagging-fraction", type=float, default=0.80)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--reg-alpha", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--demo", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config(
        timestamp_col=args.timestamp_col,
        ask_prefix=args.ask_prefix,
        bid_prefix=args.bid_prefix,
        ask_size_prefix=args.ask_size_prefix,
        bid_size_prefix=args.bid_size_prefix,
        base_frequency=args.base_frequency,
        ranking_frequency=args.ranking_frequency,
        horizon=args.horizon,
        market_open=args.market_open,
        market_close=args.market_close,
        corr_lookback_days=args.corr_lookback_days,
        corr_min_days=args.corr_min_days,
        corr_top_k=args.corr_top_k,
        corr_threshold=args.corr_threshold,
        corr_shrinkage=args.corr_shrinkage,
        relevance_bins=args.relevance_bins,
        target_mode=args.target_mode,
        top_k=args.top_k,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    if args.demo:
        raw = generate_demo_quotes(
            n_days=18,
            n_stocks=40,
            bars_per_day=240,
            base_frequency=cfg.base_frequency,
            seed=cfg.seed,
        )

        # Paramètres plus rapides pour une démonstration.
        cfg.corr_lookback_days = min(cfg.corr_lookback_days, 10)
        cfg.corr_min_days = min(cfg.corr_min_days, 3)
        cfg.early_stopping_rounds = min(cfg.early_stopping_rounds, 30)
    else:
        if not args.input:
            parser.error("--input est requis sauf avec --demo.")
        raw = read_table(args.input)

    run_pipeline(
        raw=raw,
        cfg=cfg,
        output_dir=args.output_dir,
        save_features=args.save_features,
    )


if __name__ == "__main__":
    main()
