"""Monthly rolling Ridge aggregation and daily neutral portfolio construction.

The primary inference method, ``weights_for_date``, returns an assets-by-one
DataFrame. ``walk_forward`` returns dates-by-assets for notebook compatibility.
Inputs follow this repository: alphas indexed by (date, asset), and a wide
DataFrame of daily forward arithmetic asset returns. See ``doc.md`` for timing,
initialization, and evaluation conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge

__all__ = ["AggregatorConfig", "RidgeAlphaAggregator"]


@dataclass(frozen=True)
class AggregatorConfig:
    """Selected research settings; windows and delays count observed dates."""

    ridge_lambda: float = 0.1
    training_window_dates: int = 504
    training_gap_dates: int = 1
    beta_window_dates: int = 252
    beta_return_lag_dates: int = 2
    smoothing_halflife_dates: float = 1.0
    target_gross_exposure: float = 1.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name, minimum in (
            ("training_window_dates", 2),
            ("training_gap_dates", 0),
            ("beta_window_dates", 2),
            ("beta_return_lag_dates", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in (
            "ridge_lambda", "smoothing_halflife_dates", "target_gross_exposure", "epsilon"
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number.")


class RidgeAlphaAggregator:
    """Combine normalized alphas, smooth target weights, and neutralize beta.

    ``fit`` only changes the fitted Ridge model; ``predict`` is stateless.
    ``weights_for_date`` advances a separate portfolio smoothing state and
    should be called once per allocation date in chronological order. Monthly
    refits preserve that state. Instances are intended for one sequential
    portfolio stream, not concurrent inference from multiple callers.
    """

    def __init__(self, config: AggregatorConfig | None = None) -> None:
        if config is not None and not isinstance(config, AggregatorConfig):
            raise TypeError("config must be an AggregatorConfig.")
        self.config = config if config is not None else AggregatorConfig()
        self.reset()

    def reset(self) -> None:
        """Clear the fitted model, calibration history, and portfolio state."""
        self._model: Ridge | None = None
        self._assets: pd.Index | None = None
        self._features: pd.Index | None = None
        self._training_dates: pd.DatetimeIndex | None = None
        self._calibration_date: pd.Timestamp | None = None
        self._calibrations: list[dict] = []
        self.reset_portfolio_state()

    def reset_portfolio_state(self) -> None:
        """Restart smoothing at the next target without discarding the model."""
        self._smoothed_target: np.ndarray | None = None
        self._last_weight_date: pd.Timestamp | None = None

    @property
    def coef_(self) -> pd.Series:
        """A labelled copy of the currently fitted alpha coefficients."""
        self._require_fit()
        return pd.Series(self._model.coef_.copy(), index=self._features, name="coefficient")

    @property
    def training_dates_(self) -> pd.DatetimeIndex:
        self._require_fit()
        return self._training_dates.copy()

    @property
    def calibration_date_(self) -> pd.Timestamp:
        self._require_fit()
        return self._calibration_date

    @property
    def calibration_history_(self) -> pd.DataFrame:
        """Audit of successful fits, including the exact training cutoffs."""
        columns = [
            "calibration_date", "training_start", "training_end",
            "training_dates", "observations",
        ]
        return pd.DataFrame(self._calibrations, columns=columns).set_index("calibration_date")

    def fit(
        self, alphas: pd.DataFrame, asset_returns: pd.DataFrame, *, as_of: object
    ) -> RidgeAlphaAggregator:
        """Fit on the latest eligible rolling window before ``as_of``.

        Full historical panels may be supplied, including later rows. Only the
        chosen training window is used. The default gap excludes the immediately
        preceding return row, so fitting at date position p uses p-505:p-1.
        Targets are each stock's return minus its same-date universe mean.
        Input alphas are already normalized and are not standardized again.
        """
        self._validate_panel(alphas)
        returns = self._validate_returns(asset_returns)
        date = self._timestamp(as_of)
        if self._calibration_date is not None and date < self._calibration_date:
            raise ValueError("Calibration dates must not move backwards; reset first.")
        if self._last_weight_date is not None and date <= self._last_weight_date:
            raise ValueError("Fit before producing weights for the calibration date.")

        assets = returns.columns.sort_values() if self._assets is None else self._assets
        features = alphas.columns.copy() if self._features is None else self._features
        self._check_labels(returns.columns, assets, "asset universe")
        self._check_labels(alphas.columns, features, "alpha features")

        history = self._history_window(
            returns, date, self.config.training_window_dates,
            self.config.training_gap_dates, assets,
        )
        training_dates = history.index
        x_train, _ = self._aligned_panel(alphas, training_dates, assets, features)
        outcomes = history.to_numpy(dtype=float)
        y_train = (outcomes - outcomes.mean(axis=1, keepdims=True)).reshape(-1)

        model = Ridge(
            alpha=len(x_train) * self.config.ridge_lambda,
            fit_intercept=False,
            solver="cholesky",
        )
        model.fit(x_train, y_train)

        # Commit only after all validation and fitting have succeeded.
        self._model = model
        self._assets = assets.copy()
        self._features = features.copy()
        self._training_dates = training_dates.copy()
        self._calibration_date = date
        self._calibrations.append({
            "calibration_date": date,
            "training_start": training_dates[0],
            "training_end": training_dates[-1],
            "training_dates": len(training_dates),
            "observations": len(x_train),
        })
        return self

    def predict(self, alphas: pd.DataFrame) -> pd.Series:
        """Return raw Ridge forecasts in daily relative-return units.

        This method neither refits nor advances smoothing. It uses the current
        fitted model for every supplied date; use ``walk_forward`` to schedule
        monthly refitting automatically. Row and feature order may differ from
        training: labels are aligned and predictions retain the input row order.
        """
        self._require_fit()
        self._validate_panel(alphas)
        dates = alphas.index.get_level_values("date").unique().sort_values()
        if dates[0] < self._calibration_date:
            raise ValueError("Prediction dates must be on or after the calibration date.")
        self._check_labels(alphas.columns, self._features, "alpha features")
        values, index = self._aligned_panel(alphas, dates, self._assets, self._features)
        predictions = self._model.predict(values)
        if not np.isfinite(predictions).all():
            raise ValueError("Ridge produced non-finite predictions.")
        return pd.Series(predictions, index=index, name="Ridge").reindex(alphas.index)

    def estimate_betas(self, asset_returns: pd.DataFrame, *, as_of: object) -> pd.Series:
        """Estimate betas to the equal-weight universe return using past rows.

        With the default lag, the window ends at p-2 for allocation position p.
        The sample covariance/sample variance ratio uses 252 observations.
        """
        self._require_fit()
        returns = self._validate_returns(asset_returns)
        self._check_labels(returns.columns, self._assets, "asset universe")
        history = self._history_window(
            returns, self._timestamp(as_of), self.config.beta_window_dates,
            self.config.beta_return_lag_dates - 1, self._assets,
        )
        values = history.to_numpy(dtype=float)
        market = values.mean(axis=1)
        centered_market = market - market.mean()
        market_ss = float(centered_market @ centered_market)
        if market_ss / (len(history) - 1) <= self.config.epsilon:
            raise ValueError("Historical market-proxy variance is too small to estimate beta.")
        centered = values - values.mean(axis=0)
        betas = (centered.T @ centered_market) / market_ss
        if not np.isfinite(betas).all():
            raise ValueError("Beta estimation produced non-finite values.")
        return pd.Series(betas, index=self._assets, name="estimated_beta")

    def weights_for_date(
        self, alphas: pd.DataFrame, asset_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Return final N-by-1 target weights for a single out-of-sample date.

        ``alphas`` must contain exactly that date and every fitted asset. Supply
        return history for beta estimation; current and future return values are
        ignored. Fit first in each calendar month. Calls advance smoothing and
        must have strictly increasing dates; call once per allocation date.
        """
        self._require_fit()
        self._validate_panel(alphas)
        dates = alphas.index.get_level_values("date").unique()
        if len(dates) != 1:
            raise ValueError("weights_for_date requires exactly one alpha date.")
        date = dates[0]
        if self._last_weight_date is not None and date <= self._last_weight_date:
            raise ValueError("Weight dates must be strictly increasing; reset to replay.")
        if (date.year, date.month) != (
            self._calibration_date.year, self._calibration_date.month
        ):
            raise ValueError("Refit the model before allocating in a new calendar month.")

        scores = self.predict(alphas).xs(date, level="date").reindex(self._assets).to_numpy()
        betas = self.estimate_betas(asset_returns, as_of=date).to_numpy()
        centered_scores = scores - scores.mean()
        score_gross = float(np.abs(centered_scores).sum())
        base_target = (
            centered_scores / score_gross
            if score_gross > self.config.epsilon else np.zeros(len(scores))
        )
        if self._smoothed_target is None:
            smoothed = base_target.copy()
        else:
            decay = np.exp2(-1.0 / self.config.smoothing_halflife_dates)
            smoothed = decay * self._smoothed_target + (1.0 - decay) * base_target

        residual = smoothed - smoothed.mean()
        centered_betas = betas - betas.mean()
        beta_ss = float(centered_betas @ centered_betas)
        if beta_ss > self.config.epsilon:
            residual = residual - centered_betas * (residual @ centered_betas) / beta_ss
        residual = residual - residual.mean()
        gross = float(np.abs(residual).sum())
        weights = (
            residual * (self.config.target_gross_exposure / gross)
            if gross > self.config.epsilon else np.zeros(len(residual))
        )
        if not np.isfinite(weights).all():
            raise ValueError("Portfolio construction produced non-finite weights.")

        # EWM state is the pre-projection target, not the final rescaled weights.
        self._smoothed_target = smoothed.copy()
        self._last_weight_date = date
        return pd.DataFrame({"weight": weights}, index=self._assets.rename("asset"))

    def walk_forward(
        self,
        alphas: pd.DataFrame,
        asset_returns: pd.DataFrame,
        *,
        start_date: object | None = None,
        end_date: object | None = None,
    ) -> pd.DataFrame:
        """Run a fresh monthly rolling calibration and daily allocation stream.

        Returns a dates-by-assets DataFrame, matching the research notebook.
        The optional assets-by-dates representation is ``result.T``. Each call
        resets the model and smoothing state. The first date initializes EWM;
        later monthly refits retain it. Full input history may contain future
        observations, but fitting and beta estimation select past windows only.
        """
        self._validate_panel(alphas)
        returns = self._validate_returns(asset_returns)
        warmup = max(
            self.config.training_window_dates + self.config.training_gap_dates,
            self.config.beta_window_dates + self.config.beta_return_lag_dates - 1,
        )
        if len(returns) <= warmup:
            raise ValueError("Insufficient history for any walk-forward allocation.")
        start = returns.index[warmup] if start_date is None else self._timestamp(start_date)
        end = returns.index[-1] if end_date is None else self._timestamp(end_date)
        if start > end:
            raise ValueError("start_date must not be after end_date.")
        if returns.index.searchsorted(start) < warmup:
            raise ValueError("start_date does not leave sufficient training and beta history.")
        dates = returns.index[(returns.index >= start) & (returns.index <= end)]
        if len(dates) == 0:
            raise ValueError("No return dates fall in the requested allocation period.")
        if not dates.isin(alphas.index.get_level_values("date")).all():
            raise ValueError("Alpha data is missing an allocation date.")

        self.reset()
        targets = np.empty((len(dates), returns.shape[1]))
        for i, date in enumerate(dates):
            if self._calibration_date is None or (date.year, date.month) != (
                self._calibration_date.year, self._calibration_date.month
            ):
                self.fit(alphas, returns, as_of=date)
            daily_alphas = alphas.loc[[date]]
            targets[i] = self.weights_for_date(daily_alphas, returns)["weight"].to_numpy()
        return pd.DataFrame(targets, index=dates.rename("date"), columns=self._assets.rename("asset"))

    def _require_fit(self) -> None:
        if self._model is None:
            raise NotFittedError("Call fit before requesting predictions or weights.")

    @staticmethod
    def _timestamp(value: object) -> pd.Timestamp:
        date = pd.Timestamp(value)
        if pd.isna(date) or date.tz is not None:
            raise ValueError("Dates must be finite, timezone-naive timestamps.")
        return date

    @staticmethod
    def _check_labels(actual: pd.Index, expected: pd.Index, description: str) -> None:
        if len(actual) != len(expected) or not actual.isin(expected).all():
            raise ValueError(f"The {description} does not match the fitted schema.")

    @staticmethod
    def _validate_panel(alphas: pd.DataFrame) -> None:
        if not isinstance(alphas, pd.DataFrame) or alphas.empty:
            raise ValueError("alphas must be a nonempty DataFrame.")
        if not isinstance(alphas.index, pd.MultiIndex) or list(alphas.index.names) != ["date", "asset"]:
            raise ValueError("alphas must have a MultiIndex named (date, asset).")
        dates = alphas.index.get_level_values("date")
        if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or dates.tz is not None:
            raise ValueError("Alpha dates must be a finite, timezone-naive DatetimeIndex.")
        if not alphas.index.is_unique or alphas.index.get_level_values("asset").hasnans:
            raise ValueError("Alpha (date, asset) rows must be unique and have valid asset labels.")
        if not alphas.columns.is_unique or alphas.columns.hasnans:
            raise ValueError("Alpha feature names must be unique and non-null.")

    @staticmethod
    def _validate_returns(asset_returns: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
            raise ValueError("asset_returns must be a nonempty wide DataFrame.")
        dates = asset_returns.index
        if (
            not isinstance(dates, pd.DatetimeIndex) or not dates.is_unique
            or dates.hasnans or dates.tz is not None
        ):
            raise ValueError("Return dates must be unique, finite, timezone-naive timestamps.")
        if not asset_returns.columns.is_unique or asset_returns.columns.hasnans:
            raise ValueError("Return asset columns must be unique and non-null.")
        return asset_returns if dates.is_monotonic_increasing else asset_returns.sort_index()

    @staticmethod
    def _history_window(
        returns: pd.DataFrame, date: pd.Timestamp, count: int,
        excluded_prior_dates: int, assets: pd.Index,
    ) -> pd.DataFrame:
        stop = int(returns.index.searchsorted(date, side="left")) - excluded_prior_dates
        start = stop - count
        if start < 0:
            raise ValueError(f"Insufficient historical observations before {date.date()}.")
        history = returns.iloc[start:stop].reindex(columns=assets)
        if len(history) != count or not np.isfinite(history.to_numpy(dtype=float)).all():
            raise ValueError("The eligible return window contains missing or non-finite values.")
        return history

    @staticmethod
    def _aligned_panel(
        alphas: pd.DataFrame, dates: pd.DatetimeIndex,
        assets: pd.Index, features: pd.Index,
    ) -> tuple[np.ndarray, pd.MultiIndex]:
        expected = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        selected = alphas.loc[alphas.index.get_level_values("date").isin(dates)]
        if len(selected) != len(expected) or not selected.index.isin(expected).all():
            raise ValueError("Each selected alpha date must contain every fitted asset exactly once.")
        values = selected.reindex(index=expected, columns=features).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("The selected alpha data contains missing or non-finite values.")
        return values, expected
