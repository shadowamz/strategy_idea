
# run_demo.py
# Tiny demo for midfreq_alpha_toolkit: generates a SMALL synthetic dataset and trains a toy model.
import pandas as pd
import numpy as np
from midfreq_alpha_toolkit import (
    trading_calendar_index, simulate_prices_gbm_jumps, train_simple_alpha, resample_last_price
)

if __name__ == "__main__":
    # --- Config (SMALL so it runs fast in this environment) ---
    tickers = [f"S{i:02d}" for i in range(1, 9)]   # 8 names (change to 40 for your real case)
    start_date = "2024-01-02"
    end_date   = "2024-01-10"                      # ~1 week (change to 2 years)
    freq = "1S"                                    # per-second native
    tz = "Europe/Paris"

    # Build trading calendar (09:00–17:30 local time)
    idx = trading_calendar_index(start_date, end_date, tz=tz, session_start="09:00:00", session_end="17:30:00", freq=freq)

    # Simulate price-only last prints
    df = simulate_prices_gbm_jumps(idx, tickers, seed=1)
    print("Simulated tidy price DataFrame:", df.shape, "\n", df.head())

    # Train a simple alpha on 10-second bars with 45-min horizon
    artifacts = train_simple_alpha(df, bar_rule="10S", horizon_sec=45*60)
    print("\nArtifacts keys:", list(artifacts.keys()))
    print("\nPred probabilities head:\n", artifacts["pred_proba"].head())

    # Show how to get OHLC bars quickly
    ohlc_10s = resample_last_price(df, rule="10S")
    print("\n10s OHLC sample:\n", ohlc_10s.head())
