# Well-structured Python code for a backtesting trading strategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Parameters
# ------------------------------
THRESHOLD_BPS = 5                 # Signal threshold to trigger a trade
POSITION_DURATION = 60           # Duration to hold position (in number of 10s intervals)
MAX_POSITIONS = 5                # Maximum number of simultaneous open positions
FEES_PER_TRADE = 0.0001          # Fee per trade (e.g., 1 basis point)

# ------------------------------
# Generate or Load Data
# ------------------------------
def generate_sample_data(n=6 * 60 * 24):  # One trading day sampled every 10s
    np.random.seed(42)
    timestamps = pd.date_range(start="2024-01-01", periods=n, freq="10S")
    predicted_bps = np.random.normal(loc=0.5, scale=2, size=n)
    realized_return = np.random.normal(loc=0.0, scale=1, size=n) / 10000
    return pd.DataFrame({
        "timestamp": timestamps,
        "predicted_bps": predicted_bps,
        "realized_return": realized_return
    }).set_index("timestamp")

# ------------------------------
# Strategy Logic
# ------------------------------
def apply_strategy(df):
    df["position"] = 0.0
    df["trade_open"] = False
    open_positions = []

    for i in range(len(df)):
        signal = df.iloc[i]["predicted_bps"]
        current_return = df.iloc[i]["realized_return"]

        # Clean up expired positions
        open_positions = [pos for pos in open_positions if i < pos["end_index"]]

        # Entry Condition
        if len(open_positions) < MAX_POSITIONS:
            if signal > THRESHOLD_BPS:
                direction = 1
            elif signal < -THRESHOLD_BPS:
                direction = -1
            else:
                direction = 0

            if direction != 0:
                end_index = min(i + POSITION_DURATION, len(df) - 1)
                open_positions.append({
                    "start_index": i,
                    "end_index": end_index,
                    "direction": direction
                })
                df.iloc[i, df.columns.get_loc("trade_open")] = True

        # Apply PnL from open positions
        pnl = sum(pos["direction"] * current_return for pos in open_positions)
        df.iloc[i, df.columns.get_loc("position")] = pnl

    return df

# ------------------------------
# Metrics Calculation
# ------------------------------
def calculate_metrics(df):
    df["gross_return"] = df["position"]
    df["net_return"] = df["gross_return"] - df["trade_open"].astype(int) * FEES_PER_TRADE
    df["cumulative_return"] = (1 + df["net_return"]).cumprod() - 1

    # Daily metrics
    daily_returns = df["net_return"].resample("1D").sum()

    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else np.nan
    sortino = daily_returns.mean() / daily_returns[daily_returns < 0].std() * np.sqrt(252) if daily_returns[daily_returns < 0].std() != 0 else np.nan

    rolling_max = df["cumulative_return"].cummax()
    drawdown = df["cumulative_return"] - rolling_max
    max_drawdown = drawdown.min()

    hit_rate = (df["net_return"] > 0).sum() / (df["net_return"] != 0).sum()
    total_trades = df["trade_open"].sum()
    total_fees = total_trades * FEES_PER_TRADE
    avg_return_per_trade = df["net_return"][df["trade_open"]].mean()

    return {
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Cumulative Return": df["cumulative_return"].iloc[-1],
        "Max Drawdown": max_drawdown,
        "Hit Rate": hit_rate,
        "Total Trades": total_trades,
        "Total Fees Paid": total_fees,
        "Average Return per Trade": avg_return_per_trade,
        "Daily Returns Std": daily_returns.std()
    }

# ------------------------------
# Plotting
# ------------------------------
def plot_performance(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["cumulative_return"])
    plt.title("Cumulative Net Return")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main Execution
# ------------------------------
def main():
    df = generate_sample_data()
    df = apply_strategy(df)
    metrics = calculate_metrics(df)
    plot_performance(df)
    return metrics

# Run the strategy
results = main()
results
