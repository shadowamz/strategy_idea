import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- 1. Simulate Data (Example) ---
np.random.seed(42)
n_obs = 300
time_index = pd.date_range(start='2010-01-01', periods=n_obs, freq='M')
X = np.random.rand(n_obs) * 10

# Simulate a changing relationship
beta_0_true = np.linspace(5, 10, n_obs) # gradually increasing intercept
beta_1_true = np.concatenate([np.linspace(0.5, -0.5, n_obs // 2),
                              np.linspace(-0.5, 1.5, n_obs // 2)]) # starts positive, goes negative, then positive

y = beta_0_true + beta_1_true * X + np.random.normal(0, 1.5, n_obs)
data = pd.DataFrame({'y': y, 'X': X}, index=time_index)

# --- 2. Perform Rolling Regression ---
window_size = 60 # e.g., 60 months (5 years)
rolling_betas = []
rolling_dates = []

for i in range(n_obs - window_size + 1):
    window_data = data.iloc[i : i + window_size]
    X_window = sm.add_constant(window_data['X'])
    y_window = window_data['y']

    try:
        model = sm.OLS(y_window, X_window).fit()
        rolling_betas.append(model.params)
        rolling_dates.append(window_data.index[-1]) # Store the end date of the window
    except ValueError: # Handle cases where regression might fail (e.g., singular matrix)
        rolling_betas.append([np.nan, np.nan]) # Append NaNs if model fails
        rolling_dates.append(window_data.index[-1])


rolling_betas_df = pd.DataFrame(rolling_betas, index=rolling_dates, columns=['const', 'X'])

# --- 3. Visualize Results ---
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(time_index, beta_0_true, label='True Intercept', linestyle='--')
plt.plot(rolling_betas_df.index, rolling_betas_df['const'], label=f'Rolling Intercept (Window={window_size})')
plt.title('Time-Varying Intercept (Rolling Regression)')
plt.xlabel('Date')
plt.ylabel('Intercept')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_index, beta_1_true, label='True Slope', linestyle='--')
plt.plot(rolling_betas_df.index, rolling_betas_df['X'], label=f'Rolling Slope (Window={window_size})')
plt.title('Time-Varying Slope (Rolling Regression)')
plt.xlabel('Date')
plt.ylabel('Slope')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
