import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace import sarimax # SARIMAX can be used for general state space models

# --- 1. Simulate Data (Example) ---
np.random.seed(42)
n_obs = 200
X = np.random.rand(n_obs) * 10
time_index = pd.date_range(start='2020-01-01', periods=n_obs, freq='D')

# True time-varying parameters (e.g., random walk)
beta_0_true = np.zeros(n_obs)
beta_1_true = np.zeros(n_obs)
beta_0_true[0] = 5
beta_1_true[0] = 0.5
for t in range(1, n_obs):
    beta_0_true[t] = beta_0_true[t-1] + np.random.normal(0, 0.1) # drift in intercept
    beta_1_true[t] = beta_1_true[t-1] + np.random.normal(0, 0.05) # drift in slope

y = beta_0_true + beta_1_true * X + np.random.normal(0, 1, n_obs)

data = pd.DataFrame({'y': y, 'X': X}, index=time_index)

# --- 2. Define the State Space Model ---
# In statsmodels, you define the system matrices.
# y_t = Z_t * alpha_t + epsilon_t  (Observation equation)
# alpha_t = T_t * alpha_{t-1} + R_t * eta_t (State equation)

# For a time-varying regression, the states (alpha_t) are our coefficients [beta_0, beta_1]
# Z_t (design matrix) will be [1, X_t]
# T_t (transition matrix) will be identity if parameters follow a random walk
# R_t (selection matrix) will be identity if all states have their own disturbances

# Create the state space model
# The order argument (p,d,q) is not directly used for general state space,
# but the SARIMAX class serves as a flexible base.
# We'll set order=(0,0,0) and manually define the state space matrices.

class TimeVaryingRegression(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, exog):
        super(TimeVaryingRegression, self).__init__(endog, k_states=exog.shape[1],
                                                     k_posdef=exog.shape[1], # Number of state disturbances
                                                     initialization='approximate_diffuse') # Good for unknown initial states

        self.exog = exog

        # Observation equation: y_t = Z_t * alpha_t + epsilon_t
        # Z_t is [1, X_t]
        self.ssm.bind(endog=endog, exog=exog) # This line effectively sets Z_t

        # State equation: alpha_t = T_t * alpha_{t-1} + R_t * eta_t
        # T_t (transition matrix): For random walk, it's Identity
        self.ssm['transition'] = np.eye(self.k_states)

        # R_t (selection matrix for state disturbances): Identity if each state has own shock
        self.ssm['selection'] = np.eye(self.k_states)

        # Initialize parameter vector: [variance_epsilon, variance_beta0, variance_beta1]
        self._param_names = ['var.measurement'] + [f'var.state.{i}' for i in range(self.k_states)]
        self.parameters = np.array([1.0] * (1 + self.k_states)) # Initial guess for variances

    @property
    def param_names(self):
        return self._param_names

    @property
    def start_params(self):
        return np.array([1.0] * (1 + self.k_states))

    def update(self, params, transformed=True, **kwargs):
        params = super(TimeVaryingRegression, self).update(params, transformed, **kwargs)

        # Measurement variance (epsilon_t variance)
        self.ssm['obs_cov'] = params[0]

        # State covariance (eta_t variance - for random walk)
        # Assuming state disturbances are uncorrelated
        self.ssm['state_cov'] = np.diag(params[1:])

# Prepare exogenous variables (add intercept)
X_data = sm.add_constant(data['X'], prepend=False) # Add constant at the end for consistency with internal handling

# Instantiate and fit the model
model = TimeVaryingRegression(endog=data['y'], exog=X_data)
results = model.fit(disp=False) # disp=False to suppress optimization output

print(results.summary())

# Extract estimated time-varying parameters (smoothed states)
# `smoothed_state.mean` gives the estimated mean of the states at each time point
estimated_betas = results.smoothed_state.mean
estimated_beta0 = estimated_betas[:, 0]
estimated_beta1 = estimated_betas[:, 1]

# --- 3. Visualize Results ---
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(time_index, beta_0_true, label='True Beta_0 (Intercept)', linestyle='--')
plt.plot(time_index, estimated_beta0, label='Estimated Beta_0 (Intercept)')
plt.plot(time_index, beta_1_true, label='True Beta_1 (Slope)', linestyle='--')
plt.plot(time_index, estimated_beta1, label='Estimated Beta_1 (Slope)')
plt.title('Time-Varying Regression Coefficients (Kalman Filter)')
plt.xlabel('Date')
plt.ylabel('Coefficient Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# You can also get filtered states (predictions based on data up to that point)
# filtered_betas = results.filtered_state.mean
