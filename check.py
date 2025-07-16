import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
import time

# Suppress convergence warnings from MLPRegressor for this demonstration
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Utility Functions ---

def generate_wind_speed_data(n_points=1000):
    """Generates a synthetic wind speed time series."""
    time = np.linspace(0, 100, n_points)
    # A base sinusoidal pattern representing daily cycles
    speed = 5 * np.sin(time * 2 * np.pi / 24) + 10
    # Add some higher frequency fluctuations
    speed += 2 * np.sin(time * 2 * np.pi / 6)
    # Add some non-linear trend
    speed += (time / 20) ** 1.5
    # Add random noise
    speed += np.random.normal(0, 1.5, n_points)
    speed[speed < 0] = 0 # Wind speed cannot be negative
    return speed

def create_dataset(sequence, look_back=1):
    """Transforms a time series into a supervised learning dataset."""
    X, Y = [], []
    for i in range(len(sequence) - look_back):
        a = sequence[i:(i + look_back)]
        X.append(a)
        Y.append(sequence[i + look_back])
    return np.array(X), np.array(Y)

# --- EMD (Empirical Mode Decomposition) Implementation ---
# This is a core component for FEEMD.

def get_extrema(signal):
    """Finds the local minima and maxima of a signal."""
    max_indices = (np.diff(np.sign(np.diff(signal))) < 0).nonzero()[0] + 1
    min_indices = (np.diff(np.sign(np.diff(signal))) > 0).nonzero()[0] + 1
    return min_indices, max_indices

def sift(signal):
    """Performs the sifting process to extract one Intrinsic Mode Function (IMF)."""
    h = signal.copy()
    for _ in range(10): # Sifting iterations
        min_idx, max_idx = get_extrema(h)
        if len(min_idx) < 2 or len(max_idx) < 2:
            break
        
        # Create envelopes using cubic splines
        upper_env = CubicSpline(max_idx, h[max_idx])(np.arange(len(h)))
        lower_env = CubicSpline(min_idx, h[min_idx])(np.arange(len(h)))
        
        mean_env = (upper_env + lower_env) / 2
        h_new = h - mean_env
        
        # Check stopping criterion (e.g., standard deviation)
        if np.sum((h - h_new)**2) / np.sum(h**2) < 0.2:
            break
        h = h_new
    return h

def emd(signal):
    """Performs Empirical Mode Decomposition."""
    imfs = []
    residual = signal.copy()
    while True:
        min_idx, max_idx = get_extrema(residual)
        if len(min_idx) < 2 or len(max_idx) < 2:
            break
        
        imf = sift(residual)
        imfs.append(imf)
        residual -= imf
        
        # Stop if residual is monotonic
        if (np.diff(residual) > 0).all() or (np.diff(residual) < 0).all():
            break
            
    imfs.append(residual)
    return imfs

# --- FEEMD (Fractal-based Ensemble EMD) Implementation ---

def box_counting_dimension(signal, n_boxes_start=2, n_boxes_end=256):
    """
    Calculates the Box-Counting Dimension of a 1D signal.
    The signal is normalized to fit in a unit square.
    """
    # Normalize signal to fit in a 1x1 box
    signal_normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-9)
    
    counts = []
    box_sizes = []

    n_steps = int(np.log2(n_boxes_end / n_boxes_start)) + 1
    for i in range(n_steps):
        n = n_boxes_start * (2**i)
        box_size = 1.0 / n
        
        grid = np.zeros(n)
        for j, point in enumerate(signal_normalized):
            box_index_y = int(point / box_size)
            if box_index_y >= n: box_index_y = n - 1
            grid[box_index_y] = 1
        
        counts.append(np.sum(grid))
        box_sizes.append(box_size)

    # Linear regression on log-log plot to find the slope
    coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    return -coeffs[0] # Dimension is the negative of the slope

def feemd(signal, noise_std=0.2, n_ensembles=100, box_dim_threshold=1.22):
    """
    Performs Fractal-based Ensemble EMD.
    Based on the paper's description, this method identifies and separates
    "abnormal" high-frequency components first.
    """
    print("Starting FEEMD...")
    abnormal_imfs = []
    s_current = signal.copy()
    
    for p in range(1, 10): # Limit to a max of 10 abnormal IMFs
        print(f"  Extracting potential abnormal IMF {p}...")
        # Step 1 & 2: CEEMD-like step to get the p-th component
        ensemble_imfs = []
        for i in range(n_ensembles):
            noise_p = np.random.normal(0, 1, len(signal)) * noise_std
            noise_n = -noise_p
            
            s_plus = s_current + noise_p
            s_minus = s_current + noise_n
            
            imf_p = sift(s_plus)
            imf_n = sift(s_minus)
            ensemble_imfs.append((imf_p + imf_n) / 2)
        
        # Step 3: Average to get the final IMF
        imf_p_final = np.mean(ensemble_imfs, axis=0)
        
        # Step 4: Check if it's an abnormal signal using box-counting dimension
        dim = box_counting_dimension(imf_p_final)
        print(f"    - Box-Counting Dimension: {dim:.4f}")
        
        if dim > box_dim_threshold:
            print(f"    - IMF {p} is abnormal. Separating it.")
            abnormal_imfs.append(imf_p_final)
            s_current = s_current - imf_p_final
        else:
            print(f"    - IMF {p} is normal. Stopping abnormal signal search.")
            break
            
    # Step 5: Perform standard EMD on the remaining signal
    print("  Performing final EMD on the remaining signal...")
    remaining_imfs = emd(s_current)
    
    all_components = abnormal_imfs + remaining_imfs
    print(f"FEEMD finished. Found {len(all_components)} total components.")
    return all_components

# --- Permutation Entropy (PE) Implementation ---

def permutation_entropy(signal, m=3, delay=1):
    """Calculates the Permutation Entropy of a signal."""
    n = len(signal)
    permutations = []
    for i in range(n - (m - 1) * delay):
        sorted_idx = np.argsort(signal[i : i + m * delay : delay])
        permutations.append(tuple(sorted_idx))
    
    counts = {}
    for p in permutations:
        counts[p] = counts.get(p, 0) + 1
        
    total_perms = len(permutations)
    probs = np.array(list(counts.values())) / total_perms
    
    pe = -np.sum(probs * np.log2(probs))
    return pe

# --- SSA-BP (Sparrow Search Algorithm for BP Network) ---

class SparrowSearchOptimizer:
    """
    Optimizes the initial weights of a BP Neural Network using SSA.
    """
    def __init__(self, obj_func, n_dim, pop_size=30, max_iter=50, lb=-1, ub=1,
                 discoverer_ratio=0.2, scout_ratio=0.1, safety_threshold=0.8):
        self.obj_func = obj_func
        self.n_dim = n_dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.full(n_dim, lb)
        self.ub = np.full(n_dim, ub)
        
        self.discoverer_num = int(pop_size * discoverer_ratio)
        self.scout_num = int(pop_size * scout_ratio)
        self.safety_threshold = safety_threshold
        
        self.positions = np.random.uniform(lb, ub, (pop_size, n_dim))
        self.fitness = np.full(pop_size, np.inf)
        
        self.global_best_pos = np.zeros(n_dim)
        self.global_best_fit = np.inf

    def optimize(self):
        print("Starting SSA optimization...")
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.obj_func(self.positions[i])
        
        self._update_global_best()

        for t in range(self.max_iter):
            sorted_indices = np.argsort(self.fitness)
            
            # Discoverer's update
            for i in range(self.discoverer_num):
                idx = sorted_indices[i]
                r2 = np.random.rand()
                if r2 < self.safety_threshold:
                    alpha = np.random.rand()
                    self.positions[idx] += np.random.normal(0, 1) * alpha
                else:
                    self.positions[idx] += np.random.normal(0, 1) * np.ones(self.n_dim)
            
            # Follower's update
            for i in range(self.discoverer_num, self.pop_size):
                idx = sorted_indices[i]
                if i > self.pop_size / 2:
                    self.positions[idx] = np.random.normal(0, 1) * np.exp((self.global_best_pos - self.positions[idx]) / (i**2))
                else:
                    # Find best discoverer
                    best_discoverer_pos = self.positions[sorted_indices[0]]
                    A = np.ones((1, self.n_dim)) # Simplified A matrix
                    A_plus = np.linalg.pinv(A)
                    self.positions[idx] = best_discoverer_pos + np.abs(self.positions[idx] - best_discoverer_pos) @ A_plus * np.ones(self.n_dim)

            # Scout's update
            scout_indices = np.random.choice(self.pop_size, self.scout_num, replace=False)
            for idx in scout_indices:
                if self.fitness[idx] > self.global_best_fit:
                    self.positions[idx] = self.global_best_pos + np.random.normal(0, 1, self.n_dim) * np.abs(self.positions[idx] - self.global_best_pos)
                else:
                    self.positions[idx] += np.random.uniform(-1, 1) * (np.abs(self.positions[idx] - self.global_best_pos) / (self.fitness[idx] - self.global_best_fit + 1e-9))

            # Boundary check
            self.positions = np.clip(self.positions, self.lb, self.ub)

            # Re-evaluate and update global best
            for i in range(self.pop_size):
                self.fitness[i] = self.obj_func(self.positions[i])
            self._update_global_best()
            
            if (t + 1) % 10 == 0:
                print(f"  SSA Iteration {t+1}/{self.max_iter}, Best Fitness: {self.global_best_fit:.6f}")

        print("SSA optimization finished.")
        return self.global_best_pos, self.global_best_fit

    def _update_global_best(self):
        min_fit_idx = np.argmin(self.fitness)
        if self.fitness[min_fit_idx] < self.global_best_fit:
            self.global_best_fit = self.fitness[min_fit_idx]
            self.global_best_pos = self.positions[min_fit_idx].copy()


# --- Main Execution ---
if __name__ == '__main__':
    # 1. Data Preparation
    print("--- 1. Data Preparation ---")
    look_back = 5 # Using 5 previous time steps to predict the next one
    wind_data = generate_wind_speed_data(n_points=864)
    train_size = 700
    train_data = wind_data[:train_size]
    test_data = wind_data[train_size:]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()

    # 2. FEEMD Decomposition
    print("\n--- 2. FEEMD Decomposition ---")
    components = feemd(scaled_train_data, noise_std=0.2, n_ensembles=50, box_dim_threshold=1.22)
    
    # 3. Permutation Entropy and Merging
    print("\n--- 3. Permutation Entropy & Merging ---")
    pe_values = [permutation_entropy(c, m=3, delay=1) for c in components]
    
    # Simple merging strategy: merge adjacent components if PE diff is small
    merge_threshold = 0.1
    merged_components = []
    current_merge = components[0]
    for i in range(1, len(components)):
        if abs(pe_values[i] - pe_values[i-1]) < merge_threshold:
            current_merge += components[i]
        else:
            merged_components.append(current_merge)
            current_merge = components[i]
    merged_components.append(current_merge)
    
    print(f"Original components: {len(components)}. Merged components: {len(merged_components)}")

    # 4. SSA-BP Prediction for each component
    print("\n--- 4. SSA-BP Prediction ---")
    all_predictions = []
    
    for i, nimf in enumerate(merged_components):
        print(f"\nTraining model for Merged Component (NIMF) {i+1}/{len(merged_components)}...")
        X_train, y_train = create_dataset(nimf, look_back)
        
        if len(X_train) == 0:
            print("  Skipping component with insufficient data.")
            # Predict zeros for this component
            test_input_len = len(test_data) - look_back
            all_predictions.append(np.zeros(test_input_len))
            continue

        # Define the BP network structure
        bp_model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1, warm_start=True, random_state=42)
        # Fit once to initialize weights
        bp_model.fit(X_train, y_train) 
        
        # Get weight dimensions for SSA
        coefs = bp_model.coefs_
        intercepts = bp_model.intercepts_
        n_dim = sum(c.size for c in coefs) + sum(i.size for i in intercepts)

        def objective_function(weights_flat):
            # Reshape flattened weights back to model's structure
            pointer = 0
            new_coefs = []
            for c in coefs:
                num_weights = c.size
                new_coefs.append(weights_flat[pointer : pointer + num_weights].reshape(c.shape))
                pointer += num_weights
            
            new_intercepts = []
            for inter in intercepts:
                num_biases = inter.size
                new_intercepts.append(weights_flat[pointer : pointer + num_biases].reshape(inter.shape))
                pointer += num_biases

            bp_model.coefs_ = new_coefs
            bp_model.intercepts_ = new_intercepts
            
            # Train for one epoch and get error
            bp_model.fit(X_train, y_train)
            y_pred = bp_model.predict(X_train)
            return mean_squared_error(y_train, y_pred)

        # Run SSA to find best initial weights
        ssa = SparrowSearchOptimizer(obj_func=objective_function, n_dim=n_dim, pop_size=20, max_iter=30, lb=-1, ub=1)
        best_initial_weights, _ = ssa.optimize()
        
        # Train final model with optimized initial weights
        print("  Training final BP model with optimized weights...")
        final_bp = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=200, warm_start=True, random_state=42)
        final_bp.fit(X_train, y_train) # Initialize
        
        # Set the optimized weights
        pointer = 0
        final_coefs = []
        for c in final_bp.coefs_:
            num_weights = c.size
            final_coefs.append(best_initial_weights[pointer : pointer + num_weights].reshape(c.shape))
            pointer += num_weights
        final_intercepts = []
        for inter in final_bp.intercepts_:
            num_biases = inter.size
            final_intercepts.append(best_initial_weights[pointer : pointer + num_biases].reshape(inter.shape))
            pointer += num_biases
        final_bp.coefs_ = final_coefs
        final_bp.intercepts_ = final_intercepts

        # Full training
        final_bp.fit(X_train, y_train)
        
        # Make predictions
        print("  Making predictions...")
        # We need to predict step-by-step on the test data
        current_input = scaled_train_data[-look_back:].tolist()
        nimf_predictions = []
        for _ in range(len(test_data)):
            pred = final_bp.predict(np.array(current_input).reshape(1, -1))
            nimf_predictions.append(pred[0])
            # This part is tricky. For a real scenario, we'd need predictions of ALL other components
            # to reconstruct the next input. For simplicity, we assume we can use the true scaled values
            # to form the next input window. This is a common simplification in such models.
            # A more rigorous approach would require a recursive multi-output model.
            if len(wind_data) > train_size + look_back + _:
                next_val = scaler.transform(wind_data[train_size+_:train_size+_+1].reshape(-1,1))[0,0]
                current_input.pop(0)
                current_input.append(next_val)

        all_predictions.append(np.array(nimf_predictions))

    # 5. Aggregate Predictions and Evaluate
    print("\n--- 5. Aggregation and Evaluation ---")
    # Sum the predictions from all component models
    final_prediction_scaled = np.sum(all_predictions, axis=0)
    
    # Inverse transform to get actual wind speed values
    final_prediction = scaler.inverse_transform(final_prediction_scaled.reshape(-1, 1)).flatten()
    
    # Align true test data for comparison
    true_values = test_data

    # Calculate metrics
    mae = mean_absolute_error(true_values, final_prediction)
    rmse = np.sqrt(mean_squared_error(true_values, final_prediction))
    mape = np.mean(np.abs((true_values - final_prediction) / (true_values + 1e-9))) * 100
    r2 = 1 - np.sum((true_values - final_prediction)**2) / np.sum((true_values - np.mean(true_values))**2)
    
    print("\n--- Evaluation Metrics ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R^2:  {r2:.4f}")

    # 6. Plotting Results
    print("\n--- 6. Plotting Results ---")
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Decomposition
    plt.subplot(3, 1, 1)
    for i, comp in enumerate(components):
        plt.plot(comp, label=f'IMF {i+1}')
    plt.title("FEEMD Components")
    plt.legend()
    
    # Plot 2: Final Prediction vs. Actual
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(wind_data)), wind_data, 'k-', label='Original Data')
    plt.plot(np.arange(train_size, len(wind_data)), true_values, 'b-', label='Actual Test Data')
    plt.plot(np.arange(train_size, len(wind_data)), final_prediction, 'r--', label='FEEMD-PE-SSA-BP Prediction')
    plt.axvline(x=train_size, color='gray', linestyle='--')
    plt.title("Wind Speed Prediction")
    plt.xlabel("Time Point")
    plt.ylabel("Wind Speed (m/s)")
    plt.legend()
    
    # Plot 3: Prediction Errors
    plt.subplot(3, 1, 3)
    errors = true_values - final_prediction
    plt.plot(np.arange(train_size, len(wind_data)), errors, 'g-', label='Prediction Error')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Prediction Error Over Time")
    plt.xlabel("Time Point")
    plt.ylabel("Error (m/s)")
    plt.legend()

    plt.tight_layout()
    plt.show()
