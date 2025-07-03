import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# --- Configuration Parameters ---
# Data parameters
START_DATE_STR = "2022-01-01"
END_DATE_STR = "2024-12-31" # 3 full years for synthetic data
NUM_COLUMNS = 30
FREQ_SECONDS = 10 # 10-second frequency
COLUMN_LABEL_NAME = 'col_0' # The column to sum for the output

# Time series window parameters
INPUT_SEQ_LEN_SECONDS = 3600 # 1 hour
INPUT_SEQ_LEN_SAMPLES = INPUT_SEQ_LEN_SECONDS // FREQ_SECONDS # 360 samples
OUTPUT_PRED_START_OFFSET_SECONDS = 0 # Offset after input window (0 for immediate, 10 for next sample, etc.)
OUTPUT_PRED_LEN_SAMPLES = 61 # Number of samples to sum (from 361 to 421 after input)

# DataLoader parameters
BATCH_SIZE = 24 # One day's worth of hourly samples
SHUFFLE_DAYS = True # Shuffle the order of days processed
BUFFER_DAYS = 5 # How many days to pre-load/buffer in the generator for shuffling

# Model parameters
NUM_EXPERTS = 4
EXPERT_HIDDEN_DIM = 512
GATING_HIDDEN_DIM = 256

# Training parameters
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 # Adjust based on your needs
LOAD_BALANCING_COEFF = 0.01 # Coefficient for the load balancing loss

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Synthetic Data Generation (daily chunks for generator) ---
# This function simulates reading or generating one day of data at a time.
def generate_synthetic_day_data(
    current_date,
    num_columns,
    freq_seconds,
    noise_level=0.1,
    seasonal_amp=0.5,
    daily_amp=0.2,
    base_trend_val=0.0 # To simulate a continuous trend across days
):
    """
    Generates synthetic time series data for a single day.
    """
    start_of_day = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1) - timedelta(seconds=freq_seconds)
    date_range = pd.date_range(start=start_of_day, end=end_of_day, freq=f'{freq_seconds}S')
    num_samples_day = len(date_range)

    data = np.zeros((num_samples_day, num_columns))

    time_in_hours_today = np.array([(d - start_of_day).total_seconds() / 3600 for d in date_range])
    # Total time since start for overall trend calculation
    total_time_in_days = (current_date - datetime.strptime(START_DATE_STR, "%Y-%m-%d")).total_seconds() / (3600 * 24)

    daily_cycle = np.sin(2 * np.pi * (time_in_hours_today % 24) / 24) * daily_amp
    # Simulating weekly/yearly seasonality that changes slowly
    global_seasonal_component = np.sin(2 * np.pi * total_time_in_days / 7) * seasonal_amp # Weekly
    global_seasonal_component += np.sin(2 * np.pi * total_time_in_days / 365.25) * (seasonal_amp / 2) # Yearly

    for i in range(num_columns):
        # Base trend for each column, continues from previous days
        trend = np.linspace(base_trend_val, base_trend_val + 0.01 * (1 + 0.1 * i), num_samples_day)
        random_walk = np.cumsum(np.random.normal(0, noise_level / 5, num_samples_day))
        data[:, i] = trend + daily_cycle + global_seasonal_component + random_walk + np.random.normal(0, noise_level, num_samples_day)

    df_day = pd.DataFrame(data, index=date_range, columns=[f'col_{j}' for j in range(num_columns)])
    return df_day, data[-1, 0] # Return the last value of the first column to carry over trend

# --- 2. PyTorch IterableDataset for Generator-based Data Loading ---

class TimeSeriesDataGenerator(IterableDataset):
    def __init__(self, start_date_str, end_date_str, num_columns, freq_seconds,
                 input_seq_len_samples, output_pred_len_samples, column_label_name,
                 shuffle_days=False, buffer_days=1):
        self.start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        self.num_columns = num_columns
        self.freq_seconds = freq_seconds
        self.input_seq_len_samples = input_seq_len_samples
        self.output_pred_len_samples = output_pred_len_samples
        self.column_label_idx = None # Will be set once data is available
        self.column_label_name = column_label_name

        self.shuffle_days = shuffle_days
        self.buffer_days = max(1, buffer_days) # Ensure at least 1 day buffer

        # Total number of 10-second samples in a day
        self.samples_per_day = (24 * 3600) // self.freq_seconds
        # Number of hourly inputs per day
        self.hourly_inputs_per_day = 24

        self.total_days = (self.end_date - self.start_date).days + 1
        print(f"Generator will process {self.total_days} days of data.")

    def _get_daily_data(self, current_date, base_trend_val=0.0):
        # In a real scenario, this would load data from disk (e.g., HDF5, Parquet, CSV)
        # For synthetic data, we generate it here
        df_day, last_val_col0 = generate_synthetic_day_data(
            current_date, self.num_columns, self.freq_seconds,
            base_trend_val=base_trend_val
        )
        if self.column_label_idx is None:
            self.column_label_idx = df_day.columns.get_loc(self.column_label_name)
        return df_day, last_val_col0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Each worker processes a subset of days
        all_dates = [self.start_date + timedelta(days=i) for i in range(self.total_days)]
        
        # If shuffling days, shuffle the global list of dates once per process
        if self.shuffle_days:
            random.seed(worker_id) # Ensure different shuffling for different workers if any
            random.shuffle(all_dates)

        # Divide dates among workers
        worker_dates = all_dates[worker_id::num_workers]
        print(f"Worker {worker_id} will process {len(worker_dates)} days.")

        # Initialize base trend for each worker (or load from a checkpoint)
        current_base_trend = 0.0

        # Buffer for daily DataFrames
        day_buffer = []
        day_buffer_dates = []

        # Function to yield samples from buffered data
        def yield_from_buffer(buffer_df, date_of_buffer_df):
            # The last few samples of a day might be needed for the next day's first inputs
            # To handle this, we need to fetch a bit more data for each day.
            # However, for simplicity and to prevent complex inter-day dependencies for *this* generator,
            # we will assume each yielded X,Y pair is self-contained within the current day's data.
            # This means we might skip the last hour(s) of a day if they don't have enough future data in *that* day.

            # Iterate through the 24 hourly input windows for this day
            for hour_of_day in range(self.hourly_inputs_per_day):
                input_start_sample_idx = hour_of_day * self.input_seq_len_samples

                # End index for output prediction (relative to start of input)
                # Output window starts IMMEDIATELY after input window (sample 361)
                # It covers samples [361, ..., 360 + 61]
                output_start_sample_idx = input_start_sample_idx + self.input_seq_len_samples
                output_end_sample_idx = output_start_sample_idx + self.output_pred_len_samples

                # Ensure we have enough data for both input and output sequences within this day
                if output_end_sample_idx <= self.samples_per_day:
                    X_data = buffer_df.iloc[input_start_sample_idx : input_start_sample_idx + self.input_seq_len_samples].values
                    Y_data_for_sum = buffer_df.iloc[output_start_sample_idx : output_end_sample_idx]

                    X = torch.tensor(X_data, dtype=torch.float32)
                    Y = torch.tensor(Y_data_for_sum.iloc[:, self.column_label_idx].sum(), dtype=torch.float32).unsqueeze(0)
                    yield X, Y
                # else: print(f"Skipping {date_of_buffer_df.date()} hour {hour_of_day} due to insufficient future data within day.")


        # Main loop to process days
        for i, current_date in enumerate(worker_dates):
            # Load current day's data
            df_current_day, last_val = self._get_daily_data(current_date, current_base_trend)
            current_base_trend = last_val # Carry over trend

            day_buffer.append((df_current_day, current_date))
            day_buffer_dates.append(current_date)

            # If buffer is full or this is the last day for this worker, process the buffer
            if len(day_buffer) >= self.buffer_days or i == len(worker_dates) - 1:
                # Iterate through days in buffer (in order if not shuffling, or in shuffled order if so)
                # If shuffle_days is True for the dataset, we shuffled worker_dates already
                # So we just process the buffer as it is filled.
                
                # To truly use a buffer for shuffling batches (not just days),
                # you'd collect multiple days and then shuffle the *hourly samples* from those days.
                # However, your requirement for batching by DAY (24 hourly samples per batch)
                # makes buffering days simpler, just to ensure that a full day's data is available.

                for buffered_df, buffered_date in day_buffer:
                    yield from yield_from_buffer(buffered_df, buffered_date)
                day_buffer.clear()
                day_buffer_dates.clear()

# --- 3. Mixture of Experts (MoE) Model ---
# (Identical to previous implementation as the model architecture doesn't change)
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128):
        super(GatingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, input_seq_len_samples, num_features, num_experts, expert_hidden_dim=256, gating_hidden_dim=128):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_seq_len_samples * num_features
        self.output_dim = 1

        self.experts = nn.ModuleList([
            Expert(self.input_dim, self.output_dim, expert_hidden_dim)
            for _ in range(num_experts)
        ])
        self.gating_network = GatingNetwork(self.input_dim, num_experts, gating_hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        flat_x = x.view(batch_size, -1)
        gating_logits = self.gating_network(flat_x)
        gating_weights = torch.softmax(gating_logits, dim=1)

        expert_outputs = torch.empty(batch_size, self.num_experts, self.output_dim, device=x.device)
        for i, expert in enumerate(self.experts):
            expert_outputs[:, i, :] = expert(flat_x)

        weighted_expert_outputs = gating_weights.unsqueeze(-1) * expert_outputs
        final_output = torch.sum(weighted_expert_outputs, dim=1)

        return final_output, gating_weights

# --- 4. Training Loop Components ---
# (Identical to previous implementation)
criterion = nn.MSELoss()

def moe_loss_with_load_balancing(predictions, targets, gating_weights, load_balancing_coeff=0.01):
    mse_loss = criterion(predictions, targets)
    mean_expert_prob = gating_weights.mean(dim=0)
    epsilon = 1e-8 # To prevent log(0)
    load_balancing_loss = -torch.sum(mean_expert_prob * torch.log(mean_expert_prob + epsilon))
    total_loss = mse_loss + load_balancing_coeff * load_balancing_loss
    return total_loss, mse_loss, load_balancing_loss

# --- Main Execution: Dataset, DataLoader, Model, Training ---

if __name__ == "__main__":
    print("Initializing TimeSeriesDataGenerator...")
    dataset = TimeSeriesDataGenerator(
        start_date_str=START_DATE_STR,
        end_date_str=END_DATE_STR,
        num_columns=NUM_COLUMNS,
        freq_seconds=FREQ_SECONDS,
        input_seq_len_samples=INPUT_SEQ_LEN_SAMPLES,
        output_pred_len_samples=OUTPUT_PRED_LEN_SAMPLES,
        column_label_name=COLUMN_LABEL_NAME,
        shuffle_days=SHUFFLE_DAYS,
        buffer_days=BUFFER_DAYS # Using a buffer for smoother processing/shuffling
    )
    # Note: len(dataset) is not directly available for IterableDataset.
    # You'd typically estimate total steps from total_days * hourly_inputs_per_day / BATCH_SIZE

    # Using num_workers > 0 will parallelize data loading across CPU cores.
    # This is crucial for large datasets. Make sure to adjust `buffer_days` and `shuffle_days`
    # based on how you want data to be fed across workers.
    # For a generator, drop_last=True is often important to ensure consistent batch sizes.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count() // 2, pin_memory=True, drop_last=True)
    print(f"DataLoader created with {os.cpu_count() // 2} workers.")
    print(f"Expected batch shape: (Batch Size={BATCH_SIZE}, Input Samples={INPUT_SEQ_LEN_SAMPLES}, Features={NUM_COLUMNS})")

    print("\nInstantiating MoE Model...")
    model = MoE(
        input_seq_len_samples=INPUT_SEQ_LEN_SAMPLES,
        num_features=NUM_COLUMNS,
        num_experts=NUM_EXPERTS,
        expert_hidden_dim=EXPERT_HIDDEN_DIM,
        gating_hidden_dim=GATING_HIDDEN_DIM
    ).to(DEVICE)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_epoch_loss = 0
        total_mse_loss = 0
        total_lb_loss = 0
        num_batches = 0

        # Enumerate dataloader with a try-except to handle StopIteration at the end of data
        # An IterableDataset doesn't have a fixed __len__, so we can't use `len(dataloader)`.
        # Instead, we just iterate until it's exhausted.
        for batch_idx, (X, Y) in enumerate(dataloader):
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            optimizer.zero_grad()
            predictions, gating_weights = model(X)

            loss, mse_loss_val, lb_loss_val = moe_loss_with_load_balancing(predictions, Y, gating_weights, LOAD_BALANCING_COEFF)

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_mse_loss += mse_loss_val.item()
            total_lb_loss += lb_loss_val.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, MSE: {mse_loss_val.item():.4f}, LB: {lb_loss_val.item():.4f}")

        if num_batches > 0:
            avg_loss = total_epoch_loss / num_batches
            avg_mse_loss = total_mse_loss / num_batches
            avg_lb_loss = total_lb_loss / num_batches
            print(f"Epoch {epoch+1} Complete: Avg Loss = {avg_loss:.4f}, Avg MSE Loss = {avg_mse_loss:.4f}, Avg Load Balancing Loss = {avg_lb_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} Complete: No batches processed. Data might be exhausted or too small.")

    print("\nTraining finished!")

    # --- (Optional) Inference Example with the Generator-based approach ---
    print("\nPerforming a quick inference test:")
    model.eval()
    with torch.no_grad():
        # Create a new generator for inference (or a validation set)
        inference_dataset = TimeSeriesDataGenerator(
            start_date_str="2025-01-01", # Use a new date range for "test" data
            end_date_str="2025-01-02",
            num_columns=NUM_COLUMNS,
            freq_seconds=FREQ_SECONDS,
            input_seq_len_samples=INPUT_SEQ_LEN_SAMPLES,
            output_pred_len_samples=OUTPUT_PRED_LEN_SAMPLES,
            column_label_name=COLUMN_LABEL_NAME,
            shuffle_days=False,
            buffer_days=1
        )
        inference_dataloader = DataLoader(inference_dataset, batch_size=1, num_workers=1, pin_memory=True) # Batch size 1 for single sample inference

        try:
            # Get one sample to demonstrate inference
            sample_X, sample_Y = next(iter(inference_dataloader))
            sample_X = sample_X.to(DEVICE)
            sample_Y = sample_Y.to(DEVICE)

            predicted_sum, gating_weights = model(sample_X)

            print(f"Input X shape for inference: {sample_X.shape}")
            print(f"True sum: {sample_Y.item():.4f}")
            print(f"Predicted sum: {predicted_sum.item():.4f}")
            print(f"Gating weights (expert probabilities): {gating_weights.squeeze().tolist()}")
            if gating_weights.numel() > 0: # Check if gating_weights is not empty
                print(f"Most active expert: {torch.argmax(gating_weights).item()}")
            else:
                print("No active experts (gating weights are empty).")

        except StopIteration:
            print("Inference DataLoader exhausted or empty. Cannot get a sample for inference.")
        except IndexError as e:
            print(f"Error during inference sample retrieval: {e}. Check data availability.")
