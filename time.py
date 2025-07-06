import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import deque # For the generator-based DataLoader

# --- 1. Custom MoE Layer (Simplified) ---
class MoEFeedForward(nn.Module):
    def __init__(self, model_dim, dim_feedforward, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.model_dim = model_dim
        self.dim_feedforward = dim_feedforward

        # Experts: a list of simple Feed-Forward Networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, model_dim)
            ) for _ in range(num_experts)
        ])

        # Gating network (router)
        self.gate = nn.Linear(model_dim, num_experts)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, model_dim)
        batch_size, seq_len, _ = x.shape
        flat_x = x.view(-1, self.model_dim) # Flatten for gating: (batch_size * seq_len, model_dim)

        # Get raw gating scores
        gate_logits = self.gate(flat_x) # (batch_size * seq_len, num_experts)

        # Get top-k expert indices and their weights
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1) # Normalize weights for selected experts

        # Initialize output tensor
        output = torch.zeros_like(flat_x)

        # Route tokens to experts and combine outputs
        # This naive routing is inefficient for large scale.
        # Real MoE implementations use sparse operations or custom kernels.
        for i in range(batch_size * seq_len):
            for k_idx in range(self.top_k):
                expert_idx = top_k_indices[i, k_idx]
                weight = top_k_weights[i, k_idx]
                expert_output = self.experts[expert_idx](flat_x[i].unsqueeze(0)) # Process one token with one expert
                output[i] += weight * expert_output.squeeze(0)

        return output.view(batch_size, seq_len, self.model_dim)

# --- 2. Transformer Decoder Layer with MoE ---
class MoEDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dim_feedforward, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.moe_ffn = MoEFeedForward(model_dim, dim_feedforward, num_experts, top_k)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        # MoE Feed-Forward Network
        ffn_output = self.moe_ffn(tgt)
        tgt = tgt + self.dropout2(ffn_output)
        tgt = self.norm2(tgt)
        return tgt

# --- 3. Time-MoE Transformer Decoder Model ---
class TimeMoEDecoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_decoder_layers, dim_feedforward,
                 num_experts, top_k, dropout=0.1):
        super().__init__()

        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Embedding(5000, model_dim) # Max seq length

        self.decoder_layers = nn.ModuleList([
            MoEDecoderLayer(model_dim, num_heads, dim_feedforward, num_experts, top_k, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_head = nn.Linear(model_dim, 1) # Output for the sum

    def forward(self, tgt, tgt_mask=None):
        # tgt shape: (batch_size, sequence_length, input_dim)

        # Input embedding
        tgt = self.input_embedding(tgt) # (batch_size, seq_len, model_dim)

        # Add positional encoding
        positions = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        tgt = tgt + self.positional_encoding(positions)

        for layer in self.decoder_layers:
            tgt = layer(tgt, tgt_mask=tgt_mask)

        # For forecasting, usually we take the last token's output for prediction
        # or aggregate in another way. Here, we're predicting a single sum.
        # Since it's a decoder-only for time series, it predicts the next step
        # auto-regressively. If you want a single sum for the entire forecast horizon,
        # you might need a different output head or a sequence-to-single-value approach.
        # For this problem, we'll take the last token's representation and project it.
        # The paper uses multiple forecasting heads for different resolutions.
        # Here, we simplify to one head for the sum.
        last_token_output = tgt[:, -1, :] # (batch_size, model_dim)
        output = self.output_head(last_token_output)
        return output.squeeze(-1) # (batch_size,)


# --- 4. Generator-based DataLoader for Date-by-Date Processing ---
def time_series_generator(dataframe, input_window_size=10, label_sum_length=360, target_column='last'):
    """
    A generator that yields (input_sequence, label_sum) tuples.
    It processes the dataframe date by date (row by row).
    """
    data_np = dataframe.values
    target_column_index = dataframe.columns.get_loc(target_column)
    total_rows = len(data_np)

    # We need enough data for the input window AND the label sum
    if total_rows < input_window_size + label_sum_length:
        raise ValueError("DataFrame is too short for the specified input window and label sum length.")

    for i in range(total_rows - input_window_size - label_sum_length + 1):
        input_sequence = data_np[i : i + input_window_size, :]
        label_start_idx = i + input_window_size
        label_end_idx = label_start_idx + label_sum_length
        label_sum = np.sum(data_np[label_start_idx : label_end_idx, target_column_index])

        yield torch.tensor(input_sequence, dtype=torch.float32), torch.tensor(label_sum, dtype=torch.float32)

# --- 5. Custom DataLoader for Generators ---
# This is a simple wrapper for convenience to make it feel like a PyTorch DataLoader
class GeneratorDataLoader:
    def __init__(self, generator_func, *args, batch_size=1, **kwargs):
        self.generator_func = generator_func
        self.args = args
        self.kwargs = kwargs
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for x, y in self.generator_func(*self.args, **self.kwargs):
            batch.append((x, y))
            if len(batch) == self.batch_size:
                # Transpose batch to get (inputs_batch, labels_batch)
                # where inputs_batch is (batch_size, seq_len, input_dim)
                # and labels_batch is (batch_size,)
                inputs = torch.stack([item[0] for item in batch])
                labels = torch.stack([item[1] for item in batch])
                yield inputs, labels
                batch = []
        # Yield any remaining items in the last batch
        if batch:
            inputs = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
            yield inputs, labels

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy DataFrame with 10 columns
    num_rows = 2000
    df = pd.DataFrame(np.random.rand(num_rows, 10), columns=[f'col_{i}' for i in range(9)] + ['last'])
    # Ensure there's a 'last' column for target
    df['last'] = np.random.rand(num_rows) * 100

    # Configuration
    input_window_size = 10  # 'd' from your description
    label_sum_length = 360  # Sum 'last' column from d to d+360
    batch_size = 32

    # Instantiate the generator-based DataLoader
    try:
        # Note: We pass the dataframe and other params to the generator_func
        dataloader = GeneratorDataLoader(
            time_series_generator,
            dataframe=df,
            input_window_size=input_window_size,
            label_sum_length=label_sum_length,
            target_column='last',
            batch_size=batch_size
        )
        print("DataLoader (generator-based) initialized.")
    except ValueError as e:
        print(f"Error initializing DataLoader: {e}")
        exit()

    # Get one batch to determine input_dim for the model
    sample_inputs, sample_labels = next(iter(dataloader))
    input_dim = sample_inputs.shape[-1] # Number of features (columns)
    print(f"Sample input shape from generator: {sample_inputs.shape}") # (batch_size, input_window_size, input_dim)
    print(f"Sample label shape from generator: {sample_labels.shape}") # (batch_size,)

    # Model parameters for Time-MoE (conceptual)
    model_dim = 256
    num_heads = 8
    num_decoder_layers = 4
    dim_feedforward = 512
    num_experts = 8 # Total number of experts
    top_k = 2       # How many experts to activate per token

    # Initialize the conceptual Time-MoE model
    model = TimeMoEDecoder(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        num_experts=num_experts,
        top_k=top_k
    )
    print("\nConceptual Time-MoE Model Architecture:")
    print(model)

    # --- Training Loop Placeholder ---
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nStarting a brief training simulation with generator-based DataLoader...")
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}\n")
        else:
            print(f"Epoch {epoch+1} completed, no batches processed (data might be too short).\n")

    print("Training simulation finished.")

    # --- Inference Example ---
    model.eval()
    with torch.no_grad():
        # Get one sample from the generator for inference (e.g., the first one)
        inference_generator = time_series_generator(
            dataframe=df,
            input_window_size=input_window_size,
            label_sum_length=label_sum_length,
            target_column='last'
        )
        first_input, first_actual_label = next(inference_generator)

        # Add batch dimension (1, input_window_size, input_dim)
        inference_input = first_input.unsqueeze(0)
        predicted_label = model(inference_input)

        print(f"\nExample Inference:")
        print(f"Actual Sum for the next {label_sum_length} steps: {first_actual_label.item():.4f}")
        print(f"Predicted Sum: {predicted_label.item():.4f}")
