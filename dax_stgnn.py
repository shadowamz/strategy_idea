
"""
dax_stgnn.py

Modèle spatio-temporel complet pour prédire les T' prochains rendements
de 40 actions à partir d'une fenêtre de T pas de 10 secondes.

Graphe:
- 40 nœuds actions
- 7 nœuds secteurs
- 1 nœud indice
- action <-> secteur
- action <-> indice
- secteur <-> indice
- self-loops

Architecture:
Dense GAT spatial à chaque timestamp
        -> GRU temporel par nœud
        -> décodeur multi-horizon sur les nœuds actions

Dépendances:
    pip install numpy pandas torch

Exemple avec prix:
    python dax_stgnn.py \
        --csv dax_data.csv \
        --mode prices \
        --timestamp-col timestamp \
        --ticker-col ticker \
        --sector-col sector \
        --value-col price \
        --input-window 60 \
        --horizon 6 \
        --epochs 50

Exemple avec rendements déjà calculés:
    python dax_stgnn.py \
        --csv dax_returns.csv \
        --mode returns \
        --value-col return
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class Config:
    freq: str = "10s"
    market_open: str = "09:00:00"
    market_close: str = "17:30:00"

    n_stocks: int = 40
    input_window: int = 60       # 60 x 10 s = 10 minutes
    horizon: int = 6             # 6 x 10 s = 1 minute

    gat_hidden: int = 32
    gat_heads: int = 4
    gat_layers: int = 2
    temporal_hidden: int = 64
    dropout: float = 0.10

    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 8

    train_ratio: float = 0.70
    val_ratio: float = 0.15

    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")


def extract_sector_map(
    df: pd.DataFrame,
    ticker_col: str,
    sector_col: str,
) -> Dict[str, str]:
    mapping = (
        df[[ticker_col, sector_col]]
        .dropna()
        .drop_duplicates()
    )

    counts = mapping.groupby(ticker_col)[sector_col].nunique()
    bad = counts[counts > 1]
    if not bad.empty:
        raise ValueError(
            "Certains tickers ont plusieurs secteurs: "
            + ", ".join(map(str, bad.index[:10]))
        )

    return dict(zip(mapping[ticker_col].astype(str), mapping[sector_col].astype(str)))


def filter_market_hours(
    frame: pd.DataFrame,
    market_open: str,
    market_close: str,
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("L'index doit être un DatetimeIndex.")
    return frame.between_time(market_open, market_close, inclusive="both")


# ---------------------------------------------------------------------
# Préparation des rendements
# ---------------------------------------------------------------------

def prepare_stock_returns_from_prices(
    df: pd.DataFrame,
    timestamp_col: str,
    ticker_col: str,
    sector_col: str,
    price_col: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Convertit des prix longs en rendements logarithmiques 10 secondes.

    Format d'entrée:
        timestamp | ticker | sector | price
    """
    validate_columns(df, [timestamp_col, ticker_col, sector_col, price_col])

    work = df[[timestamp_col, ticker_col, sector_col, price_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work[ticker_col] = work[ticker_col].astype(str)
    work = work.dropna(subset=[timestamp_col, ticker_col, sector_col, price_col])
    work = work[work[price_col] > 0].sort_values(timestamp_col)

    sector_map = extract_sector_map(work, ticker_col, sector_col)
    stock_order = sorted(sector_map)

    if len(stock_order) != cfg.n_stocks:
        raise ValueError(
            f"Le fichier contient {len(stock_order)} actions, "
            f"mais cfg.n_stocks={cfg.n_stocks}."
        )

    daily_returns: List[pd.DataFrame] = []

    for _, day_df in work.groupby(work[timestamp_col].dt.normalize()):
        prices = day_df.pivot_table(
            index=timestamp_col,
            columns=ticker_col,
            values=price_col,
            aggfunc="last",
        ).sort_index()

        prices = filter_market_hours(prices, cfg.market_open, cfg.market_close)
        if prices.empty:
            continue

        prices = prices.resample(cfg.freq).last()
        prices = prices.reindex(columns=stock_order)

        # À l'intérieur d'une journée, un prix sans transaction est porté en avant.
        prices = prices.ffill()

        # On conserve uniquement les lignes où tous les titres disposent d'un prix.
        prices = prices.dropna(how="any")
        if len(prices) < cfg.input_window + cfg.horizon + 2:
            continue

        returns = np.log(prices).diff().dropna()
        daily_returns.append(returns)

    if not daily_returns:
        raise ValueError("Aucune journée exploitable après préparation des prix.")

    result = pd.concat(daily_returns).sort_index()
    result.columns = result.columns.astype(str)
    return result, sector_map


def prepare_stock_returns_from_returns(
    df: pd.DataFrame,
    timestamp_col: str,
    ticker_col: str,
    sector_col: str,
    return_col: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Prépare des rendements déjà calculés.

    Format d'entrée:
        timestamp | ticker | sector | return
    """
    validate_columns(df, [timestamp_col, ticker_col, sector_col, return_col])

    work = df[[timestamp_col, ticker_col, sector_col, return_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[return_col] = pd.to_numeric(work[return_col], errors="coerce")
    work[ticker_col] = work[ticker_col].astype(str)
    work = work.dropna(subset=[timestamp_col, ticker_col, sector_col, return_col])
    work = work.sort_values(timestamp_col)

    sector_map = extract_sector_map(work, ticker_col, sector_col)
    stock_order = sorted(sector_map)

    if len(stock_order) != cfg.n_stocks:
        raise ValueError(
            f"Le fichier contient {len(stock_order)} actions, "
            f"mais cfg.n_stocks={cfg.n_stocks}."
        )

    daily_returns: List[pd.DataFrame] = []

    for _, day_df in work.groupby(work[timestamp_col].dt.normalize()):
        returns = day_df.pivot_table(
            index=timestamp_col,
            columns=ticker_col,
            values=return_col,
            aggfunc="last",
        ).sort_index()

        returns = filter_market_hours(returns, cfg.market_open, cfg.market_close)
        if returns.empty:
            continue

        returns = returns.resample(cfg.freq).last()
        returns = returns.reindex(columns=stock_order)

        # Absence d'observation sur un intervalle de 10 secondes:
        # hypothèse de rendement nul.
        returns = returns.fillna(0.0)

        if len(returns) < cfg.input_window + cfg.horizon + 1:
            continue

        daily_returns.append(returns)

    if not daily_returns:
        raise ValueError("Aucune journée exploitable après préparation des rendements.")

    result = pd.concat(daily_returns).sort_index()
    result.columns = result.columns.astype(str)
    return result, sector_map


def add_sector_and_index_nodes(
    stock_returns: pd.DataFrame,
    sector_map: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Ajoute:
    - un rendement equal-weight par secteur;
    - un rendement equal-weight de l'indice.

    Ordre final des nœuds:
        actions, secteurs, indice
    """
    stock_order = list(stock_returns.columns)
    sector_order = sorted({sector_map[s] for s in stock_order})

    sector_frames = {}
    for sector in sector_order:
        members = [s for s in stock_order if sector_map[s] == sector]
        if not members:
            raise ValueError(f"Aucune action trouvée pour le secteur {sector}.")
        sector_frames[f"SECTOR::{sector}"] = stock_returns[members].mean(axis=1)

    sector_returns = pd.DataFrame(sector_frames, index=stock_returns.index)
    index_return = stock_returns.mean(axis=1).rename("INDEX::MARKET")

    node_returns = pd.concat(
        [stock_returns, sector_returns, index_return],
        axis=1,
    )

    return node_returns, stock_order, sector_order


# ---------------------------------------------------------------------
# Split chronologique et normalisation
# ---------------------------------------------------------------------

def split_days(
    index: pd.DatetimeIndex,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]]:
    days = sorted(pd.DatetimeIndex(index.normalize().unique()))

    if len(days) < 3:
        raise ValueError("Il faut au minimum 3 journées: train, validation et test.")

    n_days = len(days)
    n_train = max(1, int(n_days * train_ratio))
    n_val = max(1, int(n_days * val_ratio))

    if n_train + n_val >= n_days:
        n_val = 1
        n_train = n_days - 2

    train_days = days[:n_train]
    val_days = days[n_train:n_train + n_val]
    test_days = days[n_train + n_val:]

    return train_days, val_days, test_days


class NodeStandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "NodeStandardScaler":
        values = frame.to_numpy(dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        values = (frame.to_numpy(dtype=np.float64) - self.mean_) / self.std_
        return pd.DataFrame(values, index=frame.index, columns=frame.columns)

    def inverse_stock_tensor(self, z: Tensor, n_stocks: int) -> Tensor:
        """
        z: [batch, horizon, n_stocks]
        """
        self._check_fitted()
        mean = torch.as_tensor(
            self.mean_[:n_stocks],
            dtype=z.dtype,
            device=z.device,
        ).view(1, 1, n_stocks)

        std = torch.as_tensor(
            self.std_[:n_stocks],
            dtype=z.dtype,
            device=z.device,
        ).view(1, 1, n_stocks)

        return z * std + mean

    def state_dict(self) -> Dict[str, List[float]]:
        self._check_fitted()
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Le scaler n'est pas ajusté.")


# ---------------------------------------------------------------------
# Dataset de fenêtres intraday
# ---------------------------------------------------------------------

class IntradayGraphWindowDataset(Dataset):
    def __init__(
        self,
        normalized_node_returns: pd.DataFrame,
        selected_days: Sequence[pd.Timestamp],
        input_window: int,
        horizon: int,
        n_stocks: int,
    ) -> None:
        self.input_window = input_window
        self.horizon = horizon
        self.n_stocks = n_stocks

        selected = {pd.Timestamp(d).normalize() for d in selected_days}
        self.daily_arrays: List[np.ndarray] = []
        self.sample_index: List[Tuple[int, int]] = []

        normalized_days = normalized_node_returns.index.normalize()

        for day in sorted(selected):
            frame = normalized_node_returns.loc[normalized_days == day]
            if frame.empty:
                continue

            array = frame.to_numpy(dtype=np.float32)
            day_id = len(self.daily_arrays)
            self.daily_arrays.append(array)

            n_samples = len(array) - input_window - horizon + 1
            for start in range(max(0, n_samples)):
                self.sample_index.append((day_id, start))

        if not self.sample_index:
            raise ValueError("Aucune fenêtre disponible pour ce split.")

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        day_id, start = self.sample_index[idx]
        arr = self.daily_arrays[day_id]

        x = arr[start:start + self.input_window]
        y_start = start + self.input_window
        y = arr[y_start:y_start + self.horizon, :self.n_stocks]

        # x: [T, N, 1]
        x_tensor = torch.from_numpy(x).unsqueeze(-1)
        # y: [H, n_stocks]
        y_tensor = torch.from_numpy(y)

        return x_tensor, y_tensor


# ---------------------------------------------------------------------
# Construction du graphe
# ---------------------------------------------------------------------

def build_adjacency(
    stock_order: Sequence[str],
    sector_order: Sequence[str],
    sector_map: Dict[str, str],
) -> Tensor:
    """
    Matrice booléenne A[i, j] = True si le nœud i peut agréger le nœud j.
    """
    n_stocks = len(stock_order)
    n_sectors = len(sector_order)
    n_nodes = n_stocks + n_sectors + 1

    sector_to_node = {
        sector: n_stocks + i
        for i, sector in enumerate(sector_order)
    }
    index_node = n_nodes - 1

    adjacency = torch.zeros((n_nodes, n_nodes), dtype=torch.bool)

    # Self-loops
    adjacency.fill_diagonal_(True)

    for stock_node, stock in enumerate(stock_order):
        sector_node = sector_to_node[sector_map[stock]]

        # Action <-> secteur
        adjacency[stock_node, sector_node] = True
        adjacency[sector_node, stock_node] = True

        # Action <-> indice
        adjacency[stock_node, index_node] = True
        adjacency[index_node, stock_node] = True

    # Secteur <-> indice
    for sector_node in sector_to_node.values():
        adjacency[sector_node, index_node] = True
        adjacency[index_node, sector_node] = True

    return adjacency


# ---------------------------------------------------------------------
# Modèle GAT dense + GRU
# ---------------------------------------------------------------------

class DenseGraphAttentionLayer(nn.Module):
    """
    GAT multi-head sans dépendance à torch-geometric.

    Entrée:
        x: [B, N, input_dim]
        adjacency: [N, N]

    Sortie:
        [B, N, heads * head_dim]
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.output_dim = head_dim * n_heads

        self.proj = nn.Linear(input_dim, self.output_dim, bias=False)
        self.attn_source = nn.Parameter(torch.empty(n_heads, head_dim))
        self.attn_target = nn.Parameter(torch.empty(n_heads, head_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.output_dim)

        self.residual = (
            nn.Identity()
            if input_dim == self.output_dim
            else nn.Linear(input_dim, self.output_dim, bias=False)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_target)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_uniform_(self.residual.weight)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        batch_size, n_nodes, _ = x.shape

        h = self.proj(x)
        h = h.view(batch_size, n_nodes, self.n_heads, self.head_dim)

        source_score = (h * self.attn_source).sum(dim=-1)
        target_score = (h * self.attn_target).sum(dim=-1)

        # e[b, i, j, h] = score_source(i) + score_target(j)
        e = source_score.unsqueeze(2) + target_score.unsqueeze(1)
        e = self.leaky_relu(e)

        mask = adjacency.view(1, n_nodes, n_nodes, 1)
        e = e.masked_fill(~mask, torch.finfo(e.dtype).min)

        attention = torch.softmax(e, dim=2)
        attention = self.dropout(attention)

        # Agrégation des voisins j vers le nœud i.
        out = torch.einsum("bijh,bjhd->bihd", attention, h)
        out = out.reshape(batch_size, n_nodes, self.output_dim)

        out = self.dropout(out)
        out = self.norm(out + self.residual(x))
        return torch.nn.functional.elu(out)


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if hidden_dim % n_heads != 0:
            raise ValueError("gat_hidden doit être divisible par gat_heads.")

        head_dim = hidden_dim // n_heads
        layers = []

        current_dim = input_dim
        for _ in range(n_layers):
            layers.append(
                DenseGraphAttentionLayer(
                    input_dim=current_dim,
                    head_dim=head_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            )
            current_dim = hidden_dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, adjacency)
        return x


class SpatioTemporalGATGRU(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_stocks: int,
        input_features: int,
        horizon: int,
        gat_hidden: int,
        gat_heads: int,
        gat_layers: int,
        temporal_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.n_nodes = n_nodes
        self.n_stocks = n_stocks
        self.horizon = horizon

        self.spatial_encoder = SpatialEncoder(
            input_dim=input_features,
            hidden_dim=gat_hidden,
            n_heads=gat_heads,
            n_layers=gat_layers,
            dropout=dropout,
        )

        # Embedding statique propre à chaque nœud.
        self.node_embedding = nn.Parameter(
            torch.zeros(1, 1, n_nodes, gat_hidden)
        )
        nn.init.normal_(self.node_embedding, mean=0.0, std=0.02)

        self.temporal_gru = nn.GRU(
            input_size=gat_hidden,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(temporal_hidden),
            nn.Linear(temporal_hidden, temporal_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden, horizon),
        )

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        """
        x: [B, T, N, F]
        output: [B, horizon, n_stocks]
        """
        batch_size, n_times, n_nodes, n_features = x.shape

        if n_nodes != self.n_nodes:
            raise ValueError(
                f"Le modèle attend {self.n_nodes} nœuds, reçu {n_nodes}."
            )

        # Application du même graphe à tous les timestamps.
        spatial_input = x.reshape(batch_size * n_times, n_nodes, n_features)
        spatial = self.spatial_encoder(spatial_input, adjacency)
        spatial = spatial.view(
            batch_size,
            n_times,
            n_nodes,
            -1,
        )

        spatial = spatial + self.node_embedding

        # Une séquence temporelle par nœud.
        temporal_input = spatial.permute(0, 2, 1, 3).contiguous()
        temporal_input = temporal_input.view(
            batch_size * n_nodes,
            n_times,
            -1,
        )

        _, hidden = self.temporal_gru(temporal_input)
        hidden = hidden[-1].view(batch_size, n_nodes, -1)

        stock_hidden = hidden[:, :self.n_stocks, :]
        prediction = self.decoder(stock_hidden)

        # [B, stocks, H] -> [B, H, stocks]
        return prediction.permute(0, 2, 1).contiguous()


# ---------------------------------------------------------------------
# Entraînement et métriques
# ---------------------------------------------------------------------

def regression_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    error = pred - target

    mse = torch.mean(error ** 2).item()
    mae = torch.mean(torch.abs(error)).item()
    rmse = math.sqrt(mse)

    directional_accuracy = (
        (torch.sign(pred) == torch.sign(target))
        .float()
        .mean()
        .item()
    )

    # Corrélation cross-sectionnelle moyenne.
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=-1)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=-1)
        * (target_centered ** 2).sum(dim=-1)
    ).clamp_min(1e-12)

    information_coefficient = (numerator / denominator).mean().item()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
        "information_coefficient": information_coefficient,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    adjacency: Tensor,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    criterion = nn.HuberLoss(delta=1.0)

    total_loss = 0.0
    total_items = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, adjacency)

        loss = criterion(prediction, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    adjacency: Tensor,
    scaler: NodeStandardScaler,
    n_stocks: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.HuberLoss(delta=1.0)

    all_pred = []
    all_target = []
    total_loss = 0.0
    total_items = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        prediction = model(x, adjacency)
        loss = criterion(prediction, y)

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

        prediction_raw = scaler.inverse_stock_tensor(prediction, n_stocks)
        target_raw = scaler.inverse_stock_tensor(y, n_stocks)

        all_pred.append(prediction_raw.cpu())
        all_target.append(target_raw.cpu())

    pred = torch.cat(all_pred, dim=0)
    target = torch.cat(all_target, dim=0)

    metrics = regression_metrics(pred, target)
    metrics["normalized_huber"] = total_loss / max(total_items, 1)
    return metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adjacency: Tensor,
    scaler: NodeStandardScaler,
    cfg: Config,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    device = torch.device(cfg.device)
    model = model.to(device)
    adjacency = adjacency.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            adjacency=adjacency,
            device=device,
            grad_clip=cfg.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            adjacency=adjacency,
            scaler=scaler,
            n_stocks=cfg.n_stocks,
            device=device,
        )

        scheduler.step(val_metrics["normalized_huber"])

        record = {
            "epoch": float(epoch),
            "train_normalized_huber": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"val={val_metrics['normalized_huber']:.6f} | "
            f"MAE={val_metrics['mae']:.8f} | "
            f"DirAcc={val_metrics['directional_accuracy']:.3f} | "
            f"IC={val_metrics['information_coefficient']:.3f}"
        )

        current_val = val_metrics["normalized_huber"]

        if current_val < best_val - 1e-8:
            best_val = current_val
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------
# Sauvegarde et prédiction
# ---------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: Config,
    scaler: NodeStandardScaler,
    stock_order: Sequence[str],
    sector_order: Sequence[str],
    sector_map: Dict[str, str],
    adjacency: Tensor,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "scaler": scaler.state_dict(),
        "stock_order": list(stock_order),
        "sector_order": list(sector_order),
        "sector_map": dict(sector_map),
        "adjacency": adjacency.cpu(),
    }
    torch.save(checkpoint, path)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    adjacency: Tensor,
    scaler: NodeStandardScaler,
    n_stocks: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    adjacency = adjacency.to(device)

    predictions = []
    targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred_z = model(x, adjacency)
        pred = scaler.inverse_stock_tensor(pred_z, n_stocks)
        target = scaler.inverse_stock_tensor(y, n_stocks)

        predictions.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(targets)


# ---------------------------------------------------------------------
# Démonstration synthétique
# ---------------------------------------------------------------------

def generate_synthetic_data(
    n_days: int = 12,
    n_stocks: int = 40,
    n_sectors: int = 7,
    steps_per_day: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Génère un petit jeu de rendements pour tester tout le pipeline.
    """
    rng = np.random.default_rng(seed)

    tickers = [f"STOCK_{i:02d}" for i in range(n_stocks)]
    sectors = [f"SECTOR_{i}" for i in range(n_sectors)]
    sector_map = {
        ticker: sectors[i % n_sectors]
        for i, ticker in enumerate(tickers)
    }

    rows = []
    first_day = pd.Timestamp("2024-01-02")

    for day_offset in range(n_days):
        day = first_day + pd.offsets.BDay(day_offset)
        timestamps = pd.date_range(
            day + pd.Timedelta(hours=9),
            periods=steps_per_day,
            freq="10s",
        )

        market_factor = rng.normal(0.0, 0.00020, size=steps_per_day)
        sector_factors = {
            sector: rng.normal(0.0, 0.00015, size=steps_per_day)
            for sector in sectors
        }

        for stock in tickers:
            sector = sector_map[stock]
            idiosyncratic = rng.normal(0.0, 0.00025, size=steps_per_day)

            ret = (
                0.45 * market_factor
                + 0.35 * sector_factors[sector]
                + idiosyncratic
            )

            for timestamp, value in zip(timestamps, ret):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "ticker": stock,
                        "sector": sector,
                        "return": float(value),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    cfg = Config(
        freq=args.freq,
        market_open=args.market_open,
        market_close=args.market_close,
        n_stocks=args.n_stocks,
        input_window=args.input_window,
        horizon=args.horizon,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        temporal_hidden=args.temporal_hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )

    set_seed(cfg.seed)

    if args.demo:
        print("Utilisation des données synthétiques.")
        df = generate_synthetic_data(
            n_days=12,
            n_stocks=cfg.n_stocks,
            n_sectors=7,
            steps_per_day=max(500, cfg.input_window + cfg.horizon + 20),
            seed=cfg.seed,
        )
        mode = "returns"
        value_col = "return"
    else:
        if not args.csv:
            raise ValueError("--csv est requis sauf avec --demo.")
        df = pd.read_csv(args.csv)
        mode = args.mode
        value_col = args.value_col

    if mode == "prices":
        stock_returns, sector_map = prepare_stock_returns_from_prices(
            df=df,
            timestamp_col=args.timestamp_col,
            ticker_col=args.ticker_col,
            sector_col=args.sector_col,
            price_col=value_col,
            cfg=cfg,
        )
    else:
        stock_returns, sector_map = prepare_stock_returns_from_returns(
            df=df,
            timestamp_col=args.timestamp_col,
            ticker_col=args.ticker_col,
            sector_col=args.sector_col,
            return_col=value_col,
            cfg=cfg,
        )

    node_returns, stock_order, sector_order = add_sector_and_index_nodes(
        stock_returns=stock_returns,
        sector_map=sector_map,
    )

    print(
        f"{len(stock_order)} actions | "
        f"{len(sector_order)} secteurs | "
        f"{node_returns.shape[1]} nœuds | "
        f"{len(node_returns)} timestamps"
    )

    train_days, val_days, test_days = split_days(
        node_returns.index,
        cfg.train_ratio,
        cfg.val_ratio,
    )

    print(
        f"Jours: train={len(train_days)}, "
        f"validation={len(val_days)}, test={len(test_days)}"
    )

    train_mask = node_returns.index.normalize().isin(train_days)
    scaler = NodeStandardScaler().fit(node_returns.loc[train_mask])
    normalized = scaler.transform(node_returns)

    train_dataset = IntradayGraphWindowDataset(
        normalized,
        train_days,
        cfg.input_window,
        cfg.horizon,
        cfg.n_stocks,
    )
    val_dataset = IntradayGraphWindowDataset(
        normalized,
        val_days,
        cfg.input_window,
        cfg.horizon,
        cfg.n_stocks,
    )
    test_dataset = IntradayGraphWindowDataset(
        normalized,
        test_days,
        cfg.input_window,
        cfg.horizon,
        cfg.n_stocks,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    adjacency = build_adjacency(
        stock_order=stock_order,
        sector_order=sector_order,
        sector_map=sector_map,
    )

    model = SpatioTemporalGATGRU(
        n_nodes=node_returns.shape[1],
        n_stocks=cfg.n_stocks,
        input_features=1,
        horizon=cfg.horizon,
        gat_hidden=cfg.gat_hidden,
        gat_heads=cfg.gat_heads,
        gat_layers=cfg.gat_layers,
        temporal_hidden=cfg.temporal_hidden,
        dropout=cfg.dropout,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables: {n_parameters:,}")
    print(f"Device: {cfg.device}")

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        adjacency=adjacency,
        scaler=scaler,
        cfg=cfg,
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        adjacency=adjacency.to(cfg.device),
        scaler=scaler,
        n_stocks=cfg.n_stocks,
        device=torch.device(cfg.device),
    )

    print("\nMétriques test:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.8f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "dax_stgnn_checkpoint.pt"
    history_path = output_dir / "training_history.json"
    predictions_path = output_dir / "test_predictions.npz"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        cfg=cfg,
        scaler=scaler,
        stock_order=stock_order,
        sector_order=sector_order,
        sector_map=sector_map,
        adjacency=adjacency,
    )

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    predictions, targets = predict_loader(
        model=model,
        loader=test_loader,
        adjacency=adjacency,
        scaler=scaler,
        n_stocks=cfg.n_stocks,
        device=torch.device(cfg.device),
    )

    np.savez_compressed(
        predictions_path,
        predictions=predictions,
        targets=targets,
        stock_order=np.asarray(stock_order),
    )

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Historique: {history_path}")
    print(f"Prédictions: {predictions_path}")
    print(f"Shape prédictions: {predictions.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GAT-GRU spatio-temporel pour rendements intraday."
    )

    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=["prices", "returns"],
        default="returns",
    )
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument("--ticker-col", type=str, default="ticker")
    parser.add_argument("--sector-col", type=str, default="sector")
    parser.add_argument(
        "--value-col",
        type=str,
        default="return",
        help="Colonne price si mode=prices, return si mode=returns.",
    )

    parser.add_argument("--freq", type=str, default="10s")
    parser.add_argument("--market-open", type=str, default="09:00:00")
    parser.add_argument("--market-close", type=str, default="17:30:00")
    parser.add_argument("--n-stocks", type=int, default=40)
    parser.add_argument("--input-window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=6)

    parser.add_argument("--gat-hidden", type=int, default=32)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--demo", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    run_pipeline(arguments)
