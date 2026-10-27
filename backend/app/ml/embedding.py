"""
Length-invariant "shape" embedding for a price window.

A window of ANY length (15 trading days or 1200) is resampled via linear
interpolation to a fixed 40-point, 3-channel sequence -- [cumulative return
from window start, short rolling volatility, running drawdown from window
start] -- so windows of wildly different absolute length (the COVID crash was
~33 trading days; the dot-com bust was ~2.5 years) become directly comparable
by shape alone.

A small Transformer encoder (2 layers, 4 heads, d_model=32) turns that fixed
(40,3) sequence into a 16-dim embedding, trained self-supervised as an
autoencoder (reconstruction loss) on a reference corpus of historical
windows. With only a few hundred small samples this does not learn anything
resembling deep "understanding" -- what it actually learns is closer to a
nonlinear, attention-based compression of the return/vol/drawdown path that
(unlike a handful of summary statistics) can represent *where in the window*
a spike sits. That is still a genuine, correctly-scoped use of self-attention,
and is described that way in the UI rather than oversold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

WINDOW_POINTS = 40
EMBED_DIM = 16
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
SHORT_VOL_WINDOW = 5


def build_channels(prices: pd.Series) -> np.ndarray:
    """
    In-window-only features -- no external lookback required, so this is
    safe to call on corpus slices as short as ~15 trading days (unlike
    regime.py's _build_features, which needs 20-60 days of PRIOR context).

    Returns an (n_days, 3) array: [cumulative return from window start,
    short rolling annualized vol, running drawdown from window start].
    """
    prices = prices.astype(float)
    cum_return = (prices / prices.iloc[0] - 1.0).values

    daily_ret = prices.pct_change().fillna(0.0)
    roll_vol = daily_ret.rolling(SHORT_VOL_WINDOW, min_periods=1).std().fillna(0.0) * np.sqrt(252)
    roll_vol = roll_vol.values

    running_max = prices.cummax()
    drawdown = (prices / running_max - 1.0).values

    return np.column_stack([cum_return, roll_vol, drawdown]).astype(np.float32)


def resample_to_fixed(channels: np.ndarray, n_points: int = WINDOW_POINTS) -> np.ndarray:
    """Per-channel linear interpolation onto n_points along a normalized [0,1] x-axis."""
    n_days = channels.shape[0]
    if n_days == 1:
        return np.repeat(channels, n_points, axis=0)
    src_x = np.linspace(0.0, 1.0, n_days)
    dst_x = np.linspace(0.0, 1.0, n_points)
    out = np.empty((n_points, channels.shape[1]), dtype=np.float32)
    for c in range(channels.shape[1]):
        out[:, c] = np.interp(dst_x, src_x, channels[:, c])
    return out


class ShapeEncoder(nn.Module):
    """(batch, 40, 3) -> (batch, 16) embedding via a small Transformer encoder."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(3, D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1, WINDOW_POINTS, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL * 2,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.out_proj = nn.Linear(D_MODEL, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.out_proj(pooled)


class ShapeAutoencoder(nn.Module):
    """Wraps ShapeEncoder with a small symmetric decoder, used only during training."""

    def __init__(self):
        super().__init__()
        self.encoder = ShapeEncoder()
        self.decode_proj = nn.Linear(EMBED_DIM, D_MODEL)
        self.decode_pos = nn.Parameter(torch.randn(1, WINDOW_POINTS, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL * 2,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.reconstruct = nn.Linear(D_MODEL, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        h = self.decode_proj(z).unsqueeze(1).expand(-1, WINDOW_POINTS, -1) + self.decode_pos
        h = self.decoder(h)
        return self.reconstruct(h)


def fit_scaler(all_channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the whole corpus (all_channels: (n, 40, 3))."""
    mean = all_channels.reshape(-1, 3).mean(axis=0)
    std = all_channels.reshape(-1, 3).std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def train_encoder(corpus_channels: np.ndarray, mean: np.ndarray, std: np.ndarray,
                   epochs: int = 120, lr: float = 2e-3, seed: int = 7) -> ShapeEncoder:
    """
    Full-batch Adam training of the autoencoder on the standardized corpus.
    corpus_channels: (n, 40, 3) raw (pre-standardization). Returns the
    trained encoder in eval mode.
    """
    torch.manual_seed(seed)
    x = torch.tensor((corpus_channels - mean) / std, dtype=torch.float32)

    model = ShapeAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        recon = model(x)
        loss = loss_fn(recon, x)
        loss.backward()
        opt.step()

    model.eval()
    return model.encoder


def embed(encoder: ShapeEncoder, channels_batch: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """channels_batch: (n, 40, 3) raw -> (n, 16) numpy embeddings."""
    x = torch.tensor((channels_batch - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        z = encoder(x)
    return z.numpy()
