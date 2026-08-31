"""Corrected temporal deep detectors with explicit sequence semantics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from analytics.base import BaseDetector
from analytics.detectors.deep import _EarlyStopping, _import_torch, _split_train_val
from analytics.time_series import WindowSpec, coerce_sequence_batch

if TYPE_CHECKING:
    import torch
    from torch import nn

ScoreArray = NDArray[np.floating[Any]]


class _TemporalDetector(BaseDetector):
    """Shared window configuration for sequence reconstruction detectors."""

    score_orientation = "lower_is_more_anomalous"

    def _window_spec(
        self,
        *,
        window_length: int | None,
        stride: int,
        horizon: int,
    ) -> WindowSpec | None:
        if window_length is None:
            return None
        return WindowSpec(
            window_length=window_length,
            stride=stride,
            horizon=horizon,
        )

    def _prepare_sequences(
        self,
        data: Any,
        *,
        window_spec: WindowSpec | None = None,
    ) -> NDArray[np.floating[Any]]:
        return coerce_sequence_batch(data, window_spec=window_spec)


class LSTMAutoencoderDetector(_TemporalDetector):
    """LSTM autoencoder operating on genuine sequences of length greater than one."""

    def get_name(self) -> str:
        return "LSTM Autoencoder"

    def fit(
        self,
        data: Any,
        epochs: int = 10,
        hidden_size: int = 16,
        lr: float = 1e-3,
        patience: int = 3,
        validation_split: float = 0.1,
        checkpoint_path: str | Path | None = None,
        window_length: int | None = None,
        stride: int = 1,
        horizon: int = 0,
        **params: Any,
    ) -> LSTMAutoencoderDetector:
        torch, nn, optim = _import_torch()
        del params
        self.window_spec = self._window_spec(
            window_length=window_length,
            stride=stride,
            horizon=horizon,
        )
        sequences = self._prepare_sequences(data, window_spec=self.window_spec)
        tensor = torch.tensor(sequences, dtype=torch.float32)
        train_seq, val_seq = _split_train_val(torch, tensor, validation_split)

        class LSTMAE(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int) -> None:
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                _, (hidden, _) = self.encoder(x)
                decoder_input = hidden.transpose(0, 1).repeat(1, x.size(1), 1)
                reconstructed, _ = self.decoder(decoder_input)
                return reconstructed

        self.sequence_length = int(tensor.size(1))
        self.input_dim = int(tensor.size(2))
        self.model = LSTMAE(self.input_dim, hidden_size)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        stopper = _EarlyStopping(torch, patience, checkpoint_path)

        for _ in range(epochs):
            optimizer.zero_grad()
            output = self.model(train_seq)
            loss = loss_fn(output, train_seq)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_seq)
                val_loss = loss_fn(val_output, val_seq).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: Any) -> ScoreArray:
        torch, _, _ = _import_torch()
        sequences = self._prepare_sequences(
            data,
            window_spec=getattr(self, "window_spec", None),
        )
        tensor = torch.tensor(sequences, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = self.model(tensor).numpy()
        errors = np.linalg.norm(sequences - reconstructed, axis=(1, 2))
        return -errors


class TransformerDetector(_TemporalDetector):
    """Transformer reconstruction detector operating on explicit sequences."""

    def get_name(self) -> str:
        return "Transformer"

    def fit(
        self,
        data: Any,
        epochs: int = 10,
        d_model: int = 16,
        nhead: int = 2,
        lr: float = 1e-3,
        patience: int = 3,
        validation_split: float = 0.1,
        checkpoint_path: str | Path | None = None,
        window_length: int | None = None,
        stride: int = 1,
        horizon: int = 0,
        **params: Any,
    ) -> TransformerDetector:
        torch, nn, optim = _import_torch()
        del params
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.window_spec = self._window_spec(
            window_length=window_length,
            stride=stride,
            horizon=horizon,
        )
        sequences = self._prepare_sequences(data, window_spec=self.window_spec)
        tensor = torch.tensor(sequences, dtype=torch.float32)
        train_seq, val_seq = _split_train_val(torch, tensor, validation_split)

        class TransAE(nn.Module):
            def __init__(self, input_dim: int, model_dim: int, heads: int) -> None:
                super().__init__()
                self.input = nn.Linear(input_dim, model_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    model_dim,
                    heads,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                decoder_layer = nn.TransformerDecoderLayer(
                    model_dim,
                    heads,
                    batch_first=True,
                )
                self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
                self.output = nn.Linear(model_dim, input_dim)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                embedded = self.input(x)
                memory = self.encoder(embedded)
                decoded = self.decoder(memory, memory)
                return self.output(decoded)

        self.sequence_length = int(tensor.size(1))
        self.input_dim = int(tensor.size(2))
        self.model = TransAE(self.input_dim, d_model, nhead)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        stopper = _EarlyStopping(torch, patience, checkpoint_path)

        for _ in range(epochs):
            optimizer.zero_grad()
            output = self.model(train_seq)
            loss = loss_fn(output, train_seq)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_seq)
                val_loss = loss_fn(val_output, val_seq).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: Any) -> ScoreArray:
        torch, _, _ = _import_torch()
        sequences = self._prepare_sequences(
            data,
            window_spec=getattr(self, "window_spec", None),
        )
        tensor = torch.tensor(sequences, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = self.model(tensor).numpy()
        errors = np.linalg.norm(sequences - reconstructed, axis=(1, 2))
        return -errors
