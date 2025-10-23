"""Deep learning-based anomaly detectors.

These detectors rely on PyTorch or TensorFlow and are imported lazily to keep
requirements optional.  All classes implement the
:class:`~analytics.base.BaseDetector` interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analytics.base import BaseDetector

if TYPE_CHECKING:
    import torch
    from torch import nn, optim


ArrayLike = Union[pd.DataFrame, NDArray[np.floating[Any]]]
ScoreArray = NDArray[np.floating[Any]]


def _import_torch() -> tuple["torch", "nn", "optim"]:
    """Import ``torch`` lazily and return the core namespaces.

    The helper centralises optional dependency checks to keep the detector
    implementations focused on modelling logic.
    """

    try:  # pragma: no cover - import guard is trivial
        import torch
        from torch import nn, optim
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError("This detector requires PyTorch to be installed") from exc
    return torch, nn, optim


def _ensure_numpy(data: ArrayLike) -> NDArray[np.floating[Any]]:
    """Convert supported inputs to a floating-point ``numpy`` array."""

    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=float)
    return np.asarray(data, dtype=float)


class _EarlyStopping:
    """Utility that monitors validation loss and restores the best weights."""

    def __init__(
        self,
        torch_module: "torch",
        patience: int,
        checkpoint_path: Union[str, Path, None] = None,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be greater than or equal to 1")
        self._torch = torch_module
        self.patience = patience
        self.checkpoint_path = (
            str(checkpoint_path) if checkpoint_path is not None else None
        )
        self.best_loss = float("inf")
        self._counter = 0
        self._best_state: dict[str, "torch.Tensor"] | None = None

    def update(self, loss: float, model: "nn.Module") -> bool:
        """Update the tracked validation loss.

        Returns ``True`` when the patience has been exhausted and training
        should stop.
        """

        improvement = loss < self.best_loss - 1e-8
        if improvement:
            self.best_loss = loss
            self._counter = 0
            self._best_state = {
                key: tensor.detach().clone()
                for key, tensor in model.state_dict().items()
            }
            if self.checkpoint_path is not None:
                self._torch.save(self._best_state, self.checkpoint_path)
            return False
        self._counter += 1
        return self._counter >= self.patience

    def restore(self, model: "nn.Module") -> None:
        """Restore the weights associated with the best validation loss."""

        if self._best_state is not None:
            model.load_state_dict(self._best_state)


def _split_train_val(
    torch_module: "torch",
    tensor: "torch.Tensor",
    validation_split: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Split ``tensor`` into train and validation subsets."""

    if tensor.size(0) < 2 or validation_split <= 0:
        return tensor, tensor
    val_size = max(1, int(tensor.size(0) * validation_split))
    if val_size >= tensor.size(0):
        val_size = tensor.size(0) - 1
    if val_size <= 0:
        return tensor, tensor
    indices = torch_module.randperm(tensor.size(0))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if train_indices.numel() == 0:
        train_indices = val_indices
    return tensor[train_indices], tensor[val_indices]


class AutoencoderDetector(BaseDetector):
    """Shallow autoencoder using reconstruction error as anomaly score."""

    def get_name(self) -> str:
        return "Autoencoder"

    def fit(self, data: ArrayLike, **params: Any) -> AutoencoderDetector:
        from sklearn.neural_network import MLPRegressor

        hidden = params.pop("hidden_layer_sizes", (32, 32, 32))
        self.model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000)
        X = _ensure_numpy(data)
        self.model.fit(X, X)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        X = _ensure_numpy(data)
        reconstructed = self.model.predict(X)
        errors = np.linalg.norm(X - reconstructed, axis=1)
        return -errors


class DenoisingAutoencoderDetector(BaseDetector):
    """Denoising autoencoder that reconstructs clean input."""

    def get_name(self) -> str:
        return "Denoising Autoencoder"

    def fit(
        self,
        data: ArrayLike,
        noise: float = 0.1,
        epochs: int = 10,
        lr: float = 1e-3,
        patience: int = 3,
        validation_split: float = 0.1,
        checkpoint_path: Union[str, Path, None] = None,
        **params: Any,
    ) -> DenoisingAutoencoderDetector:
        torch, nn, optim = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        train_tensor, val_tensor = _split_train_val(torch, tensor, validation_split)

        class AE(nn.Module):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.encoder = nn.Linear(input_dim, input_dim)
                self.decoder = nn.Linear(input_dim, input_dim)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                return self.decoder(torch.relu(self.encoder(x)))

        self.model = AE(tensor.size(-1))
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        stopper = _EarlyStopping(torch, patience, checkpoint_path)
        for _ in range(epochs):
            opt.zero_grad()
            noisy_train = train_tensor + noise * torch.randn_like(train_tensor)
            output = self.model(noisy_train)
            loss = loss_fn(output, train_tensor)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_tensor)
                val_loss = loss_fn(val_output, val_tensor).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch, _, _ = _import_torch()

        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        reconstructed = self.model(tensor).detach().numpy()
        original = tensor.detach().numpy()
        errors = np.linalg.norm(original - reconstructed, axis=1)
        return -errors


class VariationalAutoencoderDetector(BaseDetector):
    """Variational autoencoder implemented in PyTorch.

    The detector minimises the evidence lower bound (ELBO) with a simple
    multilayer perceptron encoder/decoder and reports reconstruction error as
    an anomaly score.  Training supports early stopping via a validation split
    and optional checkpoint persistence.
    """

    def get_name(self) -> str:
        return "Variational Autoencoder"

    def fit(
        self,
        data: ArrayLike,
        latent_dim: int = 8,
        hidden_dim: int = 16,
        epochs: int = 20,
        lr: float = 1e-3,
        patience: int = 5,
        validation_split: float = 0.1,
        checkpoint_path: Union[str, Path, None] = None,
        **params: Any,
    ) -> VariationalAutoencoderDetector:
        torch, nn, optim = _import_torch()
        del params  # Unused but retained for future configurability.

        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        train_tensor, val_tensor = _split_train_val(torch, tensor, validation_split)

        input_dim = tensor.size(-1)

        class VAE(nn.Module):
            def __init__(self, in_dim: int, hidden: int, latent: int) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                )
                self.z_mu = nn.Linear(hidden, latent)
                self.z_logvar = nn.Linear(hidden, latent)
                self.decoder = nn.Sequential(
                    nn.Linear(latent, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, in_dim),
                )

            def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
                h = self.encoder(x)
                mu = self.z_mu(h)
                logvar = self.z_logvar(h)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                recon = self.decoder(z)
                return recon, mu, logvar

        self.model = VAE(input_dim, hidden_dim, latent_dim)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        mse = nn.MSELoss(reduction="sum")
        stopper = _EarlyStopping(torch, patience, checkpoint_path)

        def _loss_fn(batch: "torch.Tensor") -> "torch.Tensor":
            recon, mu, logvar = self.model(batch)
            recon_loss = mse(recon, batch)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return (recon_loss + kld) / batch.size(0)

        for _ in range(epochs):
            opt.zero_grad()
            loss = _loss_fn(train_tensor)
            loss.backward()
            opt.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = _loss_fn(val_tensor).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch, _, _ = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            reconstructed, _, _ = self.model(tensor)
        errors = np.linalg.norm(tensor.numpy() - reconstructed.numpy(), axis=1)
        return -errors


class LSTMAutoencoderDetector(BaseDetector):
    """LSTM autoencoder for sequence reconstruction."""

    def get_name(self) -> str:
        return "LSTM Autoencoder"

    def fit(
        self,
        data: ArrayLike,
        epochs: int = 10,
        hidden_size: int = 16,
        lr: float = 1e-3,
        patience: int = 3,
        validation_split: float = 0.1,
        checkpoint_path: Union[str, Path, None] = None,
        **params: Any,
    ) -> LSTMAutoencoderDetector:
        torch, nn, optim = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        train_tensor, val_tensor = _split_train_val(torch, tensor, validation_split)
        train_seq = train_tensor.unsqueeze(1)
        val_seq = val_tensor.unsqueeze(1)

        class LSTMAE(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int) -> None:
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                _, (h_n, _) = self.encoder(x)
                decoder_input = h_n.transpose(0, 1).repeat(1, x.size(1), 1)
                recon, _ = self.decoder(decoder_input)
                return recon

        self.model = LSTMAE(tensor.size(-1), hidden_size)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        stopper = _EarlyStopping(torch, patience, checkpoint_path)
        for _ in range(epochs):
            opt.zero_grad()
            output = self.model(train_seq)
            loss = loss_fn(output, train_seq)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_seq)
                val_loss = loss_fn(val_output, val_seq).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch, _, _ = _import_torch()

        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        reconstructed = self.model(tensor).detach().numpy()
        original = tensor.detach().numpy()
        errors = np.linalg.norm(original - reconstructed, axis=(1, 2))
        return -errors


class TransformerDetector(BaseDetector):
    """Transformer-based autoencoder for anomaly detection."""

    def get_name(self) -> str:
        return "Transformer"

    def fit(
        self,
        data: ArrayLike,
        epochs: int = 10,
        d_model: int = 16,
        nhead: int = 2,
        lr: float = 1e-3,
        patience: int = 3,
        validation_split: float = 0.1,
        checkpoint_path: Union[str, Path, None] = None,
        **params: Any,
    ) -> TransformerDetector:
        torch, nn, optim = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        train_tensor, val_tensor = _split_train_val(torch, tensor, validation_split)
        train_seq = train_tensor.unsqueeze(1)
        val_seq = val_tensor.unsqueeze(1)

        class TransAE(nn.Module):
            def __init__(self, input_dim: int, d_model: int, nhead: int) -> None:
                super().__init__()
                self.input = nn.Linear(input_dim, d_model)
                enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc, num_layers=1)
                dec = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
                self.decoder = nn.TransformerDecoder(dec, num_layers=1)
                self.output = nn.Linear(d_model, input_dim)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                z = self.input(x)
                h = self.encoder(z)
                dec = self.decoder(h, h)
                return self.output(dec)

        self.model = TransAE(tensor.size(-1), d_model, nhead)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        stopper = _EarlyStopping(torch, patience, checkpoint_path)
        for _ in range(epochs):
            opt.zero_grad()
            output = self.model(train_seq)
            loss = loss_fn(output, train_seq)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_seq)
                val_loss = loss_fn(val_output, val_seq).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch, _, _ = _import_torch()

        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        reconstructed = self.model(tensor).detach().numpy()
        original = tensor.detach().numpy()
        errors = np.linalg.norm(original - reconstructed, axis=(1, 2))
        return -errors


class AnoGANDetector(BaseDetector):
    """AnoGAN-style detector with latent optimisation scoring.

    The detector trains a lightweight generator/discriminator pair and performs
    sample-specific latent optimisation during scoring to approximate the
    original AnoGAN formulation.
    """

    def get_name(self) -> str:
        return "AnoGAN"

    def fit(
        self,
        data: ArrayLike,
        epochs: int = 20,
        latent_dim: int = 16,
        batch_size: int = 32,
        lr: float = 2e-4,
        validation_split: float = 0.1,
        patience: int = 5,
        optimisation_steps: int = 30,
        optimisation_lr: float = 1e-2,
        checkpoint_path: Union[str, Path, None] = None,
        **params: Any,
    ) -> AnoGANDetector:
        torch, nn, optim = _import_torch()
        del params

        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        train_tensor, val_tensor = _split_train_val(torch, tensor, validation_split)
        input_dim = tensor.size(1)
        self._latent_dim = latent_dim
        self._optimisation_steps = optimisation_steps
        self._optimisation_lr = optimisation_lr

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        class _AnoGANContainer(nn.Module):
            def __init__(self, generator: "nn.Module", discriminator: "nn.Module") -> None:
                super().__init__()
                self.generator = generator
                self.discriminator = discriminator

        self._container = _AnoGANContainer(self.generator, self.discriminator)
        g_opt = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        bce = nn.BCELoss()
        n = train_tensor.size(0)
        stopper = _EarlyStopping(torch, patience, checkpoint_path)

        for _ in range(epochs):
            indices = torch.randperm(n)
            for i in range(0, n, batch_size):
                real = train_tensor[indices[i : i + batch_size]]
                if real.numel() == 0:
                    continue
                batch_size_actual = real.size(0)
                noise = torch.randn(batch_size_actual, latent_dim)
                fake = self.generator(noise)

                d_opt.zero_grad()
                loss_real = bce(
                    self.discriminator(real), torch.ones(batch_size_actual, 1)
                )
                loss_fake = bce(
                    self.discriminator(fake.detach()), torch.zeros(batch_size_actual, 1)
                )
                (loss_real + loss_fake).backward()
                d_opt.step()

                g_opt.zero_grad()
                noise = torch.randn(batch_size_actual, latent_dim)
                fake = self.generator(noise)
                g_loss = bce(
                    self.discriminator(fake), torch.ones(batch_size_actual, 1)
                )
                g_loss.backward()
                g_opt.step()

            self.generator.eval()
            self.discriminator.eval()
            with torch.no_grad():
                val_real = val_tensor
                if val_real.size(0) == 0:
                    val_real = train_tensor
                val_z = torch.randn(val_real.size(0), latent_dim)
                val_fake = self.generator(val_z)
                val_loss_real = bce(
                    self.discriminator(val_real), torch.ones(val_real.size(0), 1)
                )
                val_loss_fake = bce(
                    self.discriminator(val_fake), torch.zeros(val_real.size(0), 1)
                )
                val_loss = (val_loss_real + val_loss_fake).item()
            self.generator.train()
            self.discriminator.train()
            if stopper.update(val_loss, self._container):
                break
        stopper.restore(self._container)
        return self

    def _latent_optimisation(
        self, torch_module: "torch", sample: "torch.Tensor"
    ) -> "torch.Tensor":
        z = torch_module.zeros(1, self._latent_dim, requires_grad=True)
        optimizer = torch_module.optim.Adam([z], lr=self._optimisation_lr)
        for _ in range(self._optimisation_steps):
            optimizer.zero_grad()
            generated = self.generator(z)
            residual = torch_module.abs(generated - sample).mean()
            discr_distance = torch_module.abs(
                self.discriminator(sample) - self.discriminator(generated)
            ).mean()
            loss = residual + discr_distance
            loss.backward()
            optimizer.step()
        with torch_module.no_grad():
            generated = self.generator(z)
            residual = torch_module.abs(generated - sample).mean()
            discr_distance = torch_module.abs(
                self.discriminator(sample) - self.discriminator(generated)
            ).mean()
        return residual + discr_distance

    def score(self, data: ArrayLike) -> ScoreArray:
        torch, _, _ = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        scores = []
        for row in tensor:
            sample = row.unsqueeze(0)
            score = self._latent_optimisation(torch, sample)
            scores.append(score.item())
        return -np.asarray(scores)


class MADGANDetector(BaseDetector):
    """Lightweight MAD-GAN style detector using PyTorch."""

    def get_name(self) -> str:
        return "MAD-GAN"

    def fit(
        self,
        data: ArrayLike,
        epochs: int = 10,
        latent_dim: int = 16,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 3,
        validation_split: float = 0.1,
        checkpoint_path: Union[str, Path, None] = None,
        **params: Any,
    ) -> MADGANDetector:
        torch, nn, optim = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        train_tensor, val_tensor = _split_train_val(torch, tensor, validation_split)
        input_dim = tensor.size(1)

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        class _GANContainer(nn.Module):
            def __init__(
                self,
                generator: "nn.Module",
                discriminator: "nn.Module",
            ) -> None:
                super().__init__()
                self.generator = generator
                self.discriminator = discriminator

        self._gan_container = _GANContainer(self.generator, self.discriminator)
        g_opt = optim.Adam(self.generator.parameters(), lr=lr)
        d_opt = optim.Adam(self.discriminator.parameters(), lr=lr)
        bce = nn.BCELoss()
        n = train_tensor.size(0)
        stopper = _EarlyStopping(torch, patience, checkpoint_path)
        for _ in range(epochs):
            idx = torch.randperm(n)
            for i in range(0, n, batch_size):
                real = train_tensor[idx[i : i + batch_size]]
                z = torch.randn(real.size(0), latent_dim)
                fake = self.generator(z)

                d_opt.zero_grad()
                loss_real = bce(self.discriminator(real), torch.ones(real.size(0), 1))
                loss_fake = bce(
                    self.discriminator(fake.detach()), torch.zeros(real.size(0), 1)
                )
                (loss_real + loss_fake).backward()
                d_opt.step()

                g_opt.zero_grad()
                fake = self.generator(z)
                g_loss = bce(self.discriminator(fake), torch.ones(real.size(0), 1))
                g_loss.backward()
                g_opt.step()
            self.generator.eval()
            self.discriminator.eval()
            with torch.no_grad():
                val_real = val_tensor
                if val_real.size(0) == 0:
                    val_real = train_tensor
                val_z = torch.randn(val_real.size(0), latent_dim)
                val_fake = self.generator(val_z)
                val_loss_real = bce(
                    self.discriminator(val_real),
                    torch.ones(val_real.size(0), 1),
                )
                val_loss_fake = bce(
                    self.discriminator(val_fake),
                    torch.zeros(val_real.size(0), 1),
                )
                val_loss = (val_loss_real + val_loss_fake).item()
            self.generator.train()
            self.discriminator.train()
            if stopper.update(val_loss, self._gan_container):
                break
        stopper.restore(self._gan_container)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch, _, _ = _import_torch()

        X = _ensure_numpy(data)
        tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            scores = 1 - self.discriminator(tensor).squeeze().numpy()
        return np.asarray(scores)


__all__ = [
    "AutoencoderDetector",
    "DenoisingAutoencoderDetector",
    "VariationalAutoencoderDetector",
    "LSTMAutoencoderDetector",
    "TransformerDetector",
    "AnoGANDetector",
    "MADGANDetector",
]
