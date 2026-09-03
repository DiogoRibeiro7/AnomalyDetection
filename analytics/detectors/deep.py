"""Deep learning-based anomaly detectors.

These detectors rely on PyTorch or TensorFlow and are imported lazily to keep
requirements optional.  All classes implement the
:class:`~analytics.base.BaseDetector` interface.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from dataexcept import DependencyError, HyperparameterError
from numpy.typing import NDArray

from analytics.base import BaseDetector

if TYPE_CHECKING:
    import torch
    from torch import nn


type ArrayLike = pd.DataFrame | NDArray[np.floating[Any]]
ScoreArray = NDArray[np.floating[Any]]


def _import_torch() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Import ``torch`` lazily and return the core namespaces.

    The helper centralises optional dependency checks to keep the detector
    implementations focused on modelling logic.
    """

    try:  # pragma: no cover - import guard is trivial
        import torch
        from torch import nn, optim
    except Exception as exc:  # pragma: no cover - dependency optional
        raise DependencyError(
            "PyTorch", "this detector requires PyTorch to be installed"
        ) from exc
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
        torch_module: ModuleType,
        patience: int,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if patience < 1:
            raise HyperparameterError(
                "patience", patience, "must be greater than or equal to 1"
            )
        self._torch = torch_module
        self.patience = patience
        self.checkpoint_path = (
            str(checkpoint_path) if checkpoint_path is not None else None
        )
        self.best_loss = float("inf")
        self._counter = 0
        self._best_state: dict[str, torch.Tensor] | None = None

    def update(self, loss: float, model: nn.Module) -> bool:
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

    def restore(self, model: nn.Module) -> None:
        """Restore the weights associated with the best validation loss."""

        if self._best_state is not None:
            model.load_state_dict(self._best_state)


def _split_train_val(
    torch_module: ModuleType,
    tensor: torch.Tensor,
    validation_split: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    score_orientation = "lower_is_more_anomalous"

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

    score_orientation = "lower_is_more_anomalous"

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
        checkpoint_path: str | Path | None = None,
        **params: Any,
    ) -> DenoisingAutoencoderDetector:
        torch_module, nn_module, optim_module = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
        train_tensor, val_tensor = _split_train_val(
            torch_module, tensor, validation_split
        )

        module_base: Any = nn_module.Module

        class AE(module_base):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.encoder = nn_module.Linear(input_dim, input_dim)
                self.decoder = nn_module.Linear(input_dim, input_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.decoder(torch_module.relu(self.encoder(x)))

        self.model = AE(tensor.size(-1))
        opt = optim_module.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn_module.MSELoss()
        stopper = _EarlyStopping(torch_module, patience, checkpoint_path)
        for _ in range(epochs):
            opt.zero_grad()
            noisy_train = train_tensor + noise * torch_module.randn_like(train_tensor)
            output = self.model(noisy_train)
            loss = loss_fn(output, train_tensor)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch_module.no_grad():
                val_output = self.model(val_tensor)
                val_loss = loss_fn(val_output, val_tensor).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch_module, _, _ = _import_torch()

        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
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

    score_orientation = "lower_is_more_anomalous"

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
        checkpoint_path: str | Path | None = None,
        **params: Any,
    ) -> VariationalAutoencoderDetector:
        torch_module, nn_module, optim_module = _import_torch()
        del params  # Unused but retained for future configurability.

        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
        train_tensor, val_tensor = _split_train_val(
            torch_module, tensor, validation_split
        )

        input_dim = tensor.size(-1)

        module_base: Any = nn_module.Module

        class VAE(module_base):
            def __init__(self, in_dim: int, hidden: int, latent: int) -> None:
                super().__init__()
                self.encoder = nn_module.Sequential(
                    nn_module.Linear(in_dim, hidden),
                    nn_module.ReLU(),
                )
                self.z_mu = nn_module.Linear(hidden, latent)
                self.z_logvar = nn_module.Linear(hidden, latent)
                self.decoder = nn_module.Sequential(
                    nn_module.Linear(latent, hidden),
                    nn_module.ReLU(),
                    nn_module.Linear(hidden, in_dim),
                )

            def forward(
                self, x: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                h = self.encoder(x)
                mu = self.z_mu(h)
                logvar = self.z_logvar(h)
                std = torch_module.exp(0.5 * logvar)
                eps = torch_module.randn_like(std)
                z = mu + eps * std
                recon = self.decoder(z)
                return recon, mu, logvar

        self.model = VAE(input_dim, hidden_dim, latent_dim)
        opt = optim_module.Adam(self.model.parameters(), lr=lr)
        mse = nn_module.MSELoss(reduction="sum")
        stopper = _EarlyStopping(torch_module, patience, checkpoint_path)

        def _loss_fn(batch: torch.Tensor) -> torch.Tensor:
            recon, mu, logvar = self.model(batch)
            recon_loss = mse(recon, batch)
            kld = -0.5 * torch_module.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return (recon_loss + kld) / batch.size(0)

        for _ in range(epochs):
            opt.zero_grad()
            loss = _loss_fn(train_tensor)
            loss.backward()
            opt.step()

            self.model.eval()
            with torch_module.no_grad():
                val_loss = _loss_fn(val_tensor).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch_module, _, _ = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
        with torch_module.no_grad():
            reconstructed, _, _ = self.model(tensor)
        errors = np.linalg.norm(tensor.numpy() - reconstructed.numpy(), axis=1)
        return -errors


class LSTMAutoencoderDetector(BaseDetector):
    """LSTM autoencoder for sequence reconstruction."""

    score_orientation = "lower_is_more_anomalous"

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
        checkpoint_path: str | Path | None = None,
        **params: Any,
    ) -> LSTMAutoencoderDetector:
        torch_module, nn_module, optim_module = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
        train_tensor, val_tensor = _split_train_val(
            torch_module, tensor, validation_split
        )
        train_seq = train_tensor.unsqueeze(1)
        val_seq = val_tensor.unsqueeze(1)

        module_base: Any = nn_module.Module

        class LSTMAE(module_base):
            def __init__(self, input_dim: int, hidden_dim: int) -> None:
                super().__init__()
                self.encoder = nn_module.LSTM(input_dim, hidden_dim, batch_first=True)
                self.decoder = nn_module.LSTM(hidden_dim, input_dim, batch_first=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                _, (h_n, _) = self.encoder(x)
                decoder_input = h_n.transpose(0, 1).repeat(1, x.size(1), 1)
                recon, _ = self.decoder(decoder_input)
                return recon

        self.model = LSTMAE(tensor.size(-1), hidden_size)
        opt = optim_module.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn_module.MSELoss()
        stopper = _EarlyStopping(torch_module, patience, checkpoint_path)
        for _ in range(epochs):
            opt.zero_grad()
            output = self.model(train_seq)
            loss = loss_fn(output, train_seq)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch_module.no_grad():
                val_output = self.model(val_seq)
                val_loss = loss_fn(val_output, val_seq).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch_module, _, _ = _import_torch()

        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32).unsqueeze(1)
        reconstructed = self.model(tensor).detach().numpy()
        original = tensor.detach().numpy()
        errors = np.linalg.norm(original - reconstructed, axis=(1, 2))
        return -errors


class TransformerDetector(BaseDetector):
    """Transformer-based autoencoder for anomaly detection."""

    score_orientation = "lower_is_more_anomalous"

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
        checkpoint_path: str | Path | None = None,
        **params: Any,
    ) -> TransformerDetector:
        torch_module, nn_module, optim_module = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
        train_tensor, val_tensor = _split_train_val(
            torch_module, tensor, validation_split
        )
        train_seq = train_tensor.unsqueeze(1)
        val_seq = val_tensor.unsqueeze(1)

        module_base: Any = nn_module.Module

        class TransAE(module_base):
            def __init__(self, input_dim: int, d_model: int, nhead: int) -> None:
                super().__init__()
                self.input = nn_module.Linear(input_dim, d_model)
                enc = nn_module.TransformerEncoderLayer(
                    d_model, nhead, batch_first=True
                )
                self.encoder = nn_module.TransformerEncoder(enc, num_layers=1)
                dec = nn_module.TransformerDecoderLayer(
                    d_model, nhead, batch_first=True
                )
                self.decoder = nn_module.TransformerDecoder(dec, num_layers=1)
                self.output = nn_module.Linear(d_model, input_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                z = self.input(x)
                h = self.encoder(z)
                dec = self.decoder(h, h)
                return self.output(dec)

        self.model = TransAE(tensor.size(-1), d_model, nhead)
        opt = optim_module.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn_module.MSELoss()
        stopper = _EarlyStopping(torch_module, patience, checkpoint_path)
        for _ in range(epochs):
            opt.zero_grad()
            output = self.model(train_seq)
            loss = loss_fn(output, train_seq)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch_module.no_grad():
                val_output = self.model(val_seq)
                val_loss = loss_fn(val_output, val_seq).item()
            self.model.train()
            if stopper.update(val_loss, self.model):
                break
        stopper.restore(self.model)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch_module, _, _ = _import_torch()

        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32).unsqueeze(1)
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

    score_orientation = "lower_is_more_anomalous"

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
        checkpoint_path: str | Path | None = None,
        **params: Any,
    ) -> AnoGANDetector:
        torch_module, nn_module, optim_module = _import_torch()
        del params

        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
        train_tensor, val_tensor = _split_train_val(
            torch_module, tensor, validation_split
        )
        input_dim = tensor.size(1)
        self._latent_dim = latent_dim
        self._optimisation_steps = optimisation_steps
        self._optimisation_lr = optimisation_lr

        self.generator = nn_module.Sequential(
            nn_module.Linear(latent_dim, 64),
            nn_module.ReLU(),
            nn_module.Linear(64, input_dim),
        )
        self.discriminator = nn_module.Sequential(
            nn_module.Linear(input_dim, 64),
            nn_module.ReLU(),
            nn_module.Linear(64, 1),
            nn_module.Sigmoid(),
        )

        module_base: Any = nn_module.Module

        class _AnoGANContainer(module_base):
            def __init__(self, generator: nn.Module, discriminator: nn.Module) -> None:
                super().__init__()
                self.generator = generator
                self.discriminator = discriminator

        self._container = _AnoGANContainer(self.generator, self.discriminator)
        g_opt = optim_module.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        d_opt = optim_module.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        bce = nn_module.BCELoss()
        n = train_tensor.size(0)
        stopper = _EarlyStopping(torch_module, patience, checkpoint_path)

        for _ in range(epochs):
            indices = torch_module.randperm(n)
            for i in range(0, n, batch_size):
                real = train_tensor[indices[i : i + batch_size]]
                if real.numel() == 0:
                    continue
                batch_size_actual = real.size(0)
                noise = torch_module.randn(batch_size_actual, latent_dim)
                fake = self.generator(noise)

                d_opt.zero_grad()
                loss_real = bce(
                    self.discriminator(real), torch_module.ones(batch_size_actual, 1)
                )
                loss_fake = bce(
                    self.discriminator(fake.detach()),
                    torch_module.zeros(batch_size_actual, 1),
                )
                (loss_real + loss_fake).backward()
                d_opt.step()

                g_opt.zero_grad()
                noise = torch_module.randn(batch_size_actual, latent_dim)
                fake = self.generator(noise)
                g_loss = bce(
                    self.discriminator(fake), torch_module.ones(batch_size_actual, 1)
                )
                g_loss.backward()
                g_opt.step()

            self.generator.eval()
            self.discriminator.eval()
            with torch_module.no_grad():
                val_real = val_tensor
                if val_real.size(0) == 0:
                    val_real = train_tensor
                val_z = torch_module.randn(val_real.size(0), latent_dim)
                val_fake = self.generator(val_z)
                val_loss_real = bce(
                    self.discriminator(val_real), torch_module.ones(val_real.size(0), 1)
                )
                val_loss_fake = bce(
                    self.discriminator(val_fake),
                    torch_module.zeros(val_real.size(0), 1),
                )
                val_loss = (val_loss_real + val_loss_fake).item()
            self.generator.train()
            self.discriminator.train()
            if stopper.update(val_loss, self._container):
                break
        stopper.restore(self._container)
        return self

    def _latent_optimisation(
        self, torch_module: ModuleType, sample: torch.Tensor
    ) -> torch.Tensor:
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
        torch_module, _, _ = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
        scores = []
        for row in tensor:
            sample = row.unsqueeze(0)
            score = self._latent_optimisation(torch_module, sample)
            scores.append(score.item())
        return -np.asarray(scores)


class MADGANDetector(BaseDetector):
    """Lightweight MAD-GAN style detector using PyTorch."""

    score_orientation = "higher_is_more_anomalous"

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
        checkpoint_path: str | Path | None = None,
        **params: Any,
    ) -> MADGANDetector:
        torch_module, nn_module, optim_module = _import_torch()
        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
        train_tensor, val_tensor = _split_train_val(
            torch_module, tensor, validation_split
        )
        input_dim = tensor.size(1)

        self.generator = nn_module.Sequential(
            nn_module.Linear(latent_dim, 64),
            nn_module.ReLU(),
            nn_module.Linear(64, input_dim),
        )
        self.discriminator = nn_module.Sequential(
            nn_module.Linear(input_dim, 64),
            nn_module.ReLU(),
            nn_module.Linear(64, 1),
            nn_module.Sigmoid(),
        )

        module_base: Any = nn_module.Module

        class _GANContainer(module_base):
            def __init__(
                self,
                generator: nn.Module,
                discriminator: nn.Module,
            ) -> None:
                super().__init__()
                self.generator = generator
                self.discriminator = discriminator

        self._gan_container = _GANContainer(self.generator, self.discriminator)
        g_opt = optim_module.Adam(self.generator.parameters(), lr=lr)
        d_opt = optim_module.Adam(self.discriminator.parameters(), lr=lr)
        bce = nn_module.BCELoss()
        n = train_tensor.size(0)
        stopper = _EarlyStopping(torch_module, patience, checkpoint_path)
        for _ in range(epochs):
            idx = torch_module.randperm(n)
            for i in range(0, n, batch_size):
                real = train_tensor[idx[i : i + batch_size]]
                z = torch_module.randn(real.size(0), latent_dim)
                fake = self.generator(z)

                d_opt.zero_grad()
                loss_real = bce(
                    self.discriminator(real), torch_module.ones(real.size(0), 1)
                )
                loss_fake = bce(
                    self.discriminator(fake.detach()),
                    torch_module.zeros(real.size(0), 1),
                )
                (loss_real + loss_fake).backward()
                d_opt.step()

                g_opt.zero_grad()
                fake = self.generator(z)
                g_loss = bce(
                    self.discriminator(fake), torch_module.ones(real.size(0), 1)
                )
                g_loss.backward()
                g_opt.step()
            self.generator.eval()
            self.discriminator.eval()
            with torch_module.no_grad():
                val_real = val_tensor
                if val_real.size(0) == 0:
                    val_real = train_tensor
                val_z = torch_module.randn(val_real.size(0), latent_dim)
                val_fake = self.generator(val_z)
                val_loss_real = bce(
                    self.discriminator(val_real),
                    torch_module.ones(val_real.size(0), 1),
                )
                val_loss_fake = bce(
                    self.discriminator(val_fake),
                    torch_module.zeros(val_real.size(0), 1),
                )
                val_loss = (val_loss_real + val_loss_fake).item()
            self.generator.train()
            self.discriminator.train()
            if stopper.update(val_loss, self._gan_container):
                break
        stopper.restore(self._gan_container)
        return self

    def score(self, data: ArrayLike) -> ScoreArray:
        torch_module, _, _ = _import_torch()

        X = _ensure_numpy(data)
        tensor = torch_module.tensor(X, dtype=torch_module.float32)
        with torch_module.no_grad():
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
