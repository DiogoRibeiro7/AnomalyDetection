"""Deep learning-based anomaly detectors.

These detectors rely on PyTorch or TensorFlow and are imported lazily to keep
requirements optional.  All classes implement the
:class:`~analytics.base.BaseDetector` interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.base import BaseDetector


class AutoencoderDetector(BaseDetector):
    """Shallow autoencoder using reconstruction error as anomaly score."""

    def get_name(self) -> str:
        return "Autoencoder"

    def fit(self, data, **params):
        from sklearn.neural_network import MLPRegressor

        hidden = params.pop("hidden_layer_sizes", (32, 32, 32))
        self.model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000)
        self.model.fit(data, data)
        return self

    def score(self, data):
        reconstructed = self.model.predict(data)
        errors = np.linalg.norm(data - reconstructed, axis=1)
        return -errors


class DenoisingAutoencoderDetector(BaseDetector):
    """Denoising autoencoder that reconstructs clean input."""

    def get_name(self) -> str:
        return "Denoising Autoencoder"

    def fit(self, data, noise=0.1, epochs=10, lr=1e-3, **params):
        try:
            import torch
            from torch import nn, optim
        except Exception as e:  # pragma: no cover - dependency optional
            raise ImportError("DenoisingAutoencoderDetector requires PyTorch") from e
        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32)
        noisy = tensor + noise * torch.randn_like(tensor)

        class AE(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.encoder = nn.Linear(input_dim, input_dim)
                self.decoder = nn.Linear(input_dim, input_dim)

            def forward(self, x):
                return self.decoder(torch.relu(self.encoder(x)))

        self.model = AE(tensor.size(-1))
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad()
            output = self.model(noisy)
            loss = loss_fn(output, tensor)
            loss.backward()
            opt.step()
        return self

    def score(self, data):
        import torch

        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32)
        reconstructed = self.model(tensor).detach().numpy()
        errors = np.linalg.norm(tensor.numpy() - reconstructed, axis=1)
        return -errors


class VariationalAutoencoderDetector(BaseDetector):
    """Variational autoencoder leveraging PyOD's implementation."""

    def get_name(self) -> str:
        return "Variational Autoencoder"

    def fit(self, data, **params):
        from pyod.models.vae import VAE

        self.model = VAE(**params)
        X = data.values if isinstance(data, pd.DataFrame) else data
        self.model.fit(X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class LSTMAutoencoderDetector(BaseDetector):
    """LSTM autoencoder for sequence reconstruction."""

    def get_name(self) -> str:
        return "LSTM Autoencoder"

    def fit(self, data, epochs=10, hidden_size=16, lr=1e-3, **params):
        try:
            import torch
            from torch import nn, optim
        except Exception as e:  # pragma: no cover - dependency optional
            raise ImportError("LSTMAutoencoderDetector requires PyTorch") from e
        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        class LSTMAE(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

            def forward(self, x):
                _, h = self.encoder(x)
                recon, _ = self.decoder(h[0])
                return recon

        self.model = LSTMAE(tensor.size(-1), hidden_size)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad()
            output = self.model(tensor)
            loss = loss_fn(output, tensor)
            loss.backward()
            opt.step()
        return self

    def score(self, data):
        import torch

        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        reconstructed = self.model(tensor).detach().numpy()
        errors = np.linalg.norm(tensor.numpy() - reconstructed, axis=(1, 2))
        return -errors


class TransformerDetector(BaseDetector):
    """Transformer-based autoencoder for anomaly detection."""

    def get_name(self) -> str:
        return "Transformer"

    def fit(self, data, epochs=10, d_model=16, nhead=2, lr=1e-3, **params):
        try:
            import torch
            from torch import nn, optim
        except Exception as e:  # pragma: no cover - dependency optional
            raise ImportError("TransformerDetector requires PyTorch") from e
        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        class TransAE(nn.Module):
            def __init__(self, input_dim, d_model, nhead):
                super().__init__()
                self.input = nn.Linear(input_dim, d_model)
                enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc, num_layers=1)
                dec = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
                self.decoder = nn.TransformerDecoder(dec, num_layers=1)
                self.output = nn.Linear(d_model, input_dim)

            def forward(self, x):
                z = self.input(x)
                h = self.encoder(z)
                dec = self.decoder(h, h)
                return self.output(dec)

        self.model = TransAE(tensor.size(-1), d_model, nhead)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad()
            output = self.model(tensor)
            loss = loss_fn(output, tensor)
            loss.backward()
            opt.step()
        return self

    def score(self, data):
        import torch

        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        reconstructed = self.model(tensor).detach().numpy()
        errors = np.linalg.norm(tensor.numpy() - reconstructed, axis=(1, 2))
        return -errors


class AnoGANDetector(BaseDetector):
    """AnoGAN wrapper using PyOD's implementation."""

    def get_name(self) -> str:
        return "AnoGAN"

    def fit(self, data, **params):
        from pyod.models.anogan import AnoGAN

        self.X = data.values if isinstance(data, pd.DataFrame) else data
        self.model = AnoGAN(**params)
        self.model.fit(self.X)
        return self

    def score(self, data):
        X = data.values if isinstance(data, pd.DataFrame) else data
        return -self.model.decision_function(X)


class MADGANDetector(BaseDetector):
    """Lightweight MAD-GAN style detector using PyTorch."""

    def get_name(self) -> str:
        return "MAD-GAN"

    def fit(self, data, epochs=10, latent_dim=16, batch_size=32, lr=1e-3, **params):
        try:
            import torch
            from torch import nn, optim
        except Exception as e:  # pragma: no cover - dependency optional
            raise ImportError("MADGANDetector requires PyTorch") from e
        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32)
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
        g_opt = optim.Adam(self.generator.parameters(), lr=lr)
        d_opt = optim.Adam(self.discriminator.parameters(), lr=lr)
        bce = nn.BCELoss()
        n = tensor.size(0)
        for _ in range(epochs):
            idx = torch.randperm(n)
            for i in range(0, n, batch_size):
                real = tensor[idx[i : i + batch_size]]
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
        return self

    def score(self, data):
        import torch

        X = data.values if isinstance(data, pd.DataFrame) else data
        tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            scores = 1 - self.discriminator(tensor).squeeze().numpy()
        return scores


__all__ = [name for name in globals() if name.endswith("Detector")]
