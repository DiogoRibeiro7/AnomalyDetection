"""Integration tests for deep learning detectors."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.detectors import deep


torch = pytest.importorskip(
    "torch", reason="PyTorch is required for deep detector tests"
)


@pytest.fixture()
def toy_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(32, 4)).astype(np.float32)


@pytest.mark.parametrize(
    "detector_cls, fit_kwargs",
    [
        (deep.VariationalAutoencoderDetector, {"epochs": 5, "latent_dim": 4, "hidden_dim": 8, "patience": 2}),
        (deep.LSTMAutoencoderDetector, {"epochs": 3, "hidden_size": 4, "patience": 2}),
        (deep.TransformerDetector, {"epochs": 3, "d_model": 8, "nhead": 2, "patience": 2}),
    ],
)
def test_sequence_and_autoencoder_detectors_produce_scores(
    detector_cls, fit_kwargs, toy_data
) -> None:
    detector = detector_cls()
    detector.fit(toy_data, validation_split=0.2, **fit_kwargs)
    scores = detector.score(toy_data)
    assert scores.shape == (toy_data.shape[0],)
    assert np.isfinite(scores).all()


def test_variational_autoencoder_improves_reconstruction(toy_data: np.ndarray) -> None:
    detector = deep.VariationalAutoencoderDetector()
    detector.fit(toy_data, epochs=5, latent_dim=3, hidden_dim=6, patience=2, validation_split=0.2)
    torch_tensor = torch.tensor(toy_data, dtype=torch.float32)
    with torch.no_grad():
        recon, _, _ = detector.model(torch_tensor)
    reconstruction_error = torch.linalg.norm(torch_tensor - recon, dim=1)
    assert float(torch.mean(reconstruction_error)) >= 0


@pytest.mark.parametrize(
    "detector_cls, fit_kwargs",
    [
        (deep.AnoGANDetector, {"epochs": 5, "latent_dim": 4, "batch_size": 8, "optimisation_steps": 5, "patience": 2}),
        (deep.MADGANDetector, {"epochs": 5, "latent_dim": 4, "batch_size": 8, "patience": 2}),
    ],
)
def test_gan_based_detectors_return_scores(
    detector_cls, fit_kwargs, toy_data
) -> None:
    detector = detector_cls()
    detector.fit(toy_data, validation_split=0.2, **fit_kwargs)
    scores = detector.score(toy_data[:4])
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()

