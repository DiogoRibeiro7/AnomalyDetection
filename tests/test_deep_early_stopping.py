"""Tests for deep detector early stopping utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch is required for early stopping tests"
)

from analytics.detectors import deep


def test_early_stopping_restores_best_weights(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    stopper = deep._EarlyStopping(torch, patience=2, checkpoint_path=tmp_path / "ckpt.pt")

    # Initial improvement should save weights.
    stopper.update(1.0, model)
    expected_weights = [param.detach().clone() for param in model.parameters()]

    # Simulate deterioration in validation loss.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    assert not stopper.update(2.0, model)
    assert stopper.update(2.5, model)

    stopper.restore(model)

    for param, expected in zip(model.parameters(), expected_weights):
        assert torch.allclose(param, expected)

    assert (tmp_path / "ckpt.pt").exists()


def test_denoising_autoencoder_uses_early_stopping(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class DummyStopper:
        def __init__(
            self,
            torch_module,
            patience: int,
            checkpoint_path=None,
        ) -> None:
            calls["init_args"] = {
                "torch_module": torch_module,
                "patience": patience,
                "checkpoint_path": checkpoint_path,
            }
            calls["instance"] = self
            self.update_calls = 0
            self.restore_calls = 0

        def update(self, loss: float, model) -> bool:  # type: ignore[override]
            calls.setdefault("losses", []).append(loss)
            self.update_calls += 1
            return self.update_calls >= 1

        def restore(self, model) -> None:  # type: ignore[override]
            self.restore_calls += 1
            calls["restore_calls"] = self.restore_calls

    monkeypatch.setattr(deep, "_EarlyStopping", DummyStopper)

    detector = deep.DenoisingAutoencoderDetector()
    detector.fit(
        np.ones((4, 2)),
        epochs=1,
        patience=2,
        validation_split=0.5,
        noise=0.0,
        checkpoint_path=None,
    )

    assert isinstance(calls["losses"], list)
    assert all(isinstance(loss, float) for loss in calls["losses"])  # type: ignore[arg-type]
    assert calls["init_args"]["patience"] == 2  # type: ignore[index]
    assert calls["restore_calls"] == 1
