# Installation

```bash
pip install anomalybench
```

## Python support

AnomalyBench targets **Python 3.12 only**. 3.13 is deliberately blocked until
the dependency stack is validated there — importing the package on an
unsupported runtime raises a `DependencyError` naming the version it found
rather than failing later with an obscure error from a transitive dependency.

## Optional extras

The base install covers classical detectors, ARIMA forecasting, graph
detectors, the CLI, and the benchmark workflows. The heavier stacks are opt-in:

| Extra | Installs | Enables |
| --- | --- | --- |
| `deep` | PyTorch, TensorFlow | Autoencoder, Denoising AE, VAE, LSTM, Transformer, TCN, AnoGAN, MAD-GAN |
| `streaming` | River | Half-Space Trees, Online Isolation Forest |
| `forecasting` | Prophet | Prophet detector |
| `all-detectors` | all of the above | every detector |

```bash
pip install "anomalybench[deep]"
pip install "anomalybench[all-detectors]"
```

A detector whose extra is missing raises `DependencyError` naming the package
to install, rather than failing on an import deep inside the call.

## From source

```bash
git clone https://github.com/DiogoRibeiro7/anomalybench
cd anomalybench
poetry install
poetry run pre-commit install
```

Poetry manages dependencies through `pyproject.toml`. Extras work the same way:

```bash
poetry install -E deep
```
