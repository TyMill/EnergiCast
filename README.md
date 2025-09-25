# EnergiCast

[![CI](https://github.com/TyMill/energicast/actions/workflows/ci.yml/badge.svg)](https://github.com/TyMill/energicast/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://tymill.github.io/EnergiCast/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/TyMill/EnergiCast/blob/main/pyproject.toml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/TyMill/energicast)](https://github.com/TyMill/energicast/releases)


**EnergiCast** is a domain‑specific Python library for energy forecasting (load, PV, wind, prices) that combines:
- ⚡ **Domain‑aware AutoML** with hierarchical reconciliation and energy‑weighted cost metrics,
- 🌞 **Physics–ML hybrids** (pvlib solar/wind features + ML/DL models),
- 🔄 **Synthetic gap filling** with seasonal and generative imputation constrained by physics,
- 🌦️ **Weather scenarios** with ensemble perturbations for uncertainty quantification,
- 📊 **Benchmarks** (OpenSTEF‑like datasets) and standardized backtesting protocols.

---

## ✨ Features
- **AutoML**: tailored search spaces for energy time series (ARIMA, ETS, XGB, TFT roadmap).
- **Physics priors**: PV/wind features from pvlib and meteorological signals.
- **Synthetic imputation**: diffusion/GAN roadmap, with ramp‑rate & non‑negativity constraints.
- **Hierarchical reconciliation**: MinT‑style coherent forecasts (asset → feeder → system).
- **Metrics**: pinball loss, CRPS, and energy‑weighted MAPE with price‑aware penalties.
- **Deployment**: simple model export and CLI tools.

---

## 🚀 Quickstart

Install in editable mode for development:

```bash
git clone https://github.com/TyMill/energicast.git
cd energicast
pip install -e .
```

Run the end-to-end workflow on the bundled example data:

```bash
python -m energicast.cli train --config examples/pv_config.yaml --out runs/demo_model
python -m energicast.cli backtest --config examples/pv_config.yaml --out runs/demo_backtest
python -m energicast.cli export --model-dir runs/demo_model --fmt pickle
python -m energicast.cli report --backtest-dir runs/demo_backtest
```

---

## 📚 Dokumentacja

Dokumentację można zbudować lokalnie przy pomocy [MkDocs](https://www.mkdocs.org/):

```bash
pip install -e .[docs]
mkdocs serve
```

Po wdrożeniu GitHub Pages będzie dostępne pod adresem
[`https://TyMill.github.io/energicast/`](https://TyMill.github.io/energicast/).

---

## 📂 Repository Structure

```
energicast/
├─ pyproject.toml
├─ src/energicast/
│  ├─ data/        # loaders for entsoe, pvlib, weather
│  ├─ impute/      # synthetic gap filling
│  ├─ features/    # calendar + energy features
│  ├─ models/      # ARIMA/ETS, XGB, TFT (roadmap)
│  ├─ hier/        # reconciliation
│  ├─ automl/      # Optuna‑based AutoML
│  ├─ metrics/     # probabilistic + energy metrics
│  ├─ scenarios/   # weather ensembles
│  ├─ bench/       # benchmark datasets
│  ├─ deploy/      # export utils
│  └─ cli.py       # Typer‑based CLI
└─ examples/       # demo scripts & notebooks
```

---

## 👨‍💻 Authors
Created by 
- **Dr Tymoteusz Miller, University of Szczecin**,
- **Dr inz. Ewelina Kostecka, Maritime University of Szczecin**.

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
