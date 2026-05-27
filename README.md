# Tennis Match Prediction ML

Machine learning pipeline for ATP tennis match prediction using:
- retrospective Elo ratings,
- rolling player statistics,
- bookmaker market probabilities,
- and time-aware validation.

The project focuses primarily on:
- structured feature engineering,
- leakage prevention,
- temporal evaluation methodology,
- and reproducible analytical workflows.

---

# Current Benchmark

Primary evaluation methodology:
- Train: 2018–2023
- Validation: 2024
- Test: 2025

Current realistic pre-match benchmark:

| Model | Features | Test Accuracy | ROC AUC |
|---|---|---:|---:|
| Logistic Regression | Elo + Rolling Stats | ~63–64% | ~0.69 |
| Logistic Regression | Market + Elo + Rolling | ~67% | ~0.73–0.76 |

Notes:
- bookmaker market probabilities are currently the strongest feature group,
- Elo remains the strongest non-market feature,
- Random Forest models currently overfit more than Logistic Regression under temporal validation.

---

# Pipeline Overview

```text
Flashscore scraping
    ↓
Raw match datasets
    ↓
Preprocessing / normalization
    ↓
Retrospective feature generation
    ↓
Temporal train/validation/test split
    ↓
Feature experiments
    ↓
Model evaluation
```

---

# Repository Structure

```text
src/
    preprocessing.py
    feature_engineering.py
    rolling_stats.py
    elo.py
    model_training.py
    feature_experiments.py
    data_coverage_analysis.py
    validation_strategy_experiments.py

docs/
    methodology.md
    experiment_results.md

data/intermediate/
    feature_experiment_results.csv
    feature_importance_results.csv
    validation_strategy_results.csv
    data_coverage_by_year.csv
```

---

# Methodology Highlights

Implemented:
- retrospective Elo generation,
- rolling historical statistics,
- bookmaker implied probabilities,
- temporal validation,
- feature ablation experiments,
- feature importance analysis,
- leakage benchmarking.

The project explicitly separates:
- realistic pre-match features,
- post-match leakage features.

---

# Development Environment

Containerized workflow using:
- Python
- Pandas
- Scikit-learn
- Docker
- VS Code Dev Containers

Designed for reproducible multi-machine workflows.

---

# Documentation

Detailed methodology:
- `docs/methodology.md`

Experiment analysis:
- `docs/experiment_results.md`

Generated experiment outputs:
- `data/intermediate/*.csv`

---

# Future Work

Potential future directions:
- walk-forward validation,
- rolling-window optimization,
- probability calibration,
- gradient boosting models,
- matchup/fatigue features.