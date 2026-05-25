# Tennis Match Prediction ML

Machine learning project for predicting professional ATP tennis match outcomes using:

* Retrospective Elo ratings
* Rolling player performance statistics
* Historical recent form
* Time-aware validation
* Flashscore-based custom data scraping pipeline

The project is built with a fully containerized Python development workflow using VS Code Dev Containers and Docker.

---

# Project Goals

The main goal of this project is to build a realistic pre-match tennis prediction system using only information available before the match starts.

The project focuses heavily on:

* Proper chronological feature engineering
* Avoiding data leakage
* Reproducible ML pipelines
* Realistic sports analytics methodology
* Modular architecture for future expansion

---

# Current Features

## Retrospective Elo Ratings

The project calculates chronological Elo ratings:

* Players start from an initial rating
* Matches are processed from oldest to newest
* Elo ratings are updated after every match
* Pre-match Elo values are stored as predictive features

Generated features:

* `elo_left_before`
* `elo_right_before`
* `elo_diff_left_minus_right`

---

## Recent Form Features

Rolling recent-form features are calculated using only historical matches.

Example:

* Last 10 matches win rate
* Historical rolling averages
* Pre-match only information

Generated features:

* `left_recent_win_rate_before`
* `right_recent_win_rate_before`
* `recent_win_rate_diff_left_minus_right`

---

## Rolling Statistical Features

The project calculates rolling historical player statistics over previous matches.

Examples:

* Aces
* Double faults
* First serve percentage
* Break point conversion
* Service points won
* Return points won
* Total points won

Rolling averages are calculated retrospectively for each player before every match.

Generated features:

* `rolling_Aces_diff_left_minus_right`
* `rolling_Service Points Won_diff_left_minus_right`
* `rolling_Return Points Won_diff_left_minus_right`
* etc.

---

# Realistic vs Leakage Models

The project intentionally separates:

## Realistic Pre-Match Features

These are features available before the match starts:

* Elo
* Recent form
* Rolling historical statistics

## Post-Match Leakage Features

These use statistics generated during the match itself:

* Match aces
* Match serve percentages
* Match total points won

These features are used only as a benchmark/sanity check.

---

# Current Results

## Pre-Match Model

Features:

* Elo difference
* Recent form
* Rolling historical statistics

Validation:

* Chronological train/test split
* Time-aware evaluation

Current performance:

* Logistic Regression: ~66%
* Random Forest: ~63%

---

## Leakage Benchmark Model

Uses post-match statistics for comparison.

Current performance:

* Logistic Regression: ~92%
* Random Forest: ~92%

This confirms the pipeline works correctly while also demonstrating the importance of avoiding data leakage.

---

# Project Structure

```text
input-data/raw/
    Raw scraped tournament match data

src/
    data_loader.py
    preprocessing.py
    feature_engineering.py
    elo.py
    recent_form.py
    rolling_stats.py
    model_training.py

data/processed/
    Cleaned processed datasets

data/intermediate/
    Feature-engineered datasets

trained-models/
    Saved trained ML models

.devcontainer/
    VS Code Dev Container configuration
```

---

# Technology Stack

* Python
* Pandas
* Scikit-learn
* Docker
* VS Code Dev Containers
* Git/GitHub

---

# Data Pipeline

```text
Flashscore scraping
    ↓
Raw CSV match data
    ↓
Preprocessing
    ↓
Retrospective feature engineering
    ↓
Time-aware train/test split
    ↓
Machine learning models
```

---

# Development Environment

The project uses a containerized development workflow.

Requirements:

* Docker
* VS Code
* Dev Containers extension

The environment is fully reproducible across:

* Ubuntu
* Windows
* Multiple systems/workstations

---

# Future Improvements

Planned future work includes:

* Surface-specific Elo ratings
* Surface-specific rolling statistics
* ATP ranking integration
* Betting odds integration
* Opponent-strength adjustment
* Time-decay weighting
* Walk-forward validation
* Probability calibration
* Gradient boosting models
* Neural network experimentation
* Expanded historical dataset coverage

---

# Status

Active development.

The project currently focuses on ATP singles match prediction and feature engineering research.
