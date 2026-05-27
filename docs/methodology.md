# Methodology

## Objective

The objective of this project is to build a realistic pre-match ATP tennis prediction pipeline using historical match data, rolling player statistics, Elo ratings, and bookmaker odds.

The project focuses primarily on:
- data engineering,
- temporal feature engineering,
- leakage prevention,
- validation methodology,
- feature experimentation,
- and reproducible analytical workflows.

The goal is not to create a perfect prediction model, but to design and evaluate a structured ML pipeline under real-world data constraints.


---

# Data Sources

Primary sources:
- ATP match/tournament data
- Flashscore match statistics
- Bookmaker odds

Scraped data includes:
- match results,
- match timestamps,
- surfaces,
- player names,
- tournament metadata,
- detailed match statistics,
- bookmaker odds.


---

# Data Processing Pipeline

Pipeline stages:

1. Web-scraping pipeline (Flashscore / ATP data ingestion)
2. Raw dataset generation and normalization
3. Retrospective Elo generation
4. Rolling-stat feature engineering
5. Differential feature construction
6. Temporal train/validation/test splitting
7. Feature experimentation and model evaluation


---

# Temporal Ordering

All feature generation is performed chronologically.

Rolling features are generated using only information available before each match.

This prevents future leakage into historical predictions.


---

# Leakage Prevention

The project explicitly separates:

## Pre-match prediction features
Examples:
- Elo difference
- Recent win rate
- Rolling serve statistics
- Bookmaker probabilities

## Post-match leakage features
Examples:
- Total points won
- Service games won
- Winners / unforced errors

Post-match features are evaluated separately only as leakage/sanity-check experiments.


---

# Dataset Coverage Analysis

Coverage analysis was performed across years to evaluate:
- match counts,
- odds availability,
- rolling-stat coverage,
- surface coverage.

Results showed:
- sparse odds coverage before 2020,
- lower match volume before 2018,
- significantly improved consistency from 2018 onward.

This motivated the primary modeling regime:

- Training data focus: 2018+
- Market-aware models: 2020+ preferred


---

# Validation Strategy

Two validation approaches were compared:

## Random Split
Traditional shuffled train/test split.

## Temporal Split
Train:
- 2018–2023

Validation:
- 2024

Test:
- 2025

Temporal validation was selected as the preferred methodology because it better reflects real-world forecasting conditions and avoids future information leakage.


---

# Feature Engineering

Implemented feature groups include:

## Elo Features
- Elo difference

## Recent Form Features
- Recent win rate

## Rolling Match Statistics
Examples:
- rolling aces,
- rolling serve percentage,
- rolling service points won,
- rolling return points won.

## Market Features
Bookmaker-derived implied probabilities:
- market probability left,
- market probability right,
- probability differential.

## Experimental Features
- surface-specific rolling stats,
- opponent-strength rolling experiments.

### Similar-Opponent-Strength Features

Experimental rolling features were generated using historical performance against opponents with similar pre-match Elo strength.

The goal was to evaluate whether player performance contextualized by opponent quality provides additional predictive signal beyond global rolling averages.

These features showed promising ROC AUC improvements in some configurations, but also introduced additional sparsity and variance.

---

# Feature Experimentation

Feature-set ablation experiments were performed to evaluate:
- isolated feature groups,
- combined feature groups,
- incremental predictive contribution.

Results are stored in:
- `data/intermediate/feature_experiment_results.csv`
- `data/intermediate/feature_importance_results.csv`

Feature experimentation was evaluated incrementally using standalone and combined feature-group experiments.

---

# Models

Current benchmark models:
- Logistic Regression
- Random Forest

Evaluation metrics:
- accuracy,
- ROC AUC,
- log loss.


---

# Current Findings

Main observations so far:

- Market probabilities are highly predictive.
- Elo remains one of the strongest non-market features.
- Rolling performance statistics improve predictive power.
- Surface-specific rolling features produced limited improvement in current form.
- Logistic Regression generalizes more consistently than Random Forest on temporal test data.


---

# Current Scope

The project currently focuses on:
- tabular ML,
- structured feature engineering,
- temporal validation,
- analytical experimentation,
- and reproducible workflows.

The current implementation intentionally prioritizes:
- methodological clarity,
- realistic evaluation,
- and interpretability
over complex model architectures.