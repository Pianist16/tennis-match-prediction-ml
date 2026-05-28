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
- very high bookmaker odds coverage from 2025 onward (~98%+).

This motivated the primary modeling regime:

- Primary modeling regime: 2018+
- Market-aware modeling becomes substantially more reliable from 2020 onward.


---

# Validation Strategy

Two validation approaches were compared:

## Random Split
Traditional shuffled train/test split.

## Temporal Split
Train:
- 2018–2024

Validation:
- 2025

Test:
- 2026

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

These features produced measurable improvements in both accuracy and ROC AUC in several temporal validation configurations, particularly when combined with Elo, rolling statistics, and market features.

However, they also introduce additional sparsity and variance and require careful historical coverage handling.

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

- Market probabilities are highly predictive and remain the strongest standalone feature group.
- Elo remains one of the strongest non-market features.
- Rolling performance statistics provide measurable incremental predictive value.
- Opponent-strength contextual rolling features add additional predictive signal beyond global rolling averages.
- Surface-specific rolling features produced limited improvement in current form.
- Logistic Regression generally remains more stable than Random Forest under temporal validation.
- Modern-era ATP datasets (2018+) are substantially more complete and suitable for ML experimentation workflows.


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

Current dataset scope:
- ~28k historical ATP matches,
- 2012-2026 coverage,
- ~24k usable post-2018 modeling rows.