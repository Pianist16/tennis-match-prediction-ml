# Experiment Results

## Objective

The goal of these experiments was to evaluate:
- feature usefulness,
- validation methodologies,
- feature interactions,
- and model generalization under temporal splits.


---

# Validation Strategy Comparison

Validation strategies compared:
- random train/test split,
- temporal split.

Results:

| Strategy | Model | Accuracy | ROC AUC |
|---|---|---:|---:|
| Random Split | Logistic Regression | ~0.645 | ~0.705 |
| Temporal Split | Logistic Regression | ~0.646 | ~0.701 |
| Random Split | Random Forest | ~0.631 | ~0.690 |
| Temporal Split | Random Forest | ~0.633 | ~0.685 |

Observation:
- random split still produced slightly inflated metrics,
- especially ROC AUC,
- but the gap between random and temporal validation became relatively small after dataset expansion.

This suggests:
- improved temporal generalization,
- relatively stable feature behavior across seasons,
- and limited large-scale leakage in the current pipeline.

Temporal validation was selected as the primary methodology.


---

# Feature Experiments

Feature-set experiments were performed incrementally.

## 1. Standalone Feature Groups
## Elo Only

Result:
- reasonable baseline,
- strong standalone predictive power.

Observation:
- Elo remained one of the strongest non-market features.


---

## 2. Incremental Feature Accumulation
## Elo + Recent Form

Result:
- limited improvement over Elo alone.

Observation:
- recent win rate appeared partially redundant with Elo.


---

## Elo + Rolling Statistics

Result:
- measurable improvement over Elo baseline.

Important rolling features:
- service points won,
- return points won,
- break-point metrics,
- serve percentage metrics.

Observation:
- rolling match-quality indicators provided meaningful additional signal.


---

## Market Features

Bookmaker-derived implied probabilities produced the strongest single feature group.

Result:
- market-only models outperformed most non-market combinations.

Observation:
- bookmaker odds compress substantial real-world information efficiently.


---

## Market + Elo + Rolling

Result:
- small but measurable improvement over market-only baseline.

Observation:
- non-market features still contributed incremental information beyond bookmaker probabilities.


---

## 3. Contextual / Specialized Experiments
## Surface-Specific Rolling Features

Surface-aware rolling features were tested.

Result:
- limited or inconsistent improvement.

Possible reasons:
- feature sparsity,
- insufficient sample size per surface,
- already partially captured by market probabilities and Elo.


---

## Similar-Elo Opponent Experiments

Experimental rolling features based on opponent Elo similarity were evaluated.

Result:
- measurable improvements in both test accuracy and ROC AUC,
- particularly when combined with Elo, rolling statistics, and market probabilities.

Observation:
- opponent-strength contextual rolling features appear to contain meaningful additional predictive signal,
- especially in temporally separated future holdout evaluation.


---

# Feature Importance Analysis

Random Forest importance analysis showed strongest features included:
- Elo differential,
- market probability differential,
- rolling break-point metrics,
- rolling serve metrics,
- rolling total points won.

Observation:
- market probabilities and Elo consistently ranked among the strongest predictors.


---

# Current Benchmark

# Current Benchmark

Current realistic pre-match benchmark:
- temporal validation,
- no intentional leakage,
- rolling historical features,
- bookmaker implied probabilities,
- contextual opponent-strength rolling features.

Approximate results:

| Configuration | Test Accuracy | ROC AUC |
|---|---:|---:|
| Logistic Regression | ~68-71% | ~0.75-0.78 |
| Random Forest | ~67-71% | ~0.73-0.77 |

Best-performing configurations included:
- bookmaker implied probabilities,
- rolling match statistics,
- Elo differentials,
- contextual opponent-strength rolling features.


---

# 4. Final Conclusions

Main conclusions so far:

1. Temporal validation is required for realistic evaluation.
2. Market probabilities contain substantial predictive information.
3. Elo remains a strong foundational feature.
4. Rolling historical statistics improve model quality.
5. Opponent-strength contextual rolling features add measurable predictive signal.
6. Simpler linear models generally remain more stable under temporal validation.
7. Modern-era ATP data (2018+) is substantially more complete and suitable for ML workflows.


---

# Future Work

Potential future directions:
- rolling-window optimization,
- walk-forward cross validation,
- probability calibration,
- gradient boosting models,
- matchup/head-to-head features,
- fatigue and schedule-density features,
- betting-market inefficiency analysis,
- ensemble approaches.