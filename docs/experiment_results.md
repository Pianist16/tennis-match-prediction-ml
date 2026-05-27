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
| Random Split | Logistic Regression | ~0.637 | ~0.701 |
| Temporal Split | Logistic Regression | ~0.637 | ~0.687 |
| Random Split | Random Forest | ~0.628 | ~0.686 |
| Temporal Split | Random Forest | ~0.622 | ~0.675 |

Observation:
- random split slightly inflated metrics,
- especially ROC AUC,
- due to future information leakage across eras.

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
- mixed outcomes,
- but promising ROC AUC behavior in some configurations.

Observation:
- opponent-quality-aware features may contain additional signal,
- but require further refinement.


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

Current realistic pre-match benchmark:
- temporal validation,
- no intentional leakage,
- rolling historical features,
- market-aware features.

Approximate results:
- Logistic Regression test accuracy: ~67%
- ROC AUC: ~0.73–0.76 depending on feature configuration


---

# 4. Final Conclusions

Main conclusions so far:

1. Temporal validation is required for realistic evaluation.
2. Market probabilities contain substantial predictive information.
3. Elo remains a strong foundational feature.
4. Rolling historical statistics improve model quality.
5. Simpler linear models currently generalize more consistently than Random Forest models.
6. Experimental contextual features require careful sparsity management.


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