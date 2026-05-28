import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split


FEATURE_FILE = "data/intermediate/matches_features_v1.csv"
OUTPUT_FILE = "data/intermediate/validation_strategy_results.csv"


FEATURE_COLUMNS = [
    "elo_diff_left_minus_right",
    "recent_win_rate_diff_left_minus_right",
    "rolling_Aces_diff_left_minus_right",
    "rolling_Double Faults_diff_left_minus_right",
    "rolling_1st serve percentage_diff_left_minus_right",
    "rolling_1st serve points won_diff_left_minus_right",
    "rolling_2nd serve points won_diff_left_minus_right",
    "rolling_Break Points Saved_diff_left_minus_right",
    "rolling_Break Points Converted_diff_left_minus_right",
    "rolling_Service Points Won_diff_left_minus_right",
    "rolling_Return Points Won_diff_left_minus_right",
    "rolling_Total Points Won_diff_left_minus_right",
]


def evaluate_model(
    strategy_name,
    model_name,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
):
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "strategy": strategy_name,
        "model": model_name,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": accuracy_score(y_test, predictions),
        "auc": roc_auc_score(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities),
    }


def main():
    df = pd.read_csv(FEATURE_FILE, low_memory=False)

    model_df = df[FEATURE_COLUMNS + ["target_left_win", "date"]].copy()

    model_df["date"] = pd.to_datetime(
        model_df["date"],
        errors="coerce"
    )

    model_df = model_df.dropna()
    model_df = model_df.sort_values("date").reset_index(drop=True)

    model_df = model_df[
        model_df["date"].dt.year >= 2018
    ]

    x = model_df[FEATURE_COLUMNS]
    y = model_df["target_left_win"]

    results = []

    models = {
        "logistic_regression": LogisticRegression(max_iter=3000),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    #
    # RANDOM SPLIT
    #

    x_train_random, x_test_random, y_train_random, y_test_random = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    for model_name, model in models.items():
        result = evaluate_model(
            strategy_name="random_split",
            model_name=model_name,
            model=model,
            x_train=x_train_random,
            y_train=y_train_random,
            x_test=x_test_random,
            y_test=y_test_random,
        )

        results.append(result)

    #
    # TEMPORAL SPLIT
    #

    train_df = model_df[
        model_df["date"].dt.year <= 2025
    ]

    test_df = model_df[
        model_df["date"].dt.year == 2026
    ]

    x_train_temporal = train_df[FEATURE_COLUMNS]
    y_train_temporal = train_df["target_left_win"]

    x_test_temporal = test_df[FEATURE_COLUMNS]
    y_test_temporal = test_df["target_left_win"]

    for model_name, model in models.items():
        result = evaluate_model(
            strategy_name="temporal_split",
            model_name=model_name,
            model=model,
            x_train=x_train_temporal,
            y_train=y_train_temporal,
            x_test=x_test_temporal,
            y_test=y_test_temporal,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(results_df.to_string(index=False))
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()