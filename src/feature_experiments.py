import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


FEATURE_FILE = "data/intermediate/matches_features_v1.csv"
RESULTS_FILE = "data/intermediate/feature_experiment_results.csv"
IMPORTANCE_FILE = "data/intermediate/feature_importance_results.csv"


ELO_FEATURES = [
    "elo_diff_left_minus_right",
]

RECENT_FEATURES = [
    "recent_win_rate_diff_left_minus_right",
]

ROLLING_FEATURES = [
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

MARKET_FEATURES = [
    "market_prob_left",
    "market_prob_right",
    "market_prob_diff_left_minus_right",
]

SURFACE_ROLLING_FEATURES = [
    "surface_rolling_Aces_diff_left_minus_right",
    "surface_rolling_Double Faults_diff_left_minus_right",
    "surface_rolling_1st serve percentage_diff_left_minus_right",
    "surface_rolling_1st serve points won_diff_left_minus_right",
    "surface_rolling_2nd serve points won_diff_left_minus_right",
    "surface_rolling_Break Points Saved_diff_left_minus_right",
    "surface_rolling_Break Points Converted_diff_left_minus_right",
    "surface_rolling_Service Points Won_diff_left_minus_right",
    "surface_rolling_Return Points Won_diff_left_minus_right",
    "surface_rolling_Total Points Won_diff_left_minus_right",
]

SIMILAR_ELO_FEATURES = [
    "rolling_similar_elo_Aces_diff_left_minus_right",
    "rolling_similar_elo_Double Faults_diff_left_minus_right",
    "rolling_similar_elo_1st serve percentage_diff_left_minus_right",
    "rolling_similar_elo_1st serve points won_diff_left_minus_right",
    "rolling_similar_elo_2nd serve points won_diff_left_minus_right",
    "rolling_similar_elo_Break Points Saved_diff_left_minus_right",
    "rolling_similar_elo_Break Points Converted_diff_left_minus_right",
    "rolling_similar_elo_Service Points Won_diff_left_minus_right",
    "rolling_similar_elo_Return Points Won_diff_left_minus_right",
    "rolling_similar_elo_Total Points Won_diff_left_minus_right",
]


FEATURE_SETS = {
    "elo_only": ELO_FEATURES,
    "elo_recent": ELO_FEATURES + RECENT_FEATURES,
    "elo_recent_rolling": ELO_FEATURES + RECENT_FEATURES + ROLLING_FEATURES,
    "market_only": MARKET_FEATURES,
    "elo_market": ELO_FEATURES + MARKET_FEATURES,
    "elo_recent_rolling_market": (
        ELO_FEATURES
        + RECENT_FEATURES
        + ROLLING_FEATURES
        + MARKET_FEATURES
    ),
    "elo_recent_rolling_market_surface": (
        ELO_FEATURES
        + RECENT_FEATURES
        + ROLLING_FEATURES
        + MARKET_FEATURES
        + SURFACE_ROLLING_FEATURES
    ),
    "similar_elo_experiment": (
        ELO_FEATURES
        + RECENT_FEATURES
        + ROLLING_FEATURES
        + MARKET_FEATURES
        + SIMILAR_ELO_FEATURES
    ),
}


def prepare_model_df(df, feature_columns):
    model_df = df[feature_columns + ["target_left_win", "date"]].copy()

    model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce")
    model_df = model_df.dropna()
    model_df = model_df.sort_values("date").reset_index(drop=True)

    model_df = model_df[model_df["date"].dt.year >= 2018]

    train_df = model_df[model_df["date"].dt.year <= 2023]
    validation_df = model_df[model_df["date"].dt.year == 2024]
    test_df = model_df[model_df["date"].dt.year == 2025]

    return model_df, train_df, validation_df, test_df


def evaluate_model(model, model_name, feature_set_name, feature_columns, train_df, validation_df, test_df):
    x_train = train_df[feature_columns]
    y_train = train_df["target_left_win"]

    x_validation = validation_df[feature_columns]
    y_validation = validation_df["target_left_win"]

    x_test = test_df[feature_columns]
    y_test = test_df["target_left_win"]

    model.fit(x_train, y_train)

    validation_pred = model.predict(x_validation)
    test_pred = model.predict(x_test)

    validation_proba = model.predict_proba(x_validation)[:, 1]
    test_proba = model.predict_proba(x_test)[:, 1]

    result = {
        "feature_set": feature_set_name,
        "model": model_name,
        "feature_count": len(feature_columns),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "validation_accuracy": accuracy_score(y_validation, validation_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "validation_auc": roc_auc_score(y_validation, validation_proba),
        "test_auc": roc_auc_score(y_test, test_proba),
        "validation_log_loss": log_loss(y_validation, validation_proba),
        "test_log_loss": log_loss(y_test, test_proba),
    }

    importance_rows = []

    if hasattr(model, "feature_importances_"):
        for feature, importance in zip(feature_columns, model.feature_importances_):
            importance_rows.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "feature": feature,
                "importance": importance,
            })

    return result, importance_rows


def main():
    df = pd.read_csv(FEATURE_FILE, low_memory=False)

    results = []
    importance_results = []

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        model_df, train_df, validation_df, test_df = prepare_model_df(
            df,
            feature_columns
        )

        if train_df.empty or validation_df.empty or test_df.empty:
            print(f"Skipping {feature_set_name}: insufficient data")
            continue

        models = {
            "logistic_regression": LogisticRegression(max_iter=3000),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        }

        for model_name, model in models.items():
            result, importances = evaluate_model(
                model=model,
                model_name=model_name,
                feature_set_name=feature_set_name,
                feature_columns=feature_columns,
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
            )

            results.append(result)
            importance_results.extend(importances)

            print(
                feature_set_name,
                model_name,
                "validation_accuracy:",
                round(result["validation_accuracy"], 4),
                "test_accuracy:",
                round(result["test_accuracy"], 4),
                "test_auc:",
                round(result["test_auc"], 4),
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["test_accuracy", "test_auc"],
        ascending=False
    )

    results_df.to_csv(RESULTS_FILE, index=False)

    importance_df = pd.DataFrame(importance_results)
    importance_df.to_csv(IMPORTANCE_FILE, index=False)

    print("\nSaved:", RESULTS_FILE)
    print("Saved:", IMPORTANCE_FILE)
    print("\nTop results:")
    print(results_df.head(10))


if __name__ == "__main__":
    main()