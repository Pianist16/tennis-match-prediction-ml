import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


FEATURE_FILE = "data/intermediate/matches_features_v1.csv"


FEATURE_COLUMNS = [
    "Aces_diff_left_minus_right",
    "Double Faults_diff_left_minus_right",
    "1st serve percentage_diff_left_minus_right",
    "1st serve points won_diff_left_minus_right",
    "2nd serve points won_diff_left_minus_right",
    "Break Points Saved_diff_left_minus_right",
    "1st return points won_diff_left_minus_right",
    "2nd return points won_diff_left_minus_right",
    "Break Points Converted_diff_left_minus_right",
    "Service Points Won_diff_left_minus_right",
    "Return Points Won_diff_left_minus_right",
    "Total Points Won_diff_left_minus_right",
    "Service games won_diff_left_minus_right",
    "Return games won_diff_left_minus_right",
    "Total games won_diff_left_minus_right",
    "Winners_diff_left_minus_right",
    "Unforced errors_diff_left_minus_right",
]

PRE_MATCH_FEATURES = [
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
    "market_prob_left",
    "market_prob_right",
    "market_prob_diff_left_minus_right",
]

POST_MATCH_FEATURES = [
    "Aces_diff_left_minus_right",
    "Double Faults_diff_left_minus_right",
    "1st serve percentage_diff_left_minus_right",
    "1st serve points won_diff_left_minus_right",
    "2nd serve points won_diff_left_minus_right",
    "Break Points Saved_diff_left_minus_right",
    "1st return points won_diff_left_minus_right",
    "2nd return points won_diff_left_minus_right",
    "Break Points Converted_diff_left_minus_right",
    "Service Points Won_diff_left_minus_right",
    "Return Points Won_diff_left_minus_right",
    "Total Points Won_diff_left_minus_right",
    "Service games won_diff_left_minus_right",
    "Return games won_diff_left_minus_right",
    "Total games won_diff_left_minus_right",
    "Winners_diff_left_minus_right",
    "Unforced errors_diff_left_minus_right",
]

def train_and_evaluate(df, feature_columns, model_name):
    model_df = df[feature_columns + ["target_left_win", "date"]].copy()
    model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce")
    model_df = model_df.dropna()
    model_df = model_df.sort_values("date").reset_index(drop=True)

    model_df = model_df[model_df["date"].dt.year >= 2018]

    train_df = model_df[model_df["date"].dt.year <= 2024]
    validation_df = model_df[model_df["date"].dt.year == 2025]
    test_df = model_df[model_df["date"].dt.year == 2026]

    X_train = train_df[feature_columns]
    y_train = train_df["target_left_win"]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df["target_left_win"]

    X_test = test_df[feature_columns]
    y_test = test_df["target_left_win"]

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    logistic_validation_predictions = logistic_model.predict(X_validation)
    logistic_test_predictions = logistic_model.predict(X_test)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    rf_validation_predictions = rf_model.predict(X_validation)
    rf_test_predictions = rf_model.predict(X_test)

    importances = pd.DataFrame({
        "feature": feature_columns,
        "importance": rf_model.feature_importances_
    })

    importances = importances.sort_values(
        "importance",
        ascending=False
    )

    print("\nTop Random Forest Features:")
    print(importances.head(15))

    print(f"\n{model_name}")
    print("Rows used:", len(model_df))
    print("Train rows:", len(train_df))
    print("Validation rows:", len(validation_df))
    print("Test rows:", len(test_df))
    print("Features:", feature_columns)
    print("Logistic Validation Accuracy:", round(accuracy_score(y_validation, logistic_validation_predictions), 4))
    print("Logistic Test Accuracy:", round(accuracy_score(y_test, logistic_test_predictions), 4))
    print("Random Forest Validation Accuracy:", round(accuracy_score(y_validation, rf_validation_predictions), 4))
    print("Random Forest Test Accuracy:", round(accuracy_score(y_test, rf_test_predictions), 4))


def main():
    df = pd.read_csv(FEATURE_FILE)

    train_and_evaluate(
        df,
        PRE_MATCH_FEATURES,
        "Pre-match model: Elo + recent form + rolling stats"
    )

    train_and_evaluate(
        df,
        POST_MATCH_FEATURES,
        "Post-match leakage model: match stats"
    )


if __name__ == "__main__":
    main()