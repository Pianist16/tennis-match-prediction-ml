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
    model_df = model_df.dropna()
    model_df = model_df.sort_values("date").reset_index(drop=True)

    X = model_df[feature_columns]
    y = model_df["target_left_win"]

    split_index = int(len(model_df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    print(f"\n{model_name}")
    print("Rows used:", len(model_df))
    print("Features:", feature_columns)
    print("Logistic Regression Accuracy:", round(logistic_accuracy, 4))
    print("Random Forest Accuracy:", round(rf_accuracy, 4))


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