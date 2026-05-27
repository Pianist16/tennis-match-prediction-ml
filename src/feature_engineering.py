import pandas as pd

from elo import add_elo_features
from recent_form import add_recent_form_features
from rolling_stats import (
    add_rolling_stat_features,
    add_rolling_stat_features_vs_similar_elo,
    add_surface_rolling_stat_features,
)

PROCESSED_INPUT = "data/processed/matches_processed.csv"
FEATURE_OUTPUT = "data/intermediate/matches_features_v1.csv"


def add_stat_differentials(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    left_cols = [c for c in df.columns if c.endswith("_left_numeric")]

    for left_col in left_cols:
        right_col = left_col.replace("_left_numeric", "_right_numeric")

        if right_col not in df.columns:
            continue

        base_name = left_col.replace("_left_numeric", "")

        left_num = pd.to_numeric(df[left_col], errors="coerce")
        right_num = pd.to_numeric(df[right_col], errors="coerce")

        if left_num.notna().sum() == 0 and right_num.notna().sum() == 0:
            continue

        df[base_name + "_diff_left_minus_right"] = left_num - right_num

    return df


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["odds_left_numeric"] = pd.to_numeric(df["odds_left"], errors="coerce")
    df["odds_right_numeric"] = pd.to_numeric(df["odds_right"], errors="coerce")

    df["odds_diff_left_minus_right"] = (
        df["odds_left_numeric"] - df["odds_right_numeric"]
    )

    df["implied_prob_left"] = 1 / df["odds_left_numeric"]
    df["implied_prob_right"] = 1 / df["odds_right_numeric"]

    implied_sum = df["implied_prob_left"] + df["implied_prob_right"]

    df["market_prob_left"] = df["implied_prob_left"] / implied_sum
    df["market_prob_right"] = df["implied_prob_right"] / implied_sum

    df["market_prob_diff_left_minus_right"] = (
        df["market_prob_left"] - df["market_prob_right"]
    )

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_elo_features(df)
    df = add_recent_form_features(df)
    df = add_rolling_stat_features(df)
    df = add_surface_rolling_stat_features(df)
    df = add_rolling_stat_features_vs_similar_elo(df)
    df = add_stat_differentials(df)
    df = add_odds_features(df)

    df["target_left_win"] = (df["winner_side"] == "left").astype(int)

    return df


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_INPUT)
    features_df = build_features(df)

    features_df.to_csv(FEATURE_OUTPUT, index=False)

    print("Input rows:", len(df))
    print("Output rows:", len(features_df))
    print("Input columns:", len(df.columns))
    print("Output columns:", len(features_df.columns))
    print("Saved:", FEATURE_OUTPUT)

    diff_cols = [c for c in features_df.columns if c.endswith("_diff_left_minus_right")]
    print("Differential columns created:", len(diff_cols))
    print(diff_cols[:20])