import pandas as pd


ROLLING_WINDOW = 10


ROLLING_STAT_COLUMNS = [
    "Aces",
    "Double Faults",
    "1st serve percentage",
    "1st serve points won",
    "2nd serve points won",
    "Break Points Saved",
    "Break Points Converted",
    "Service Points Won",
    "Return Points Won",
    "Total Points Won",
]


def add_rolling_stat_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.sort_values("date").reset_index(drop=True)

    player_history = {}

    feature_storage = {}

    for stat in ROLLING_STAT_COLUMNS:
        feature_storage[f"rolling_{stat}_left_before"] = []
        feature_storage[f"rolling_{stat}_right_before"] = []
        feature_storage[f"rolling_{stat}_diff_left_minus_right"] = []

    for _, row in df.iterrows():

        left_player = row["player_left"]
        right_player = row["player_right"]

        if left_player not in player_history:
            player_history[left_player] = {}

        if right_player not in player_history:
            player_history[right_player] = {}

        for stat in ROLLING_STAT_COLUMNS:

            left_col = f"{stat}_left_numeric"
            right_col = f"{stat}_right_numeric"

            left_history = player_history[left_player].get(stat, [])
            right_history = player_history[right_player].get(stat, [])

            left_recent = left_history[-ROLLING_WINDOW:]
            right_recent = right_history[-ROLLING_WINDOW:]

            left_avg = (
                sum(left_recent) / len(left_recent)
                if left_recent else None
            )

            right_avg = (
                sum(right_recent) / len(right_recent)
                if right_recent else None
            )

            feature_storage[f"rolling_{stat}_left_before"].append(left_avg)
            feature_storage[f"rolling_{stat}_right_before"].append(right_avg)

            if left_avg is not None and right_avg is not None:
                diff = left_avg - right_avg
            else:
                diff = None

            feature_storage[f"rolling_{stat}_diff_left_minus_right"].append(diff)

            left_value = row.get(left_col)
            right_value = row.get(right_col)

            if pd.notna(left_value):
                player_history[left_player].setdefault(stat, []).append(left_value)

            if pd.notna(right_value):
                player_history[right_player].setdefault(stat, []).append(right_value)

    for key, values in feature_storage.items():
        df[key] = values

    return df