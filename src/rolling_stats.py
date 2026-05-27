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


def add_rolling_stat_features_vs_similar_elo(
    df: pd.DataFrame,
    elo_tolerance: float = 150,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    player_history = {}

    feature_storage = {}

    for stat in ROLLING_STAT_COLUMNS:
        feature_storage[f"rolling_similar_elo_{stat}_left_before"] = []
        feature_storage[f"rolling_similar_elo_{stat}_right_before"] = []
        feature_storage[f"rolling_similar_elo_{stat}_diff_left_minus_right"] = []

    for _, row in df.iterrows():
        left_player = row["player_left"]
        right_player = row["player_right"]

        left_current_elo = row["elo_left_before"]
        right_current_elo = row["elo_right_before"]

        left_history = player_history.get(left_player, [])
        right_history = player_history.get(right_player, [])

        for stat in ROLLING_STAT_COLUMNS:
            left_values = [
                item[stat]
                for item in left_history
                if abs(item["opponent_elo"] - right_current_elo) <= elo_tolerance
                and item[stat] is not None
            ]

            right_values = [
                item[stat]
                for item in right_history
                if abs(item["opponent_elo"] - left_current_elo) <= elo_tolerance
                and item[stat] is not None
            ]

            left_avg = (
                sum(left_values[-ROLLING_WINDOW:]) / len(left_values[-ROLLING_WINDOW:])
                if left_values
                else None
            )

            right_avg = (
                sum(right_values[-ROLLING_WINDOW:]) / len(right_values[-ROLLING_WINDOW:])
                if right_values
                else None
            )

            feature_storage[f"rolling_similar_elo_{stat}_left_before"].append(left_avg)
            feature_storage[f"rolling_similar_elo_{stat}_right_before"].append(right_avg)

            if left_avg is not None and right_avg is not None:
                diff = left_avg - right_avg
            else:
                diff = None

            feature_storage[f"rolling_similar_elo_{stat}_diff_left_minus_right"].append(diff)

        left_record = {"opponent_elo": right_current_elo}
        right_record = {"opponent_elo": left_current_elo}

        for stat in ROLLING_STAT_COLUMNS:
            left_col = f"{stat}_left_numeric"
            right_col = f"{stat}_right_numeric"

            left_record[stat] = pd.to_numeric(row.get(left_col), errors="coerce")
            right_record[stat] = pd.to_numeric(row.get(right_col), errors="coerce")

        player_history.setdefault(left_player, []).append(left_record)
        player_history.setdefault(right_player, []).append(right_record)

    for col, values in feature_storage.items():
        df[col] = values

    return df

def add_surface_rolling_stat_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    player_surface_history = {}
    feature_storage = {}

    for stat in ROLLING_STAT_COLUMNS:
        feature_storage[f"surface_rolling_{stat}_left_before"] = []
        feature_storage[f"surface_rolling_{stat}_right_before"] = []
        feature_storage[f"surface_rolling_{stat}_diff_left_minus_right"] = []

    for _, row in df.iterrows():
        left_player = row["player_left"]
        right_player = row["player_right"]
        surface = row.get("match_surface", "")

        left_key = (left_player, surface)
        right_key = (right_player, surface)

        left_history = player_surface_history.get(left_key, {})
        right_history = player_surface_history.get(right_key, {})

        for stat in ROLLING_STAT_COLUMNS:
            left_values = left_history.get(stat, [])[-ROLLING_WINDOW:]
            right_values = right_history.get(stat, [])[-ROLLING_WINDOW:]

            left_avg = sum(left_values) / len(left_values) if left_values else None
            right_avg = sum(right_values) / len(right_values) if right_values else None

            feature_storage[f"surface_rolling_{stat}_left_before"].append(left_avg)
            feature_storage[f"surface_rolling_{stat}_right_before"].append(right_avg)

            diff = left_avg - right_avg if left_avg is not None and right_avg is not None else None
            feature_storage[f"surface_rolling_{stat}_diff_left_minus_right"].append(diff)

        for player, key, side in [
            (left_player, left_key, "left"),
            (right_player, right_key, "right"),
        ]:
            if key not in player_surface_history:
                player_surface_history[key] = {}

            for stat in ROLLING_STAT_COLUMNS:
                col = f"{stat}_{side}_numeric"
                value = pd.to_numeric(row.get(col), errors="coerce")

                if pd.notna(value):
                    player_surface_history[key].setdefault(stat, []).append(value)

    for col, values in feature_storage.items():
        df[col] = values

    return df