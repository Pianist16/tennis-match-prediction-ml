import pandas as pd


DEFAULT_RECENT_WIN_RATE = 0.5
ROLLING_WINDOW = 10


def add_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    player_results = {}

    left_rates = []
    right_rates = []

    for _, row in df.iterrows():
        left_player = row["player_left"]
        right_player = row["player_right"]

        left_history = player_results.get(left_player, [])
        right_history = player_results.get(right_player, [])

        left_recent = left_history[-ROLLING_WINDOW:]
        right_recent = right_history[-ROLLING_WINDOW:]

        left_rate = (
            sum(left_recent) / len(left_recent)
            if left_recent
            else DEFAULT_RECENT_WIN_RATE
        )

        right_rate = (
            sum(right_recent) / len(right_recent)
            if right_recent
            else DEFAULT_RECENT_WIN_RATE
        )

        left_rates.append(left_rate)
        right_rates.append(right_rate)

        if row["winner_side"] == "left":
            player_results.setdefault(left_player, []).append(1)
            player_results.setdefault(right_player, []).append(0)
        else:
            player_results.setdefault(left_player, []).append(0)
            player_results.setdefault(right_player, []).append(1)

    df["left_recent_win_rate_before"] = left_rates
    df["right_recent_win_rate_before"] = right_rates
    df["recent_win_rate_diff_left_minus_right"] = (
        df["left_recent_win_rate_before"] - df["right_recent_win_rate_before"]
    )

    return df