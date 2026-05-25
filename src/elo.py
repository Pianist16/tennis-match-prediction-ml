import pandas as pd


ELO_INITIAL = 1500

K_LOW = 40
K_MID = 20
K_HIGH = 10

ELO_LOW = 2100
ELO_MID = 2400


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def determine_k_factor(elo):
    if elo < ELO_LOW:
        return K_LOW
    elif elo < ELO_MID:
        return K_MID
    else:
        return K_HIGH


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.sort_values("date").reset_index(drop=True)

    elo_ratings = {}

    elo_left_before = []
    elo_right_before = []

    elo_left_after = []
    elo_right_after = []

    for _, row in df.iterrows():

        left_player = row["player_left"]
        right_player = row["player_right"]

        left_elo = elo_ratings.get(left_player, ELO_INITIAL)
        right_elo = elo_ratings.get(right_player, ELO_INITIAL)

        elo_left_before.append(left_elo)
        elo_right_before.append(right_elo)

        expected_left = expected_score(left_elo, right_elo)
        expected_right = expected_score(right_elo, left_elo)

        k_left = determine_k_factor(left_elo)
        k_right = determine_k_factor(right_elo)

        if row["winner_side"] == "left":
            left_result = 1
            right_result = 0
        else:
            left_result = 0
            right_result = 1

        left_new_elo = left_elo + k_left * (left_result - expected_left)
        right_new_elo = right_elo + k_right * (right_result - expected_right)

        elo_ratings[left_player] = left_new_elo
        elo_ratings[right_player] = right_new_elo

        elo_left_after.append(left_new_elo)
        elo_right_after.append(right_new_elo)

    df["elo_left_before"] = elo_left_before
    df["elo_right_before"] = elo_right_before

    df["elo_left_after"] = elo_left_after
    df["elo_right_after"] = elo_right_after

    df["elo_diff_left_minus_right"] = (
        df["elo_left_before"] - df["elo_right_before"]
    )

    return df