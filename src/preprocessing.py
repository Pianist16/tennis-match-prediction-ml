import re
import pandas as pd

from data_loader import load_raw_match_data


def percent_to_float(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(r"(\d+)%", text)
    if match:
        return float(match.group(1)) / 100

    return pd.to_numeric(text, errors="coerce")


def fraction_to_rate(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(r"\((\d+)/(\d+)\)", text)
    if match:
        won = float(match.group(1))
        total = float(match.group(2))
        return won / total if total else None

    match = re.search(r"^(\d+)/(\d+)$", text)
    if match:
        won = float(match.group(1))
        total = float(match.group(2))
        return won / total if total else None

    return pd.to_numeric(text, errors="coerce")


def parse_match_result_sets(value):
    if pd.isna(value):
        return None, None

    parts = str(value).replace("\n", "-").split("-")

    if len(parts) != 2:
        return None, None

    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def preprocess_matches(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    left_sets, right_sets = zip(*df["match_result_sets"].apply(parse_match_result_sets))
    df["sets_left"] = left_sets
    df["sets_right"] = right_sets

    df["winner_side"] = df.apply(
        lambda row: "left" if row["sets_left"] > row["sets_right"] else "right",
        axis=1
    )

    df["winner_player"] = df.apply(
        lambda row: row["player_left"] if row["winner_side"] == "left" else row["player_right"],
        axis=1
    )

    df["loser_player"] = df.apply(
        lambda row: row["player_right"] if row["winner_side"] == "left" else row["player_left"],
        axis=1
    )

    for col in df.columns:
        if col.endswith("_left") or col.endswith("_right"):
            if "percentage" in col or "points won" in col or "games won" in col:
                df[col + "_rate"] = df[col].apply(percent_to_float)
            elif "Break Points" in col:
                df[col + "_rate"] = df[col].apply(fraction_to_rate)

    return df


if __name__ == "__main__":
    raw_df = load_raw_match_data()
    processed_df = preprocess_matches(raw_df)

    output_path = "data/processed/matches_processed.csv"
    processed_df.to_csv(output_path, index=False)

    print("Rows:", len(processed_df))
    print("Columns:", len(processed_df.columns))
    print("Saved:", output_path)

    print(processed_df[[
        "date",
        "player_left",
        "player_right",
        "match_result_sets",
        "score_full",
        "winner_side",
        "winner_player",
        "loser_player"
    ]].head(20))