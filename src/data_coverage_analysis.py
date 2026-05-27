import pandas as pd


FEATURE_FILE = "data/intermediate/matches_features_v1.csv"
OUTPUT_FILE = "data/intermediate/data_coverage_by_year.csv"


def main():
    df = pd.read_csv(FEATURE_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    summary = (
        df.groupby("year")
        .agg(
            rows=("date", "size"),
            unique_tournaments=("tournament_slug", "nunique"),
            rows_with_odds=("market_prob_left", lambda x: x.notna().sum()),
            rows_with_elo=("elo_diff_left_minus_right", lambda x: x.notna().sum()),
            rows_with_rolling_stats=("rolling_Total Points Won_diff_left_minus_right", lambda x: x.notna().sum()),
            rows_with_surface=("match_surface", lambda x: x.notna().sum()),
        )
        .reset_index()
    )

    summary["odds_coverage_pct"] = summary["rows_with_odds"] / summary["rows"]
    summary["rolling_stats_coverage_pct"] = summary["rows_with_rolling_stats"] / summary["rows"]
    summary["surface_coverage_pct"] = summary["rows_with_surface"] / summary["rows"]

    summary.to_csv(OUTPUT_FILE, index=False)

    print(summary.to_string(index=False))
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()