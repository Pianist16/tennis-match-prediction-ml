from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("input-data/raw")


def load_raw_match_data(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    csv_files = sorted(raw_data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    frames = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df["source_file"] = file_path.name
        df["tournament_slug"] = file_path.stem
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_raw_match_data()

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("\nFiles loaded:", df["source_file"].nunique())
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nSample:")
    print(df.head())