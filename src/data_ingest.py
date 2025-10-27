import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED, DATA_INTERIM

def main():
    train = pd.read_parquet(DATA_RAW / "training-set.parquet")
    test = pd.read_parquet(DATA_RAW / "testing-set.parquet")


    df = pd.concat([train, test], ignore_index=True)

    df.to_csv(DATA_INTERIM / "unsw_full.csv", index=False)
    print(f"Data merged: {df.shape}")

if __name__ == "__main__":
    main()