import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump
from src.config import DATA_RAW, DATA_PROCESSED, MODELS
from pathlib import Path

def main():
    # güvenlik: klasörler varsa oluştur
    Path(MODELS).mkdir(parents=True, exist_ok=True)
    Path(DATA_PROCESSED).mkdir(parents=True, exist_ok=True)

    # parquet dosyalarını oku
    train = pd.read_parquet(DATA_RAW / "training-set.parquet")
    test = pd.read_parquet(DATA_RAW / "testing-set.parquet")

    # hedef (label)
    y_train = train["label"]
    y_test = test["label"]

    # attack_cat çıkar (cevabı içeriyor)
    X_train = train.drop(columns=["label", "attack_cat"])
    X_test = test.drop(columns=["label", "attack_cat"])

    # tüm sayısal tipleri yakala
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    print(f"Numeric cols: {len(num_cols)}, Categorical cols: {len(cat_cols)}")
    print("Categorical columns:", cat_cols)

    # preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    # fit-transform
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # kaydet
    dump(preprocessor, MODELS / "preprocess_ct.joblib")

    # dönüştürülmüş veriyi kaydet
    pd.DataFrame.sparse.from_spmatrix(X_train_t).to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    pd.DataFrame.sparse.from_spmatrix(X_test_t).to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)

    print("✅ Preprocessing complete")
    print("X_train:", X_train_t.shape, " X_test:", X_test_t.shape)

if __name__ == "__main__":
    main()
