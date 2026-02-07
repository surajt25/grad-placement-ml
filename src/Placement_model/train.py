import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_OUT = BASE_DIR / "data" / "processed" / "placement_clean_train.csv"
TARGET_COL = "Placement Status"


def train_and_evaluate():
    df = pd.read_csv(TRAIN_OUT)
    print(f"Training data shape: {df.shape}")

    #    Sanity check
    if df[TARGET_COL].isna().any():
        raise ValueError("Training data contains NAN targets. Preprocessingis broken.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    print(
    X.isna()
     .sum()
     .sort_values(ascending=False)
     .head(10)
    )


    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    print("\n =============== Validation Results =============== ")
    print(classification_report(y_val, preds))

    return model



if __name__ == "__main__":
    train_and_evaluate()
