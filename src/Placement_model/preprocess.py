import pandas as pd
from pathlib import Path

# ================= PATHS =================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = RAW_DIR / "placement_train.xlsx"
TEST_PATH = RAW_DIR / "placement_test.xlsx"

TRAIN_OUT = PROCESSED_DIR / "placement_clean_train.csv"
UNLABELED_OUT = PROCESSED_DIR / "placement_unlabeled.csv"
TEST_OUT = PROCESSED_DIR / "placement_clean_test.csv"

TARGET_COL = "Placement Status"

# ================= COLUMNS =================

KEEP_COLUMNS = [
    "CGPA",
    "Speaking Skills",
    "ML Knowledge",
    "Year of Graduation",
    "College Name",
    TARGET_COL
]

DROP_COLUMNS = [
    "First Name", "Email ID", "Quantity", "Price Tier", "Ticket Type",
    "Attendee #", "Group", "Order Type", "Currency", "Total Paid",
    "Fees Paid", "Eventbrite Fees", "Eventbrite Payment Processing",
    "Attendee Status", "How did you come to know about this event?",
    "Specify in \"Others\"", "Designation"
]

# Functions 
def clean_dataframe(df, is_train=True):
    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])
    if is_train:
        return df[KEEP_COLUMNS]
    return df[[col for col in KEEP_COLUMNS if col != TARGET_COL]]


def impute_missing_values(train_df, test_df):
    numeric_cols = ["CGPA", "Speaking Skills", "ML Knowledge"]

    for col in numeric_cols:
        median = train_df[col].median()
        train_df[col].fillna(median, inplace=True)
        test_df[col].fillna(median, inplace=True)

    year_mode = train_df["Year of Graduation"].mode()[0]
    train_df["Year of Graduation"].fillna(year_mode, inplace=True)
    test_df["Year of Graduation"].fillna(year_mode, inplace=True)

    train_df["College Name"].fillna("Unknown", inplace=True)
    test_df["College Name"].fillna("Unknown", inplace=True)

    return train_df, test_df


def encode_year_of_graduation(df):
    mapping = {
        "First year": 1,
        "Second year": 2,
        "Third year": 3,
        "Final year": 4
    }

    df["Year of Graduation"] = (
        df["Year of Graduation"]
        .astype(str)
        .str.strip()
        .map(mapping)
    )

    return df


def encode_target(df):
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df[TARGET_COL] = df[TARGET_COL].map({
        "placed": 1,
        "not placed": 0
    })

    return df


def one_hot_encode(train_df, test_df):
    combined = pd.concat(
        [train_df.drop(columns=[TARGET_COL]), test_df],
        axis=0
    )

    combined_encoded = pd.get_dummies(
        combined,
        columns=["College Name"],
        drop_first=False
    )

    train_encoded = combined_encoded.iloc[:len(train_df)]
    test_encoded = combined_encoded.iloc[len(train_df):]

    train_encoded[TARGET_COL] = train_df[TARGET_COL].values

    return train_encoded, test_encoded


# MAIN 

def run_preprocessing():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Loading raw data
    train_df = pd.read_excel(TRAIN_PATH)
    test_df = pd.read_excel(TEST_PATH)

    # Column selection
    train_df = clean_dataframe(train_df, True)
    test_df = clean_dataframe(test_df, False)

    
    # Encoding categorical fields FIRST
    train_df = encode_year_of_graduation(train_df)
    test_df = encode_year_of_graduation(test_df)

    # Impute after encoding
    train_df, test_df = impute_missing_values(train_df, test_df)

    # Encoding target last
    train_df = encode_target(train_df)

    
    
    # Separating labeled and unlabeled
    labeled_df = train_df[train_df[TARGET_COL].notna()].copy()
    unlabeled_df = train_df[train_df[TARGET_COL].isna()].copy()

    # One-hot encoding (fit on labeled only)
    train_final, test_final = one_hot_encode(labeled_df, test_df)

    # Save o/p
    train_final.to_csv(TRAIN_OUT, index=False)
    test_final.to_csv(TEST_OUT, index=False)
    unlabeled_df.to_csv(UNLABELED_OUT, index=False)

    print(" Preprocessing completed")
    print(f" Labeled training rows   : {len(train_final)}")
    print(f" Unlabeled rows saved to : {UNLABELED_OUT}")
    print(f" Test rows               : {len(test_final)}")




if __name__ == "__main__":
    run_preprocessing()
