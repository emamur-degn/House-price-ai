from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "house_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


# ==========================================
# LOAD DATA
# ==========================================

print("\n====================================")
print("      HOUSE PRICE AI TRAINING")
print("====================================\n")

print("Loading dataset...")


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\n{DATA_PATH.name} পাওয়া যায়নি!\n"
        "house_data.csv file-টি project folder-এ রাখুন."
    )


df = pd.read_csv(DATA_PATH)


print("Dataset loaded successfully.")
print("Rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# REQUIRED COLUMNS
# ==========================================

features = [
    "bedrooms",
    "bathrooms",
    "floors",
    "year_built",
    "area_sqft"
]

target = "price"


required_columns = features + [target]


# ==========================================
# CHECK COLUMNS
# ==========================================

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nCSV-তে '{column}' column পাওয়া যায়নি.\n"
            f"Required columns: {required_columns}"
        )


# ==========================================
# CLEAN DATA
# ==========================================

df = df[required_columns].copy()

df = df.dropna()


for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna()


# ==========================================
# REMOVE INVALID DATA
# ==========================================

df = df[
    (df["bedrooms"] > 0) &
    (df["bathrooms"] > 0) &
    (df["floors"] > 0) &
    (df["year_built"] >= 1800) &
    (df["area_sqft"] > 0) &
    (df["price"] > 0)
]


print("\nValid rows:", len(df))


if len(df) < 10:

    raise ValueError(
        "Training-এর জন্য কমপক্ষে 10টি valid data দরকার."
    )


# ==========================================
# X AND Y
# ==========================================

X = df[features]

y = df[target]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print("\nTraining rows:", len(X_train))

print("Testing rows:", len(X_test))


# ==========================================
# CREATE MODEL
# ==========================================

print("\nCreating Random Forest model...")


model = RandomForestRegressor(

    n_estimators=200,

    max_depth=15,

    random_state=42,

    n_jobs=-1
)


# ==========================================
# TRAIN
# ==========================================

print("Training model...")

model.fit(
    X_train,
    y_train
)


print("Training completed!")


# ==========================================
# TEST
# ==========================================

predictions = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)


rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5


r2 = r2_score(
    y_test,
    predictions
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_PATH
)


# ==========================================
# RESULT
# ==========================================

print("\n====================================")
print("       MODEL TRAINING COMPLETE")
print("====================================")

print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R2   : {r2:.4f}")

print("\nModel saved successfully!")

print(
    f"Location:\n{MODEL_PATH}"
)

print("====================================\n")