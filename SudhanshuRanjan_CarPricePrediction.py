from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cars_ds_final_2021.csv"
MODEL_PATH = BASE_DIR / "car_price_model.joblib"

TARGET = "Ex-Showroom_Price"
FEATURES = [
    "Make", "Model", "Variant", "Displacement", "Cylinders", "Fuel_Tank_Capacity",
    "Fuel_Type", "Height", "Length", "Width", "Body_Type", "Doors",
    "ARAI_Certified_Mileage", "Kerb_Weight", "Gears", "Ground_Clearance",
    "Front_Brakes", "Rear_Brakes", "Power", "Torque", "Seating_Capacity",
    "Type", "Start_/_Stop_Button", "Airbags", "Fasten_Seat_Belt_Warning",
    "Gear_Shift_Reminder",
]
UNIT_COLUMNS = {
    "Displacement": r"[^0-9.]", "Fuel_Tank_Capacity": r"[^0-9.]", "Height": r"[^0-9.]",
    "Length": r"[^0-9.]", "Width": r"[^0-9.]", "Kerb_Weight": r"[^0-9.]",
    "Ground_Clearance": r"[^0-9.]", "Seating_Capacity": r"[^0-9.]",
}


def _number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def load_clean_data(path=DATA_PATH):
    raw = pd.read_csv(path)
    available = [column for column in FEATURES + [TARGET] if column in raw.columns]
    data = raw[available].copy()
    data[TARGET] = data[TARGET].map(_number)
    data = data.dropna(subset=[TARGET])
    for column, pattern in UNIT_COLUMNS.items():
        if column in data:
            data[column] = data[column].map(_number)
    for column in ["Doors", "Gears", "Cylinders"]:
        if column in data:
            data[column] = data[column].map(_number)
    for column in data.columns:
        if data[column].dtype == object:
            data[column] = data[column].replace({"?": np.nan, "": np.nan}).astype(object)
    return data


def build_model(data):
    features = [column for column in FEATURES if column in data.columns]
    X, y = data[features], data[TARGET]
    numeric = X.select_dtypes(exclude=["object", "string"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, max_depth=18)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions),
    }
    return pipeline, metrics, (X_test, y_test, predictions), features


def train_and_save():
    data = load_clean_data()
    pipeline, metrics, _, features = build_model(data)
    joblib.dump({"pipeline": pipeline, "features": features, "metrics": metrics}, MODEL_PATH)
    return data, metrics


def predict_one(values):
    artifact = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{feature: values.get(feature) for feature in artifact["features"]}])
    return float(artifact["pipeline"].predict(row)[0])


if __name__ == "__main__":
    data, metrics = train_and_save()
    print(f"Rows: {len(data)}")
    print(metrics)
    print(f"Saved model: {MODEL_PATH}")