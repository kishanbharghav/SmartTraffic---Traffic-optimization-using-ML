# models/trainer.py

import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Import the different regressor models ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# --- 1. SETUP PATHS AND LOAD DATA ---
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "synthetic_traffic_dataset_6000.csv"
MODEL_DIR = ROOT / "models"

# Ensure the models directory exists
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
print("✅ Data loaded successfully.")

# --- 2. FEATURE ENGINEERING & DATA SPLITTING ---
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

TARGET_COLUMN = 'vehicle_count'
FEATURES_TO_USE = [
    'avg_speed_kmph', 'queue_length_m', 'avg_wait_time_sec', 'hour', 'minute',
    'is_peak', 'is_holiday', 'pedestrian_count', 'occupancy_percent', 'signal_phase',
    'signal_phase_duration_sec', 'travel_time_sec', 'incident_flag', 'weather',
    'road_capacity_veh_per_hour', 'day_of_week', 'incident_type'
]

X = df[FEATURES_TO_USE]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows).")

# --- 3. DEFINE MODELS AND PREPROCESSOR ---
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# ✅ NEW: Define all the models you want to train
models_to_train = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Ridge Regression": Ridge(random_state=42)
}

# --- 4. TRAIN AND SAVE A PIPELINE FOR EACH MODEL ---
print("\n--- Starting Model Training ---")
for name, model in models_to_train.items():
    print(f"🚀 Training {name}...")
    
    # Create a full pipeline with the preprocessor and the specific model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Save the trained pipeline to a file
    file_name = f"{name.replace(' ', '_')}_pipeline.pkl"
    joblib.dump(pipeline, MODEL_DIR / file_name)
    
    print(f"✅ Saved {name} pipeline to {file_name}\n")

print("--- All models trained successfully! ---")