import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_model():
    """
    Loads data, trains a new XGBoost model, evaluates it, and saves the final version.
    """
    print("--- Starting model training script ---")

    # --- 1. Load and Preprocess Data ---
    print("Loading and preprocessing data...")
    try:
        # Assuming the dataset is in the 'data' folder
        df = pd.read_csv('data/dataset.csv')
    except FileNotFoundError:
        print("Error: 'data/dataset.csv' not found. Please check the path.")
        return

    # Basic data cleaning and feature engineering
    df.dropna(subset=['price'], inplace=True)
    df = df[df['price'] > 0]
    df['vehicle_age'] = 2025 - df['year']
    for col in ['mileage', 'cylinders']:
        df[col].fillna(df[col].median(), inplace=True)
    for col in ['fuel', 'transmission', 'body', 'doors', 'make', 'model', 'trim', 'exterior_color', 'interior_color', 'drivetrain']:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    df['cylinders'] = df['cylinders'].astype(int)
    df['doors'] = df['doors'].astype(int)
    
    # Simple feature engineering: extract brand from the 'name' column
    df['brand'] = df['name'].apply(lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) > 1 else 'Other')

    df['log_price'] = np.log1p(df['price'])

    # --- 2. Define Features, Target, and Split Data for Evaluation ---
    # Now keeping more features, as recommended
    features_to_drop = ['price', 'log_price', 'year', 'description', 'engine', 'name']
    X = df.drop(columns=features_to_drop, axis=1)
    y = df['log_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Define Preprocessing and Model Pipeline ---
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # Replaced RandomForestRegressor with XGBoost
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', 
                                       n_estimators=1000, 
                                       learning_rate=0.05, 
                                       max_depth=5,
                                       subsample=0.7,
                                       colsample_bytree=0.7,
                                       random_state=42))
    ])

    # --- 4. Train and Evaluate the Model ---
    print("Training model on the training set for evaluation...")
    model_pipeline.fit(X_train, y_train)

    print("Evaluating model performance...")
    y_pred = model_pipeline.predict(X_test)

    # Calculate metrics on original price scale
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

    print("\n--- Model Evaluation Metrics ---")
    print(f"R-squared (R²): {r2:.3f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print("--------------------------------\n")

    # --- 5. Retrain on Full Dataset and Save ---
    print("Retraining model on the entire dataset...")
    model_pipeline.fit(X, y) # Retrain on all data

    print("Saving the final model...")
    os.makedirs('models', exist_ok=True)
    save_path = 'models/vehicle_price_model.joblib'
    joblib.dump(model_pipeline, save_path)

    print(f"--- Script finished. Final model saved to {save_path} ---")

if __name__ == '__main__':
    train_model()