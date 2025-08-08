import pandas as pd
import numpy as np
import joblib
import os

def predict_price(vehicle_data):
    """
    Loads the trained model and predicts the price for a given vehicle's data.
    """
    try:
        # Load the trained model pipeline
        model_path = 'models/vehicle_price_model.joblib'
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        print("Please run train.py first to create and save the model.")
        return

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([vehicle_data])

    # Predict the log price
    log_prediction = model.predict(input_df)

    # Inverse transform the prediction to get the actual price
    prediction = np.expm1(log_prediction)[0]

    return prediction

# --- Main execution block ---
if __name__ == '__main__':
    # Example: Let's predict the price for a new vehicle with all features
    sample_vehicle = {
        'make': 'Jeep',
        'model': 'Wagoneer',
        'trim': 'A-Series II',
        'cylinders': 6,
        'fuel': 'Gasoline',
        'mileage': 10,
        'transmission': 'Automatic',
        'body': 'SUV',
        'doors': 4,
        'drivetrain': 'Four-wheel Drive',
        'exterior_color': 'White',
        'interior_color': 'Global Black',
        'brand': 'Jeep',
        'vehicle_age': 1  # A 1-year-old vehicle (2025 - 2024)
    }

    predicted_price = predict_price(sample_vehicle)

    if predicted_price is not None:
        print("\n--- Vehicle Price Prediction ---")
        print(f"Predicted Price for the vehicle is: ${predicted_price:,.2f}")
        print("---------------------------------")