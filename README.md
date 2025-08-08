# Vehicle Price Prediction 🚗💨

This project uses machine learning to predict the price of vehicles based on their specifications. It includes a complete workflow from data cleaning and feature engineering to model training, evaluation, and prediction.

---

## Model Performance 📈

The final model is a Random Forest Regressor which was evaluated on a held-out test set.

* **R-squared ($R^2$)**: **0.814**
    * This means the model can explain about 81.4% of the variance in vehicle prices, indicating a strong fit.
* **Mean Absolute Error (MAE)**: **~$5,678**
    * On average, the model's price prediction is off by approximately $5,678.

---

## Project Structure

```
├── data/
│   └── dataset.csv         # The raw dataset
├── models/
│   └── (This folder is created by train.py to store the model)
├── .gitignore              # Specifies files for Git to ignore
├── predict.py              # Loads the model to predict the price of a sample vehicle
├── README.md               # This file
├── requirements.txt        # Required Python libraries
└── train.py                # Script to train, evaluate, and save the model
```

---

## How to Run This Project

Follow these steps to run the project locally.

### 1. Setup

First, clone the repository and install the necessary dependencies.

```bash
# Clone the repository
git clone <your-repository-url>

# Navigate into the project directory
cd vehicle-price-prediction

# Install the required libraries
pip install -r requirements.txt
```

### 2. Train the Model

Run the train.py script. This will process the data, train the model, print its evaluation scores, and save the trained model pipeline to the models/ directory.
```
python train.py
```

### 3. Make a Prediction

Run the predict.py script to use the saved model to predict the price of a sample vehicle defined within the script.
```
python predict.py
```
