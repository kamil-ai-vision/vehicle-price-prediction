# 🚗 Vehicle Price Prediction

A machine learning project that predicts vehicle prices based on specifications using a powerful **XGBoost** regressor.

---

## 🌟 Features

- ✅ **High R-squared:** Achieves a strong R² score of **0.915** on the validation set.
- 🧠 **Robust Model:** Utilizes **XGBoost**, a state-of-the-art gradient boosting algorithm.
- 🛠️ **Complete Workflow:** Scripts for training (`train.py`) and prediction (`predict.py`).
- 📈 **Detailed Evaluation:** Model performance measured using:
  - **R² Score**
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**

 ---

 ## 🛠️ Setup and Installation

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd vehicle-price-prediction
```

### 2️⃣ Download Dataset
- Dataset: [Download Here](https://drive.google.com/file/d/1DCcHXU6uhXkYds9qlr5qXBbWYreGwq_L/view?usp=sharing)  
- Unzip and place the dataset in the `data/` folder in the project root.

### 3️⃣ Set Up Virtual Environment
```
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Use

### 1️⃣ Train the Model
To train the model and save the final pipeline:  
```bash
python train.py
```

### 2️⃣ Predict a Vehicle's Price
To predict the price of a new vehicle:
1. Open predict.py and modify the sample_vehicle dictionary with your desired features.
2. Run the script:
   ```
   python predict.py
   ```

---

## 📊 Performance Metrics
- **R-squared (R²):** 0.915  
- **Mean Absolute Error (MAE):** $3,750.05  
- **Root Mean Squared Error (RMSE):** $6,202.96  

---

## 👤 Author
Kamil - [https://github.com/kamil-ai-vision](https://github.com/kamil-ai-vision)
