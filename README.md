# Laptop-Price-Prediction-using-Random-Forest-Regression


## Overview

This project implements a machine learning model for predicting laptop prices based on hardware specifications and product features. The system utilizes a Random Forest Regressor to learn relationships between laptop configurations and market prices, enabling accurate price estimation for unseen devices.

The project covers the complete machine learning pipeline, including data preprocessing, feature engineering, model training, evaluation, and prediction.

---

## Objectives

- Predict laptop prices using machine learning techniques.
- Analyze the impact of hardware specifications on pricing.
- Build an end-to-end regression pipeline.
- Compare actual and predicted laptop prices.
- Demonstrate practical applications of supervised learning in price forecasting.

---

## Dataset

The dataset contains laptop specifications and pricing information.

Typical features include:

| Feature |
|----------|
| Brand |
| Laptop Type |
| RAM |
| Processor |
| Storage Capacity |
| Storage Type |
| GPU |
| Screen Size |
| Operating System |
| Weight |
| Resolution |
| Price |

The target variable is:

```text
Price
```

---

## Machine Learning Pipeline

### 1. Data Collection

The dataset is loaded into a Pandas DataFrame for analysis and preprocessing.

```python
import pandas as pd

df = pd.read_csv("laptop_data.csv")
```

---

### 2. Data Cleaning

Data preprocessing includes:

- Handling missing values
- Removing unnecessary columns
- Standardizing categorical features
- Preparing data for model training

---

### 3. Feature Engineering

Additional features are generated from raw specifications to improve model performance.

Examples include:

- Screen resolution processing
- Processor categorization
- Memory and storage transformations
- Hardware specification encoding

These engineered features help the model better understand relationships between laptop components and pricing.

---

### 4. Data Encoding

Categorical variables are transformed into numerical representations using encoding techniques.

Examples:

- Brand encoding
- Processor encoding
- Operating system encoding
- Storage type encoding

---

### 5. Train-Test Split

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 6. Model Training

A Random Forest Regressor is used for training.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
```

Why Random Forest?

- Handles nonlinear relationships effectively
- Robust against overfitting
- Works well with mixed feature types
- Provides strong predictive performance

---

### 7. Model Evaluation

The model is evaluated using regression metrics.

Common evaluation metrics include:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Example:

```python
from sklearn.metrics import r2_score

score = r2_score(y_test, predictions)
```

---

## Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-Learn

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Development Environment

- Jupyter Notebook
- Google Colab

---

## Project Structure

```text
.
├── RandomForestRegressor_laptop_price_predictor.ipynb
├── laptop_data.csv
├── model.pkl
├── preprocessing.pkl
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/laptop-price-prediction.git

cd laptop-price-prediction
```

### Install Dependencies

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

Or install all at once:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the notebook:

```bash
jupyter notebook RandomForestRegressor_laptop_price_predictor.ipynb
```

Execute all cells to:

1. Load data
2. Preprocess features
3. Train model
4. Evaluate performance
5. Generate predictions

---

## Making Predictions

Example usage:

```python
predicted_price = model.predict(input_features)
```

Sample workflow:

```python
laptop_specs = {
    "Brand": "Dell",
    "RAM": 16,
    "SSD": 512,
    "Processor": "Intel Core i7"
}

price = model.predict(processed_input)
```

Output:

```text
Predicted Laptop Price: ₹75,000
```

---

## Applications

### E-Commerce Platforms

Estimate fair market prices for laptops.

### Retail Analytics

Support pricing and inventory decisions.

### Consumer Decision Support

Help users evaluate laptop value before purchasing.

### Market Research

Analyze relationships between specifications and pricing trends.

---

## Model Advantages

### Random Forest Benefits

- High prediction accuracy
- Handles complex feature interactions
- Less sensitive to outliers
- Minimal feature scaling requirements
- Strong generalization performance

---

## Future Improvements

- Hyperparameter optimization using GridSearchCV.
- Incorporate additional laptop specifications.
- Deploy the model as a web application.
- Add real-time prediction APIs.
- Compare performance with XGBoost and Gradient Boosting models.
- Create an interactive dashboard for predictions.
- Support automated dataset updates.

---

## Results

The trained Random Forest model successfully learns relationships between laptop specifications and market prices, providing reliable price predictions for unseen devices.

The project demonstrates the effectiveness of ensemble learning methods for regression-based price forecasting problems.

---

## License

This project is licensed under the MIT License.

---

## Author

Developed as a machine learning project for laptop price prediction using Random Forest Regression and feature engineering techniques.
