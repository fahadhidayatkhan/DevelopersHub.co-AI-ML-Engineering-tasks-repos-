# Task 6: House Price Prediction

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load dataset (replace with your Kaggle file path)
df = pd.read_csv("house_prices.csv")

# 2. Preprocessing
df = df.dropna()  # drop missing values
X = df[["square_feet", "bedrooms", "location"]]  # example features
y = df["price"]

# Encode categorical 'location'
X = pd.get_dummies(X, columns=["location"], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
X[["square_feet", "bedrooms"]] = scaler.fit_transform(X[["square_feet", "bedrooms"]])

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Choose model (Linear Regression or Gradient Boosting)
# model = LinearRegression()
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("MAE:", mae)
print("RMSE:", rmse)

# 7. Visualization
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()