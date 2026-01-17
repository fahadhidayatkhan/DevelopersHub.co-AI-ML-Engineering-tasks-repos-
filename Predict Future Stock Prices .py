import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Load historical stock data (example: Apple)
df = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

# 2. Prepare features and target
X = df[["Open", "High", "Low", "Volume"]]
y = df["Close"].shift(-1).dropna()   # next day's Close
X = X[:-1]                           # align with shifted target

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Choose model (Linear Regression or Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
# model = LinearRegression()  # alternative
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))

# 7. Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual", color="blue")
plt.plot(y_pred, label="Predicted", color="red")
plt.title("Actual vs Predicted Closing Prices")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()