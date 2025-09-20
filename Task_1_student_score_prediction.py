import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset
data = pd.read_csv("StudentPerformanceFactors.csv")
print(data.head())

# Check the missing values
print(data.isnull().sum())

# Handle missing values
data['Teacher_Quality'] = data['Teacher_Quality'].fillna(data['Teacher_Quality'].mode()[0])
data['Parental_Education_Level'] = data['Parental_Education_Level'].fillna(data['Parental_Education_Level'].mode()[0])
data['Distance_from_Home'] = data['Distance_from_Home'].fillna(data['Distance_from_Home'].mode()[0])

# Check again to confirm
print(data.isnull().sum())

# Encode categorical columns automatically
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Features and Target
X = data.drop('Exam_Score', axis=1)
y = data['Exam_Score']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- LINEAR REGRESSION ----------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# ---------------- POLYNOMIAL REGRESSION ----------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

# ---------------- METRICS ----------------
# Linear Regression metrics
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# Polynomial Regression metrics
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("Linear Regression -> MSE:", mse_lin, " R2:", r2_lin)
print("Polynomial Regression -> MSE:", mse_poly, " R2:", r2_poly)

# ---------------- COMPARISON TABLE ----------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression"],
    "MSE": [mse_lin, mse_poly],
    "R2 Score": [r2_lin, r2_poly]
})

print("\nComparison Table:")
print(results)

# --------------- PLOT COMPARISON -----------------
plt.figure(figsize=(12,5))

# Linear Regression
plt.subplot(1,2,1)
plt.scatter(y_test, y_pred_lin, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression: Actual vs Predicted")

# Polynomial Regression
plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_poly, color="purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial Regression: Actual vs Predicted")

plt.show()
